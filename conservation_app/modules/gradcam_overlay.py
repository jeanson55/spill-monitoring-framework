"""Grad-CAM utilities for the project's Ultralytics YOLO spill detector.

The map is a qualitative visualization of detector class-score sensitivity. It
is not a segmentation mask, probability map, or physical spill boundary.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


def generate_yolo_gradcam(
    yolo_model: Any,
    frame_bgr: np.ndarray,
    *,
    image_size: int = 640,
    alpha: float = 0.38,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Return a BGR Grad-CAM overlay and metadata for an Ultralytics YOLO model."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Grad-CAM requires a decoded color image")

    detector = getattr(yolo_model, "model", None)
    if detector is None or not hasattr(detector, "model"):
        raise TypeError("Expected an Ultralytics YOLO detection model")
    layers = detector.model
    head = layers[-1]
    source_indices = getattr(head, "f", None)
    target_index = source_indices[-1] if isinstance(source_indices, (list, tuple)) else len(layers) - 2
    target_layer = layers[target_index]

    height, width = frame_bgr.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (image_size - resized_width) / 2
    pad_y = (image_size - resized_height) / 2
    left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    right, bottom = image_size - resized_width - left, image_size - resized_height - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    parameter = next(detector.parameters())
    tensor = (tensor.unsqueeze(0).to(device=parameter.device, dtype=torch.float32) / 255.0).requires_grad_(True)

    captured: dict[str, torch.Tensor] = {}

    def save_activation(_module, _inputs, output):
        activation = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(activation):
            captured["activation"] = activation

    hook = target_layer.register_forward_hook(save_activation)
    was_training = detector.training
    was_head_training = head.training
    detector.eval()
    # The YOLO eval head decodes predictions using a detached/inference path.
    # Its training branch exposes differentiable class scores and is used here
    # solely to compute an explanation from the same trained weights.
    head.train()
    try:
        with torch.enable_grad():
            # Ultralytics DetectionModel.forward is inference-mode decorated in
            # recent releases. Walk the same saved/skip connections directly so
            # autograd remains enabled for the class-score explanation.
            current = tensor
            saved_outputs = []
            for module in layers:
                if module.f != -1:
                    current = (saved_outputs[module.f] if isinstance(module.f, int)
                               else [current if source == -1 else saved_outputs[source]
                                     for source in module.f])
                current = module(current)
                saved_outputs.append(current if module.i in detector.save else None)
            output = current
            if isinstance(output, dict) and torch.is_tensor(output.get("scores")):
                class_scores = output["scores"]
            elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
                prediction = output[0]
                if prediction.ndim != 3 or prediction.shape[1] <= 4:
                    raise RuntimeError("Unsupported YOLO prediction output for Grad-CAM")
                class_scores = prediction[:, 4:, :]
            else:
                raise RuntimeError("Unsupported YOLO prediction output for Grad-CAM")
            activation = captured.get("activation")
            if activation is None or activation.ndim != 4:
                raise RuntimeError("Could not capture a spatial feature map for Grad-CAM")
            target_score = class_scores.max()
            gradient = torch.autograd.grad(target_score, activation, retain_graph=False,
                                           create_graph=False, allow_unused=True)[0]
            if gradient is None:
                raise RuntimeError("The selected YOLO feature map has no gradient to the class score")
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1))[0]
            cam = cam.detach().float().cpu().numpy()
            score = float(target_score.detach().cpu())
    finally:
        hook.remove()
        detector.train(was_training)
        head.train(was_head_training)

    cam = cv2.resize(cam, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    cam = cam[top:top + resized_height, left:left + resized_width]
    cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    maximum = float(cam.max())
    if maximum > 0:
        cam /= maximum
    heatmap = cv2.applyColorMap(np.uint8(np.clip(cam, 0, 1) * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap, alpha, 0)
    return overlay, {"method": "Grad-CAM", "target_score": score}
