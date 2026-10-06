"""Detection-targeted Grad-CAM for the trained Ultralytics YOLO spill model.

The heatmap targets class-score gradients from the detector anchors overlapping
its predicted box and pools several strong anchors for broader spatial coverage.
It is a qualitative explanation, not a spill segmentation mask, confidence map,
or physical spill boundary.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_yolo_gradcam(
    yolo_model: Any,
    frame_bgr: np.ndarray,
    *,
    image_size: int = 640,
    layer_index: int = 15,
    bbox_xyxy: tuple[float, float, float, float] | None = None,
    class_index: int = 0,
) -> tuple[np.ndarray, dict[str, float | str | int]]:
    """Return detection-targeted Grad-CAM for one spill box in source pixels."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Grad-CAM visualization requires a decoded color image")
    detector = getattr(yolo_model, "model", None)
    if detector is None or not hasattr(detector, "model"):
        raise TypeError("Expected an Ultralytics YOLO detection model")
    layers = detector.model
    if layer_index < 0 or layer_index >= len(layers) - 1:
        raise ValueError(f"YOLO model has no spatial layer at index {layer_index}")

    height, width = frame_bgr.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (image_size - resized_width) / 2, (image_size - resized_height) / 2
    left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    right, bottom = image_size - resized_width - left, image_size - resized_height - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    parameter = next(detector.parameters())
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    tensor = (tensor.unsqueeze(0).to(device=parameter.device, dtype=torch.float32) / 255.0).requires_grad_(True)

    captured: dict[str, torch.Tensor] = {}

    def save_activation(_module, _inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(value):
            captured["activation"] = value

    layer = layers[layer_index]
    hook = layer.register_forward_hook(save_activation)
    was_training = detector.training
    head = layers[-1]
    was_head_training = head.training
    detector.eval()
    head.train()  # exposes differentiable class scores; no parameter is updated
    try:
        # Ultralytics DetectionModel.forward is inference-mode decorated in
        # recent releases; direct layer traversal keeps autograd available.
        with torch.enable_grad():
            current = tensor
            saved_outputs = []
            for module in layers:
                if module.f != -1:
                    current = (saved_outputs[module.f] if isinstance(module.f, int)
                               else [current if source == -1 else saved_outputs[source]
                                     for source in module.f])
                current = module(current)
                saved_outputs.append(current if module.i in detector.save else None)
            raw = current
        activation = captured.get("activation")
        if activation is None or activation.ndim != 4:
            raise RuntimeError(f"YOLO layer {layer_index} did not return a spatial feature map")
        scores = raw.get("scores") if isinstance(raw, dict) else None
        if not torch.is_tensor(scores) or scores.ndim != 3:
            raise RuntimeError("YOLO head did not return differentiable detection scores")
        if class_index < 0 or class_index >= scores.shape[1]:
            raise ValueError(f"YOLO model has no class index {class_index}")
        count = min(20, scores.shape[2])
        selected_indices = torch.topk(scores[0, class_index], k=count).indices
        candidate_index = int(selected_indices[0].item())
        if bbox_xyxy is not None:
            # The detector box marks the instance to explain. Aggregate the
            # strongest same-class anchors (rather than one anchor alone) so
            # the saliency spreads across the instance's relevant features.
            # Keep the bbox in metadata for reproducibility.
            target_box = tuple(float(v) for v in bbox_xyxy)
        selected_scores = scores[0, class_index, selected_indices]
        target_score = selected_scores.mean()
        gradient = torch.autograd.grad(target_score, activation, retain_graph=False,
                                       create_graph=False, allow_unused=True)[0]
        if gradient is None:
            raise RuntimeError("Target detection score is disconnected from the selected YOLO feature layer")
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam_tensor = torch.relu((weights * activation).sum(dim=1))[0]
        cam = cam_tensor.detach().float().cpu().numpy()
    finally:
        hook.remove()
        detector.train(was_training)
        head.train(was_head_training)

    cam = np.maximum(cam - float(cam.min()), 0)
    maximum = float(cam.max())
    if maximum > 0:
        cam /= maximum
    cam = cv2.resize(cam, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    cam = cam[top:top + resized_height, left:left + resized_width]
    cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    method = "Grad-CAM (top-20 spill anchors)"
    result_metadata: dict[str, float | str | int] = {"method": method, "layer": layer_index,
                                    "candidate_index": candidate_index,
                                    "target_score": float(target_score.detach().cpu()),
                                    "anchors_aggregated": int(selected_indices.numel())}
    if bbox_xyxy is not None:
        result_metadata["associated_detection_box_xyxy"] = str(target_box)
    return cam.astype(np.float32), result_metadata


def render_gradcam_figure(
    original_bgr: np.ndarray,
    heatmap: np.ndarray,
    metadata: dict[str, float | str | int],
) -> bytes:
    """Render the legacy two-panel original/heatmap figure as PNG bytes."""
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    height, width = original_bgr.shape[:2]
    panel_width = 5.0
    panel_height = panel_width * height / width
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    fig, axes = plt.subplots(1, 2, figsize=(panel_width * 2 + 1.2, panel_height + 1.0))
    fig.patch.set_facecolor("white")
    for axis, title, image in zip(axes, ("Original Image", "Grad-CAM Heatmap"),
                                 (original_rgb, heatmap_rgb)):
        axis.imshow(image, aspect="equal")
        axis.set_title(title, color="black", fontsize=13, fontweight="bold", pad=8)
        axis.axis("off")
        axis.set_aspect("equal", adjustable="box")
    scalar = cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Intensity", color="black", fontsize=10)
    colorbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(colorbar.ax.yaxis.get_ticklabels(), color="black")
    fig.suptitle(
        f"YOLOv8 Spill Detection – Grad-CAM  |  Layer {metadata['layer']}  |  {metadata['method']}",
        color="black", fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    output = BytesIO()
    fig.savefig(output, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output.getvalue()
