"""Original whole-image Layer 21 Activation Norm map for the YOLO spill model."""

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
    layer_index: int = 21,
    bbox_xyxy: tuple[float, float, float, float] | None = None,
    class_index: int = 0,
) -> tuple[np.ndarray, dict[str, float | str | int]]:
    """Return the original Layer 21 activation-norm heatmap for an image."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Grad-CAM visualization requires a decoded color image")
    detector = getattr(yolo_model, "model", None)
    if detector is None or not hasattr(detector, "model"):
        raise TypeError("Expected an Ultralytics YOLO detection model")
    layers = detector.model
    if layer_index < 0 or layer_index >= len(layers) - 1:
        raise ValueError(f"YOLO model has no spatial layer at index {layer_index}")

    height, width = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    parameter = next(detector.parameters())
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    tensor = tensor.unsqueeze(0).to(device=parameter.device, dtype=torch.float32) / 255.0

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
    head.train()
    try:
        # Direct layer traversal exposes the selected feature map without
        # invoking Ultralytics' inference-mode decorated forward method.
        with torch.no_grad():
            current = tensor
            saved_outputs = []
            for module in layers:
                if module.f != -1:
                    current = (saved_outputs[module.f] if isinstance(module.f, int)
                               else [current if source == -1 else saved_outputs[source]
                                     for source in module.f])
                current = module(current)
                saved_outputs.append(current if module.i in detector.save else None)
        activation = captured.get("activation")
        if activation is None or activation.ndim != 4:
            raise RuntimeError(f"YOLO layer {layer_index} did not return a spatial feature map")
        # This reproduces the original script's Layer 21 activation-norm map.
        cam_tensor = activation[0].norm(dim=0)
        cam = cam_tensor.detach().float().cpu().numpy()
    finally:
        hook.remove()
        detector.train(was_training)
        head.train(was_head_training)

    cam = np.maximum(cam - float(cam.min()), 0)
    maximum = float(cam.max())
    if maximum > 0:
        cam /= maximum
    cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    method = "Activation Norm"
    result_metadata: dict[str, float | str | int] = {"method": method, "layer": layer_index}
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
