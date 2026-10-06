# Spill Conservation Monitor (first integration slice)

A local Flask dashboard that processes uploaded videos and still images with the project-trained YOLO detector and the existing Random Forest fluid classifier. It also loads the conservation-conditioned spill PINN and exposes a physical-input prediction route.

## Models used by default

- Detector: `New/Spill_model/training_runs/baseline_yolov8n/weights/best.pt` (the current project-trained detector)
- Conservation field model: `New/models/spill_pinn_stage2_ic_conservation_best.pt`
- Fluid classifier: `Old/current_fluid/fluid_rf.joblib`, loaded by its matching `Old/current_fluid/modules/fluid_classifier.py`

The previous framework detector can be selected for diagnostic comparison by setting `$env:SPILL_DETECTOR_PATH="..\..\Old\current_framework\spill_best.pt"` in PowerShell from this app folder. Paths can be overridden with `SPILL_DETECTOR_PATH`, `CONSERVATION_MODEL_PATH`, `FLUID_CLASSIFIER_PATH`, and `FLUID_CLASSIFIER_MODULE` environment variables. The defaults are derived from this app's location.

## Start

From this folder, with the project virtual environment active:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`, upload an MP4/AVI/MOV/MKV/MPEG video or a JPG/PNG/WEBP/BMP image, then review the uploaded video with time-matched detector boxes/confidences overlaid, the annotated video download, and recorded JSON results. For still images, the app also generates an on-demand qualitative Grad-CAM overlay that can be toggled beside the standard detection result. The browser overlay uses the original uploaded video so it does not depend on MP4 codec support.

## Model boundary

The video pipeline records detector boxes in pixels and fluid class probabilities. The conservation model uses 16 normalized inputs derived from physical units: spatial and time coordinates, density, viscosity, spill geometry, and the initial spill center, width scales, and thickness amplitude. The app therefore does not infer a conservation result from a pixel box. The separate `POST /api/conservation/predict` endpoint accepts physical values and returns predicted thickness in metres; it is a field prediction, not a detection or pass/fail score.

The conservation prediction request body is:

```json
{
  "points": [
    {
      "x_m": 0.2, "y_m": 0.2, "time_s": 30,
      "rho_kg_m3": 950, "mu_pa_s": 0.04,
      "xc_m": 0.2, "yc_m": 0.2,
      "area_m2": 0.1, "width_m": 0.3, "height_m": 0.2,
      "aspect_ratio": 1.5,
      "initial_xc_m": 0.2, "initial_yc_m": 0.2,
      "initial_sigma_x_m": 0.05, "initial_sigma_y_m": 0.04,
      "initial_amplitude_m": 0.001
    }
  ]
}
```

Those values are an example of the request shape only; use measured/calibrated values from the specific event. The model was trained on synthetic trajectories, so field use needs separate validation.

For uploaded still images, the app produces a two-panel original/heatmap figure with a color bar. It computes detection-targeted Grad-CAM at Layer 15 by pooling class-score gradients from strong detector anchors overlapping each predicted box. This gives more spatial detail and coverage than the earlier Layer 21 activation-norm diagnostic. The map remains a qualitative explanation, not a spill segmentation mask, a calibrated probability map, or a physical boundary. This explanation is generated for still images only; video processing is unchanged.

The first slice processes uploaded video and image files. The dashboard keeps Webcam and RTSP controls in the reference layout but leaves them disabled until live-source processing is implemented. Event tracking/temporal confirmation and automatic derivation of calibrated model inputs are also later steps.






