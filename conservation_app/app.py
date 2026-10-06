from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from modules.gradcam_overlay import generate_yolo_gradcam, render_gradcam_figure

APP_DIR = Path(__file__).resolve().parent
NEW_ROOT = APP_DIR.parent
PROJECT_ROOT = NEW_ROOT.parent
OLD_ROOT = PROJECT_ROOT / "Old"
OUTPUT_DIR = APP_DIR / "instance" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_PATH = Path(os.environ.get(
    "SPILL_DETECTOR_PATH",
    NEW_ROOT / "models" / "best_spill_detector.pt",
))
PINN_PATH = Path(os.environ.get(
    "CONSERVATION_MODEL_PATH",
    NEW_ROOT / "models" / "spill_pinn_stage2_ic_conservation_best.pt",
))
FLUID_MODULE_PATH = Path(os.environ.get(
    "FLUID_CLASSIFIER_MODULE",
    NEW_ROOT / "fluid" / "fluid_classifier.py",
))
FLUID_MODEL_PATH = Path(os.environ.get(
    "FLUID_CLASSIFIER_PATH",
    NEW_ROOT / "fluid" / "fluid_rf.joblib",
))
DEVICE = os.environ.get("SPILL_APP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

_detector = None
_fluid_classifier = None
_conservation_model = None
_model_status: dict[str, str] = {}


class ConservationPINN(nn.Module):
    """Inference architecture matching the stage-2 conservation checkpoint."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(16, 96), nn.Tanh()]
        for _ in range(4):
            layers.extend((nn.Linear(96, 96), nn.Tanh()))
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(96, 1)
        self.output_activation = nn.Softplus(beta=1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return 0.006 * self.output_activation(self.output_layer(self.network(inputs)))


def _load_fluid_classifier():
    if not FLUID_MODULE_PATH.is_file() or not FLUID_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Fluid classifier module/model not found: {FLUID_MODULE_PATH} ; {FLUID_MODEL_PATH}"
        )
    module_name = "spill_app_fluid_classifier"
    spec = importlib.util.spec_from_file_location(module_name, FLUID_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import classifier module: {FLUID_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.FluidClassifier(method="random_forest", model_path=str(FLUID_MODEL_PATH))


def _load_models():
    global _detector, _fluid_classifier, _conservation_model
    if _detector is None:
        try:
            from ultralytics import YOLO
            if not DETECTOR_PATH.is_file():
                raise FileNotFoundError(DETECTOR_PATH)
            _detector = YOLO(str(DETECTOR_PATH))
            _model_status["detector"] = f"Loaded: {DETECTOR_PATH}"
        except Exception as exc:
            _model_status["detector"] = f"Unavailable: {exc}"
    if _fluid_classifier is None:
        try:
            _fluid_classifier = _load_fluid_classifier()
            _model_status["fluid_classifier"] = f"Loaded: {FLUID_MODEL_PATH.name}"
        except Exception as exc:
            _model_status["fluid_classifier"] = f"Unavailable: {exc}"
    if _conservation_model is None:
        try:
            if not PINN_PATH.is_file():
                raise FileNotFoundError(PINN_PATH)
            model = ConservationPINN().to(DEVICE)
            checkpoint = torch.load(PINN_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            _conservation_model = model
            _model_status["conservation_model"] = f"Loaded: {PINN_PATH.name}"
        except Exception as exc:
            _model_status["conservation_model"] = f"Unavailable: {exc}"


def _serialize_prediction(prediction: Any) -> dict:
    return {
        "label": str(prediction.label),
        "confidence": float(prediction.confidence),
        "probabilities": {str(k): float(v) for k, v in prediction.probabilities.items()},
    }


def _annotate_video(video_path: Path, output_path: Path) -> tuple[list[dict], dict]:
    _load_models()
    if _detector is None:
        raise RuntimeError(_model_status.get("detector", "Detector is unavailable"))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the annotated video output.")

    records: list[dict] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            results = _detector.predict(frame, conf=0.25, verbose=False, device=DEVICE)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    bbox = tuple(float(v) for v in box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    fluid = {"label": "unavailable", "confidence": 0.0, "probabilities": {}}
                    if _fluid_classifier is not None:
                        fluid = _serialize_prediction(_fluid_classifier.predict(frame, bbox))
                    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                    color = (40, 190, 255)
                    caption = f"spill {confidence:.2f} | fluid: {fluid['label']} {fluid['confidence']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, caption, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
                    records.append({
                        "frame": frame_index,
                        "time_s": round(frame_index / fps, 4),
                        "bbox_px": [round(v, 2) for v in bbox],
                        "detector_confidence": confidence,
                        "fluid": fluid,
                        "conservation_status": "not evaluated: physical inputs/calibration required",
                    })
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return records, {"frames": frame_index, "fps": fps, "width": width, "height": height}


# Training reference scales and input order from the conservation model's dataset builder.
INPUT_KEYS = (
    "x_m", "y_m", "time_s", "rho_kg_m3", "mu_pa_s", "xc_m", "yc_m",
    "area_m2", "width_m", "height_m", "aspect_ratio", "initial_xc_m",
    "initial_yc_m", "initial_sigma_x_m", "initial_sigma_y_m", "initial_amplitude_m",
)


def _normalize_physical_input(row: dict[str, float]) -> list[float]:
    missing = [key for key in INPUT_KEYS if key not in row]
    if missing:
        raise ValueError(f"Missing physical inputs: {', '.join(missing)}")
    values = [float(row[key]) for key in INPUT_KEYS]
    x, y, t, rho, mu, xc, yc, area, width, height, ar, ixc, iyc, isx, isy, iamp = values
    return [
        x / 1.0, y / 1.0, t / 300.0, rho / 950.0, mu / 0.04,
        xc / 1.0, yc / 1.0, area / 0.5, width / 1.0, height / 1.0, ar,
        ixc / 1.0, iyc / 1.0, isx / 1.0, isy / 1.0, iamp / 0.006,
    ]


@app.get("/")
def index():
    _load_models()
    return render_template("index.html", model_status=_model_status)


@app.get("/api/status")
def status():
    _load_models()
    return jsonify({"models": _model_status, "device": DEVICE})


@app.post("/api/analyze-video")
def analyze_video():
    uploaded = request.files.get("video")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a video file first."}), 400
    safe_name = secure_filename(uploaded.filename)
    if Path(safe_name).suffix.lower() not in VIDEO_EXTENSIONS:
        return jsonify({"error": "Upload an MP4, AVI, MOV, MKV, MPEG, or MPG video."}), 400
    job_id = uuid.uuid4().hex[:12]
    input_path = OUTPUT_DIR / f"{job_id}_{safe_name}"
    output_name = f"{job_id}_annotated.mp4"
    output_path = OUTPUT_DIR / output_name
    result_path = OUTPUT_DIR / f"{job_id}_results.json"
    uploaded.save(input_path)
    try:
        records, video_info = _annotate_video(input_path, output_path)
        payload = {
            "job_id": job_id,
            "source_name": safe_name,
            "video": video_info,
            "models": _model_status,
            "detections": records,
            "summary": {
                "candidate_count": len(records),
                "fluid_labels": sorted({r["fluid"]["label"] for r in records}),
                "conservation_note": "The conservation model is available for explicit physical-input queries. A pixel box alone is insufficient to infer calibrated lengths, initial spill conditions, density, or viscosity.",
            },
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return jsonify({**payload, "annotated_video_url": f"/outputs/{output_name}", "results_url": f"/outputs/{job_id}_results.json"})
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc), "models": _model_status}), 500
    finally:
        input_path.unlink(missing_ok=True)


@app.post("/api/analyze-image")
def analyze_image():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose an image file first."}), 400
    safe_name = secure_filename(uploaded.filename)
    if Path(safe_name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return jsonify({"error": "Upload a JPG, PNG, WEBP, or BMP image."}), 400
    _load_models()
    if _detector is None:
        return jsonify({"error": _model_status.get("detector", "Detector is unavailable")}), 503
    raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "The uploaded image could not be decoded."}), 400
    job_id = uuid.uuid4().hex[:12]
    detections = []
    results = _detector.predict(frame, conf=0.25, verbose=False, device=DEVICE)
    gradcam_frame = None
    gradcam_info = None
    gradcam_figure = None
    gradcam_error = None
    try:
        heatmap, gradcam_info = generate_yolo_gradcam(_detector, frame)
        gradcam_figure = render_gradcam_figure(frame, heatmap, gradcam_info)
    except Exception as exc:
        gradcam_error = str(exc)
        app.logger.warning("Still-image Grad-CAM unavailable: %s", exc)
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            bbox = tuple(float(v) for v in box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            fluid = {"label": "unavailable", "confidence": 0.0, "probabilities": {}}
            if _fluid_classifier is not None:
                fluid = _serialize_prediction(_fluid_classifier.predict(frame, bbox))
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            caption = f"spill {confidence:.2f} | fluid: {fluid['label']} {fluid['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 190, 255), 2)
            cv2.putText(frame, caption, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 190, 255), 2)
            detections.append({
                "bbox_px": [round(v, 2) for v in bbox],
                "detector_confidence": confidence,
                "fluid": fluid,
                "conservation_status": "not evaluated: physical inputs/calibration required",
            })
    image_name = f"{job_id}_annotated.jpg"
    gradcam_name = f"{job_id}_gradcam.jpg"
    results_name = f"{job_id}_results.json"
    cv2.imwrite(str(OUTPUT_DIR / image_name), frame)
    gradcam_url = None
    if gradcam_figure is not None:
        (OUTPUT_DIR / gradcam_name).write_bytes(gradcam_figure)
        gradcam_url = f"/outputs/{gradcam_name}"
    payload = {"job_id": job_id, "source_name": safe_name, "models": _model_status,
               "detections": detections, "gradcam": gradcam_info,
               "gradcam_error": gradcam_error}
    (OUTPUT_DIR / results_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return jsonify({**payload, "annotated_image_url": f"/outputs/{image_name}",
                    "gradcam_image_url": gradcam_url, "results_url": f"/outputs/{results_name}"})

@app.post("/api/conservation/predict")
def conservation_predict():
    _load_models()
    if _conservation_model is None:
        return jsonify({"error": _model_status.get("conservation_model", "Model unavailable")}), 503
    payload = request.get_json(silent=True) or {}
    rows = payload.get("points")
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "Provide a non-empty 'points' list with all required physical inputs."}), 400
    try:
        normalized = np.asarray([_normalize_physical_input(row) for row in rows], dtype=np.float32)
        with torch.inference_mode():
            tensor = torch.from_numpy(normalized).to(DEVICE)
            predicted_m = _conservation_model(tensor).cpu().numpy().reshape(-1)
        return jsonify({
            "predicted_thickness_m": [float(value) for value in predicted_m],
            "interpretation": "Model-predicted thickness field values for the supplied physical coordinates and initial conditions; this is not a spill detection decision.",
        })
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/outputs/<path:filename>")
def outputs(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)




