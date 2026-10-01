from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "yolo" / "data.yaml"
WEIGHTS = ROOT / "models" / "best_spill_detector.pt"
RUNS = ROOT / "results" / "yolo" / "runs"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Recreates the baseline YOLOv8n training configuration.
# Use --resume only if continuing an interrupted run; default starts fresh.
def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA), project=str(RUNS), name="baseline_yolov8n",
        epochs=100, imgsz=640, batch=16, device=DEVICE, workers=4,
        seed=42, deterministic=True, patience=20, task="detect",
        save=True, save_period=10, val=True, exist_ok=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
    )

if __name__ == "__main__":
    main()
