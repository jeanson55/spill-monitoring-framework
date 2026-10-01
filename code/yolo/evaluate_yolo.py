from pathlib import Path
import argparse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "yolo" / "data.yaml"
WEIGHTS = ROOT / "models" / "best_spill_detector.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=("val", "test"), default="test")
parser.add_argument("--device", default="0")
args = parser.parse_args()
model = YOLO(str(WEIGHTS))
metrics = model.val(data=str(DATA), split=args.split, imgsz=640, batch=16,
                    device=args.device, plots=True, save_json=True,
                    project=str(ROOT / "results" / "yolo" / "evaluation"),
                    name=args.split, exist_ok=True)
print({"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
       "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)})
