"""
Plain YOLOv8m FT + Scenario split — fair 비교 baseline.

목적:
  SGLDet+freeze+scn (방금 학습 끝) 과 fair 비교 위해 같은 split 으로 plain FT.

데이터/split: configs/morai.yaml (현재 train.txt/val.txt = scenario-level)
Hyperparam:   configs/sgldet_config.yaml 매칭

결과 저장: runs/detect/runs/baseline_ft/yolov8m_ft_902_scn/weights/best.pt
"""
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / "configs/sgldet_config.yaml") as f:
    cfg = yaml.safe_load(f)["train"]

print("=" * 60)
print("  YOLOv8m + Scenario Split (no SGLDet, no freeze)")
print(f"  Train: 4176 (scenario split) / Val: 203 (origins only)")
print(f"  Epochs: {cfg['epochs']} (patience 20)")
print("=" * 60)

model = YOLO("yolov8m.pt")
model.train(
    data=str(PROJECT_ROOT / "configs/morai.yaml"),
    epochs=cfg["epochs"],
    batch=cfg["batch_size"],
    imgsz=cfg["img_size"],
    optimizer=cfg["optimizer"],
    lr0=cfg["lr"],
    momentum=cfg["momentum"],
    weight_decay=cfg["weight_decay"],
    warmup_epochs=cfg["warmup_epochs"],
    cos_lr=True,
    workers=cfg["num_workers"],
    device=cfg["device"],
    patience=20,
    project="runs/baseline_ft_scn",
    name="yolov8m_ft_902_scn",
    save=True,
    exist_ok=True,
    verbose=True,
)
print(f"\n학습 완료. Best: runs/baseline_ft_scn/yolov8m_ft_902_scn/weights/best.pt")
