"""
Plain YOLOv8m FT + Backbone Freeze — pedestrian feature 보존 실험.

가설:
  - 우리 902장 fine-tune 이 COCO 의 person prior 를 손상시킴 (catastrophic forgetting)
  - Backbone (layer 0~9, CSPDarknet) freeze 하면 COCO person/car feature 그대로 보존
  - Neck + Head 만 우리 데이터/2 class 에 적응 → 도메인 적응 + prior 보존 trade-off 해결

YOLOv8m 구조:
  layer 0~9   : Backbone (CSPDarknet)   ← freeze
  layer 10~21 : Neck (PANet/FPN)         ← train (도메인 적응)
  layer 22    : Head (Detect)            ← train (우리 2 class)

데이터/split: configs/morai.yaml (sgldet 학습과 동일 — fair ablation)
Hyperparam:   configs/sgldet_config.yaml 매칭

결과: runs/baseline_freeze/yolov8m_freeze_902/weights/best.pt
"""
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / "configs/sgldet_config.yaml") as f:
    cfg = yaml.safe_load(f)["train"]

data_yaml = PROJECT_ROOT / "configs/morai.yaml"
train_txt = PROJECT_ROOT / "data/morai/train.txt"
val_txt = PROJECT_ROOT / "data/morai/val.txt"
if not train_txt.exists() or not val_txt.exists():
    print(f"[Error] train.txt / val.txt 없음 — python3 scripts/make_split_files.py 먼저 실행")
    sys.exit(1)

print("=" * 60)
print("  YOLOv8m Backbone-Freeze Fine-Tune")
print(f"  Freeze: layer 0~9 (Backbone, CSPDarknet)")
print(f"  Train: Neck (10~21) + Head (22)")
print(f"  Epochs: {cfg['epochs']}  |  Batch: {cfg['batch_size']}  |  ImgSz: {cfg['img_size']}")
print(f"  Optimizer: {cfg['optimizer']}  |  lr: {cfg['lr']}  |  Cosine scheduler")
print("=" * 60)

model = YOLO("yolov8m.pt")

model.train(
    data=str(data_yaml),
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
    patience=23,
    project="runs/baseline_freeze",
    name="yolov8m_freeze_902",
    freeze=10,                            # ← Backbone (0~9) freeze
    save=True,
    exist_ok=True,
    verbose=True,
)

print("\n학습 완료.")
print(f"Best weights: runs/baseline_freeze/yolov8m_freeze_902/weights/best.pt")
