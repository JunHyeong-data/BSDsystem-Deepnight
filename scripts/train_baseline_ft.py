"""
Plain YOLOv8m baseline fine-tune — SGLDet ablation 용.

목적:
  - SGLDet vs no-SGLDet 진정한 ablation
  - 같은 데이터, 같은 split, 같은 hyperparam → 차이는 오직 SGLDet wrapping

학습 데이터: configs/morai.yaml 의 train/val (sgldet 학습과 동일 split)
Hyperparam: configs/sgldet_config.yaml 의 train 섹션 매칭

결과 저장: runs/baseline_ft/yolov8m_ft_902/weights/best.pt
"""
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Hyperparam 로드 (SGLDet 학습과 동일)
with open(PROJECT_ROOT / "configs/sgldet_config.yaml") as f:
    cfg = yaml.safe_load(f)["train"]

# 데이터 yaml 존재 확인
data_yaml = PROJECT_ROOT / "configs/morai.yaml"
train_txt = PROJECT_ROOT / "data/morai/train.txt"
val_txt = PROJECT_ROOT / "data/morai/val.txt"
if not train_txt.exists() or not val_txt.exists():
    print(f"[Error] train.txt / val.txt 없음")
    print(f"  → python3 scripts/make_split_files.py 먼저 실행")
    sys.exit(1)

print("=" * 60)
print("  YOLOv8m Baseline Fine-Tune (no SGLDet)")
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
    cos_lr=True,                          # scheduler: cosine
    workers=cfg["num_workers"],
    device=cfg["device"],
    patience=23,                          # Ours 도 23 epoch plateau 에서 early stop
    project="runs/baseline_ft",
    name="yolov8m_ft_902",
    save=True,
    exist_ok=True,
    verbose=True,
)

print("\n학습 완료.")
print(f"Best weights: runs/baseline_ft/yolov8m_ft_902/weights/best.pt")
print(f"Last weights: runs/baseline_ft/yolov8m_ft_902/weights/last.pt")
