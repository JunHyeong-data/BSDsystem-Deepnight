"""
MoraiDataset 의 split 로직을 ultralytics 호환 train.txt/val.txt 로 export.

- 원본 902장만 seed=42 로 shuffle (sgldet 학습과 동일)
- 80/20 split (train_ratio=0.8)
- train origins + 그 aug 파일들 → train.txt
- val origins (원본만, aug 제외) → val.txt

ultralytics YOLO.train() 시 data='configs/morai.yaml' 로 호출하면
자동으로 이 파일들 읽어서 학습/검증 데이터로 사용.
"""
import random
from pathlib import Path

ROOT = (Path(__file__).resolve().parent.parent / "data" / "morai").resolve()
CONDS = ["night", "dusk"]
SEED = 42
TRAIN_RATIO = 0.8

orig_map = {}
aug_map = {}

for cond in CONDS:
    img_dir = ROOT / cond / "images"
    if not img_dir.exists():
        print(f"[Warning] {img_dir} 없음")
        continue
    for img in sorted(img_dir.glob("*")):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        stem = img.stem
        if "_aug" in stem:
            base = "_aug".join(stem.split("_aug")[:-1])
            aug_map.setdefault(base, []).append(img)
        else:
            orig_map[stem] = img

orig_list = list(orig_map.items())
random.Random(SEED).shuffle(orig_list)

n_train = int(len(orig_list) * TRAIN_RATIO)
train_origs = orig_list[:n_train]
val_origs = orig_list[n_train:]

train_files = [p for _, p in train_origs]
for stem, _ in train_origs:
    train_files.extend(aug_map.get(stem, []))

val_files = [p for _, p in val_origs]

train_txt = ROOT / "train.txt"
val_txt = ROOT / "val.txt"
train_txt.write_text("\n".join(str(p) for p in train_files) + "\n")
val_txt.write_text("\n".join(str(p) for p in val_files) + "\n")

print(f"원본 origins: {len(orig_list)}")
print(f"  train origins: {len(train_origs)}, val origins: {len(val_origs)}")
print(f"train.txt: {len(train_files)} files ({len(train_origs)} origins + {len(train_files) - len(train_origs)} augs)")
print(f"val.txt:   {len(val_files)} files (origins only)")
print(f"저장: {train_txt}")
print(f"저장: {val_txt}")
