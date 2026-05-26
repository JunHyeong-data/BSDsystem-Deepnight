"""
Scenario-level train/val split — Lesson #2 적용.

기존 make_split_files.py 는 origins 를 random shuffle 했음 (sister frame leakage):
  - 0.5s 간격 frame 이 train/val 양쪽에 흩어져 model 이 외움
  - Random val mAP 가 inflated (97.7%)

이 스크립트는 scenario-level 로 분할:
  - Timestamp gap > 10s 인 곳에서 scenario boundary 분리
  - Scenarios 단위로 shuffle (seed=42) → 80/20 split
  - 같은 scenario 의 frames 는 train 또는 val 한쪽에만 들어감

결과:
  data/morai/train.txt  ← scenario 80% (origins + augs)
  data/morai/val.txt    ← scenario 20% (origins only)

Lesson #2 의 실시간 적용 — 학습 시 monitoring 도 fair, in-domain test 의 평가도 fair.
"""
import random
from pathlib import Path

ROOT = (Path(__file__).resolve().parent.parent / "data" / "morai").resolve()
CONDS = ["night", "dusk"]
SEED = 42
TRAIN_RATIO = 0.8
GAP_THRESHOLD_MS = 10_000   # 10초


def parse_timestamp(stem: str) -> int | None:
    """Origin filename = unix timestamp (ms). aug 파일은 None."""
    if "_aug" in stem:
        return None
    return int(stem) if stem.isdigit() else None


# 1) Origin / aug 분리 + scenario grouping
orig_by_cond = {c: [] for c in CONDS}    # cond → [(ts, path)]
aug_map = {}                               # origin_stem → [aug paths]

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
            ts = parse_timestamp(stem)
            if ts is not None:
                orig_by_cond[cond].append((ts, img))

# 2) Scenario 분할 (timestamp gap > 10s 기준)
scenarios = []   # [(cond, [(ts, path), ...]), ...]
for cond, ts_list in orig_by_cond.items():
    if not ts_list:
        continue
    ts_list.sort(key=lambda x: x[0])
    current = [ts_list[0]]
    prev_ts = ts_list[0][0]
    for ts, p in ts_list[1:]:
        if ts - prev_ts <= GAP_THRESHOLD_MS:
            current.append((ts, p))
        else:
            scenarios.append((cond, current))
            current = [(ts, p)]
        prev_ts = ts
    if current:
        scenarios.append((cond, current))

# 3) Scenarios 단위로 shuffle → 80/20 split
random.Random(SEED).shuffle(scenarios)
n_train = int(len(scenarios) * TRAIN_RATIO)
train_scenarios = scenarios[:n_train]
val_scenarios   = scenarios[n_train:]

# 4) Train: origins + augs / Val: origins only
train_files, val_files = [], []
train_origins_count = 0
for cond, ts_list in train_scenarios:
    for ts, p in ts_list:
        train_files.append(p)
        train_origins_count += 1
        train_files.extend(aug_map.get(p.stem, []))

val_origins_count = 0
for cond, ts_list in val_scenarios:
    for ts, p in ts_list:
        val_files.append(p)
        val_origins_count += 1

# 5) 저장
train_txt = ROOT / "train.txt"
val_txt = ROOT / "val.txt"
train_txt.write_text("\n".join(str(p) for p in train_files) + "\n")
val_txt.write_text("\n".join(str(p) for p in val_files) + "\n")

# 6) 통계
total_scenarios = len(scenarios)
night_scenarios = sum(1 for c, _ in scenarios if c == "night")
dusk_scenarios = sum(1 for c, _ in scenarios if c == "dusk")
print(f"전체 scenarios: {total_scenarios}  (night {night_scenarios}, dusk {dusk_scenarios})")
print(f"  Scenario sizes: min={min(len(t) for _,t in scenarios)}, "
      f"max={max(len(t) for _,t in scenarios)}, "
      f"avg={sum(len(t) for _,t in scenarios)/total_scenarios:.1f}")
print(f"")
print(f"Train: {len(train_scenarios)} scenarios, {train_origins_count} origins + "
      f"{len(train_files) - train_origins_count} augs = {len(train_files)} files")
print(f"Val:   {len(val_scenarios)} scenarios, {val_origins_count} origins (no augs)")
print(f"")
print(f"저장: {train_txt}")
print(f"저장: {val_txt}")
print(f"")
print(f"⚠️ Lesson #2 적용: sister frame leakage 제거됨. val 은 train 과 다른 scenario.")
