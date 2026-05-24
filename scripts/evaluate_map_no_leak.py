"""
누수 없는 mAP 측정 (Scenario-Level Split)
=========================================
파일명이 Unix timestamp 라는 점을 이용해 시나리오 단위로 train/val 분할.
같은 시나리오의 인접 frame은 모두 같은 split 에 귀속 → temporal leakage 제거.

⚠️ 주의: 이 스크립트의 val 은 학습 시 split 과 다르므로,
   학습된 모델이 일부 새 val frame 을 이미 봤을 수 있음 (less leaky 추정치).

진짜 정확한 평가는 scenario-aware split 으로 처음부터 재학습 필요.

실행:
    python3 scripts/evaluate_map_no_leak.py
"""

import argparse
import datetime
import random
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.sgldet_yolov8 import SGLDetYOLO
from ultralytics import YOLO


def find_scenarios(root: str = "data/morai", gap_threshold_ms: int = 10000,
                   conditions=None):
    """Timestamp gap > 10초 인 곳에서 시나리오 분할."""
    conditions = conditions or ["dusk", "night"]
    scenarios = []   # [{"cond": str, "frames": [Path]}]

    for cond in conditions:
        img_dir = Path(root) / cond / "images"
        if not img_dir.exists():
            continue

        # Timestamp 순 정렬 (원본만)
        files = [f for f in img_dir.glob("*")
                 if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
                 and "_aug" not in f.stem]
        ts_files = []
        for f in files:
            try:
                ts_files.append((int(f.stem), f))
            except ValueError:
                continue
        ts_files.sort()

        # Gap > threshold 인 곳에서 분할
        current_scenario = []
        prev_ts = None
        for ts, f in ts_files:
            if prev_ts is None or (ts - prev_ts) <= gap_threshold_ms:
                current_scenario.append(f)
            else:
                if current_scenario:
                    scenarios.append({"cond": cond, "frames": current_scenario})
                current_scenario = [f]
            prev_ts = ts
        if current_scenario:
            scenarios.append({"cond": cond, "frames": current_scenario})

    return scenarios


def split_scenarios(scenarios: list, val_ratio: float = 0.2, seed: int = 42):
    """시나리오 단위로 셔플 후 train/val 분할."""
    rng = random.Random(seed)
    shuffled = scenarios.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_scenarios = shuffled[:n_val]
    train_scenarios = shuffled[n_val:]
    return train_scenarios, val_scenarios


def collect_frames(scenarios: list):
    """시나리오 리스트에서 모든 frame 경로 수집."""
    frames = []
    for sc in scenarios:
        frames.extend(sc["frames"])
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   default="checkpoints/best_model.pt")
    parser.add_argument("--data-root", default="data/morai")
    parser.add_argument("--img-size",  type=int, default=640)
    parser.add_argument("--batch",     type=int, default=8)
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--conf",      type=float, default=0.001)
    parser.add_argument("--iou",       type=float, default=0.6)
    parser.add_argument("--gap-sec",   type=float, default=10.0,
                        help="시나리오 경계로 인식할 gap (초)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── 1. 시나리오 분할 ──────────────────────────────────────
    scenarios = find_scenarios(
        args.data_root, gap_threshold_ms=int(args.gap_sec * 1000),
    )
    print(f"\n[시나리오 분석]")
    print(f"  총 시나리오: {len(scenarios)} 개")
    print(f"  Dusk: {sum(1 for s in scenarios if s['cond']=='dusk')} 개")
    print(f"  Night: {sum(1 for s in scenarios if s['cond']=='night')} 개")
    print(f"  평균 frame/시나리오: "
          f"{sum(len(s['frames']) for s in scenarios) / len(scenarios):.1f}")

    train_sc, val_sc = split_scenarios(
        scenarios, val_ratio=args.val_ratio, seed=args.seed,
    )
    val_frames = collect_frames(val_sc)
    train_frames = collect_frames(train_sc)
    print(f"\n[Scenario-level Split]")
    print(f"  Train: {len(train_sc)} 시나리오, {len(train_frames)} frame")
    print(f"  Val:   {len(val_sc)} 시나리오, {len(val_frames)} frame  ⭐")
    print(f"  → 같은 시나리오는 한 split 에만 귀속 (temporal leakage 제거)")

    if not val_frames:
        print("[Error] Val frame 없음")
        return

    # ── 2. val.txt + data.yaml 생성 ─────────────────────────
    val_list_path = Path("eval_val_noleak.txt").resolve()
    with open(val_list_path, "w") as f:
        for img in val_frames:
            f.write(str(img.resolve()) + "\n")

    yaml_path = Path("eval_data_noleak.yaml").resolve()
    data = {
        "path":  str(Path.cwd().resolve()),
        "train": str(val_list_path),
        "val":   str(val_list_path),
        "nc":    2,
        "names": ["vehicle", "pedestrian"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True)

    # ── 3. SGLDet 모델 로드 ───────────────────────────────────
    print(f"\n[Loading] {args.weights}")
    model = SGLDetYOLO(
        yolo_weights="yolov8m.pt", lambda_self=0.01, num_classes=2,
    ).to(device)
    model.warmup(img_size=args.img_size, device=device)

    state_dict = torch.load(args.weights, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()

    # ── 4. detector → ultralytics 호환 .pt ────────────────────
    model.detector.names = {0: "vehicle", 1: "pedestrian"}
    tmp_pt = Path("checkpoints/best_yolo_only.pt")
    ckpt = {
        "model":         model.detector.float(),
        "epoch":         -1,
        "best_fitness":  None,
        "ema":           None,
        "updates":       0,
        "optimizer":     None,
        "train_args":    {"data": str(yaml_path), "imgsz": args.img_size},
        "train_metrics": {},
        "train_results": None,
        "date":          datetime.datetime.now().isoformat(),
        "version":       "8.0.0",
    }
    torch.save(ckpt, tmp_pt)

    # ── 5. 평가 ───────────────────────────────────────────────
    print(f"\n[Evaluating] Scenario-aware Val 에서 mAP 측정 중...")
    yolo = YOLO(str(tmp_pt))
    results = yolo.val(
        data=str(yaml_path), imgsz=args.img_size, batch=args.batch,
        device=device, conf=args.conf, iou=args.iou,
        save_json=False, verbose=True, plots=True,
    )

    # ── 6. 결과 ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" " * 18 + "📊 누수 없는 mAP 측정 결과")
    print("=" * 72)

    box = results.box
    print(f"\n[전체 Metric]  (val = {len(val_frames)} frame, scenario-level)")
    print(f"  mAP@0.5      : {box.map50:.4f}   ({box.map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95 : {box.map:.4f}   ({box.map*100:.2f}%)")
    print(f"  Precision    : {box.mp:.4f}   ({box.mp*100:.2f}%)")
    print(f"  Recall       : {box.mr:.4f}   ({box.mr*100:.2f}%)")

    print(f"\n[Class 별]")
    names = ["vehicle", "pedestrian"]
    header = f"  {'Class':<12} {'AP@0.5':>10} {'AP@.5:.95':>12} {'Precision':>10} {'Recall':>10}"
    print(header); print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(names):
        try:
            ap50 = box.ap50[i]; ap = box.ap[i]
            p = box.p[i];       r  = box.r[i]
            print(f"  {name:<12} {ap50:>10.4f} {ap:>12.4f} {p:>10.4f} {r:>10.4f}")
        except (IndexError, AttributeError, TypeError):
            print(f"  {name:<12} (해당 클래스 검출 없음)")

    print("\n" + "=" * 72)
    print(f"[비교]")
    print(f"  Random shuffle val (이전, 누수 有): mAP@0.5 = 97.7%")
    print(f"  Scenario-level val (현재, 누수 ↓): mAP@0.5 = {box.map50*100:.1f}%")
    print(f"  → 차이: {(0.977 - box.map50)*100:.1f}%p  (= 데이터 누수의 부풀린 정도)")
    print("=" * 72)


if __name__ == "__main__":
    main()
