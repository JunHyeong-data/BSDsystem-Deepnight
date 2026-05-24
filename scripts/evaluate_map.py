"""
mAP 측정 스크립트
=================
학습된 SGLDet 모델을 val 데이터(원본 181장)에서 평가하여 mAP 산출.

ultralytics YOLO 표준 validation 사용 (검증된 mAP 계산).

실행:
    python3 scripts/evaluate_map.py
    python3 scripts/evaluate_map.py --weights checkpoints/best_model.pt
"""

import argparse
import datetime
import random
import sys
from pathlib import Path

import torch
import yaml

# 프로젝트 루트 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.sgldet_yolov8 import SGLDetYOLO
from ultralytics import YOLO


def get_val_images(root: str = "data/morai", train_ratio: float = 0.8,
                   seed: int = 42, conditions=None):
    """morai_dataset.py 와 동일한 train/val split 로직 재현."""
    conditions = conditions or ["dusk", "night"]
    orig_files = {}

    for cond in conditions:
        img_dir = Path(root) / cond / "images"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            stem = img.stem
            if "_aug" in stem:
                continue   # val 은 원본만
            orig_files[stem] = img

    orig_list = sorted(orig_files.items())
    random.Random(seed).shuffle(orig_list)
    n_train = int(len(orig_list) * train_ratio)
    val_pairs = orig_list[n_train:]
    return [img for _, img in val_pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   default="checkpoints/best_model.pt")
    parser.add_argument("--data-root", default="data/morai")
    parser.add_argument("--img-size",  type=int, default=640)
    parser.add_argument("--batch",     type=int, default=8)
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--conf",      type=float, default=0.001,
                        help="mAP 측정용 매우 낮은 confidence (표준)")
    parser.add_argument("--iou",       type=float, default=0.6,
                        help="NMS IoU threshold")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── 1. Val 이미지 목록 (학습과 동일한 split) ──────────────
    val_images = get_val_images(args.data_root)
    print(f"\n[Val] {len(val_images)} 이미지")
    if not val_images:
        print("[Error] Val 이미지를 찾을 수 없음")
        return

    # ── 2. val.txt 생성 (ultralytics 표준 포맷) ────────────────
    val_list_path = Path("eval_val.txt").resolve()
    with open(val_list_path, "w") as f:
        for img in val_images:
            f.write(str(img.resolve()) + "\n")

    # ── 3. data.yaml 생성 ───────────────────────────────────
    yaml_path = Path("eval_data.yaml").resolve()
    data = {
        "path":  str(Path.cwd().resolve()),
        "train": str(val_list_path),       # dummy, 사용 안 됨
        "val":   str(val_list_path),
        "nc":    2,
        "names": ["vehicle", "pedestrian"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True)
    print(f"[Config] {yaml_path}")

    # ── 4. SGLDet 모델 로드 + weight 적용 ─────────────────────
    print(f"\n[Loading] {args.weights}")
    model = SGLDetYOLO(
        yolo_weights = "yolov8m.pt",
        lambda_self  = 0.01,
        num_classes  = 2,
    )
    model.to(device)     # ← warmup 전에 device 이동
    model.warmup(img_size=args.img_size, device=device)

    state_dict = torch.load(args.weights, map_location=device)
    # last_model.pt (전체 체크포인트) 호환
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[OK] 모델 로드 완료 (device={device})")

    # ── 5. detector 만 ultralytics 호환 .pt 로 저장 ──────────
    # SGLDetYOLO.detector 는 PyTorch DetectionModel (YOLO wrapper 아님).
    # ultralytics YOLO 가 인식하는 dict 포맷으로 감싸 저장 → 다시 로드.
    model.detector.names = {0: "vehicle", 1: "pedestrian"}
    tmp_yolo_pt = Path("checkpoints/best_yolo_only.pt")
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
    torch.save(ckpt, tmp_yolo_pt)
    print(f"[Save] {tmp_yolo_pt} (ultralytics 호환 format)")

    # ── 6. ultralytics YOLO 로 다시 로드 → .val() ─────────────
    print(f"\n[Evaluating] mAP 계산 중... (val={len(val_images)}, batch={args.batch})")
    yolo = YOLO(str(tmp_yolo_pt))
    results = yolo.val(
        data       = str(yaml_path),
        imgsz      = args.img_size,
        batch      = args.batch,
        device     = device,
        conf       = args.conf,
        iou        = args.iou,
        save_json  = False,
        verbose    = True,
        plots      = True,    # PR curve, confusion matrix 자동 생성
    )

    # ── 6. 결과 출력 ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" " * 24 + "📊 mAP 측정 결과")
    print("=" * 72)

    box = results.box

    print(f"\n[전체 Metric]")
    print(f"  mAP@0.5      : {box.map50:.4f}   ({box.map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95 : {box.map:.4f}   ({box.map*100:.2f}%)")
    print(f"  Precision    : {box.mp:.4f}   ({box.mp*100:.2f}%)")
    print(f"  Recall       : {box.mr:.4f}   ({box.mr*100:.2f}%)")

    f1 = 2 * box.mp * box.mr / max(box.mp + box.mr, 1e-6)
    print(f"  F1 Score     : {f1:.4f}")

    print(f"\n[Class 별 성능]")
    names = ["vehicle", "pedestrian"]
    header = f"  {'Class':<12} {'AP@0.5':>10} {'AP@.5:.95':>12} {'Precision':>10} {'Recall':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, name in enumerate(names):
        try:
            ap50 = box.ap50[i] if hasattr(box, "ap50") else float("nan")
            ap   = box.ap[i]   if hasattr(box, "ap")   else float("nan")
            p    = box.p[i]    if hasattr(box, "p")    else float("nan")
            r    = box.r[i]    if hasattr(box, "r")    else float("nan")
            print(f"  {name:<12} {ap50:>10.4f} {ap:>12.4f} {p:>10.4f} {r:>10.4f}")
        except (IndexError, AttributeError, TypeError):
            print(f"  {name:<12} (해당 클래스 검출 없음)")

    print("\n" + "=" * 72)
    print(f"[발표용 한 줄 요약]")
    print(f"  ▶  mAP@0.5 = {box.map50*100:.1f}%  |  "
          f"mAP@0.5:0.95 = {box.map*100:.1f}%")
    print("=" * 72)

    # 부가 정보: 추론 시간
    if hasattr(results, "speed"):
        speed = results.speed
        total_ms = sum(speed.values())
        fps = 1000 / total_ms if total_ms > 0 else 0
        print(f"\n[추론 속도]")
        print(f"  Pre-process : {speed.get('preprocess', 0):.2f} ms")
        print(f"  Inference   : {speed.get('inference', 0):.2f} ms")
        print(f"  Post-process: {speed.get('postprocess', 0):.2f} ms")
        print(f"  Total       : {total_ms:.2f} ms  →  {fps:.1f} FPS")

    # 자동 저장된 plots 위치 안내
    print(f"\n[자동 생성된 시각화]")
    print(f"  PR curve, Confusion Matrix → runs/detect/val*/ 에 저장됨")
    print(f"  확인: ls runs/detect/")


if __name__ == "__main__":
    main()
