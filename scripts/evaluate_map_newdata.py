"""
새 데이터 (학습에 안 쓴 unseen) 로 mAP 측정
============================================
누수 없는 진짜 평가. 학습된 모델 그대로 사용.

데이터 구조 (예시):
  data/morai_test/
  ├── dusk/
  │   ├── images/ *.jpg
  │   └── labels/ *.txt
  └── night/
      ├── images/
      └── labels/

실행:
    python3 scripts/evaluate_map_newdata.py
    python3 scripts/evaluate_map_newdata.py --data-root data/morai_test
"""

import argparse
import datetime
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.sgldet_yolov8 import SGLDetYOLO
from ultralytics import YOLO


def get_all_images(root: str, conditions=None):
    """지정 폴더의 모든 원본 이미지 수집 (aug 제외)."""
    conditions = conditions or ["dusk", "night"]
    all_imgs = []
    for cond in conditions:
        img_dir = Path(root) / cond / "images"
        if not img_dir.exists():
            print(f"  [Skip] {img_dir} 없음")
            continue
        files = sorted([
            f for f in img_dir.glob("*")
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and "_aug" not in f.stem
        ])
        all_imgs.extend(files)
        print(f"  {cond}: {len(files)} 이미지")
    return all_imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   default="checkpoints/best_model.pt")
    parser.add_argument("--data-root", default="data/morai_test",
                        help="새 평가용 데이터 폴더 (학습에 안 쓴 것)")
    parser.add_argument("--img-size",  type=int, default=640)
    parser.add_argument("--batch",     type=int, default=8)
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--conf",      type=float, default=0.001)
    parser.add_argument("--iou",       type=float, default=0.6)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── 1. 평가 이미지 수집 ──────────────────────────────────
    print(f"\n[새 데이터 수집] {args.data_root}/")
    test_images = get_all_images(args.data_root)
    print(f"  총 {len(test_images)} 이미지")

    if not test_images:
        print(f"\n[Error] {args.data_root} 에 이미지가 없습니다.")
        print(f"  → collect_data.py 로 새 데이터를 먼저 수집하세요:")
        print(f"     python3 collect_data.py --condition night --output {args.data_root} --auto")
        return

    # 라벨 존재 확인
    label_count = 0
    for img in test_images:
        lbl_dir = img.parent.parent / "labels"
        if (lbl_dir / (img.stem + ".txt")).exists():
            label_count += 1
    print(f"  라벨 존재: {label_count}/{len(test_images)}")

    if label_count == 0:
        print("\n[Error] 라벨 파일이 없습니다. collect_data.py 로 GT 라벨도 함께 수집했는지 확인하세요.")
        return

    # ── 2. val.txt + data.yaml 생성 ─────────────────────────
    test_list_path = Path("eval_newdata.txt").resolve()
    with open(test_list_path, "w") as f:
        for img in test_images:
            f.write(str(img.resolve()) + "\n")

    yaml_path = Path("eval_newdata.yaml").resolve()
    data = {
        "path":  str(Path.cwd().resolve()),
        "train": str(test_list_path),  # dummy
        "val":   str(test_list_path),
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

    # ── 4. detector → ultralytics 호환 .pt ───────────────────
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
    print(f"\n[Evaluating] 새 데이터 {len(test_images)} 이미지에서 mAP 측정...")
    yolo = YOLO(str(tmp_pt))
    results = yolo.val(
        data=str(yaml_path), imgsz=args.img_size, batch=args.batch,
        device=device, conf=args.conf, iou=args.iou,
        save_json=False, verbose=True, plots=True,
    )

    # ── 6. 결과 ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" " * 14 + "🎯 진짜 Unseen Data mAP 측정 결과")
    print("=" * 72)

    box = results.box
    print(f"\n[전체 Metric]  (완전히 새로운 {len(test_images)} 이미지)")
    print(f"  mAP@0.5      : {box.map50:.4f}   ({box.map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95 : {box.map:.4f}   ({box.map*100:.2f}%)")
    print(f"  Precision    : {box.mp:.4f}   ({box.mp*100:.2f}%)")
    print(f"  Recall       : {box.mr:.4f}   ({box.mr*100:.2f}%)")

    f1 = 2 * box.mp * box.mr / max(box.mp + box.mr, 1e-6)
    print(f"  F1 Score     : {f1:.4f}")

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
            print(f"  {name:<12} (해당 클래스 없음)")

    print("\n" + "=" * 72)
    print("[3가지 평가 비교]")
    print(f"  1. Random val (누수 有)         : mAP@0.5 = 97.7%")
    print(f"  2. Scenario val (부분 누수)     : mAP@0.5 = 95.9%")
    print(f"  3. New data (누수 X) ⭐         : mAP@0.5 = {box.map50*100:.1f}%  ← 진짜!")
    print("=" * 72)


if __name__ == "__main__":
    main()
