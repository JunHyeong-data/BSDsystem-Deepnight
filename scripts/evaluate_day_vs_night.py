"""
주간 vs 야간 정량 비교 평가
=============================
동일 학습 모델 (Plain FT + Scenario split 최종 model) 을 여러 조건 데이터셋에
적용하여 mAP/Precision/Recall 차이를 표·JSON 으로 산출.

목적: "야간 BSD 는 어렵다" 라는 프로젝트 motivation 을 데이터로 입증.

데이터 구조 (각 condition 별 동일):
  <root>/
  ├── <condition>/
  │   ├── images/  *.jpg|png
  │   └── labels/  *.txt   (YOLO format: class cx cy w h)

실행:
    python3 scripts/evaluate_day_vs_night.py \
        --weights checkpoints/best_yolo_only.pt \
        --datasets night=data/morai/night day=data/morai_day/day

    # 또는 여러 조건 동시:
    python3 scripts/evaluate_day_vs_night.py \
        --weights checkpoints/best_yolo_only.pt \
        --datasets day=data/morai_day/day dusk=data/morai/dusk night=data/morai/night
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


CLASS_NAMES = ["vehicle", "pedestrian"]


def collect_images(root: Path, condition: str) -> list[Path]:
    """<root>/<condition>/images 에서 aug 제외 원본 이미지 수집."""
    img_dir = root / condition / "images"
    if not img_dir.exists():
        return []
    return sorted([
        f for f in img_dir.glob("*")
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and "_aug" not in f.stem
    ])


def has_label(img: Path) -> bool:
    return (img.parent.parent / "labels" / (img.stem + ".txt")).exists()


def build_eval_yaml(images: list[Path], out_yaml: Path) -> Path:
    """Ultralytics val() 용 dataset yaml + image list 작성."""
    list_txt = out_yaml.with_suffix(".txt")
    with open(list_txt, "w") as f:
        for img in images:
            f.write(str(img.resolve()) + "\n")

    data = {
        "path":  str(Path.cwd().resolve()),
        "train": str(list_txt.resolve()),   # dummy
        "val":   str(list_txt.resolve()),
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(out_yaml, "w") as f:
        yaml.dump(data, f, allow_unicode=True)
    return out_yaml


def evaluate_one(
    label: str,
    images: list[Path],
    weights: str,
    img_size: int,
    batch: int,
    device: str,
    conf: float,
    iou: float,
    tmp_root: Path,
) -> dict:
    """한 조건의 val() 실행 후 metric dict 반환."""
    yaml_path = tmp_root / f"eval_{label}.yaml"
    build_eval_yaml(images, yaml_path)

    yolo = YOLO(weights)
    # name 인자로 결과 폴더 분리 — runs/detect/eval_<label>
    results = yolo.val(
        data=str(yaml_path), imgsz=img_size, batch=batch,
        device=device, conf=conf, iou=iou,
        save_json=False, verbose=False, plots=False,
        name=f"eval_{label}",
    )
    b = results.box
    metrics = {
        "n_images":   len(images),
        "mAP50":      float(b.map50),
        "mAP50_95":   float(b.map),
        "precision":  float(b.mp),
        "recall":     float(b.mr),
    }

    # class-wise (vehicle 만이 BSD scope 주 metric)
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        try:
            per_class[name] = {
                "AP50":      float(b.ap50[i]),
                "AP50_95":   float(b.ap[i]),
                "precision": float(b.p[i]),
                "recall":    float(b.r[i]),
            }
        except (IndexError, AttributeError, TypeError):
            per_class[name] = None
    metrics["per_class"] = per_class

    return metrics


def parse_datasets(spec_list: list[str]) -> list[tuple[str, Path, str]]:
    """
    "<label>=<root>" 또는 "<label>=<root>:<condition>" 파싱.

    예) "night=data/morai/night"           → label=night, root=data/morai, condition=night
        "day=data/morai_day:day"           → label=day,   root=data/morai_day, condition=day
        "night=data/morai/night"           → root 끝 폴더를 condition 으로 자동 추정
    """
    out = []
    for spec in spec_list:
        if "=" not in spec:
            raise ValueError(f"--datasets 항목은 'label=path' 형식이어야 합니다: {spec}")
        label, path = spec.split("=", 1)
        if ":" in path:
            root_str, condition = path.split(":", 1)
            root = Path(root_str)
        else:
            # path 가 .../<condition> 구조라고 가정 → root = path.parent, condition = path.name
            p = Path(path)
            root = p.parent
            condition = p.name
        out.append((label.strip(), root.resolve(), condition.strip()))
    return out


def print_comparison_table(all_metrics: dict[str, dict]) -> None:
    """모든 조건의 metric 을 가로 비교 표로 출력."""
    labels = list(all_metrics.keys())
    print("\n" + "=" * 78)
    print(" " * 22 + "🌗 주간 / 야간 정량 비교 결과")
    print("=" * 78)

    # 전체 metric
    print(f"\n[Overall — {', '.join(labels)}]")
    header = f"  {'Metric':<14}" + "".join(f"{lbl:>12}" for lbl in labels)
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = [
        ("# images",    "n_images"),
        ("mAP@0.5",     "mAP50"),
        ("mAP@.5:.95",  "mAP50_95"),
        ("Precision",   "precision"),
        ("Recall",      "recall"),
    ]
    for name, key in rows:
        line = f"  {name:<14}"
        for lbl in labels:
            v = all_metrics[lbl][key]
            line += f"{v:>12.4f}" if isinstance(v, float) else f"{v:>12d}"
        print(line)

    # Δ (조건이 정확히 2개일 때만 의미 있음)
    if len(labels) == 2:
        a, b = labels
        print("  " + "-" * (len(header) - 2))
        line = f"  {'Δ ('+b+' - '+a+')':<14}"
        for name, key in rows:
            if key == "n_images":
                line += f"{'-':>12}"
            else:
                d = all_metrics[b][key] - all_metrics[a][key]
                line += f"{d:>+12.4f}"
        print(line)

    # class-wise (BSD scope: vehicle 만)
    print("\n[Class-wise: vehicle (BSD 1차 평가 대상)]")
    header = f"  {'Metric':<14}" + "".join(f"{lbl:>12}" for lbl in labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, key in [("AP@0.5", "AP50"), ("AP@.5:.95", "AP50_95"),
                      ("Precision", "precision"), ("Recall", "recall")]:
        line = f"  {name:<14}"
        for lbl in labels:
            v = all_metrics[lbl]["per_class"].get("vehicle")
            line += f"{v[key]:>12.4f}" if v else f"{'(N/A)':>12}"
        print(line)

    print("\n[Class-wise: pedestrian (참고용 — BSD scope 외)]")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, key in [("AP@0.5", "AP50"), ("AP@.5:.95", "AP50_95"),
                      ("Precision", "precision"), ("Recall", "recall")]:
        line = f"  {name:<14}"
        for lbl in labels:
            v = all_metrics[lbl]["per_class"].get("pedestrian")
            line += f"{v[key]:>12.4f}" if v else f"{'(N/A)':>12}"
        print(line)

    print("\n" + "=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Day vs Night 정량 비교 평가")
    parser.add_argument("--weights", default="checkpoints/best_yolo_only.pt",
                        help="평가에 사용할 ultralytics-호환 weight (final model)")
    parser.add_argument("--datasets", nargs="+", required=True,
                        metavar="LABEL=PATH[:CONDITION]",
                        help="비교할 조건들. 예: day=data/morai_day/day night=data/morai/night")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--conf",     type=float, default=0.001,
                        help="ultralytics val 표준값 (mAP 계산용). 배포 임계값과 다름.")
    parser.add_argument("--iou",      type=float, default=0.6)
    parser.add_argument("--output",   default="day_vs_night_results.json")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[Error] weights 없음: {weights}")
        sys.exit(1)

    datasets = parse_datasets(args.datasets)
    print(f"\n[비교 대상] {len(datasets)} 조건")
    for label, root, cond in datasets:
        print(f"  - {label}: {root}/{cond}/")

    tmp_root = Path("/tmp")  # eval yaml 임시 저장

    all_metrics: dict[str, dict] = {}
    for label, root, condition in datasets:
        imgs = collect_images(root, condition)
        labeled = [im for im in imgs if has_label(im)]
        if not labeled:
            print(f"\n[Skip] {label}: 라벨 있는 이미지 없음 ({root}/{condition}/labels/)")
            print(f"  → collect_data.py 로 GT 라벨도 같이 수집했는지 확인")
            continue

        print(f"\n[Evaluating] {label} — {len(labeled)} images "
              f"(of {len(imgs)} found, {len(imgs)-len(labeled)} 미라벨 제외)")
        metrics = evaluate_one(
            label    = label,
            images   = labeled,
            weights  = str(weights),
            img_size = args.img_size,
            batch    = args.batch,
            device   = device,
            conf     = args.conf,
            iou      = args.iou,
            tmp_root = tmp_root,
        )
        all_metrics[label] = metrics

    if not all_metrics:
        print("\n[Error] 평가 가능한 조건이 하나도 없음")
        sys.exit(1)

    print_comparison_table(all_metrics)

    # JSON 저장
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"weights": str(weights), "results": all_metrics},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out_path}")

    # 해석 가이드 한 줄
    if len(all_metrics) == 2:
        a, b = list(all_metrics.keys())
        d_map = all_metrics[b]["mAP50"] - all_metrics[a]["mAP50"]
        print(f"\n💡 Interpretation:")
        print(f"   mAP@0.5 difference ({b} - {a}) = {d_map:+.4f}")
        if d_map < -0.05:
            print(f"   → {a} 가 {b} 보다 의미 있게 어려움 (Δ {abs(d_map)*100:.1f} %p drop).")
            print(f"     '야간 BSD 가 어렵다' motivation 이 데이터로 입증됨.")
        elif d_map > 0.05:
            print(f"   → {b} 가 {a} 보다 어려움 (역방향).")
        else:
            print(f"   → 두 조건 차이 미미. motivation 재검토 필요.")


if __name__ == "__main__":
    main()
