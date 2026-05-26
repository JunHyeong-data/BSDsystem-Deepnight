"""
Baseline comparison — fine-tune 효과 측정
==========================================
같은 in-domain test 60장에서 4개 모델 비교:
  1. YOLOv8n (COCO pretrained, no fine-tune)
  2. YOLOv8s (COCO pretrained, no fine-tune)
  3. YOLOv8m (COCO pretrained, no fine-tune)
  4. Ours (SGLDet + YOLOv8m, fine-tuned on 902 MORAI frames)

COCO 모델의 80 class → 우리 2 class 매핑:
  COCO car(2) / motorcycle(3) / bus(5) / truck(7)  →  vehicle (0)
  COCO person(0)                                    →  pedestrian (1)

출력:
  - 콘솔: 4-row mAP@0.5 + class-wise 표
  - compare_baseline_results.json: 전체 결과
  - compare_visual.png: COCO YOLOv8m vs Ours 4장 side-by-side
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COCO_VEHICLE = {2, 3, 5, 7}
COCO_PERSON = {0}
CLS_MAP_COCO = {**{c: 0 for c in COCO_VEHICLE}, **{c: 1 for c in COCO_PERSON}}


def load_gt(label_path, W, H):
    if not label_path.exists():
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    txt = label_path.read_text().strip()
    if not txt:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    boxes, classes = [], []
    for line in txt.split("\n"):
        parts = line.split()
        c = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        boxes.append([(xc - w/2) * W, (yc - h/2) * H, (xc + w/2) * W, (yc + h/2) * H])
        classes.append(c)
    return np.array(boxes), np.array(classes, dtype=int)


def predict_mapped(model, img_path, class_map=None, conf=0.001, iou=0.6):
    """Run model, optionally remap class indices and filter unknown classes."""
    res = model.predict(str(img_path), conf=conf, iou=iou, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    if class_map is None:
        return boxes, scores, classes
    mask = np.array([c in class_map for c in classes])
    if not mask.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return (
        boxes[mask],
        scores[mask],
        np.array([class_map[c] for c in classes[mask]], dtype=int),
    )


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def compute_ap50(predictions, ground_truths, num_classes=2, iou_thr=0.5):
    """Pascal-VOC style AP per class."""
    per_class = {c: {"tp": [], "fp": [], "scores": [], "n_gt": 0} for c in range(num_classes)}

    for (pb, ps, pc), (gb, gc) in zip(predictions, ground_truths):
        for c in range(num_classes):
            per_class[c]["n_gt"] += int((gc == c).sum())
        if len(pb) == 0:
            continue
        order = np.argsort(-ps)
        pb, ps, pc = pb[order], ps[order], pc[order]
        matched = set()
        for b, s, c in zip(pb, ps, pc):
            best_iou, best_j = 0.0, -1
            for j in range(len(gb)):
                if j in matched or gc[j] != c:
                    continue
                io = iou_xyxy(b, gb[j])
                if io > best_iou:
                    best_iou, best_j = io, j
            if best_iou >= iou_thr:
                per_class[int(c)]["tp"].append(1)
                per_class[int(c)]["fp"].append(0)
                matched.add(best_j)
            else:
                per_class[int(c)]["tp"].append(0)
                per_class[int(c)]["fp"].append(1)
            per_class[int(c)]["scores"].append(float(s))

    out = {}
    for c, d in per_class.items():
        if d["n_gt"] == 0:
            out[c] = {"AP": 0.0, "P": 0.0, "R": 0.0, "TP": 0, "FP": sum(d["fp"]), "FN": 0}
            continue
        order = np.argsort(-np.array(d["scores"])) if d["scores"] else np.array([], dtype=int)
        tp = np.array(d["tp"])[order] if len(order) else np.zeros(0)
        fp = np.array(d["fp"])[order] if len(order) else np.zeros(0)
        cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
        recall = cum_tp / d["n_gt"]
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[0.0], precision, [0.0]])
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        out[c] = {
            "AP": ap,
            "P": float(precision[-1]) if len(precision) else 0.0,
            "R": float(recall[-1]) if len(recall) else 0.0,
            "TP": int(cum_tp[-1]) if len(cum_tp) else 0,
            "FP": int(cum_fp[-1]) if len(cum_fp) else 0,
            "FN": d["n_gt"] - (int(cum_tp[-1]) if len(cum_tp) else 0),
        }
    return out


def visualize_3col(images, coco_pair, ft_pair, ours_pair, out_path, conf_show=0.3,
                    titles=None):
    """3-column 시각 비교 (기본 라벨: SGLDet 원본 / Plain FT+Scn / Ours)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    coco_model, coco_map = coco_pair
    ft_model, ft_map = ft_pair
    ours_model, ours_map = ours_pair
    CLR = {0: "lime", 1: "red"}
    NAME = {0: "vehicle", 1: "pedestrian"}

    fig, axes = plt.subplots(len(images), 3, figsize=(18, 4 * len(images)))
    if len(images) == 1:
        axes = axes[None, :]

    for row, (img_path, _) in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        coco_det = predict_mapped(coco_model, img_path, coco_map, conf=conf_show)
        ft_det = predict_mapped(ft_model, img_path, ft_map, conf=conf_show)
        ours_det = predict_mapped(ours_model, img_path, ours_map, conf=conf_show)
        triplets = [
            (*coco_det, f"SGLDet (Full FT + random) — {len(coco_det[0])} det"),
            (*ft_det,   f"Plain FT + Scenario split — {len(ft_det[0])} det"),
            (*ours_det, f"Ours: SGLDet+Freeze+Scn — {len(ours_det[0])} det"),
        ]
        for col, (bx, sc, cl, title) in enumerate(triplets):
            ax = axes[row, col]
            ax.imshow(img)
            for b, s, c in zip(bx, sc, cl):
                x1, y1, x2, y2 = b
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                               fill=False, edgecolor=CLR[c], linewidth=2))
                ax.text(x1, max(0, y1 - 4), f"{NAME[c]} {s:.2f}",
                        color=CLR[c], fontsize=8, weight="bold",
                        bbox=dict(facecolor="black", alpha=0.4, pad=1))
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    # 5개 모델 — (가중치 경로, 클래스 매핑). 4-stage ablation 진행 보여줌.
    BASELINE_FT_RANDOM = "runs/detect/runs/baseline_ft/yolov8m_ft_902/weights/best.pt"
    BASELINE_FT_SCN    = "runs/detect/runs/baseline_ft_scn/yolov8m_ft_902_scn/weights/best.pt"
    SGLDET_ORIG        = "checkpoints/best_yolo_only_sgldet_orig.pt"
    SGLDET_FREEZE_SCN  = "checkpoints/best_yolo_only_freeze_scn.pt"
    model_specs = [
        ("COCO YOLOv8m (no FT)",                  "yolov8m.pt",         CLS_MAP_COCO),
        ("Plain FT (random split)",               BASELINE_FT_RANDOM,   None),
        ("Plain FT + Scenario split",             BASELINE_FT_SCN,      None),
        ("SGLDet (Full FT + random split)",       SGLDET_ORIG,          None),
        ("Ours: SGLDet + Freeze + Scn",           SGLDET_FREEZE_SCN,    None),
    ]

    # 60장 수집
    images = []
    for cond in ["night", "dusk"]:
        img_dir = Path(f"data/morai_test/{cond}/images")
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            if lbl.exists():
                images.append((img, lbl))
    print(f"평가 이미지: {len(images)} 장")

    # GT 한 번만 로드
    ground_truths = []
    for img_path, lbl_path in images:
        W, H = Image.open(img_path).size
        ground_truths.append(load_gt(lbl_path, W, H))

    # 각 모델 × conf 두 가지 평가
    all_results = {}      # {name: {conf: results_dict}}
    loaded_models = {}
    CONFS = [0.001, 0.5]

    for name, weights, cmap in model_specs:
        print(f"\n[{name}] loading {weights}...")
        model = YOLO(weights)
        loaded_models[name] = (model, cmap)
        all_results[name] = {}
        for conf in CONFS:
            preds = [predict_mapped(model, img, cmap, conf=conf) for img, _ in images]
            r = compute_ap50(preds, ground_truths)
            all_results[name][conf] = r
            mAP = (r[0]["AP"] + r[1]["AP"]) / 2
            print(f"  conf={conf:.3f}  mAP@0.5={mAP*100:6.2f}%  "
                  f"V-AP={r[0]['AP']*100:6.2f}%  P-AP={r[1]['AP']*100:6.2f}%  "
                  f"V-Recall={r[0]['R']*100:6.2f}%  P-Recall={r[1]['R']*100:6.2f}%")

    # 요약 표 (conf 별 분리)
    for conf in CONFS:
        print("\n" + "=" * 96)
        print(f" In-domain Test 60장 — conf={conf}")
        print("-" * 96)
        print(f"{'Model':<33} {'mAP@0.5':>10} {'Vehicle AP':>12} {'Ped AP':>10} {'V-Recall':>10} {'P-Recall':>10}")
        print("-" * 96)
        for name in [s[0] for s in model_specs]:
            r = all_results[name][conf]
            mAP = (r[0]["AP"] + r[1]["AP"]) / 2
            print(f"{name:<33} {mAP*100:>9.2f}% {r[0]['AP']*100:>11.2f}% {r[1]['AP']*100:>9.2f}% "
                  f"{r[0]['R']*100:>9.2f}% {r[1]['R']*100:>9.2f}%")
        print("=" * 96)

    # JSON 저장
    json_out = {
        name: {f"conf_{c}": {str(k): v for k, v in r.items()} for c, r in conf_results.items()}
        for name, conf_results in all_results.items()
    }
    Path("compare_baseline_results.json").write_text(json.dumps(json_out, indent=2))
    print("\n결과 저장: compare_baseline_results.json")

    # 시각 비교 — SGLDet 원본 vs Plain FT (scn) vs Ours (freeze+scn), 4장 3-column
    print("\n[시각 비교 생성] SGLDet 원본 / Plain FT+Scn / Ours, 4장 sample 3-column...")
    sample = images[::max(1, len(images) // 4)][:4]
    visualize_3col(
        sample,
        loaded_models["SGLDet (Full FT + random split)"],
        loaded_models["Plain FT + Scenario split"],
        loaded_models["Ours: SGLDet + Freeze + Scn"],
        "compare_visual.png",
    )
    print("  저장: compare_visual.png")


if __name__ == "__main__":
    main()
