"""
YOLO 라벨 자동 생성기 (Windows 전용, ROS2 불필요)
===================================================
Linux에서 collect_raw.py 로 수집한 이미지 쌍을 처리.

입력:
  raw_data/{condition}/
    {ts}_rgb.jpg    ← RGB 이미지
    {ts}_mask.jpg   ← Instance 마스크

출력 (train.py가 기대하는 구조):
  data/morai/{condition}/images/{ts}.jpg
  data/morai/{condition}/labels/{ts}.txt   ← YOLO 형식

실행:
  python generate_labels.py --condition night
  python generate_labels.py --condition night --input raw_data/night
  python generate_labels.py --condition dusk  --preview   # 결과 확인용 창
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ── BSD 클래스 ────────────────────────────────────────────────────────────────

BSD_CLASSES = {0: "car", 1: "pedestrian", 2: "truck"}

COCO_TO_BSD = {
    0: (1, "pedestrian"),
    2: (0, "car"),
    5: (2, "truck"),
    7: (2, "truck"),
}

CLASS_COLORS = {
    0: (0, 200, 0),
    1: (255, 140, 0),
    2: (0, 0, 255),
    3: (255, 0, 255),
}


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def iou(b1: tuple, b2: tuple) -> float:
    """IoU 계산. b = (x, y, w, h)."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2]); y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def size_heuristic(w: int, h: int) -> tuple[int, str]:
    """크기/종횡비로 클래스 추정 (YOLO 매칭 실패 폴백)."""
    aspect = h / (w + 1e-6)
    area   = w * h
    if aspect > 1.5:                    return (1, "pedestrian")
    elif area > 40000:                  return (2, "truck")
    else:                               return (0, "car")


# ── 핵심 처리 함수 ────────────────────────────────────────────────────────────

def extract_instance_bboxes(
    mask: np.ndarray,
    min_area: int,
    fisheye_cx: int = 640,
    fisheye_cy: int = 360,
    fisheye_r:  int = 350,
) -> list[tuple]:
    """
    Instance 마스크에서 연결 컴포넌트 bbox 추출.
    흰색 배경(≥240) 제외.

    Fish-eye ROI 필터:
      179° 어안렌즈의 유효 원형 영역 밖(검은 원형 테두리)은
      connectedComponents 이전에 마스킹하여 가짜 bbox 원천 차단.
      기본값: 중심 (640, 360), 반지름 350 px (1280×720 기준).
    """
    H, W = mask.shape[:2]

    # ── ① 흰색 배경 제거 ─────────────────────────────────────
    bg = np.all(mask >= 240, axis=2)
    obj_bin = (~bg).astype(np.uint8) * 255

    # ── ② Fish-eye 유효 원형 ROI 마스크 적용 ─────────────────
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(roi_mask,
               center=(fisheye_cx, fisheye_cy),
               radius=fisheye_r,
               color=255,
               thickness=-1)
    obj_bin = cv2.bitwise_and(obj_bin, roi_mask)

    # ── ③ 모폴로지 오픈으로 노이즈 제거 ──────────────────────
    kernel = np.ones((3, 3), np.uint8)
    obj_bin = cv2.morphologyEx(obj_bin, cv2.MORPH_OPEN, kernel)

    # ── ④ 연결 컴포넌트 ──────────────────────────────────────
    n, _, stats, _ = cv2.connectedComponentsWithStats(obj_bin, connectivity=8)
    bboxes = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = max(0, stats[i, cv2.CC_STAT_LEFT])
        y = max(0, stats[i, cv2.CC_STAT_TOP])
        w = min(stats[i, cv2.CC_STAT_WIDTH],  W - x)
        h = min(stats[i, cv2.CC_STAT_HEIGHT], H - y)
        if w < 10 or h < 10:
            continue
        bboxes.append((x, y, w, h))

    return bboxes


def classify_with_yolo(
    rgb: np.ndarray,
    inst_bboxes: list[tuple],
    yolo: YOLO,
    conf_thres: float,
    iou_match: float,
) -> list[dict]:
    """
    YOLOv8 전체 추론 → IoU 매칭 → 각 인스턴스에 클래스 부여.
    """
    H, W = rgb.shape[:2]

    # YOLOv8 추론
    results = yolo.predict(rgb, conf=conf_thres, verbose=False)
    yolo_dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cid = int(box.cls[0].item())
            if cid not in COCO_TO_BSD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bsd_id, bsd_name = COCO_TO_BSD[cid]
            yolo_dets.append({
                "bbox": (x1, y1, x2-x1, y2-y1),
                "cls_id": bsd_id,
                "cls_name": bsd_name,
                "conf": float(box.conf[0].item()),
            })

    # IoU 매칭
    matched = set()
    detections = []

    for ibbox in inst_bboxes:
        best_iou = 0.0
        best_det = None
        best_idx = -1

        for j, yd in enumerate(yolo_dets):
            if j in matched:
                continue
            score = iou(ibbox, yd["bbox"])
            if score > best_iou:
                best_iou = score
                best_det = yd
                best_idx = j

        if best_iou >= iou_match and best_det is not None:
            cls_id   = best_det["cls_id"]
            cls_name = best_det["cls_name"]
            conf     = best_det["conf"]
            matched.add(best_idx)
        else:
            # 폴백: 크기 휴리스틱
            cls_id, cls_name = size_heuristic(ibbox[2], ibbox[3])
            conf = 0.40

        x, y, w, h = ibbox
        cx_n = max(0.0, min(1.0, (x + w/2) / W))
        cy_n = max(0.0, min(1.0, (y + h/2) / H))
        w_n  = max(0.0, min(1.0, w / W))
        h_n  = max(0.0, min(1.0, h / H))

        detections.append({
            "cls_id":   cls_id,
            "cls_name": cls_name,
            "conf":     conf,
            "bbox":     ibbox,
            "yolo":     (cx_n, cy_n, w_n, h_n),
        })

    return detections


def make_preview(
    rgb: np.ndarray,
    mask: np.ndarray,
    detections: list[dict],
    filename: str,
) -> np.ndarray:
    """시각화 이미지 생성."""
    vis = rgb.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        color = CLASS_COLORS.get(det["cls_id"], (200, 200, 200))
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        label = f"{det['cls_name']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x, y-th-6), (x+tw+4, y), color, -1)
        cv2.putText(vis, label, (x+2, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    cv2.putText(vis, filename, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    return vis


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="YOLO 라벨 자동 생성기 (Windows)")
    p.add_argument("--condition", "-c",
                   choices=["dusk", "night"], default="night")
    p.add_argument("--input", "-i", default=None,
                   help="원시 이미지 폴더 (기본: raw_data/{condition})")
    p.add_argument("--output", "-o", default="data/morai",
                   help="YOLO 데이터셋 출력 루트 (기본: data/morai)")
    p.add_argument("--weights",    default="yolov8m.pt")
    p.add_argument("--conf",       type=float, default=0.20)
    p.add_argument("--iou-match",  type=float, default=0.30)
    p.add_argument("--min-area",   type=int,   default=800)
    p.add_argument("--preview",    action="store_true",
                   help="처리 결과 미리보기 창")
    p.add_argument("--skip-heuristic", action="store_true",
                   help="YOLO 매칭 실패 시 저장 자체를 건너뜀")

    # Fish-eye ROI (어안렌즈 가장자리 검은 테두리 제거)
    p.add_argument("--fisheye-cx", type=int, default=640,
                   help="Fish-eye 유효 원 중심 x (기본: 640)")
    p.add_argument("--fisheye-cy", type=int, default=360,
                   help="Fish-eye 유효 원 중심 y (기본: 360)")
    p.add_argument("--fisheye-r",  type=int, default=350,
                   help="Fish-eye 유효 반지름 px (기본: 350)")

    args = p.parse_args()

    # 경로 설정
    in_dir  = Path(args.input) if args.input else Path("raw_data") / args.condition
    out_img = Path(args.output) / args.condition / "images"
    out_lbl = Path(args.output) / args.condition / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[오류] 입력 폴더 없음: {in_dir}")
        print(f"  → Linux에서 collect_raw.py 로 먼저 수집하세요.")
        return

    # RGB 파일 목록 수집
    rgb_files = sorted(in_dir.glob("*_rgb.jpg"))
    if not rgb_files:
        print(f"[오류] {in_dir} 에 *_rgb.jpg 파일이 없습니다.")
        return

    print(f"\n{'='*55}")
    print(f"  입력 : {in_dir}  ({len(rgb_files)}쌍)")
    print(f"  출력 : {out_img.parent}")
    print(f"  조도 : {args.condition}")
    print(f"{'='*55}\n")

    # YOLOv8 로드
    print("YOLOv8 로드 중...")
    yolo = YOLO(args.weights)
    print("YOLOv8 로드 완료\n")

    saved = skipped = heuristic = 0

    for i, rgb_path in enumerate(rgb_files):
        # 대응하는 마스크 파일
        mask_path = in_dir / rgb_path.name.replace("_rgb.jpg", "_mask.jpg")
        if not mask_path.exists():
            print(f"  [SKIP] 마스크 없음: {mask_path.name}")
            skipped += 1
            continue

        # 이미지 로드
        rgb  = cv2.imread(str(rgb_path))
        mask = cv2.imread(str(mask_path))
        if rgb is None or mask is None:
            skipped += 1
            continue

        # ① 인스턴스 bbox 추출 (fish-eye ROI 필터 적용)
        inst_bboxes = extract_instance_bboxes(
            mask, args.min_area,
            fisheye_cx=args.fisheye_cx,
            fisheye_cy=args.fisheye_cy,
            fisheye_r=args.fisheye_r,
        )
        if not inst_bboxes:
            skipped += 1
            continue

        # ② YOLOv8 분류 + IoU 매칭
        detections = classify_with_yolo(
            rgb, inst_bboxes, yolo, args.conf, args.iou_match,
        )

        # 휴리스틱만 있는 경우 건너뛰기 (옵션)
        if args.skip_heuristic:
            detections = [d for d in detections if d["conf"] > 0.41]

        if not detections:
            skipped += 1
            continue

        # 타임스탬프 추출 (파일명에서)
        ts = rgb_path.stem.replace("_rgb", "")

        # ③ 이미지 복사 + 라벨 저장
        shutil.copy2(str(rgb_path), str(out_img / f"{ts}.jpg"))

        lbl_path = out_lbl / f"{ts}.txt"
        lines = []
        for det in detections:
            cx, cy, w, h = det["yolo"]
            lines.append(f"{det['cls_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 휴리스틱 사용 개수 추적
        heuristic += sum(1 for d in detections if d["conf"] <= 0.41)
        saved += 1

        # 진행 상황
        obj_str = " ".join(
            f"{d['cls_name']}({'H' if d['conf']<=0.41 else f'{d[\"conf\"]:.2f}'})"
            for d in detections
        )
        print(f"  [{i+1:4d}/{len(rgb_files)}] {ts}.jpg | {len(detections)}개: {obj_str}")

        # 미리보기
        if args.preview:
            vis = make_preview(rgb, mask, detections, rgb_path.name)
            mask_small = cv2.resize(mask, (vis.shape[1]//2, vis.shape[0]//2))
            cv2.imshow("Result", vis)
            cv2.imshow("Mask",   mask_small)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cv2.destroyAllWindows()

    # 최종 요약
    print(f"\n{'='*55}")
    print(f"  처리 완료")
    print(f"  저장: {saved}개 | 스킵: {skipped}개")
    if heuristic > 0:
        print(f"  ※ 휴리스틱 사용: {heuristic}개 (YOLO 미매칭 → 크기로 추정)")
    print(f"\n  출력 경로:")
    print(f"    이미지: {out_img}")
    print(f"    라벨:   {out_lbl}")
    print(f"\n  이제 학습 실행:")
    print(f"    python train.py --condition {args.condition}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
