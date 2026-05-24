"""
BSD 통합 데모 스크립트 (정적 이미지)
====================================
학습 데이터의 이미지에서 detection + SORT + BSD warning 전체 파이프라인 시연.
결과를 demo_output/ 에 저장.

실행:
    python3 scripts/demo_bsd.py
    python3 scripts/demo_bsd.py --condition dusk --n 10
"""

import argparse
import random
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector       import SGLDetInference
from src.inference.bsd_interface  import BSDInterface
from models.sort_tracker          import SORTTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   default="checkpoints/best_model.pt")
    parser.add_argument("--data-root", default="data/morai")
    parser.add_argument("--condition", default="dusk", choices=["dusk", "night"])
    parser.add_argument("--camera-cfg", default="configs/camera_config.yaml")
    parser.add_argument("--n",         type=int, default=10,
                        help="시연할 이미지 수")
    parser.add_argument("--output",    default="demo_output")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--mix-size",  action="store_true",
                        help="bbox 크기로 균형 선택 (멀리/중간/가까이)")
    parser.add_argument("--conf",      type=float, default=0.5,
                        help="confidence threshold (낮으면 false positive ↑)")
    args = parser.parse_args()

    # ── 1. 이미지 샘플 ────────────────────────────────────────
    img_dir = Path(args.data_root) / args.condition / "images"
    all_imgs = sorted([
        f for f in img_dir.glob("*")
        if f.suffix.lower() in {".jpg", ".png"} and "_aug" not in f.stem
    ])
    if not all_imgs:
        print(f"[Error] {img_dir} 에 이미지 없음")
        return

    if args.mix_size:
        # bbox 면적 기준으로 정렬 → 작은/중간/큰 균형 선택
        def get_max_bbox_area(img_path):
            lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if not lbl_path.exists():
                return 0.0
            areas = []
            with open(lbl_path) as f:
                for line in f:
                    vals = line.strip().split()
                    if len(vals) == 5:
                        try:
                            w, h = float(vals[3]), float(vals[4])
                            areas.append(w * h)
                        except ValueError:
                            pass
            return max(areas) if areas else 0.0

        imgs_with_size = [(img, get_max_bbox_area(img)) for img in all_imgs]
        imgs_with_size = [x for x in imgs_with_size if x[1] > 0]
        imgs_with_size.sort(key=lambda x: x[1])

        # 3분할: 작은(멀리) / 중간 / 큰(가까이)
        n_each = max(1, args.n // 3)
        third = len(imgs_with_size) // 3
        small = imgs_with_size[:third][:n_each]
        medium = imgs_with_size[third:2*third]
        random.Random(args.seed).shuffle(medium)
        medium = medium[:n_each]
        large = imgs_with_size[2*third:][-n_each:]

        samples = [img for img, _ in small + medium + large]
        print(f"[Demo] bbox 크기 균형 선택:")
        print(f"  작은 (멀리)  {len(small)} 장 | 최대 bbox 면적 ≤ {imgs_with_size[third-1][1]:.4f}")
        print(f"  중간          {len(medium)} 장")
        print(f"  큰 (가까이)  {len(large)} 장 | 최대 bbox 면적 ≥ {imgs_with_size[2*third][1]:.4f}")
    else:
        random.Random(args.seed).shuffle(all_imgs)
        samples = all_imgs[:args.n]
        print(f"[Demo] {args.condition} 에서 {len(samples)} 장 선택")

    # ── 2. 모델 / 트래커 / BSD 초기화 ─────────────────────────
    print(f"[Loading] {args.weights}")
    detector = SGLDetInference(weights=args.weights, mode="auto",
                                conf_thres=args.conf)
    print(f"  Confidence threshold: {args.conf}")
    tracker  = SORTTracker(max_age=3, min_hits=1, iou_threshold=0.3)
    bsd      = BSDInterface(args.camera_cfg)

    # ── 3. 출력 폴더 ──────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    print(f"[Output] {out_dir}/")

    # ── 4. 각 이미지 처리 ──────────────────────────────────────
    n_warning = 0
    n_total_det = 0
    for i, img_path in enumerate(samples):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [Skip] {img_path.name} 로드 실패")
            continue

        # Detection
        detections = detector.detect(frame)
        n_total_det += len(detections)

        # SORT 추적
        sort_input  = bsd.format_sort_input(detections)
        sort_output = tracker.update(sort_input)
        _, track_ids = bsd.parse_sort_output(sort_output, detections)
        if len(track_ids) != len(detections):
            track_ids = list(range(len(detections)))

        # BSD 경고 판단 (우측 카메라)
        h, w = frame.shape[:2]
        tracked_objs, any_danger = bsd.process(
            detections, side="right",
            tracked_ids=track_ids, img_w=w, img_h=h,
        )

        # 시각화
        bsd_indices = [j for j, o in enumerate(tracked_objs) if o.is_bsd]
        vis = detector.visualize(frame, detections, bsd_indices, track_ids)

        # 경고 표시
        if any_danger:
            cv2.putText(vis, "BSD WARNING!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.rectangle(vis, (10, 10), (w-10, h-10), (0, 0, 255), 8)
            n_warning += 1
        else:
            cv2.putText(vis, "SAFE", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)

        # 정보 오버레이
        info = f"Frame {i+1:02d} | Detections: {len(detections)} | BSD: {'DANGER' if any_danger else 'safe'}"
        cv2.putText(vis, info, (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 저장
        out_path = out_dir / f"demo_{i:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  [{i+1:2d}/{len(samples)}] {out_path.name}  "
              f"det={len(detections)}  bsd={'⚠️' if any_danger else 'safe'}")

    # ── 5. 요약 ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[BSD 데모 요약]")
    print(f"  처리 이미지       : {len(samples)} 장")
    print(f"  총 검출 객체      : {n_total_det} 개")
    print(f"  BSD 경고 발생     : {n_warning} 장 ({n_warning/len(samples)*100:.1f}%)")
    print(f"  결과 저장 위치    : {out_dir}/")
    print("=" * 60)
    print(f"\n→ {out_dir}/ 에서 demo_*.jpg 파일들 확인하세요.")


if __name__ == "__main__":
    main()
