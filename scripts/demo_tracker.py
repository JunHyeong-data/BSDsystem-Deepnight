"""
SORT Tracker 데모 (연속 frame 시퀀스)
======================================
같은 시나리오의 연속 frame 에서 SORT tracker 가 ID를 유지하는 것 시연.
결과: 시각화 이미지 + mp4 영상.

실행:
    python3 scripts/demo_tracker.py
    python3 scripts/demo_tracker.py --condition night --seq-idx 0
"""

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector       import SGLDetInference
from src.inference.bsd_interface  import BSDInterface
from models.sort_tracker          import SORTTracker


def find_sequences(root: str, condition: str, min_len: int = 8,
                   gap_ms: int = 2000):
    """연속된 timestamp 시퀀스들 찾기."""
    img_dir = Path(root) / condition / "images"
    files = sorted([
        f for f in img_dir.glob("*")
        if f.suffix.lower() in {".jpg", ".png"} and "_aug" not in f.stem
    ])
    ts_files = sorted([(int(f.stem), f) for f in files if f.stem.isdigit()])

    sequences, current = [], [ts_files[0]] if ts_files else []
    for i in range(1, len(ts_files)):
        if ts_files[i][0] - ts_files[i-1][0] <= gap_ms:
            current.append(ts_files[i])
        else:
            if len(current) >= min_len: sequences.append(current)
            current = [ts_files[i]]
    if len(current) >= min_len: sequences.append(current)
    sequences.sort(key=len, reverse=True)
    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   default="checkpoints/best_yolo_only.pt")
    parser.add_argument("--data-root", default="data/morai")
    parser.add_argument("--condition", default="night", choices=["dusk", "night"])
    parser.add_argument("--camera-cfg", default="configs/camera_config.yaml")
    parser.add_argument("--seq-idx",   type=int, default=0,
                        help="시퀀스 인덱스 (0=가장 긴 시퀀스)")
    parser.add_argument("--output",    default="demo_tracker")
    parser.add_argument("--conf",      type=float, default=0.5)
    parser.add_argument("--fps",       type=int, default=2,
                        help="출력 영상 FPS (느린 게 보기 좋음)")
    args = parser.parse_args()

    # ── 1. 시퀀스 선택 ────────────────────────────────────────
    sequences = find_sequences(args.data_root, args.condition)
    if not sequences:
        print(f"[Error] 연속 시퀀스 없음")
        return

    if args.seq_idx >= len(sequences):
        print(f"[Error] seq-idx {args.seq_idx} 범위 초과 (max {len(sequences)-1})")
        return

    seq = sequences[args.seq_idx]
    duration = (seq[-1][0] - seq[0][0]) / 1000
    print(f"[Selected] 시퀀스 #{args.seq_idx}: "
          f"{len(seq)} frame, {duration:.1f}초")

    # ── 2. 모델 / 트래커 ───────────────────────────────────────
    print(f"[Loading] {args.weights}")
    detector = SGLDetInference(weights=args.weights, mode="auto",
                                conf_thres=args.conf)
    tracker  = SORTTracker(max_age=5, min_hits=1, iou_threshold=0.3)
    bsd      = BSDInterface(args.camera_cfg)
    print(f"  Confidence threshold: {args.conf}")

    # ── 3. 출력 폴더 ──────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    print(f"[Output] {out_dir}/")

    # ── 4. 각 frame 처리 (순서대로!) ──────────────────────────
    vis_frames = []
    track_history = {}                                     # id -> frame count
    track_positions = defaultdict(lambda: deque(maxlen=5))  # id -> [(X, Y, ts)]
    approach_velocity = {}  # id -> 접근 속도 (m/s, + = 접근)

    for i, (ts, img_path) in enumerate(seq):
        frame = cv2.imread(str(img_path))
        if frame is None: continue

        detections = detector.detect(frame)

        # SORT 추적 (연속이라 ID 유지되어야 함!)
        sort_input  = bsd.format_sort_input(detections)
        sort_output = tracker.update(sort_input)
        _, track_ids = bsd.parse_sort_output(sort_output, detections)
        if len(track_ids) != len(detections):
            track_ids = list(range(len(detections)))

        for tid in track_ids:
            track_history[tid] = track_history.get(tid, 0) + 1

        h, w = frame.shape[:2]
        tracked_objs, _ = bsd.process(
            detections, side="right",
            tracked_ids=track_ids, img_w=w, img_h=h,
        )

        # ── 접근 속도 계산 (track별 X_fwd 변화율) ──────────────
        # X_fwd < 0 = 자차 후방. 객체 접근 = X_fwd 가 0 으로 증가 (dX > 0).
        # 양의 dX = 접근 (위험), 음의 dX = 멀어짐 (안전)
        for obj in tracked_objs:
            track_positions[obj.track_id].append((obj.X_fwd, obj.Y_lat, ts))

        # 3단계 alert 재판정: SAFE / WARNING / DANGER
        any_danger = False
        any_warning = False
        for obj in tracked_objs:
            if not obj.is_bsd:
                obj.alert_level = "SAFE"
                continue

            # zone 안에 있음. 접근 중인지 판단.
            history = track_positions[obj.track_id]
            if len(history) < 2:
                # history 부족 → 일단 WARNING
                obj.alert_level = "WARNING"
                any_warning = True
                continue

            X_old, _, t_old = history[0]
            X_new, _, t_new = history[-1]
            dt = (t_new - t_old) / 1000.0  # ms → s
            if dt <= 0:
                obj.alert_level = "WARNING"
                any_warning = True
                continue

            dX_dt = (X_new - X_old) / dt   # 양수 = 접근, 음수 = 멀어짐
            approach_velocity[obj.track_id] = dX_dt

            # 접근 판정: 자차 후방(X<0)에서 다가오거나, 전방(X>0)에서 멀어지지 않음
            # 단순 근사: dX_dt > 0.3 m/s 이면 접근으로 간주 (threshold)
            if dX_dt > 0.3:
                obj.alert_level = "DANGER"   # 접근 중
                any_danger = True
            else:
                obj.alert_level = "WARNING"  # zone 안이지만 정적/멀어짐
                any_warning = True

        # 시각화 색상 매핑
        danger_indices  = [j for j, o in enumerate(tracked_objs) if o.alert_level == "DANGER"]
        warning_indices = [j for j, o in enumerate(tracked_objs) if o.alert_level == "WARNING"]
        # detector.visualize 에는 danger 만 전달 (빨간색)
        vis = detector.visualize(frame, detections, danger_indices, track_ids)

        # WARNING 객체는 노란색 박스로 덮어쓰기
        for idx in warning_indices:
            if idx < len(detections):
                x1, y1, x2, y2 = detections[idx]["bbox"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(vis, "APPROACHING?", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # 큰 타이틀 (최고 위험도 우선)
        if any_danger:
            cv2.putText(vis, "BSD DANGER!", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
            cv2.rectangle(vis, (5, 5), (w-5, h-5), (0, 0, 255), 8)
        elif any_warning:
            cv2.putText(vis, "BSD WARNING", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 3)
            cv2.rectangle(vis, (5, 5), (w-5, h-5), (0, 165, 255), 4)
        else:
            cv2.putText(vis, "SAFE", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 0), 3)

        # 시간/ID 정보
        elapsed = (ts - seq[0][0]) / 1000
        info1 = f"Frame {i+1:02d}/{len(seq)}  |  t={elapsed:.1f}s"
        info2 = f"Active tracks: {sorted(set(track_ids))} | Detections: {len(detections)}"
        cv2.putText(vis, info1, (30, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, info2, (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        out_path = out_dir / f"track_{i:02d}.jpg"
        cv2.imwrite(str(out_path), vis)
        vis_frames.append(vis)

        ids_str = sorted(set(track_ids)) if track_ids else "-"
        state = "🔴 DANGER" if any_danger else ("🟡 WARNING" if any_warning else "🟢 SAFE")
        # 접근 속도 표시 (가장 빠른 접근 객체)
        max_approach = max((approach_velocity.get(tid, 0) for tid in track_ids), default=0)
        print(f"  [{i+1:2d}/{len(seq)}] t={elapsed:4.1f}s  "
              f"det={len(detections)}  IDs={ids_str}  "
              f"approach={max_approach:+.1f}m/s  {state}")

    # ── 5. mp4 영상 생성 ──────────────────────────────────────
    if vis_frames:
        out_video = Path("demo_tracker.mp4")
        h, w = vis_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, args.fps, (w, h))
        for f in vis_frames:
            writer.write(f)
        writer.release()
        print(f"\n[Video] {out_video} 생성 ({args.fps} FPS, {len(vis_frames)/args.fps:.1f}초)")

    # ── 6. 요약 ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[Tracker 데모 요약]")
    print(f"  처리 frame      : {len(vis_frames)}")
    print(f"  관찰된 track ID : {sorted(track_history.keys())}")
    print(f"  ID별 등장 횟수  : {dict(sorted(track_history.items()))}")
    if track_history:
        max_id, max_count = max(track_history.items(), key=lambda x: x[1])
        if max_count > 1:
            print(f"  ⭐ Track #{max_id} 가 {max_count} frame 동안 같은 ID 유지!")
    print(f"  결과 영상       : demo_tracker.mp4")
    print(f"  결과 이미지     : {out_dir}/track_*.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()
