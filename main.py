"""
BSD 시스템 메인 진입점 (DeepNight)
------------------------------------
Part A: 학습       → python main.py --mode train
Part A: 사전학습   → python main.py --mode train --pretrain
Part B: 영상 추론  → python main.py --mode run --source video.mp4
Part B: 카메라     → python main.py --mode run --source 0
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="BSD DeepNight System")
    parser.add_argument("--mode",         choices=["train", "run"], default="train")
    parser.add_argument("--config",       default="configs/sgldet_config.yaml")
    parser.add_argument("--camera-config", default="configs/camera_config.yaml")
    parser.add_argument("--weights",      default="checkpoints/best_model.pt")
    parser.add_argument("--source",       default="0",
                        help="영상 경로 또는 카메라 인덱스 (0,1,...)")
    parser.add_argument("--pretrain",     action="store_true")
    parser.add_argument("--resume",       action="store_true",
                        help="(train) checkpoints/last_model.pt 에서 이어서 학습")
    # train 모드 전달 옵션
    parser.add_argument("--epochs",       type=int, default=None,
                        help="(train) 학습 epoch 수. 미지정시 config 값 사용")
    parser.add_argument("--batch",        type=int, default=None,
                        help="(train) 배치 크기. 미지정시 config 값 사용")
    parser.add_argument("--device",       default=None,
                        help="(train) 디바이스 (cuda/cpu). 미지정시 config 값 사용")
    return parser.parse_args()


def run_training(args):
    """Part A: SGLDet 학습 (train.py 위임)."""
    import subprocess
    cmd = [sys.executable, "train.py", "--config", args.config]
    if args.pretrain:
        cmd.append("--pretrain")
    if args.resume:
        cmd.append("--resume")
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch is not None:
        cmd.extend(["--batch", str(args.batch)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    subprocess.run(cmd)


def run_inference(args):
    """Part B: 실시간 BSD 추론 파이프라인."""
    import cv2
    import numpy as np

    from src.preprocessing.calibration import CameraCalibration
    from src.inference.detector       import SGLDetInference
    from src.inference.bsd_interface  import BSDInterface
    from models.sort_tracker          import SORTTracker

    # 초기화
    calib    = CameraCalibration(args.camera_config)
    detector = SGLDetInference(weights=args.weights)
    tracker  = SORTTracker(max_age=3, min_hits=1, iou_threshold=0.3)
    bsd      = BSDInterface(args.camera_config)

    # 영상 소스
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] 영상 소스 열기 실패: {args.source}")
        return

    print(f"[BSD DeepNight] 추론 시작 | 소스: {args.source}")
    print("  q: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 카메라 왜곡 보정
        frame_corrected = calib.undistort(frame)

        # 2. SGLDet 탐지 (보조 파이프라인 없이 경량 추론)
        detections = detector.detect(frame_corrected)

        # 3. SORT 추적
        sort_input  = bsd.format_sort_input(detections)
        sort_output = tracker.update(sort_input)
        track_ids   = tracker.get_track_ids(sort_output)

        # 4. BSD 경고 판단
        h, w = frame.shape[:2]
        tracked_objs, any_danger = bsd.process(
            detections, tracked_ids=track_ids, img_w=w, img_h=h,
        )

        # 5. 시각화
        bsd_indices = [i for i, o in enumerate(tracked_objs) if o.is_bsd]
        vis = detector.visualize(frame_corrected, detections, bsd_indices)

        if any_danger:
            cv2.putText(vis, "BSD WARNING!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("BSD DeepNight", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    elif args.mode == "run":
        run_inference(args)
