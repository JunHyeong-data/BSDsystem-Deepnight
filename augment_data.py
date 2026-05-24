"""
BSD 야간 데이터 증강기 (albumentations 기반, v2)
================================================
data/morai/{condition}/ 의 YOLO 데이터셋에 현실적 증강 적용.

핵심 설계:
  - albumentations 의 검증된 transform 만 사용 (자체 합성/필터 제거)
  - bbox 자동 변환 + min_visibility / min_area 로 잘못된 bbox 자동 필터
  - GT 정확도 보존

야간 BSD 현실 augmentation 구성:
  Photometric  : Brightness/Contrast, Hue/Saturation/Value, RGB shift
  Sensor noise : GaussNoise, ISONoise (실제 카메라 센서 모델)
  Blur         : Motion / Gaussian / Median (주행 진동, defocus)
  Weather      : RandomFog, RandomRain (실제 야간 환경)
  Contrast     : CLAHE (어두운 영역 디테일)
  Occlusion    : CoarseDropout (부분 가림 robustness)
  Geometric    : Affine (mild ±5° rotate / ±5% translate / 0.92~1.08 scale)
  Flip         : HorizontalFlip (옵션, 기본 OFF — BSD 우측 카메라 전용)

입력 : data/morai/{condition}/{images,labels}
출력 : 동일 경로에 *_augN.jpg / *_augN.txt 추가

실행:
  python augment_data.py --condition night --n-aug 5
  python augment_data.py --condition dusk  --n-aug 5
  python augment_data.py --condition night --n-aug 5 --preview
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import albumentations as A


# ── YOLO 라벨 I/O ────────────────────────────────────────────────────────────

def load_label(path: Path):
    """YOLO .txt → (boxes [[cx,cy,w,h],...], classes [cls,...])."""
    if not path.exists():
        return [], []
    boxes, classes = [], []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        # 정규화 좌표가 [0,1] 범위 벗어나면 스킵 (잘못된 라벨)
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue
        boxes.append([cx, cy, w, h])
        classes.append(cls)
    return boxes, classes


def save_label(path: Path, boxes, classes):
    """albumentations bbox 결과 → YOLO .txt"""
    lines = []
    for (cx, cy, w, h), cls in zip(boxes, classes):
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── albumentations 파이프라인 ────────────────────────────────────────────────

def build_pipeline(allow_flip: bool = False) -> A.Compose:
    """야간 BSD 현실적 augmentation pipeline. GT bbox 자동 변환."""
    transforms = [
        # Photometric — 조도/색상 변화
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.15),   # 더 어둡게 비대칭 (야간 변동)
            contrast_limit=0.25,
            p=0.7,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.4,
        ),
        A.RGBShift(
            r_shift_limit=12,
            g_shift_limit=8,
            b_shift_limit=12,
            p=0.3,
        ),

        # Sensor noise — 저조도 ISO 모델
        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.12), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.5),

        # Blur — 주행 진동 / defocus
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.25),

        # Weather — 야간 환경 (가끔)
        A.OneOf([
            A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=1.0),
            A.RandomRain(slant_range=(-10, 10), drop_length=15, drop_width=1, p=1.0),
        ], p=0.15),

        # CLAHE — 어두운 영역 디테일 복원
        A.CLAHE(clip_limit=(1.0, 3.0), p=0.2),

        # Occlusion — 부분 가림 robustness
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(10, 40),
            hole_width_range=(10, 40),
            fill=0,
            p=0.25,
        ),

        # Geometric — mild affine (bbox 자동 변환)
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.92, 1.08),
            rotate=(-5, 5),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
    ]

    if allow_flip:
        transforms.append(A.HorizontalFlip(p=0.5))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.4,   # bbox 40% 미만 visible 이면 제거
            min_area=20,          # 너무 작은 bbox 자동 제거
            clip=True,            # bbox 좌표 [0,1] 클리핑
        ),
    )


# ── 시각화 ──────────────────────────────────────────────────────────────────

CLASS_COLORS = {0: (0, 200, 0), 1: (255, 128, 0)}
CLASS_NAMES = {0: "vehicle", 1: "pedestrian"}


def draw_boxes(img, boxes, classes):
    H, W = img.shape[:2]
    vis = img.copy()
    for (cx, cy, w, h), c in zip(boxes, classes):
        x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
        color = CLASS_COLORS.get(int(c), (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, CLASS_NAMES.get(int(c), str(c)), (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


# ── 안전장치 ─────────────────────────────────────────────────────────────────

def clean_existing_augs(img_dir: Path, lbl_dir: Path) -> int:
    """이전 _aug* 파일 삭제."""
    targets = (
        list(img_dir.glob("*_aug*.jpg"))
        + list(img_dir.glob("*_aug*.png"))
        + list(lbl_dir.glob("*_aug*.txt"))
    )
    for f in targets:
        try:
            f.unlink()
        except OSError:
            pass
    return len(targets)


# ── 메인 ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="BSD 야간 데이터 증강기 (albumentations, GT bbox 정확)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
주의:
  ① 좌우 반전 기본 OFF — BSD 우측 카메라 전용. 좌측도 학습시키려면 --allow-flip
  ② 기존 _aug* 자동 정리 — 중복 방지. 끄려면 --no-clean
  ③ --preview 모드는 한 장씩 보여주며 저장 안 함. 결과 빠른 검증용
""",
    )
    p.add_argument("--condition", "-c", choices=["dusk", "night"], default="night")
    p.add_argument("--root", "-r", default="data/morai")
    p.add_argument("--n-aug", "-n", type=int, default=5)
    p.add_argument("--allow-flip", action="store_true",
                   help="좌우 반전 허용 (기본 OFF — BSD 우측 카메라)")
    p.add_argument("--no-clean", action="store_true",
                   help="기존 _aug* 유지 (기본 자동 정리)")
    p.add_argument("--yes", "-y", action="store_true")
    p.add_argument("--preview", action="store_true",
                   help="저장 없이 한 장씩 미리보기 (스페이스/엔터=다음, Q/ESC=종료)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    img_dir = Path(args.root) / args.condition / "images"
    lbl_dir = Path(args.root) / args.condition / "labels"

    if not img_dir.exists():
        print(f"[오류] {img_dir} 없음.")
        return

    orig_imgs = sorted([p for p in img_dir.glob("*.jpg") if "_aug" not in p.stem])
    if not orig_imgs:
        print(f"[오류] {img_dir} 에 원본 이미지 없음.")
        return

    pipeline = build_pipeline(allow_flip=args.allow_flip)

    existing_augs = (
        list(img_dir.glob("*_aug*.jpg"))
        + list(lbl_dir.glob("*_aug*.txt"))
    )

    expected_total = len(orig_imgs) * (1 + args.n_aug)

    print(f"\n{'='*60}")
    print(f"  조건         : {args.condition}")
    print(f"  원본         : {len(orig_imgs)}장")
    print(f"  배수         : ×{args.n_aug}")
    print(f"  예상 총량    : {expected_total}장 (원본 + 증강)")
    print(f"  좌우 반전    : {'ON' if args.allow_flip else 'OFF (BSD 우측 카메라)'}")
    if existing_augs:
        print(f"  기존 _aug*   : {len(existing_augs)}개 → {'유지' if args.no_clean else '자동 삭제'}")
    print(f"  모드         : {'PREVIEW (저장 안 함)' if args.preview else '저장'}")
    print(f"{'='*60}\n")

    if existing_augs and not args.no_clean and not args.preview:
        deleted = clean_existing_augs(img_dir, lbl_dir)
        print(f"[정리] 기존 증강 {deleted}개 삭제\n")

    saved = skipped_empty = skipped_no_label = errors = 0

    for i, img_path in enumerate(orig_imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # albumentations 는 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes, classes = load_label(lbl_path)
        if not boxes:
            skipped_no_label += 1
            continue

        for k in range(args.n_aug):
            try:
                result = pipeline(
                    image=img_rgb,
                    bboxes=boxes,
                    class_labels=classes,
                )
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [경고] {img_path.name} aug{k}: {type(e).__name__}: {e}")
                continue

            aug_img_rgb = result["image"]
            aug_boxes = result["bboxes"]
            aug_classes = result["class_labels"]

            if not aug_boxes:
                skipped_empty += 1
                continue

            aug_img = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)

            if args.preview:
                vis_orig = draw_boxes(img, boxes, classes)
                vis_aug = draw_boxes(aug_img, aug_boxes, aug_classes)
                # 가로로 concat
                if vis_orig.shape == vis_aug.shape:
                    combined = np.hstack([vis_orig, vis_aug])
                    cv2.imshow("Original | Augmented", combined)
                else:
                    cv2.imshow("Original", vis_orig)
                    cv2.imshow("Augmented", vis_aug)
                key = cv2.waitKey(0) & 0xFF
                if key in (ord('q'), 27):
                    cv2.destroyAllWindows()
                    return
            else:
                stem = f"{img_path.stem}_aug{k}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"),
                            aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                save_label(lbl_dir / f"{stem}.txt", aug_boxes, aug_classes)
                saved += 1

        if (i + 1) % 50 == 0 or i == len(orig_imgs) - 1:
            print(f"  [{i+1:4d}/{len(orig_imgs)}]  저장: +{saved}")

    if args.preview:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass   # headless opencv 환경

    print(f"\n{'='*60}")
    print(f"  완료")
    print(f"    저장          : +{saved}장")
    print(f"    bbox 사라짐   : {skipped_empty}건 (transform 후 visibility 미만)")
    print(f"    라벨 없음     : {skipped_no_label}건")
    print(f"    에러          : {errors}건")
    print(f"    최종 총량     : {len(orig_imgs) + saved}장")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
