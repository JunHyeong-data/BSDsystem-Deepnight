"""
BSD 야간 데이터 증강기 (Windows 전용, 학습 데이터 부풀리기)
=============================================================
data/morai/{condition}/ 의 YOLO 데이터셋에 증강 적용.

야간 특화 증강 (Night-Specific):
  1. 감마/밝기 다운     → 더 어두운 밤 시나리오 시뮬레이션
  2. ISO 노이즈         → 센서 노이즈 (저조도 환경)
  3. 헤드라이트 글레어  → 후방 차량 헤드라이트 (BSD 핵심 시나리오)
  4. 가로등 점광원      → 도심 야간 도로 환경
  5. 색온도 시프트      → 노란 가로등 / 파란 달빛
  6. 비네팅             → 어안렌즈 가장자리 어두움 강화
  7. 모션 블러          → 주행 중 카메라 흔들림

기하학적 증강 (Bbox 변환 포함):
  8. 좌우 반전          → 클래스 보존 (★ BSD 측면 카메라엔 신중히 사용)
  9. 소량 회전/줌       → 차량 흔들림 시뮬레이션
  10. Cutout/Erasing    → 부분 가림 학습

입력 : data/morai/{condition}/images, labels
출력 : data/morai/{condition}/images, labels  (원본 + _augN.jpg/_augN.txt 추가)

실행:
  python augment_data.py --condition night --n-aug 5
  python augment_data.py --condition dusk  --n-aug 3 --no-flip
  python augment_data.py --condition night --n-aug 5 --preview
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


# ── YOLO 라벨 I/O ────────────────────────────────────────────────────────────

def load_label(path: Path) -> list[tuple]:
    """YOLO .txt → [(cls, cx, cy, w, h), ...] (정규화 좌표)"""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        boxes.append((cls, cx, cy, w, h))
    return boxes


def save_label(path: Path, boxes: list[tuple]) -> None:
    """[(cls, cx, cy, w, h), ...] → YOLO .txt"""
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (c, cx, cy, w, h) in boxes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clip_boxes(boxes: list[tuple], min_size: float = 0.005) -> list[tuple]:
    """0~1 범위 클리핑, 너무 작은 박스는 제거."""
    out = []
    for c, cx, cy, w, h in boxes:
        x1 = max(0.0, cx - w/2)
        y1 = max(0.0, cy - h/2)
        x2 = min(1.0, cx + w/2)
        y2 = min(1.0, cy + h/2)
        nw, nh = x2 - x1, y2 - y1
        if nw < min_size or nh < min_size:
            continue
        out.append((c, (x1+x2)/2, (y1+y2)/2, nw, nh))
    return out


# ── 야간 특화 증강 ──────────────────────────────────────────────────────────

def aug_darken(img: np.ndarray, factor: float | None = None) -> np.ndarray:
    """감마/밝기 감소 → 더 어두운 시나리오."""
    if factor is None:
        factor = random.uniform(0.4, 0.85)
    gamma = random.uniform(1.2, 2.2)
    # 감마 보정 (어둡게)
    inv = np.power(img.astype(np.float32) / 255.0, gamma)
    out = (inv * 255.0 * factor).clip(0, 255).astype(np.uint8)
    return out


def aug_iso_noise(img: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """저조도 ISO 노이즈 (Gaussian)."""
    if sigma is None:
        sigma = random.uniform(8, 25)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_headlight_glare(img: np.ndarray, n: int | None = None) -> np.ndarray:
    """
    후방/측방 차량 헤드라이트 글레어 시뮬레이션.
    BSD의 핵심 케이스 — 야간에 옆 차 헤드라이트가 카메라에 직접 들어옴.
    """
    H, W = img.shape[:2]
    if n is None:
        n = random.randint(1, 3)
    out = img.astype(np.float32)

    for _ in range(n):
        # 헤드라이트 위치 (이미지 어딘가)
        cx = random.randint(W // 4, 3 * W // 4)
        cy = random.randint(H // 3, 2 * H // 3)
        radius = random.randint(30, 100)
        intensity = random.uniform(180, 255)

        # 가우시안 점광원
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        gain = intensity * np.exp(-(dist**2) / (2 * (radius**2)))

        # 약간 노란 색조 (5000K 헤드라이트)
        out[..., 0] += gain * 0.7   # B
        out[..., 1] += gain * 0.95  # G
        out[..., 2] += gain * 1.0   # R

    return np.clip(out, 0, 255).astype(np.uint8)


def aug_streetlight(img: np.ndarray) -> np.ndarray:
    """가로등 점광원 (이미지 상단에 노란빛)."""
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    n = random.randint(1, 4)

    for _ in range(n):
        cx = random.randint(0, W)
        cy = random.randint(0, H // 3)              # 상단
        radius = random.randint(40, 150)
        intensity = random.uniform(100, 200)

        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        gain = intensity * np.exp(-(dist**2) / (2 * (radius**2)))

        # 노란빛 (3000K)
        out[..., 0] += gain * 0.3   # B
        out[..., 1] += gain * 0.7   # G
        out[..., 2] += gain * 0.9   # R

    return np.clip(out, 0, 255).astype(np.uint8)


def aug_color_temp(img: np.ndarray) -> np.ndarray:
    """색온도 시프트 — 따뜻함(노랑) ↔ 차가움(파랑)."""
    out = img.astype(np.float32)
    # warm: B↓ R↑   cool: B↑ R↓
    shift = random.uniform(-25, 25)
    out[..., 0] -= shift          # B
    out[..., 2] += shift          # R
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_vignette(img: np.ndarray, strength: float | None = None) -> np.ndarray:
    """비네팅 — 가장자리 어둡게 (어안렌즈 효과)."""
    if strength is None:
        strength = random.uniform(0.3, 0.7)
    H, W = img.shape[:2]
    cx, cy = W / 2, H / 2
    max_d = np.sqrt(cx**2 + cy**2)

    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2) / max_d
    mask = 1 - strength * (dist**2)
    mask = np.clip(mask, 0, 1)

    out = img.astype(np.float32) * mask[..., None]
    return out.astype(np.uint8)


def aug_motion_blur(img: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
    """주행 중 모션 블러."""
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7])
    angle = random.uniform(-30, 30)
    # 방향성 모션 블러 커널
    k = np.zeros((kernel_size, kernel_size))
    k[kernel_size // 2, :] = 1.0 / kernel_size
    # 회전
    M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
    k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
    k /= k.sum() + 1e-6
    return cv2.filter2D(img, -1, k)


# ── 기하학적 증강 (Bbox 변환 포함) ──────────────────────────────────────────

def aug_hflip(img: np.ndarray, boxes: list[tuple]) -> tuple:
    """좌우 반전 + bbox 변환."""
    img_out = cv2.flip(img, 1)
    boxes_out = [(c, 1.0 - cx, cy, w, h) for (c, cx, cy, w, h) in boxes]
    return img_out, boxes_out


def aug_rotation(img: np.ndarray, boxes: list[tuple], max_deg: float = 8) -> tuple:
    """소량 회전 (±max_deg). bbox는 회전된 외접 사각형으로."""
    H, W = img.shape[:2]
    angle = random.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
    img_out = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)

    boxes_out = []
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    for c, cx, cy, w, h in boxes:
        # 픽셀로 변환
        cxp, cyp = cx * W, cy * H
        wp, hp   = w * W, h * H
        # 네 꼭짓점
        corners = np.array([
            [cxp - wp/2, cyp - hp/2],
            [cxp + wp/2, cyp - hp/2],
            [cxp + wp/2, cyp + hp/2],
            [cxp - wp/2, cyp + hp/2],
        ])
        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])     # (4, 3)
        rotated = (M @ corners_h.T).T              # (4, 2)
        x_min, y_min = rotated.min(axis=0)
        x_max, y_max = rotated.max(axis=0)
        # 정규화 좌표로 복귀
        ncx = ((x_min + x_max) / 2) / W
        ncy = ((y_min + y_max) / 2) / H
        nw  = (x_max - x_min) / W
        nh  = (y_max - y_min) / H
        boxes_out.append((c, ncx, ncy, nw, nh))
    return img_out, clip_boxes(boxes_out)


def aug_zoom(img: np.ndarray, boxes: list[tuple], scale_range=(0.85, 1.15)) -> tuple:
    """랜덤 줌 (확대/축소) + bbox 변환."""
    H, W = img.shape[:2]
    s = random.uniform(*scale_range)
    M = cv2.getRotationMatrix2D((W/2, H/2), 0, s)
    img_out = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)

    # bbox는 중심 기준 스케일
    boxes_out = []
    for c, cx, cy, w, h in boxes:
        # 중심으로부터의 오프셋도 스케일링
        ncx = 0.5 + (cx - 0.5) * s
        ncy = 0.5 + (cy - 0.5) * s
        nw, nh = w * s, h * s
        boxes_out.append((c, ncx, ncy, nw, nh))
    return img_out, clip_boxes(boxes_out)


def aug_random_erasing(img: np.ndarray, boxes: list[tuple]) -> tuple:
    """
    랜덤 영역 검은 사각형으로 가림 (부분 가림 학습).
    Bbox는 그대로 유지 (모델이 가린 채로 학습).
    """
    H, W = img.shape[:2]
    out = img.copy()
    n = random.randint(1, 3)
    for _ in range(n):
        rw = random.randint(W // 20, W // 8)
        rh = random.randint(H // 20, H // 8)
        rx = random.randint(0, W - rw)
        ry = random.randint(0, H - rh)
        out[ry:ry+rh, rx:rx+rw] = random.randint(0, 30)
    return out, boxes


# ── 증강 파이프라인 ────────────────────────────────────────────────────────

def build_random_pipeline(allow_flip: bool = True) -> list[str]:
    """이미지마다 랜덤하게 적용할 증강 조합 생성."""
    chosen = []

    # 야간 특화 (1~3개 적용)
    night_pool = ["darken", "iso_noise", "headlight", "streetlight",
                  "color_temp", "vignette", "motion_blur"]
    chosen.extend(random.sample(night_pool, k=random.randint(2, 4)))

    # 기하학 (0~2개)
    geo_pool = ["rotation", "zoom", "erasing"]
    if allow_flip:
        geo_pool.append("hflip")
    if random.random() < 0.7:
        chosen.extend(random.sample(geo_pool, k=random.randint(1, 2)))

    return chosen


def apply_pipeline(
    img: np.ndarray,
    boxes: list[tuple],
    ops: list[str],
) -> tuple[np.ndarray, list[tuple]]:
    """증강 시퀀스 적용 (기하학 → 픽셀 순서 권장)."""
    # 기하학 먼저 (bbox 변환)
    if "hflip" in ops:
        img, boxes = aug_hflip(img, boxes)
    if "rotation" in ops:
        img, boxes = aug_rotation(img, boxes)
    if "zoom" in ops:
        img, boxes = aug_zoom(img, boxes)
    if "erasing" in ops:
        img, boxes = aug_random_erasing(img, boxes)

    # 픽셀 (bbox 불변)
    if "darken" in ops:        img = aug_darken(img)
    if "iso_noise" in ops:     img = aug_iso_noise(img)
    if "headlight" in ops:     img = aug_headlight_glare(img)
    if "streetlight" in ops:   img = aug_streetlight(img)
    if "color_temp" in ops:    img = aug_color_temp(img)
    if "vignette" in ops:      img = aug_vignette(img)
    if "motion_blur" in ops:   img = aug_motion_blur(img)

    return img, boxes


# ── 시각화 ──────────────────────────────────────────────────────────────────

CLASS_COLORS = {0: (0, 200, 0), 1: (255, 128, 0), 2: (0, 0, 255)}
CLASS_NAMES  = {0: "car",       1: "pedestrian",   2: "truck"}


def draw_boxes(img: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """미리보기용 bbox 그리기."""
    H, W = img.shape[:2]
    vis = img.copy()
    for c, cx, cy, w, h in boxes:
        x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
        x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
        color = CLASS_COLORS.get(c, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, CLASS_NAMES.get(c, str(c)), (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


# ── 메인 ────────────────────────────────────────────────────────────────────

# ── 안전장치: 기존 증강 파일 자동 정리 ──────────────────────────────────────

def clean_existing_augs(img_dir: Path, lbl_dir: Path) -> int:
    """이전 실행에서 생성된 *_aug*.jpg / *_aug*.txt 삭제."""
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


# ── 안전장치: 데이터량 권장치 검증 ──────────────────────────────────────────

MIN_RECOMMENDED_TOTAL = 500          # 학습 권장 최소 총 장수
GOOD_RECOMMENDED_TOTAL = 1500        # 충분히 좋은 수준


def assess_data_volume(n_orig: int, n_aug: int) -> tuple[str, int]:
    """원본+증강 총량 기준으로 권장 등급 반환."""
    total = n_orig + n_orig * n_aug
    if total < MIN_RECOMMENDED_TOTAL:
        # 부족 → 권장 n_aug 계산
        needed = (MIN_RECOMMENDED_TOTAL - n_orig) // max(n_orig, 1) + 1
        return ("부족", needed)
    elif total < GOOD_RECOMMENDED_TOTAL:
        return ("적정", n_aug)
    else:
        return ("충분", n_aug)


def main():
    p = argparse.ArgumentParser(
        description="BSD 야간 데이터 증강기 (안전장치 내장)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
기본 동작 (주의사항 자동 적용):
  ① 좌우 반전 OFF        — BSD 우측 카메라 전용이므로 기본 비활성화
                            좌측까지 학습하려면 --allow-flip 명시
  ② 기존 _aug* 자동 정리 — 중복 누적 방지. 끄려면 --no-clean
  ③ 데이터량 자동 검증   — 총 장수 부족 시 권장 n-aug 안내 후 확인
""",
    )
    p.add_argument("--condition", "-c", choices=["dusk", "night"], default="night")
    p.add_argument("--root", "-r", default="data/morai",
                   help="데이터셋 루트 (기본: data/morai)")
    p.add_argument("--n-aug", "-n", type=int, default=5,
                   help="이미지당 생성할 증강 샘플 수 (기본: 5)")

    # ── 안전장치 옵션 (기본값이 안전한 쪽) ────────────────────
    p.add_argument("--allow-flip", action="store_true",
                   help="좌우 반전 허용 (기본: 비활성화. BSD 우측 카메라 전용이므로)")
    p.add_argument("--no-clean", action="store_true",
                   help="기존 _aug* 자동 정리 비활성화 (기본: 자동 정리 ON)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="모든 확인 프롬프트 자동 승인 (스크립트 자동화용)")

    p.add_argument("--preview", action="store_true",
                   help="증강 결과 미리보기 (저장 안 함)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    img_dir = Path(args.root) / args.condition / "images"
    lbl_dir = Path(args.root) / args.condition / "labels"

    if not img_dir.exists():
        print(f"[오류] {img_dir} 가 없습니다.")
        return

    # 원본 파일 목록 (_aug 가 붙은 건 제외)
    orig_imgs = sorted([p for p in img_dir.glob("*.jpg") if "_aug" not in p.stem])
    if not orig_imgs:
        print(f"[오류] {img_dir} 에 이미지 없음.")
        return

    # ── 안전장치 ③ : 데이터량 검증 ─────────────────────────────
    level, recommended_n = assess_data_volume(len(orig_imgs), args.n_aug)
    expected_total = len(orig_imgs) + len(orig_imgs) * args.n_aug

    # ── 안전장치 ① : 좌우 반전 정책 (기본 OFF) ────────────────
    flip_status = "ON (--allow-flip)" if args.allow_flip else "OFF (BSD 우측 카메라 기본)"

    # ── 안전장치 ② : 기존 _aug* 자동 정리 (기본 ON) ───────────
    existing_augs = (
        list(img_dir.glob("*_aug*.jpg"))
        + list(lbl_dir.glob("*_aug*.txt"))
    )

    # 요약 출력
    print(f"\n{'='*60}")
    print(f"  조도 조건  : {args.condition}")
    print(f"  원본 이미지: {len(orig_imgs)}장")
    print(f"  배수       : ×{args.n_aug}")
    print(f"  예상 총량  : {expected_total}장 ({len(orig_imgs)} 원본 + {expected_total-len(orig_imgs)} 증강)  → {level}")
    print(f"  좌우 반전  : {flip_status}")
    if existing_augs:
        action = "유지 (--no-clean)" if args.no_clean else "자동 삭제 후 재생성"
        print(f"  기존 _aug* : {len(existing_augs)}개 발견 → {action}")
    print(f"  모드       : {'PREVIEW (저장 안 함)' if args.preview else '저장'}")
    print(f"{'='*60}")

    # 데이터량 부족 경고
    if level == "부족" and not args.preview:
        print(f"\n[경고] 총 {expected_total}장 < 권장 {MIN_RECOMMENDED_TOTAL}장")
        print(f"       → --n-aug {recommended_n} 이상 사용 권장")
        if not args.yes:
            ans = input("그래도 계속하시겠습니까? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                print("중단.")
                return

    # ── 안전장치 ② 실행: 기존 _aug* 삭제 ──────────────────────
    if existing_augs and not args.no_clean and not args.preview:
        deleted = clean_existing_augs(img_dir, lbl_dir)
        print(f"\n[정리] 기존 증강 파일 {deleted}개 삭제 완료\n")
    else:
        print()

    # ── 본 증강 루프 ──────────────────────────────────────────
    saved = skipped = 0

    for i, img_path in enumerate(orig_imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes = load_label(lbl_path)
        if not boxes:
            skipped += 1
            print(f"  [SKIP] 라벨 없음: {img_path.name}")
            continue

        for k in range(args.n_aug):
            ops = build_random_pipeline(allow_flip=args.allow_flip)
            aug_img, aug_boxes = apply_pipeline(img.copy(), list(boxes), ops)

            if not aug_boxes:
                continue   # 변환 후 모든 bbox 사라지면 스킵

            if args.preview:
                vis_orig = draw_boxes(img, boxes)
                vis_aug  = draw_boxes(aug_img, aug_boxes)
                cv2.imshow("Original", vis_orig)
                cv2.imshow(f"Aug [{', '.join(ops)}]", vis_aug)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow(f"Aug [{', '.join(ops)}]")
                if key in (ord('q'), 27):
                    cv2.destroyAllWindows()
                    return
            else:
                out_stem = f"{img_path.stem}_aug{k}"
                cv2.imwrite(str(img_dir / f"{out_stem}.jpg"),
                            aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                save_label(lbl_dir / f"{out_stem}.txt", aug_boxes)
                saved += 1

        if (i + 1) % 50 == 0 or i == len(orig_imgs) - 1:
            print(f"  [{i+1:4d}/{len(orig_imgs)}] 진행")

    cv2.destroyAllWindows()

    # ── 최종 요약 ─────────────────────────────────────────────
    final_total = len(orig_imgs) + saved
    print(f"\n{'='*60}")
    print(f"  증강 완료")
    print(f"    저장: +{saved}장   스킵: {skipped}장")
    print(f"    최종: {final_total}장 = {len(orig_imgs)} 원본 + {saved} 증강")

    # 최종 데이터량 평가
    final_level, _ = assess_data_volume(len(orig_imgs), saved // max(len(orig_imgs), 1))
    if final_level == "부족":
        print(f"\n  [주의] {final_total}장은 학습에 부족할 수 있습니다.")
        print(f"         → MORAI에서 추가 수집 또는 --n-aug 증가 권장")
    elif final_level == "적정":
        print(f"\n  학습 가능 수준 (권장 {GOOD_RECOMMENDED_TOTAL}장 이상).")
    else:
        print(f"\n  학습에 충분한 데이터 확보.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
