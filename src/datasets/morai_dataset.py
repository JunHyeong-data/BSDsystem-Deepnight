"""
MORAI 합성 데이터셋 로더
-----------------------
MORAI 시뮬레이션으로 생성된 야간 BSD 시나리오 데이터를 로드.

데이터 구조 (YOLO 형식):
  data/morai/
  ├── dusk/
  │   ├── images/  *.png or *.jpg
  │   └── labels/  *.txt  (class cx cy w h, 정규화 좌표)
  └── night/

클래스:
  0: car  1: pedestrian  2: truck
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = ["car", "pedestrian", "truck"]
LIGHTING_CONDITIONS = ["dusk", "night"]


class MoraiDataset(Dataset):
    """MORAI 시뮬레이션 합성 데이터셋."""

    def __init__(
        self,
        root: str = "data/morai",
        conditions: list | None = None,
        split: str = "train",
        train_ratio: float = 0.8,
        img_size: int = 640,
        transform=None,
    ):
        self.root = Path(root)
        self.conditions = conditions or LIGHTING_CONDITIONS
        self.split = split
        self.img_size = img_size
        self.transform = transform

        self.samples = self._load_samples(train_ratio)
        print(f"[MoraiDataset] {split}: {len(self.samples)} samples "
              f"from {self.conditions}")

    def _load_samples(self, train_ratio: float) -> list:
        all_samples = []
        for cond in self.conditions:
            img_dir = self.root / cond / "images"
            lbl_dir = self.root / cond / "labels"
            if not img_dir.exists():
                print(f"[Warning] {img_dir} 없음 → 스킵")
                continue

            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                all_samples.append({
                    "img": img_path,
                    "lbl": lbl_path if lbl_path.exists() else None,
                    "condition": cond,
                })

        # 재현성 위한 시드 고정 셔플
        random.Random(42).shuffle(all_samples)

        n_train = int(len(all_samples) * train_ratio)
        return all_samples[:n_train] if self.split == "train" else all_samples[n_train:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        img = cv2.imread(str(sample["img"]))
        if img is None:
            raise FileNotFoundError(f"이미지 로드 실패: {sample['img']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = img.shape[:2]

        # Letterbox 리사이즈 (스케일/패딩 정보 반환)
        img, scale, (pad_w, pad_h) = self._letterbox(img, self.img_size)

        # YOLO 라벨 로드 + letterbox 좌표 변환
        boxes = self._load_labels(
            sample["lbl"],
            orig_w=w_orig, orig_h=h_orig,
            scale=scale, pad_w=pad_w, pad_h=pad_h,
            target=self.img_size,
        )

        # Augmentation
        if self.transform is not None:
            img, boxes = self.transform(img, boxes)

        # HWC uint8 → CHW float [0,1]
        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0

        return {
            "image":     img_tensor,
            "boxes":     boxes,
            "condition": sample["condition"],
            "img_path":  str(sample["img"]),
        }

    @staticmethod
    def _letterbox(img: np.ndarray, target: int,
                   color: tuple = (114, 114, 114)) -> tuple:
        """비율 유지 리사이즈 + 패딩. (image, scale, (pad_w, pad_h)) 반환."""
        h, w = img.shape[:2]
        scale = min(target / h, target / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h_top = (target - new_h) // 2
        pad_h_bot = target - new_h - pad_h_top
        pad_w_lef = (target - new_w) // 2
        pad_w_rig = target - new_w - pad_w_lef

        img = cv2.copyMakeBorder(
            img, pad_h_top, pad_h_bot, pad_w_lef, pad_w_rig,
            cv2.BORDER_CONSTANT, value=color,
        )
        return img, scale, (pad_w_lef, pad_h_top)

    @staticmethod
    def _load_labels(
        lbl_path,
        orig_w: int, orig_h: int,
        scale: float, pad_w: int, pad_h: int,
        target: int,
    ) -> torch.Tensor:
        """
        YOLO txt → letterbox 변환 후 정규화 (N, 5) [cls, cx, cy, w, h].

        원본 라벨 (orig_w × orig_h 정규화)을 target × target 정규화로 변환:
          cx_new = (cx_orig * orig_w * scale + pad_w) / target
          cy_new = (cy_orig * orig_h * scale + pad_h) / target
          w_new  = w_orig  * orig_w * scale / target
          h_new  = h_orig  * orig_h * scale / target
        """
        if lbl_path is None or not Path(lbl_path).exists():
            return torch.zeros((0, 5))

        rows = []
        with open(lbl_path) as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) != 5:
                    continue
                cls, cx, cy, w, h = (float(v) for v in vals)

                # letterbox 변환
                cx_new = (cx * orig_w * scale + pad_w) / target
                cy_new = (cy * orig_h * scale + pad_h) / target
                w_new  = w  * orig_w * scale / target
                h_new  = h  * orig_h * scale / target

                # 경계 클램프 (혹시 모를 outlier 대비)
                cx_new = max(0.0, min(1.0, cx_new))
                cy_new = max(0.0, min(1.0, cy_new))
                w_new  = max(0.0, min(1.0, w_new))
                h_new  = max(0.0, min(1.0, h_new))

                rows.append([cls, cx_new, cy_new, w_new, h_new])

        return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 5))


def collate_fn(batch: list) -> dict:
    """
    Ultralytics YOLOv8 호환 batch 포맷으로 변환.
    {
      'img'      : (B, 3, H, W)        - 이미지 텐서
      'cls'      : (N, 1)              - 클래스 인덱스
      'bboxes'   : (N, 4)              - [cx, cy, w, h] 정규화
      'batch_idx': (N,)                - 각 box가 속한 batch 인덱스
    }
    """
    images = torch.stack([b["image"] for b in batch])

    cls_list, box_list, idx_list = [], [], []
    for i, b in enumerate(batch):
        if b["boxes"].shape[0] == 0:
            continue
        cls_list.append(b["boxes"][:, 0:1])           # (N, 1)
        box_list.append(b["boxes"][:, 1:5])           # (N, 4)
        idx_list.append(torch.full((b["boxes"].shape[0],), i, dtype=torch.long))

    if cls_list:
        cls       = torch.cat(cls_list, dim=0)
        bboxes    = torch.cat(box_list, dim=0)
        batch_idx = torch.cat(idx_list, dim=0)
    else:
        cls       = torch.zeros((0, 1), dtype=torch.float32)
        bboxes    = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx = torch.zeros((0,),   dtype=torch.long)

    return {
        "img":        images,
        "cls":        cls,
        "bboxes":     bboxes,
        "batch_idx":  batch_idx,
        "conditions": [b["condition"] for b in batch],
        "img_paths":  [b["img_path"]  for b in batch],
    }


def build_dataloader(
    root: str = "data/morai",
    conditions: list | None = None,
    split: str = "train",
    img_size: int = 640,
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    """DataLoader 생성 헬퍼."""
    dataset = MoraiDataset(root=root, conditions=conditions, split=split,
                           img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


if __name__ == "__main__":
    loader = build_dataloader(root="data/morai", split="train", batch_size=2)
    for batch in loader:
        print("img:      ", batch["img"].shape)
        print("cls:      ", batch["cls"].shape)
        print("bboxes:   ", batch["bboxes"].shape)
        print("batch_idx:", batch["batch_idx"].shape)
        break
