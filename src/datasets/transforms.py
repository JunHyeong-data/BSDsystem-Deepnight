"""
데이터 증강 (MORAI 야간 데이터 특화)
-------------------------------------
저조도 탐지 성능 향상을 위한 augmentation 전략.
MORAI 시뮬레이션 데이터는 GT가 완벽하므로 강한 augmentation 적용 가능.
"""

import random
import numpy as np
import cv2
import torch


class NightAugmentation:
    """
    야간/저조도 환경 특화 augmentation.

    적용 순서:
      1. 랜덤 수평 플립
      2. 랜덤 밝기/대비 조정 (조도 조건 시뮬레이션)
      3. 가우시안 노이즈 추가 (센서 노이즈 시뮬레이션)
      4. 랜덤 모자이크 (선택)
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        brightness_range: tuple = (0.5, 1.2),   # 저조도 편향
        noise_std: float = 0.03,
        apply_noise: bool = True,
    ):
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.noise_std = noise_std
        self.apply_noise = apply_noise

    def __call__(
        self,
        img: np.ndarray,
        boxes: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """
        Args:
            img  : (H, W, 3) uint8 RGB
            boxes: (N, 5) [cls, cx, cy, w, h] 정규화 좌표
        Returns:
            aug_img, aug_boxes
        """
        # 1. 수평 플립
        if random.random() < self.flip_prob:
            img, boxes = self._hflip(img, boxes)

        # 2. 랜덤 밝기 조정
        factor = random.uniform(*self.brightness_range)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # 3. 가우시안 노이즈
        if self.apply_noise:
            noise = np.random.normal(0, self.noise_std * 255, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img, boxes

    @staticmethod
    def _hflip(
        img: np.ndarray,
        boxes: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """수평 플립 + bbox 좌우 반전."""
        img = np.fliplr(img).copy()
        if boxes.shape[0] > 0:
            boxes = boxes.clone()
            boxes[:, 1] = 1.0 - boxes[:, 1]   # cx 반전
        return img, boxes


class ValTransform:
    """검증용 (증강 없음, 정규화만)."""

    def __call__(
        self,
        img: np.ndarray,
        boxes: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor]:
        return img, boxes
