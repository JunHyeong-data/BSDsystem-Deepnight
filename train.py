"""
SGLDet 학습 스크립트 (Part A: 개발 및 검증 단계)
-------------------------------------------------
MORAI 합성 데이터로 SGLDet 프레임워크 학습.

실행:
  python train.py
  python train.py --epochs 100 --batch 8
  python train.py --pretrain          # SCI/SDAP 사전학습 포함
"""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from models.sgldet_yolov8 import (
    SGLDetYOLO,
    pretrain_enhancer,
    pretrain_denoiser,
)
from src.datasets.morai_dataset import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="SGLDet Training")
    parser.add_argument("--config", default="configs/sgldet_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch",  type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pretrain", action="store_true",
                        help="SGLDet 본 학습 전 SCI/SDAP 사전학습 실행")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch: dict, device: str) -> dict:
    """배치 dict의 모든 텐서를 device로 이동."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def train_one_epoch(model, loader, optimizer, device, epoch, writer) -> dict:
    """한 에폭 학습."""
    model.train()
    sums = {"det": 0.0, "self": 0.0, "total": 0.0}
    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        try:
            out = model(batch)
            loss = out["loss"]
            loss.backward()
            optimizer.step()

            sums["det"]   += out["det_loss"].item()
            sums["self"]  += out["self_loss"].item()
            sums["total"] += loss.item()
            n += 1

        except Exception as e:
            print(f"  [Skip batch] {type(e).__name__}: {e}")
            continue

    n = max(n, 1)
    avg = {k: v / n for k, v in sums.items()}

    writer.add_scalar("Loss/det",   avg["det"],   epoch)
    writer.add_scalar("Loss/self",  avg["self"],  epoch)
    writer.add_scalar("Loss/total", avg["total"], epoch)
    return avg


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    epochs     = args.epochs or cfg["train"]["epochs"]
    batch_size = args.batch  or cfg["train"]["batch_size"]
    device     = args.device or cfg["train"]["device"]
    device     = device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"  SGLDet Training")
    print(f"  Device: {device}  |  Epochs: {epochs}  |  Batch: {batch_size}")
    print("=" * 60)

    # ── DataLoader ────────────────────────────────────────────
    train_loader = build_dataloader(
        root        = cfg["data"]["root"],
        split       = "train",
        img_size    = cfg["train"]["img_size"],
        batch_size  = batch_size,
        num_workers = cfg["train"]["num_workers"],
    )

    if len(train_loader.dataset) == 0:
        print("\n[Error] 학습 데이터가 없습니다!")
        print(f"  → MORAI 데이터를 {cfg['data']['root']}/dusk(night)/images, labels 에 배치하세요.")
        return

    # ── 모델 ──────────────────────────────────────────────────
    model = SGLDetYOLO(
        yolo_weights = cfg["model"]["yolo_weights"],
        lambda_self  = cfg["model"]["lambda_self"],
        num_classes  = cfg["model"].get("num_classes"),
    ).to(device)

    # ★ 중요: optimizer 생성 전 AuxDecoder 파라미터 메모리 강제 할당
    #         (lazy init 시점이 첫 forward여서 optimizer가 모를 수 있음)
    model.warmup(img_size=cfg["train"]["img_size"], device=device)

    # ── 사전학습 ─────────────────────────────────────────────
    if args.pretrain:
        print("\n[Step 1] SCI Enhancer 사전학습...")
        pretrained_e = pretrain_enhancer(
            train_loader, cfg["pretrain"]["enhancer_epochs"], device,
            lr=cfg["pretrain"]["pretrain_lr"],
        )
        model.enhancer.load_state_dict(pretrained_e.state_dict())

        print("\n[Step 2] SDAP Denoiser 사전학습...")
        pretrained_d = pretrain_denoiser(
            train_loader, cfg["pretrain"]["denoiser_epochs"], device,
            lr=cfg["pretrain"]["pretrain_lr"],
        )
        model.denoiser.load_state_dict(pretrained_d.state_dict())

        # 보조 모듈 freeze (사전학습 가중치 보존)
        for p in model.enhancer.parameters(): p.requires_grad = False
        for p in model.denoiser.parameters(): p.requires_grad = False

        print("\n[Step 3] SGLDet 본 학습 시작\n")

    # ── Optimizer / Scheduler ────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        trainable,
        lr           = cfg["train"]["lr"],
        momentum     = cfg["train"]["momentum"],
        weight_decay = cfg["train"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir="logs/tensorboard")
    Path("checkpoints").mkdir(exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        m = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
        scheduler.step()
        dt = time.time() - t0

        print(f"Epoch [{epoch:3d}/{epochs}] "
              f"total={m['total']:.4f}  det={m['det']:.4f}  "
              f"self={m['self']:.4f}  ({dt:.1f}s)")

        if m["total"] < best_loss:
            best_loss = m["total"]
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  ✓ best_model.pt 저장 (loss={best_loss:.4f})")

        torch.save(model.state_dict(), "checkpoints/last_model.pt")

    writer.close()
    print(f"\n학습 완료! best loss: {best_loss:.4f}")
    print(f"checkpoints/best_model.pt 사용 권장")


if __name__ == "__main__":
    main()
