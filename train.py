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
    parser.add_argument("--resume", action="store_true",
                        help="checkpoints/last_model.pt 에서 이어서 학습")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="YOLOv8m backbone (layer 0~9, CSPDarknet) freeze "
                             "— small data 에서 COCO prior 보존")
    parser.add_argument("--suffix", default="",
                        help="checkpoint 이름 접미사 (e.g. '_freeze' → best_model_freeze.pt)")
    parser.add_argument("--patience", type=int, default=0,
                        help="early stopping patience (0=비활성). best_loss 가 N epoch 연속 갱신 안 되면 학습 종료")
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

    writer.add_scalar("Loss/train/det",   avg["det"],   epoch)
    writer.add_scalar("Loss/train/self",  avg["self"],  epoch)
    writer.add_scalar("Loss/train/total", avg["total"], epoch)
    return avg


@torch.no_grad()
def validate(model, loader, device, epoch, writer) -> dict:
    """검증 루프 — overfitting 감지 + 정확한 best_model 선정."""
    model.eval()
    sums = {"det": 0.0, "self": 0.0, "total": 0.0}
    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        try:
            out = model(batch)
            sums["det"]   += out["det_loss"].item()
            sums["self"]  += out["self_loss"].item()
            sums["total"] += out["loss"].item()
            n += 1
        except Exception as e:
            print(f"  [Skip val batch] {type(e).__name__}: {e}")
            continue

    n = max(n, 1)
    avg = {k: v / n for k, v in sums.items()}

    writer.add_scalar("Loss/val/det",   avg["det"],   epoch)
    writer.add_scalar("Loss/val/self",  avg["self"],  epoch)
    writer.add_scalar("Loss/val/total", avg["total"], epoch)
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
    val_loader = build_dataloader(
        root        = cfg["data"]["root"],
        split       = "val",
        img_size    = cfg["train"]["img_size"],
        batch_size  = batch_size,
        num_workers = cfg["train"]["num_workers"],
    )

    if len(train_loader.dataset) == 0:
        print("\n[Error] 학습 데이터가 없습니다!")
        print(f"  → MORAI 데이터를 {cfg['data']['root']}/dusk(night)/images, labels 에 배치하세요.")
        return
    if len(val_loader.dataset) == 0:
        print("\n[경고] 검증 데이터가 비어있습니다 — train loss 만 사용해 best 선정.")

    # ── 모델 ──────────────────────────────────────────────────
    model = SGLDetYOLO(
        yolo_weights = cfg["model"]["yolo_weights"],
        lambda_self  = cfg["model"]["lambda_self"],
        num_classes  = cfg["model"].get("num_classes"),
    ).to(device)

    # ★ 중요: optimizer 생성 전 AuxDecoder 파라미터 메모리 강제 할당
    #         (lazy init 시점이 첫 forward여서 optimizer가 모를 수 있음)
    model.warmup(img_size=cfg["train"]["img_size"], device=device)

    # ── Resume: 모델 state 먼저 로드 (optimizer/scheduler는 생성 후) ──
    ckpt_path = Path("checkpoints/last_model.pt")
    ckpt = None
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            print(f"\n[Resume] checkpoint 로드 (epoch {ckpt['epoch']} 완료, "
                  f"pretrained={ckpt.get('pretrained', False)})")
        else:
            print(f"\n[Resume 실패] {ckpt_path} 는 구버전 포맷. 처음부터 학습합니다.")
            ckpt = None
    elif args.resume:
        print(f"\n[Resume] {ckpt_path} 없음 → 처음부터 학습합니다.")

    # ── 사전학습 (resume 시 이전에 했으면 스킵) ──────────────
    do_pretrain = args.pretrain and not (ckpt and ckpt.get("pretrained", False))
    pretrained = (ckpt and ckpt.get("pretrained", False)) or do_pretrain

    if do_pretrain:
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
        print("\n[Step 3] SGLDet 본 학습 시작\n")
    elif ckpt and ckpt.get("pretrained", False):
        print("[Resume] 이전 학습이 pretrain 완료 상태 — Enhancer/Denoiser freeze 재적용")

    # Enhancer/Denoiser freeze (pretrain 했거나 이전에 했던 경우)
    if pretrained:
        for p in model.enhancer.parameters(): p.requires_grad = False
        for p in model.denoiser.parameters(): p.requires_grad = False

    # ── Backbone freeze (옵션) — small data 에서 COCO prior 보존 ─
    if args.freeze_backbone:
        # YOLOv8m: layer 0~9 = Backbone (CSPDarknet), 10~21 = Neck, 22 = Head
        for i in range(10):
            for p in model.detector.model[i].parameters():
                p.requires_grad = False
        n_frozen = sum(p.numel() for p in model.detector.parameters() if not p.requires_grad)
        n_total  = sum(p.numel() for p in model.detector.parameters())
        print(f"[Freeze] Backbone (layer 0~9): {n_frozen:,}/{n_total:,} params frozen "
              f"({100*n_frozen/n_total:.1f}%)")

    # ── Optimizer / Scheduler (freeze 적용 후 생성) ──────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        trainable,
        lr           = cfg["train"]["lr"],
        momentum     = cfg["train"]["momentum"],
        weight_decay = cfg["train"]["weight_decay"],
    )
    # ── Warmup + Cosine (논문 4.2 절: warmup 10% + cosine one-cycle) ─
    warmup_epochs = min(cfg["train"].get("warmup_epochs", max(1, epochs // 10)), epochs - 1)
    warmup  = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    print(f"  Scheduler: LinearLR({warmup_epochs} ep warmup) → "
          f"CosineAnnealingLR({epochs - warmup_epochs} ep)")

    writer = SummaryWriter(log_dir="logs/tensorboard")
    Path("checkpoints").mkdir(exist_ok=True)

    best_loss = float("inf")
    has_val = len(val_loader.dataset) > 0
    start_epoch = 1
    epochs_since_best = 0  # early stopping counter

    # ── Resume: optimizer/scheduler state 복원 ───────────────
    if ckpt is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            print(f"[Resume] optimizer/scheduler 복원 → "
                  f"epoch {start_epoch} 부터 재개 (best_loss={best_loss:.4f})")
        except (ValueError, KeyError) as e:
            print(f"[Resume 경고] optimizer/scheduler 복원 실패 ({e}) — "
                  f"모델 weight만 복원하고 epoch 1 부터 재학습")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
        if has_val:
            va = validate(model, val_loader, device, epoch, writer)
        scheduler.step()
        dt = time.time() - t0

        if has_val:
            print(f"Epoch [{epoch:3d}/{epochs}] "
                  f"train={tr['total']:.4f}  val={va['total']:.4f}  "
                  f"(det={va['det']:.4f} self={va['self']:.4f})  ({dt:.1f}s)")
        else:
            print(f"Epoch [{epoch:3d}/{epochs}] "
                  f"total={tr['total']:.4f}  det={tr['det']:.4f}  "
                  f"self={tr['self']:.4f}  ({dt:.1f}s)")

        # best 선정 — val 가능하면 val loss, 아니면 train loss
        score = va["total"] if has_val else tr["total"]
        if score < best_loss:
            best_loss = score
            epochs_since_best = 0
            torch.save(model.state_dict(), f"checkpoints/best_model{args.suffix}.pt")
            label = "val" if has_val else "train"
            print(f"  ✓ best_model{args.suffix}.pt 저장 ({label} loss={best_loss:.4f})")
        else:
            epochs_since_best += 1
            if args.patience > 0:
                print(f"  · best 미갱신 {epochs_since_best}/{args.patience}")

        # last_model.pt: 전체 상태 저장 (resume 용)
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "best_loss":  best_loss,
            "pretrained": pretrained,
        }, f"checkpoints/last_model{args.suffix}.pt")

        # ── Early stopping ──────────────────────────────────────
        if args.patience > 0 and epochs_since_best >= args.patience:
            print(f"\n[EarlyStopping] best_loss={best_loss:.4f} 이 "
                  f"{args.patience} epoch 연속 미갱신 — 학습 종료 (epoch {epoch}/{epochs})")
            break

    writer.close()
    print(f"\n학습 완료! best loss: {best_loss:.4f}")
    print(f"checkpoints/best_model.pt 사용 권장")


if __name__ == "__main__":
    main()
