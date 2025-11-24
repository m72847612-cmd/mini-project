import argparse
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp

from config import Config
from data import create_dataloaders
from model_factory import build_model
from train_utils import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced training pipeline for the deforestation model")
    parser.add_argument("--train-image-dir", type=str, default=Config.train_image_dir)
    parser.add_argument("--train-mask-dir", type=str, default=Config.train_mask_dir)
    parser.add_argument("--val-image-dir", type=str, default=Config.val_image_dir)
    parser.add_argument("--val-mask-dir", type=str, default=Config.val_mask_dir)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-arch", type=str, default="smp_unet", choices=["baseline_unet", "smp_unet"])
    parser.add_argument("--encoder-name", type=str, default="timm-efficientnet-b3")
    parser.add_argument("--encoder-weights", type=str, default="imagenet", help="Set to 'None' to train from scratch")
    parser.add_argument("--checkpoint-dir", type=str, default=Config.checkpoint_dir)
    parser.add_argument("--best-model-path", type=str, default="checkpoints/best_model_enhanced.pth")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or mps")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint to resume training from")
    parser.add_argument("--dice-threshold", type=float, default=0.5)
    return parser.parse_args()


def get_device(arg_device: str | None) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ComboLoss(nn.Module):
    """
    Blend BCE, Dice, and Focal losses for better boundary recall and robustness to class imbalance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="binary", alpha=0.8, gamma=2.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice(logits, targets)
        focal = self.focal(logits, targets)
        return 0.4 * bce + 0.4 * dice + 0.2 * focal


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    loss_fn: nn.Module,
    scaler: GradScaler,
    grad_clip: float,
    dice_threshold: float,
) -> Dict[str, float]:
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
        epoch_dice += dice_score(logits.detach(), masks, threshold=dice_threshold).item()
        epoch_iou += iou_score(logits.detach(), masks, threshold=dice_threshold).item()

    num_batches = len(dataloader)
    return {
        "loss": epoch_loss / num_batches,
        "dice": epoch_dice / num_batches,
        "iou": epoch_iou / num_batches,
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader,
    device: torch.device,
    loss_fn: nn.Module,
    dice_threshold: float,
) -> Dict[str, float]:
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        epoch_loss += loss.item()
        epoch_dice += dice_score(logits, masks, threshold=dice_threshold).item()
        epoch_iou += iou_score(logits, masks, threshold=dice_threshold).item()

    num_batches = len(dataloader)
    return {
        "loss": epoch_loss / num_batches,
        "dice": epoch_dice / num_batches,
        "iou": epoch_iou / num_batches,
    }


def maybe_resume(model: nn.Module, optimizer, scaler: GradScaler, checkpoint_path: str, device: torch.device) -> int:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0

    print(f"Resuming training from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint.get("epoch", 0)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        train_image_dir=args.train_image_dir,
        train_mask_dir=args.train_mask_dir,
        val_image_dir=args.val_image_dir,
        val_mask_dir=args.val_mask_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_albumentations=True,
    )

    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = build_model(
        architecture=args.model_arch,
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=Config.in_channels,
        out_channels=Config.out_channels,
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1_000,
    )
    scaler = GradScaler()
    loss_fn = ComboLoss()

    start_epoch = 0
    if args.resume_from:
        start_epoch = maybe_resume(model, optimizer, scaler, args.resume_from, device)

    best_dice = 0.0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            loss_fn=loss_fn,
            scaler=scaler,
            grad_clip=args.grad_clip,
            dice_threshold=args.dice_threshold,
        )
        print(
            f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}"
        )

        val_metrics = validate_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
            dice_threshold=args.dice_threshold,
        )
        print(
            f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            print(f"🟢 New best Dice {best_dice:.4f}. Saving enhanced checkpoint to {args.best_model_path}")
            save_checkpoint(model, args.best_model_path)

        # Optionally save last checkpoint for resuming
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_dice": best_dice,
            },
            os.path.join(args.checkpoint_dir, "last_enhanced.pth"),
        )


if __name__ == "__main__":
    main()


