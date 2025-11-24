import argparse
import os
from typing import Dict

import torch
from torch import optim

from config import Config
from data import create_dataloaders
from losses import BCEDiceLoss
from model_factory import build_model
from train_utils import train_one_epoch, validate, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net for deforestation detection")
    parser.add_argument("--train-image-dir", type=str, default=Config.train_image_dir)
    parser.add_argument("--train-mask-dir", type=str, default=Config.train_mask_dir)
    parser.add_argument("--val-image-dir", type=str, default=Config.val_image_dir)
    parser.add_argument("--val-mask-dir", type=str, default=Config.val_mask_dir)
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--lr", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--checkpoint-dir", type=str, default=Config.checkpoint_dir)
    parser.add_argument("--best-model-path", type=str, default=Config.best_model_path)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or mps")
    parser.add_argument(
        "--advanced-augs",
        action="store_true",
        help="Use Albumentations-based strong augmentations and normalization",
    )
    return parser.parse_args()


def get_device(arg_device: str | None) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        use_albumentations=args.advanced_augs,
    )

    model = build_model()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = BCEDiceLoss()

    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics: Dict[str, float] = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
        )
        print(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}"
        )

        val_metrics: Dict[str, float] = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
        )
        print(
            f"Val   - Loss: {val_metrics['loss']:.4f}, "
            f"IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}"
        )

        current_dice = val_metrics["dice"]
        if current_dice > best_val_dice:
            best_val_dice = current_dice
            print(f"New best Dice: {best_val_dice:.4f}. Saving model to {args.best_model_path}")
            save_checkpoint(model, args.best_model_path)


if __name__ == "__main__":
    main()


