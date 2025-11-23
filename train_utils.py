import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import BCEDiceLoss
from metrics import iou_score, dice_coef


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: BCEDiceLoss,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss, bce_l, dice_l = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            batch_iou = iou_score(logits, masks).item()
            batch_dice = dice_coef(logits, masks).item()

        running_loss += loss.item()
        running_bce += bce_l.item()
        running_dice_loss += dice_l.item()
        running_iou += batch_iou
        running_dice += batch_dice
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{running_loss / num_batches:.4f}",
                "IoU": f"{running_iou / num_batches:.4f}",
                "Dice": f"{running_dice / num_batches:.4f}",
            }
        )

    return {
        "loss": running_loss / num_batches,
        "bce": running_bce / num_batches,
        "dice_loss": running_dice_loss / num_batches,
        "iou": running_iou / num_batches,
        "dice": running_dice / num_batches,
    }


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: BCEDiceLoss,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_bce = 0.0
    running_dice_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Val", leave=False)
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss, bce_l, dice_l = loss_fn(logits, masks)

            batch_iou = iou_score(logits, masks).item()
            batch_dice = dice_coef(logits, masks).item()

            running_loss += loss.item()
            running_bce += bce_l.item()
            running_dice_loss += dice_l.item()
            running_iou += batch_iou
            running_dice += batch_dice
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{running_loss / num_batches:.4f}",
                    "IoU": f"{running_iou / num_batches:.4f}",
                    "Dice": f"{running_dice / num_batches:.4f}",
                }
            )

    return {
        "loss": running_loss / num_batches,
        "bce": running_bce / num_batches,
        "dice_loss": running_dice_loss / num_batches,
        "iou": running_iou / num_batches,
        "dice": running_dice / num_batches,
    }


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


