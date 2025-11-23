from typing import Tuple

import torch
import torch.nn as nn


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes Dice loss for binary segmentation.
    Expects logits (unnormalized) and binary targets {0,1}.
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    loss = 1.0 - dice
    return loss.mean()


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy with logits and Dice loss.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce_loss = self.bce(logits, targets)
        d_loss = dice_loss(logits, targets)
        loss = self.bce_weight * bce_loss + self.dice_weight * d_loss
        return loss, bce_loss, d_loss


