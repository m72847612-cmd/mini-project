from __future__ import annotations

from typing import Optional

import segmentation_models_pytorch as smp

from config import Config
from model import UNet


def build_model(
    architecture: Optional[str] = None,
    *,
    encoder_name: Optional[str] = None,
    encoder_weights: Optional[str] = None,
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
):
    """
    Centralized factory so training/inference use the same model definition.
    """

    arch = (architecture or Config.model_architecture).lower()
    in_c = in_channels or Config.in_channels
    out_c = out_channels or Config.out_channels

    if arch == "baseline_unet":
        return UNet(in_channels=in_c, out_channels=out_c)

    if arch == "smp_unet":
        enc = encoder_name or Config.encoder_name
        weights = encoder_weights if encoder_weights is not None else Config.encoder_weights
        return smp.Unet(
            encoder_name=enc,
            encoder_weights=weights,
            in_channels=in_c,
            classes=out_c,
            activation=None,
        )

    raise ValueError(f"Unknown architecture '{architecture}'. Supported: baseline_unet, smp_unet")


