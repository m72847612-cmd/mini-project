from dataclasses import dataclass


@dataclass
class Config:
    # Data
    train_image_dir: str = "data/train/images"
    train_mask_dir: str = "data/train/masks"
    val_image_dir: str = "data/val/images"
    val_mask_dir: str = "data/val/masks"
    image_size: int = 256

    # Training
    batch_size: int = 4
    num_workers: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Model / checkpointing
    in_channels: int = 3
    out_channels: int = 1
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pth"
    model_architecture: str = "baseline_unet"  # baseline_unet or smp_unet
    encoder_name: str = "timm-efficientnet-b3"
    encoder_weights: str | None = "imagenet"


