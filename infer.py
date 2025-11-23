import argparse
import os
from typing import List

import torch
from PIL import Image
import torchvision.transforms as T

from config import Config
from model import UNet
from train_utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with trained U-Net model")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/test/images",
        help="Directory with input images",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=Config.best_model_path,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/masks",
        help="Directory to save predicted masks",
    )
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or mps")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask")
    return parser.parse_args()


def get_device(arg_device: str | None) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device, image_size: int) -> UNet:
    model = UNet(in_channels=Config.in_channels, out_channels=Config.out_channels)
    model = load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()
    return model


def list_images(image_dir: str) -> List[str]:
    return sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
    )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, device, args.image_size)

    transform = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
        ]
    )

    if not os.path.exists(args.image_dir):
        raise RuntimeError(
            f"Image directory does not exist: {args.image_dir}\n"
            f"Please provide a valid path with --image-dir"
        )
    
    image_paths = list_images(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.image_dir}")

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > args.threshold).float()

        mask_np = pred_mask.squeeze().cpu().numpy()
        mask_img = Image.fromarray((mask_np * 255).astype("uint8"))

        # Calculate deforestation percentage
        total_pixels = mask_np.size
        deforested_pixels = mask_np.sum()
        deforestation_percentage = (deforested_pixels / total_pixels) * 100.0

        save_path = os.path.join(args.output_dir, img_name)
        mask_img.save(save_path)
        print(f"Saved mask for {img_name} to {save_path}")
        print(f"  Deforestation: {deforestation_percentage:.2f}% ({deforested_pixels}/{total_pixels} pixels)")


if __name__ == "__main__":
    main()


