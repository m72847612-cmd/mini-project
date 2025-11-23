import argparse
import os
from typing import List, Tuple

import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

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


def create_deforestation_charts(
    img_name: str,
    deforestation_percentage: float,
    forest_percentage: float,
    deforested_pixels: int,
    forest_pixels: int,
    total_pixels: int,
    output_dir: str,
) -> None:
    """
    Create pie chart and bar chart for deforestation visualization.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie Chart
    labels = ['Forest', 'Deforested']
    sizes = [forest_percentage, deforestation_percentage]
    colors = ['#2d5016', '#d62728']  # Dark green for forest, red for deforested
    explode = (0, 0.1)  # Explode the deforested slice
    
    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.2f%%',
        shadow=True,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    ax1.set_title(f'Deforestation Analysis: {img_name}\n', fontsize=14, fontweight='bold')
    ax1.axis('equal')
    
    # Bar Chart
    categories = ['Forest', 'Deforested']
    percentages = [forest_percentage, deforestation_percentage]
    bars = ax2.bar(categories, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{pct:.2f}%\n({int(forest_pixels if i == 0 else deforested_pixels)} pixels)',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Deforestation Rate Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add summary text
    summary_text = (
        f'Total Pixels: {total_pixels:,}\n'
        f'Forest Area: {forest_percentage:.2f}% ({forest_pixels:,} pixels)\n'
        f'Deforested Area: {deforestation_percentage:.2f}% ({int(deforested_pixels):,} pixels)'
    )
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save chart
    chart_name = os.path.splitext(img_name)[0] + '_deforestation_chart.png'
    chart_path = os.path.join(output_dir, chart_name)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Saved chart: {chart_path}")


def create_summary_chart(
    results: List[Tuple[str, float, float]],
    output_dir: str,
) -> None:
    """
    Create a summary bar chart comparing deforestation rates across all images.
    """
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    image_names = [r[0] for r in results]
    deforestation_rates = [r[1] for r in results]
    forest_rates = [r[2] for r in results]
    
    x = np.arange(len(image_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, forest_rates, width, label='Forest', color='#2d5016', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, deforestation_rates, width, label='Deforested', color='#d62728', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 1:  # Only label if > 1% to avoid clutter
                ax.text(
                    bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold'
                )
    
    ax.set_xlabel('Images', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Deforestation Rate Summary - All Images', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([os.path.splitext(name)[0] for name in image_names], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save summary chart
    summary_path = os.path.join(output_dir, 'deforestation_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Summary chart saved: {summary_path}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.checkpoint}\n"
            f"Please train the model first or specify the correct checkpoint path with --checkpoint"
        )

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

    # Store results for summary chart
    results: List[Tuple[str, float, float]] = []

    for img_path in image_paths:
        try:
            img_name = os.path.basename(img_path)
            print(f"Processing {img_name}...")
            
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)
                pred_mask = (probs > args.threshold).float()
                
                # Debug: Print probability statistics
                prob_min = probs.min().item()
                prob_max = probs.max().item()
                prob_mean = probs.mean().item()
                print(f"  Probability range: [{prob_min:.4f}, {prob_max:.4f}], mean: {prob_mean:.4f}")

            # Handle different tensor shapes
            mask_np = pred_mask.squeeze().cpu().numpy()
            
            # Ensure 2D array for PIL Image
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            if mask_np.ndim != 2:
                raise ValueError(f"Unexpected mask shape: {mask_np.shape}")
            
            mask_img = Image.fromarray((mask_np * 255).astype("uint8"), mode='L')

            # Calculate deforestation percentage
            total_pixels = mask_np.size
            deforested_pixels = float(mask_np.sum())
            forest_pixels = total_pixels - deforested_pixels
            deforestation_percentage = (deforested_pixels / total_pixels) * 100.0
            forest_percentage = (forest_pixels / total_pixels) * 100.0

            save_path = os.path.join(args.output_dir, img_name)
            mask_img.save(save_path)
            print(f"✓ Saved mask for {img_name} to {save_path}")
            print(f"  Deforestation: {deforestation_percentage:.2f}% ({int(deforested_pixels)}/{total_pixels} pixels)")
            
            # Create visualization charts
            create_deforestation_charts(
                img_name=img_name,
                deforestation_percentage=deforestation_percentage,
                forest_percentage=forest_percentage,
                deforested_pixels=int(deforested_pixels),
                forest_pixels=int(forest_pixels),
                total_pixels=total_pixels,
                output_dir=args.output_dir,
            )
            
            # Store results for summary
            results.append((img_name, deforestation_percentage, forest_percentage))
        except Exception as e:
            print(f"✗ Error processing {img_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary chart if we processed multiple images
    if len(results) > 1:
        create_summary_chart(results, args.output_dir)


if __name__ == "__main__":
    main()


