## Deforestation Detection with U-Net (PyTorch)

This project implements a U-Net convolutional neural network for **deforestation detection** from satellite remote sensing imagery.  
The model performs **binary semantic segmentation**, predicting a per-pixel mask of deforested vs non-deforested areas.

### Project Structure

- `requirements.txt` – Python dependencies.
- `config.py` – Simple configuration (paths, hyperparameters).
- `data.py` – Dataset and DataLoader utilities for satellite images and masks.
- `model.py` – U-Net architecture implementation.
- `losses.py` – Binary Cross Entropy + Dice loss.
- `metrics.py` – IoU and Dice coefficient metrics.
- `train_utils.py` – Training and validation loops.
- `train.py` – Main script to run training and validation, save best checkpoints.
- `infer.py` – Inference script to run the trained model on new images.
- `scripts/prepare_xingu_subset.py` – Helper to download & preprocess a tiny subset of the open Xingu deforestation dataset into PNG images/masks.

### Expected Data Layout

By default, the project expects data in this structure (you can change paths in `config.py` or via CLI args):

```text
data/
  train/
    images/
      img_001.png
      img_002.png
      ...
    masks/
      img_001_mask.png
      img_002_mask.png
      ...
  val/
    images/
    masks/
  test/
    images/
```

Image and mask filenames should correspond (e.g., `img_001.png` ↔ `img_001_mask.png`).
Masks should be single-channel, binary (0 for background, 1 for deforestation).

### Sample dataset (Xingu subset)

To get you started quickly, this repo includes a lightweight subset (8 scenes) extracted from the open **Xingu deforestation dataset** (`github.com/ebouhid/Xingu_Dataset`).  
Run the helper script to (re)generate the PNG images/masks under `data/`:

```bash
python scripts/prepare_xingu_subset.py
```

The script downloads the required Landsat bands + masks directly from GitHub, combines bands B4/B3/B2 into RGB, normalizes pixel ranges, and drops the processed pairs into:

- `data/train` – 5 scenes
- `data/val` – 1 scene (adjust the split in the script if you need more)
- `data/test` – 2 scenes (images only, for inference demos)

> **Attribution:** Source imagery and masks come from the [Xingu_Dataset](https://github.com/ebouhid/Xingu_Dataset). Please review that project's license before redistributing.

### Setup

```bash
cd /Users/amruthck/Desktop/mini-project
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Training

```bash
python train.py \
  --train-image-dir data/train/images \
  --train-mask-dir data/train/masks \
  --val-image-dir data/val/images \
  --val-mask-dir data/val/masks \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-3
```

The script will:

- Train U-Net on the training set
- Evaluate on the validation set each epoch
- Save the **best model checkpoint** (highest validation Dice) to `checkpoints/best_model.pth`

#### Enhanced training pipeline (recommended)

For a stronger model with aggressive augmentations, Imagenet-pretrained encoder, mixed precision, and OneCycle learning-rate scheduling, run:

```bash
python train_enhanced.py \
  --train-image-dir data/train/images \
  --train-mask-dir data/train/masks \
  --val-image-dir data/val/images \
  --val-mask-dir data/val/masks \
  --image-size 512 \
  --batch-size 6 \
  --epochs 60 \
  --encoder-name timm-efficientnet-b3 \
  --best-model-path checkpoints/best_model_enhanced.pth
```

Key upgrades:

- **Albumentations** pipeline with color/contrast, geometric, and noise transforms to improve generalization.
- **Segmentation Models PyTorch** U-Net backbone with pretrained EfficientNet encoders.
- **Combo loss** blending BCE, Dice, and Focal to improve boundary recall on small clearings.
- **Mixed precision + OneCycleLR** for smoother convergence and faster training.

Once satisfied with the new weights, either update `Config.best_model_path` or start the web app with `--checkpoint` pointing to `checkpoints/best_model_enhanced.pth`.

### Inference

Once you have a trained model:

```bash
python infer.py \
  --image-dir data/test/images \
  --checkpoint checkpoints/best_model.pth \
  --output-dir outputs/masks
```

This will:

- Load the trained U-Net model
- Run prediction on each image in `image-dir`
- Save binary mask images in `output-dir`

### Interactive Web App

Spin up the FastAPI-powered UI to upload images through the browser and visualize the model output:

```bash
uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) and either drop your own satellite capture or try one of the bundled demo scenes. The page renders:

- The predicted binary mask
- A color overlay on the original image
- Summary metrics (percent deforested, pixel counts)
- Adjustable probability threshold slider

> Make sure `checkpoints/best_model.pth` exists before starting the server. The model is loaded once on startup so subsequent requests stay responsive.
> If you trained the enhanced SMP model, update `Config.model_architecture = "smp_unet"` and set `Config.best_model_path` (or pass `--checkpoint` when running `uvicorn`) so the server loads the new weights.

### Notes

- The input images are resized to **256×256** by default (configurable in `config.py`).
- Basic augmentations (random flips, rotations) are applied during training.
- You can extend this baseline with multi-class segmentation, additional spectral bands, or more advanced augmentations as needed.


