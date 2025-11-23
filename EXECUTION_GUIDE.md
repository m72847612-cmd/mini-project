# Execution Guide - Deforestation Detection Project

This guide provides step-by-step commands to execute the deforestation detection project.

## Prerequisites

- Python 3.8 or higher
- pip package manager

---

## Step 1: Navigate to Project Directory

```bash
cd /Users/amruthck/Desktop/mini-project/code
```

---

## Step 2: Set Up Virtual Environment (Optional but Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Prepare Data

If you haven't already processed the raw data, run the data preparation script:

```bash
python scripts/prepare_xingu_subset.py
```

This will:
- Process raw TIF band files into RGB PNG images
- Convert .npy mask files to PNG format
- Organize data into train/val/test directories

**Expected output:**
- `data/train/images/` - 5 training images
- `data/train/masks/` - 5 training masks
- `data/val/images/` - 1 validation image
- `data/val/masks/` - 1 validation mask
- `data/test/images/` - 2 test images

---

## Step 5: Train the Model

### Basic Training (Using Default Config)

```bash
python train.py
```

### Training with Custom Parameters

```bash
python train.py \
  --train-image-dir data/train/images \
  --train-mask-dir data/train/masks \
  --val-image-dir data/val/images \
  --val-mask-dir data/val/masks \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-3 \
  --image-size 256
```

### Training on Specific Device

```bash
# Use CUDA (NVIDIA GPU)
python train.py --device cuda

# Use MPS (Apple Silicon GPU)
python train.py --device mps

# Use CPU
python train.py --device cpu
```

**What happens during training:**
- Model trains for specified number of epochs (default: 50)
- Training metrics (Loss, IoU, Dice) are printed each epoch
- Validation metrics are computed after each training epoch
- Best model (highest validation Dice) is saved to `checkpoints/best_model.pth`

**Expected output:**
```
Using device: cuda (or mps/cpu)
Epoch 1/50
Train - Loss: 0.xxxx, IoU: 0.xxxx, Dice: 0.xxxx
Val   - Loss: 0.xxxx, IoU: 0.xxxx, Dice: 0.xxxx
New best Dice: 0.xxxx. Saving model to checkpoints/best_model.pth
...
```

---

## Step 6: Run Inference on Test Images

### Basic Inference (Using Default Paths)

```bash
python infer.py
```

### Inference with Custom Parameters

```bash
python infer.py \
  --image-dir data/test/images \
  --checkpoint checkpoints/best_model.pth \
  --output-dir outputs/masks \
  --threshold 0.5
```

### Inference on Demo Images

```bash
python infer.py \
  --image-dir data/demo/forest \
  --checkpoint checkpoints/best_model.pth \
  --output-dir outputs/demo/forest_masks
```

```bash
python infer.py \
  --image-dir data/demo/deforested \
  --checkpoint checkpoints/best_model.pth \
  --output-dir outputs/demo/deforested_masks
```

**What happens during inference:**
- Loads the trained model from checkpoint
- Processes each image in the specified directory
- Generates binary masks (white = deforested, black = forest)
- Calculates deforestation percentage for each image
- Saves masks to output directory

**Expected output:**
```
Using device: cuda (or mps/cpu)
Saved mask for AE-X07.png to outputs/masks/AE-X07.png
  Deforestation: 12.34% (xxxxx/xxxxx pixels)
Saved mask for AE-X08.png to outputs/masks/AE-X08.png
  Deforestation: 8.76% (xxxxx/xxxxx pixels)
```

---

## Complete Workflow (All Steps in Sequence)

```bash
# 1. Navigate to project
cd /Users/amruthck/Desktop/mini-project/code

# 2. Create and activate virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare data (if not already done)
python scripts/prepare_xingu_subset.py

# 5. Train the model
python train.py --epochs 50 --batch-size 4 --lr 1e-3

# 6. Run inference on test images
python infer.py --image-dir data/test/images --output-dir outputs/masks

# 7. Run inference on demo images
python infer.py --image-dir data/demo/forest --output-dir outputs/demo/forest_masks
python infer.py --image-dir data/demo/deforested --output-dir outputs/demo/deforested_masks
```

---

## Command Line Arguments Reference

### train.py Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-image-dir` | `data/train/images` | Directory with training images |
| `--train-mask-dir` | `data/train/masks` | Directory with training masks |
| `--val-image-dir` | `data/val/images` | Directory with validation images |
| `--val-mask-dir` | `data/val/masks` | Directory with validation masks |
| `--image-size` | `256` | Image size (width and height) |
| `--batch-size` | `4` | Batch size for training |
| `--num-workers` | `4` | Number of data loader workers |
| `--epochs` | `50` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--weight-decay` | `1e-5` | Weight decay for optimizer |
| `--checkpoint-dir` | `checkpoints` | Directory to save checkpoints |
| `--best-model-path` | `checkpoints/best_model.pth` | Path to save best model |
| `--device` | `auto` | Device: cuda, mps, or cpu |

### infer.py Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image-dir` | `data/test/images` | Directory with input images |
| `--checkpoint` | `checkpoints/best_model.pth` | Path to trained model |
| `--output-dir` | `outputs/masks` | Directory to save predicted masks |
| `--image-size` | `256` | Image size (should match training) |
| `--threshold` | `0.5` | Threshold for binary mask (0-1) |
| `--device` | `auto` | Device: cuda, mps, or cpu |

---

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size: `python train.py --batch-size 2`

### Issue: "No images found in directory"
**Solution:** Check that image directory exists and contains PNG/JPG files

### Issue: "Could not find mask for image"
**Solution:** Ensure mask files have matching names with images in the mask directory

### Issue: Model checkpoint not found during inference
**Solution:** Train the model first or specify correct checkpoint path: `--checkpoint path/to/model.pth`

---

## Quick Test Commands

### Test if everything is set up correctly:

```bash
# Check Python version
python --version

# Check if PyTorch is installed
python -c "import torch; print(torch.__version__)"

# Check if data directories exist
ls data/train/images/
ls data/val/images/

# Quick training test (1 epoch)
python train.py --epochs 1

# Quick inference test
python infer.py --image-dir data/test/images
```

---

## Expected Directory Structure After Execution

```
code/
├── checkpoints/
│   └── best_model.pth          # Trained model (after training)
├── outputs/
│   ├── masks/                   # Test predictions
│   │   ├── AE-X07.png
│   │   └── AE-X08.png
│   └── demo/                    # Demo predictions
│       ├── forest_masks/
│       └── deforested_masks/
├── data/
│   ├── train/                   # Training data
│   ├── val/                     # Validation data
│   ├── test/                    # Test data
│   └── demo/                    # Demo images
└── ...
```

---

## Notes

- Training time depends on your hardware (GPU recommended)
- Default training takes ~50 epochs, which may take 10-30 minutes on GPU
- Inference is fast (~1-2 seconds per image)
- Model checkpoints are saved automatically when validation Dice improves
- All paths can be customized via command-line arguments

