# Quick Start Commands

## 🚀 Complete Setup & Execution (Copy-Paste Ready)

```bash
# Navigate to project
cd /Users/amruthck/Desktop/mini-project/code

# Install dependencies
pip install -r requirements.txt

# Prepare data (if needed)
python scripts/prepare_xingu_subset.py

# Train the model
python train.py

# Run inference
python infer.py
```

---

## 📋 Individual Commands

### Setup
```bash
pip install -r requirements.txt
```

### Data Preparation
```bash
python scripts/prepare_xingu_subset.py
```

### Training
```bash
# Basic training
python train.py

# Custom training
python train.py --epochs 50 --batch-size 4 --lr 1e-3 --device cuda
```

### Inference
```bash
# Test images
python infer.py --image-dir data/test/images --output-dir outputs/masks

# Demo images
python infer.py --image-dir data/demo/forest --output-dir outputs/demo/forest_masks
```

---

## 🎯 Most Common Use Cases

**1. First Time Setup:**
```bash
pip install -r requirements.txt
python scripts/prepare_xingu_subset.py
python train.py
```

**2. Re-train with different settings:**
```bash
python train.py --epochs 100 --batch-size 8 --lr 5e-4
```

**3. Run inference on new images:**
```bash
python infer.py --image-dir path/to/your/images --output-dir outputs/predictions
```

**4. Quick test (1 epoch):**
```bash
python train.py --epochs 1
python infer.py
```

