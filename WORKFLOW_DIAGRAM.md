# Project Workflow Diagram

## Complete System Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: DATA PREPARATION                        │
└─────────────────────────────────────────────────────────────────────────┘

    Raw Data (data/raw/)
    ├── AE-X01_B2.tif, AE-X01_B3.tif, AE-X01_B4.tif  (Landsat bands)
    └── truth_x01.npy  (Ground truth mask)
            │
            │ python scripts/prepare_xingu_subset.py
            ▼
    ┌───────────────────────────────────────┐
    │  Data Processing Script                │
    │  • Combine B4/B3/B2 → RGB              │
    │  • Normalize to 0-255                  │
    │  • Convert masks to PNG                │
    │  • Split into train/val/test           │
    └───────────────────────────────────────┘
            │
            ▼
    Processed Data
    ├── data/train/ (5 scenes)
    │   ├── images/ (AE-X01.png ... AE-X05.png)
    │   └── masks/  (AE-X01.png ... AE-X05.png)
    ├── data/val/ (1 scene)
    │   ├── images/ (AE-X06.png)
    │   └── masks/  (AE-X06.png)
    └── data/test/ (2 scenes)
        └── images/ (AE-X07.png, AE-X08.png)


┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2: MODEL TRAINING                           │
└─────────────────────────────────────────────────────────────────────────┘

    Training Data
            │
            │ python train.py
            ▼
    ┌───────────────────────────────────────┐
    │  DataLoader                            │
    │  • Load image-mask pairs              │
    │  • Resize to 256×256                  │
    │  • Apply augmentation (flips, rotate) │
    │  • Batch size: 4                      │
    └───────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │  U-Net Model                           │
    │  • Input: 3-channel RGB (256×256)     │
    │  • Encoder-Decoder architecture       │
    │  • Skip connections                   │
    │  • Output: 1-channel logits          │
    └───────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │  Loss Function                         │
    │  • BCE Loss (50%)                     │
    │  • Dice Loss (50%)                    │
    │  • Combined: BCE + Dice               │
    └───────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │  Optimizer                             │
    │  • Adam optimizer                     │
    │  • Learning rate: 1e-3                │
    │  • Weight decay: 1e-5                 │
    └───────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │  Training Loop (50 epochs)             │
    │                                        │
    │  For each epoch:                      │
    │  1. Train on training set             │
    │  2. Validate on validation set        │
    │  3. Calculate metrics (Loss, IoU, Dice)│
    │  4. Save best model if Dice improves  │
    └───────────────────────────────────────┘
            │
            ▼
    checkpoints/best_model.pth
    (Model with highest validation Dice)


┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 3: INFERENCE                                │
└─────────────────────────────────────────────────────────────────────────┘

    Test Images (data/test/images/)
            │
            │ python infer.py
            ▼
    ┌───────────────────────────────────────┐
    │  Load Trained Model                    │
    │  • Load from checkpoints/best_model.pth│
    │  • Set to evaluation mode              │
    └───────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │  Image Preprocessing                   │
    │  • Load RGB image                      │
    │  • Resize to 256×256                   │
    │  • Convert to tensor                   │
    └───────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │  Model Forward Pass                    │
    │  • Input: Image tensor                 │
    │  • Output: Logits                      │
    │  • Apply sigmoid → probabilities       │
    │  • Threshold at 0.5 → binary mask      │
    └───────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │  Post-processing                       │
    │  • Convert to numpy array              │
    │  • Scale to 0-255                      │
    │  • Calculate deforestation %           │
    └───────────────────────────────────────┘
            │
            ▼
    outputs/masks/
    ├── AE-X07.png (Binary mask)
    └── AE-X08.png (Binary mask)
    
    Console Output:
    • Deforestation percentage per image
    • Pixel counts


┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW SUMMARY                                 │
└─────────────────────────────────────────────────────────────────────────┘

Raw TIF Bands + .npy Masks
        │
        │ [prepare_xingu_subset.py]
        ▼
PNG Images + PNG Masks (train/val/test)
        │
        │ [train.py]
        ▼
Trained U-Net Model (checkpoints/best_model.pth)
        │
        │ [infer.py]
        ▼
Predicted Binary Masks (outputs/masks/)


┌─────────────────────────────────────────────────────────────────────────┐
│                         KEY COMPONENTS                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   data.py    │────▶│   model.py   │────▶│  losses.py   │
│              │     │              │     │              │
│ • Dataset    │     │ • U-Net      │     │ • BCE+Dice   │
│ • DataLoader │     │ • Encoder    │     │ • Combined   │
│ • Augment    │     │ • Decoder    │     │   Loss       │
└──────────────┘     └──────────────┘     └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    train.py      │
                    │                  │
                    │ • Training loop  │
                    │ • Validation     │
                    │ • Checkpointing  │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    infer.py      │
                    │                  │
                    │ • Load model     │
                    │ • Predict        │
                    │ • Save masks     │
                    └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                         METRICS & EVALUATION                             │
└─────────────────────────────────────────────────────────────────────────┘

During Training:
    • Loss: Combined BCE + Dice loss
    • IoU: Intersection over Union
    • Dice: Dice Coefficient
    
During Inference:
    • Binary mask (0 = forest, 1 = deforested)
    • Deforestation percentage
    • Pixel-level statistics


┌─────────────────────────────────────────────────────────────────────────┐
│                         FILE STRUCTURE                                   │
└─────────────────────────────────────────────────────────────────────────┘

code/
├── scripts/
│   └── prepare_xingu_subset.py  → Data preparation
├── data.py                       → Dataset & DataLoader
├── model.py                      → U-Net architecture
├── losses.py                     → Loss functions
├── metrics.py                    → IoU & Dice metrics
├── train_utils.py                → Training utilities
├── train.py                      → Training script
├── infer.py                      → Inference script
├── config.py                     → Configuration
└── checkpoints/
    └── best_model.pth            → Trained model

