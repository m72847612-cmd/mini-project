# Model Accuracy Test Results

## Test Date
November 23, 2024

## Test Setup
- **Model**: U-Net (checkpoints/best_model.pth)
- **Test Images**: Forest images from `data/demo/forest/` and deforested images from `data/demo/deforested/`
- **Expected Results**: 
  - Forest images should show **0-5% deforestation** (healthy forests)
  - Deforested images should show **high deforestation** (50-100%)

---

## Results Summary

### Forest Images (Should be ~0% deforestation)

| Image | Threshold 0.5 | Threshold 0.55 | Threshold 0.6 | Probability Range | Mean Prob |
|-------|---------------|----------------|---------------|-------------------|-----------|
| forest_1.png | **99.69%** ❌ | 80.72% ❌ | 15.41% ⚠️ | [0.4800, 0.6202] | 0.5736 |
| forest_2.png | **99.99%** ❌ | 66.20% ❌ | 31.63% ❌ | [0.4964, 0.7002] | 0.5793 |
| forest_demo_1.png | **87.37%** ❌ | 56.74% ❌ | 28.63% ❌ | [0.4724, 0.6957] | 0.5654 |
| forest_demo_2.png | **95.44%** ❌ | 57.76% ❌ | 4.98% ✅ | [0.4658, 0.6187] | 0.5548 |
| forest_demo_3.png | **99.64%** ❌ | 89.39% ❌ | 46.03% ❌ | [0.4770, 0.7358] | 0.6035 |
| healthy_forest_example.png | **99.99%** ❌ | 26.97% ❌ | **0.08%** ✅ | [0.4946, 0.6027] | 0.5447 |

### Deforested Images (Should be high deforestation)

| Image | Threshold 0.5 | Threshold 0.6 | Probability Range | Mean Prob |
|-------|---------------|---------------|-------------------|-----------|
| deforested_example_1.png | 100.00% ✅ | **0.00%** ❌ | [0.4995, 0.5942] | 0.5362 |
| deforested_example_2.png | 98.57% ✅ | **0.00%** ❌ | [0.4895, 0.5659] | 0.5226 |

---

## Key Findings

### ❌ **Critical Issues Identified:**

1. **Low Model Confidence**: 
   - All probability ranges are clustered around 0.5 (0.47-0.74)
   - Mean probabilities are 0.52-0.60, indicating high uncertainty
   - Model cannot confidently distinguish between forest and deforested areas

2. **Threshold Sensitivity**:
   - **Threshold 0.5**: Predicts almost everything as deforested (99%+)
   - **Threshold 0.55**: Still predicts 26-89% deforestation on healthy forests
   - **Threshold 0.6**: Better for some images but still inconsistent
     - Forest images: 0.08% to 46.03% (should be ~0%)
     - Deforested images: 0.00% (should be high!)

3. **Incorrect Predictions**:
   - **Forest images** with threshold 0.6: 0.08% to 46.03% deforestation (inconsistent)
   - **Deforested images** with threshold 0.6: 0.00% deforestation (completely wrong!)

4. **Model Performance**:
   - The model appears to be **not properly trained** or **underfitted**
   - Predictions are essentially random/unconfident
   - Cannot reliably distinguish between forest and deforested areas

---

## Recommendations

### Immediate Actions:

1. **Check Training History**:
   - Review training logs to see final validation metrics
   - Check if model converged during training
   - Verify training loss decreased and validation metrics improved

2. **Retrain the Model**:
   - The current model seems underfitted or not properly trained
   - Consider training for more epochs
   - Check if data augmentation is appropriate
   - Verify training/validation data quality

3. **Threshold Tuning**:
   - Current threshold (0.5) is too low given model's uncertainty
   - Need to find optimal threshold based on validation set
   - Consider using ROC curve to find best threshold

4. **Data Quality Check**:
   - Verify training masks are correct
   - Check if training data is representative of test data
   - Ensure image preprocessing is consistent

### For Your Mentor:

**"We tested the model on healthy forest images and found that it's predicting with very low confidence (probabilities around 0.5-0.6). With the default threshold of 0.5, it incorrectly classifies almost all healthy forests as deforested. Even with higher thresholds, the model shows inconsistent behavior - sometimes predicting high deforestation on healthy forests and low deforestation on deforested areas. This suggests the model may need more training or there's an issue with the training process. We recommend reviewing the training metrics and potentially retraining the model."**

---

## Test Commands Used

```bash
# Forest images with threshold 0.5
python infer.py --image-dir data/demo/forest --output-dir outputs/demo/forest_masks --threshold 0.5

# Forest images with threshold 0.55
python infer.py --image-dir data/demo/forest --output-dir outputs/demo/forest_masks_th055 --threshold 0.55

# Forest images with threshold 0.6
python infer.py --image-dir data/demo/forest --output-dir outputs/demo/forest_masks_th060 --threshold 0.6

# Deforested images with threshold 0.6
python infer.py --image-dir data/demo/deforested --output-dir outputs/demo/deforested_masks_th060 --threshold 0.6
```

---

## Conclusion

**Model Status**: ⚠️ **Needs Improvement**

The model is currently **not reliable** for deforestation detection. It shows:
- Low confidence predictions
- Inconsistent results across different thresholds
- Incorrect classifications on both forest and deforested images

**Next Steps**: Review training process, check data quality, and consider retraining with better hyperparameters or more epochs.

