# Deforestation Visualization Feature

## Overview

The inference script now automatically generates **visualization charts** to help users easily understand deforestation rates. This makes it much easier to study and analyze deforestation patterns.

## Features

### 1. Individual Image Charts

For each processed image, the script creates a **dual-chart visualization**:

- **Pie Chart** (Left):
  - Shows forest vs deforested area as percentages
  - Color-coded: Dark green (Forest) and Red (Deforested)
  - Exploded slice for better visibility
  - Percentage labels on each slice

- **Bar Chart** (Right):
  - Side-by-side comparison of forest and deforested percentages
  - Shows both percentage and pixel counts
  - Grid lines for easy reading
  - Color-coded bars matching the pie chart

**File Format**: `{image_name}_deforestation_chart.png`

**Example**: `forest_1_deforestation_chart.png`

### 2. Summary Comparison Chart

When processing multiple images, a **summary bar chart** is automatically created:

- Compares deforestation rates across all processed images
- Side-by-side bars for Forest and Deforested areas
- Image names on x-axis
- Percentage labels on bars
- Easy comparison of multiple images at once

**File Format**: `deforestation_summary.png`

## Usage

The visualization feature is **automatic** - no additional commands needed!

```bash
# Run inference as usual
python infer.py --image-dir data/test/images --output-dir outputs/masks

# Charts are automatically generated in the same output directory
```

## Output Files

After running inference, you'll find:

```
outputs/masks/
├── AE-X07.png                          # Binary mask
├── AE-X07_deforestation_chart.png      # Individual chart
├── AE-X08.png                          # Binary mask
├── AE-X08_deforestation_chart.png      # Individual chart
└── deforestation_summary.png            # Summary chart (if multiple images)
```

## Chart Details

### Individual Chart Components

1. **Title**: Image name and "Deforestation Analysis"
2. **Pie Chart**: Visual representation of forest vs deforested ratio
3. **Bar Chart**: Numerical comparison with exact percentages
4. **Summary Box**: Total pixels, forest pixels, and deforested pixels

### Summary Chart Components

1. **Title**: "Deforestation Rate Summary - All Images"
2. **X-axis**: Image names (truncated for readability)
3. **Y-axis**: Percentage (0-100%)
4. **Legend**: Forest (green) and Deforested (red)
5. **Value Labels**: Percentage shown on each bar

## Benefits

✅ **Easy to Understand**: Visual representation is more intuitive than numbers  
✅ **Quick Comparison**: Summary chart allows comparing multiple images at once  
✅ **Professional**: High-quality charts suitable for presentations  
✅ **Automatic**: No extra steps needed - charts are generated automatically  
✅ **Comprehensive**: Shows both percentages and pixel counts  

## Example Output

When you run inference, you'll see:

```
Processing forest_1.png...
✓ Saved mask for forest_1.png to outputs/masks/forest_1.png
  Deforestation: 15.41% (10102/65536 pixels)
  📊 Saved chart: outputs/masks/forest_1_deforestation_chart.png

Processing forest_2.png...
✓ Saved mask for forest_2.png to outputs/masks/forest_2.png
  Deforestation: 31.63% (20730/65536 pixels)
  📊 Saved chart: outputs/masks/forest_2_deforestation_chart.png

📊 Summary chart saved: outputs/masks/deforestation_summary.png
```

## Technical Details

- **Chart Format**: PNG (150 DPI for high quality)
- **Chart Size**: 14x6 inches (individual), 12x6 inches (summary)
- **Colors**: 
  - Forest: `#2d5016` (Dark Green)
  - Deforested: `#d62728` (Red)
- **Library**: matplotlib
- **File Size**: ~120-140 KB per chart

## Customization

To modify chart appearance, edit the `create_deforestation_charts()` and `create_summary_chart()` functions in `infer.py`:

- Change colors: Modify the `colors` variable
- Change size: Modify `figsize` parameter
- Change style: Modify matplotlib styling options

## Notes

- Charts are saved in the same directory as masks
- Summary chart is only created when processing 2+ images
- All charts use high resolution (150 DPI) for quality
- Charts include all necessary labels and legends

