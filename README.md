# Agentic Data Quality Auditor for Computer Vision

An intelligent agent that automatically audits and preprocesses image datasets **before** model training. This tool analyzes images for blur, noise, brightness, contrast, saturation, and sharpness issues, then provides actionable recommendations and can automatically fix detected issues.

## Features

### 1. **Comprehensive Data Quality Analysis**
- **Blur Detection**: Uses Laplacian variance to detect blurry images
- **Sharpness Analysis**: Detects over-sharpened images (Laplacian variance > threshold)
- **Noise Estimation**: Statistical method to estimate noise levels
- **Brightness Analysis**: Mean pixel intensity analysis with automatic correction
- **Contrast Evaluation**: Standard deviation of pixel intensities (low/high contrast detection)
- **Saturation Analysis**: Mean saturation value detection (under/over-saturated)
- **Class Distribution**: Detects class imbalance issues
- **Consistent Image Sizing**: Automatically resizes all images to 224x224 with aspect ratio preservation

### 2. **Agentic Decision Engine**
A rule-based agent that makes human-readable decisions:
- "35% images are blurry → recommend denoising"
- "Class X is under-represented → recommend augmentation"
- "Dataset too dark → recommend brightness normalization"
- "X images have low contrast → applying CLAHE enhancement"
- "X images have high saturation → applying saturation reduction"

### 3. **Auto-Fix Module with Verification Loop**
The system applies fixes iteratively with verification:
- **Image Denoising**: Fast Non-Local Means denoising for noisy images
- **Brightness Normalization**: Adaptive brightness adjustment toward target mean (128)
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for low contrast
- **Contrast Reduction**: Dynamic range compression for high contrast images
- **Sharpness Normalization**: Gaussian smoothing for over-sharpened images
- **Saturation Adjustment**: HSV-based saturation correction
- **Image Resizing**: Consistent 224x224 size with padding (preserves aspect ratio)
- **Class Imbalance Augmentation**: Automatic augmentation (flips, rotations, brightness jitter) for minority classes
  
### 4. **Visualization Module**
Generates comprehensive visualizations:
- Histogram of image brightness distribution (with threshold markers)
- Bar chart of class distribution (color-coded for imbalance)
- Blur score distribution plot
- Noise score distribution plot
- **Contrast score distribution plot** (NEW)
- **Saturation score distribution plot** (NEW)

### 5. **Model Evaluation** (Optional)
Train a lightweight CNN baseline model to compare performance:
- Compares accuracy on original vs. cleaned dataset
- Provides quantitative justification for preprocessing
- Generates classification reports

### 6. **Report Generation**
Automatically generates:
- Comprehensive text report with before/after statistics
- JSON report for programmatic access
- All visualizations saved as high-resolution PNG files
- Detailed logging of all fixes applied

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: PyTorch and torchvision are optional dependencies (only needed for model evaluation). If you don't plan to use the `--evaluate` flag, you can skip installing them.

## Dataset Structure

The tool expects an **ImageFolder-style** structure:
```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── class3/
    └── ...
```

## Usage

### Basic Usage (Analysis Only)
```bash
python main.py --dataset path/to/your/dataset
```

### With Auto-Fixes
```bash
python main.py --dataset path/to/your/dataset --fix
```

### With Model Evaluation
```bash
python main.py --dataset path/to/your/dataset --fix --evaluate
```

### Custom Output Directory
```bash
python main.py --dataset path/to/your/dataset --output my_output_dir
```

### Skip Visualizations
```bash
python main.py --dataset path/to/your/dataset --no-viz
```

### Custom Evaluation Epochs
```bash
python main.py --dataset path/to/your/dataset --fix --evaluate --eval-epochs 10
```

### Full Example
```bash
python main.py --dataset ./my_dataset --output results --fix --evaluate
```

## Output

After running the audit, you'll find:

1. **Reports** (in `output/` directory):
   - `quality_report.txt` - Human-readable text report with before/after metrics
   - `quality_report.json` - Machine-readable JSON report

2. **Visualizations** (in `output/` directory):
   - `brightness_histogram.png`
   - `class_distribution.png`
   - `blur_distribution.png`
   - `noise_distribution.png`
   - `contrast_distribution.png`
   - `saturation_distribution.png` 

3. **Cleaned Dataset** (in `cleaned_dataset/` directory, if `--fix` is used):
   - All images resized to 224x224
   - Fixed images with preserved directory structure
   - Corrupted and severely blurry images removed
   - Augmented images for minority classes

## Project Structure

```
.
├── data_loader.py          # Loads images from ImageFolder structure
├── quality_metrics.py      # Computes all quality metrics (blur, noise, brightness, contrast, saturation, etc.)
├── agent.py                # Rule-based decision engine
├── visualizer.py           # Generates plots and charts
├── fixer.py                # Applies fixes with verification loop
├── preprocessor.py         # Main preprocessing pipeline orchestrator
├── report_generator.py      # Generates text and JSON reports
├── model_evaluator.py      # Optional: Model evaluation module
├── main.py                 # Entry point with CLI
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## How It Works

1. **Data Loading**: Scans the dataset directory and loads all images
2. **Quality Metrics**: Computes blur, noise, brightness, contrast, saturation, sharpness, and corruption for each image
3. **Agent Analysis**: Rule-based agent analyzes statistics and makes decisions
4. **Visualization**: Generates plots for visual inspection
5. **Preprocessing** (optional, with `--fix`):
   - Removes severely blurry images
   - Applies fixes iteratively with verification loop
   - Resizes all images to 224x224 (consistent size)
   - Augments minority classes to balance distribution
6. **Model Evaluation** (optional, with `--evaluate`): Trains baseline CNN and compares performance
7. **Report Generation**: Creates comprehensive reports with before/after statistics

## Quality Thresholds

Default thresholds (can be modified in `agent.py`):
- **Blur**: Laplacian variance < 100 = blurry (removed if severely blurry)
- **Sharpness**: Laplacian variance > 500 = over-sharpened
- **Noise**: Noise score > 20 = noisy
- **Brightness**: 
  - < 50 = too dark
  - > 200 = too bright
  - Target mean: 128
- **Contrast**:
  - < 30 = low contrast (applies CLAHE)
  - > 80 = high contrast (applies reduction)
- **Saturation**:
  - < 30 = under-saturated
  - > 200 = over-saturated
- **Class Imbalance**: Class with < 30% of average count = imbalanced (triggers augmentation)

## Verification Loop

The preprocessing system uses an iterative verification approach:
1. Apply fixes based on agent decisions
2. Re-measure all quality metrics
3. If issues persist, apply stronger corrections
4. If still problematic after max iterations, remove the image
5. Log all actions taken for transparency

## Example Output

```
AGENTIC DATA QUALITY AUDITOR FOR COMPUTER VISION
================================================================================
Generated: 2024-01-15 10:30:00

DATASET SUMMARY
--------------------------------------------------------------------------------
Dataset Path: ./my_dataset
Total Images: 1250
Number of Classes: 5

CLASS DISTRIBUTION
--------------------------------------------------------------------------------
  cat: 300 images (24.0%)
  dog: 350 images (28.0%)
  bird: 200 images (16.0%)
  fish: 250 images (20.0%)
  rabbit: 150 images (12.0%)

AGENT ANALYSIS
--------------------------------------------------------------------------------
Detected 5 issue(s):
  1. [MEDIUM] BLUR
     35% of images are blurry (437/1250)
  2. [HIGH] CLASS_IMBALANCE
     Class "rabbit" is under-represented (150 images, 60.0% of average)
  3. [MEDIUM] BRIGHTNESS_DARK
     120 images have low brightness (9.6%)
  4. [MEDIUM] CONTRAST_LOW
     85 images have low contrast (6.8%)
  5. [LOW] SATURATION_HIGH
     45 images have high saturation (3.6%)

Agent Decisions:
  1. 35% images are blurry → applying denoising
  2. Class "rabbit" is under-represented → applying augmentation
  3. 120 images have low brightness → applying brightness normalization
  4. 85 images have low contrast → applying CLAHE contrast enhancement
  5. 45 images have high saturation → applying saturation reduction

Model Evaluation Results:
  Original Dataset: 73.97% accuracy
  Cleaned Dataset: 73.12% accuracy
  Improvement: -0.85% (-1.1% relative)
```

## Limitations

- Works with RGB images only
- Model evaluation uses a lightweight CNN (for demonstration purposes)
- Auto-fixes are conservative to preserve image quality
- Some severely corrupted images are permanently removed

## Requirements

- Python 3.7+
- OpenCV 4.8+
- NumPy 1.24+
- Matplotlib 3.7+
- scikit-learn (for model evaluation metrics)
- PyTorch 2.0+ (optional, only for model evaluation)
- torchvision (optional, only for model evaluation)

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to extend this project with:
- Additional quality metrics
- More sophisticated preprocessing methods
- PDF report generation
- Support for other image formats
- Batch processing capabilities
- Advanced augmentation strategies
