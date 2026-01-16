# Agentic Data Quality Auditor for Computer Vision

An intelligent agent that automatically audits the quality of image datasets **before** model training. This tool analyzes images for blur, noise, brightness issues, and class imbalance, then provides actionable recommendations.

## Features

### 1. **Data Quality Analysis**
- **Blur Detection**: Uses Laplacian variance to detect blurry images
- **Noise Estimation**: Statistical method to estimate noise levels
- **Brightness Analysis**: Mean pixel intensity analysis
- **Class Distribution**: Detects class imbalance issues

### 2. **Agentic Decision Engine**
A rule-based agent that makes human-readable decisions:
- "35% images are blurry → recommend denoising"
- "Class X is under-represented → recommend oversampling"
- "Dataset too dark → recommend brightness normalization"

### 3. **Auto-Fix Module (Safe Fixes Only)**
- Image denoising using OpenCV (Fast Non-Local Means or Bilateral Filter)
- Brightness normalization
- **Note**: Does NOT modify class distribution automatically (only recommends)

### 4. **Visualization Module**
Generates:
- Histogram of image brightness distribution
- Bar chart of class distribution (color-coded for imbalance)
- Blur score distribution plot
- Noise score distribution plot

### 5. **Report Generation**
Automatically generates:
- Comprehensive text report
- JSON report for programmatic access
- All visualizations saved as high-resolution PNG files

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

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

### Custom Output Directory
```bash
python main.py --dataset path/to/your/dataset --output my_output_dir
```

### Skip Visualizations
```bash
python main.py --dataset path/to/your/dataset --no-viz
```

### Full Example
```bash
python main.py --dataset ./my_dataset --output results --fix
```

## Output

After running the audit, you'll find:

1. **Reports** (in `output/` directory):
   - `quality_report.txt` - Human-readable text report
   - `quality_report.json` - Machine-readable JSON report

2. **Visualizations** (in `output/` directory):
   - `brightness_histogram.png`
   - `class_distribution.png`
   - `blur_distribution.png`
   - `noise_distribution.png`

3. **Fixed Images** (in `fixed_images/` directory, if `--fix` is used):
   - Fixed images with preserved directory structure

## Project Structure

```
.
├── data_loader.py          # Loads images from ImageFolder structure
├── quality_metrics.py      # Computes blur, noise, brightness metrics
├── agent.py                # Rule-based decision engine
├── visualizer.py           # Generates plots and charts
├── fixer.py                # Applies safe auto-fixes
├── report_generator.py      # Generates text and JSON reports
├── main.py                 # Entry point with CLI
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## How It Works

1. **Data Loading**: Scans the dataset directory and loads all images
2. **Quality Metrics**: Computes blur, noise, and brightness for each image
3. **Agent Analysis**: Rule-based agent analyzes statistics and makes decisions
4. **Visualization**: Generates plots for visual inspection
5. **Auto-Fix** (optional): Applies denoising and brightness normalization
6. **Report Generation**: Creates comprehensive reports

## Quality Thresholds

Default thresholds (can be modified in `agent.py`):
- **Blur**: Laplacian variance < 100 = blurry
- **Noise**: Noise score > 20 = noisy
- **Brightness**: 
  - < 50 = too dark
  - > 200 = too bright
- **Class Imbalance**: Class with < 30% of average count = imbalanced

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
Detected 2 issue(s):
  1. [MEDIUM] BLUR
     35% of images are blurry (437/1250)
  2. [HIGH] CLASS_IMBALANCE
     Class "rabbit" is under-represented (150 images, 60.0% of average)

Agent Decisions:
  1. 35% images are blurry → recommend denoising
  2. Class "rabbit" is under-represented → recommend oversampling
```

## Limitations

- Works with RGB images only
- No deep learning models (pure computer vision)
- Auto-fixes are conservative (only denoising and brightness)
- Class distribution fixes require manual intervention

## Requirements

- Python 3.7+
- OpenCV 4.8+
- NumPy 1.24+
- Matplotlib 3.7+

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to extend this project with:
- Additional quality metrics
- More sophisticated denoising methods
- PDF report generation
- Support for other image formats
- Batch processing capabilities
