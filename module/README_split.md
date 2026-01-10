# ColabNAS Split Implementation

This is a modular version of ColabNAS split into multiple Python files for better organization and easier deployment on H100 or other high-performance GPUs.

## Files Structure

- `run_colabnas.py` - **Main execution script** (run this file)
- `colabnas_core.py` - Core ColabNAS class with training and evaluation logic
- `nas_algorithm.py` - Neural Architecture Search algorithm implementation
- `dataset_utils.py` - Dataset downloading and GPU utilities
- `requirements_split.txt` - Required Python packages

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements_split.txt
```

2. **Run the main script:**
```bash
python run_colabnas.py
```

## Configuration

The main configuration is in `run_colabnas.py`:

```python
# Target: STM32L412KBU3
input_shape = (50, 50, 3)
peak_RAM_upper_bound = 40960    # 40 KB
Flash_upper_bound = 131072      # 128 KB  
MACC_upper_bound = 2730000      # CoreMark * 1e4
val_split = 0.3
cache = True
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (H100, RTX series, etc.)
- **RAM**: 8GB+ system RAM
- **Storage**: 5GB+ free space for dataset and models

## Output

The script will:
1. Download the flower dataset automatically
2. Search for optimal CNN architectures
3. Train and evaluate candidate models
4. Generate a quantized TFLite model ready for STM32 deployment
5. Save results in `./results/` directory

## Expected Runtime

- **H100**: 30-60 minutes
- **RTX 4090**: 1-2 hours  
- **RTX 3080**: 2-4 hours
- **CPU only**: 8-12 hours

## Deployment

The final `.tflite` model can be deployed on STM32L412KBU3 using:
- STM32CubeAI
- TensorFlow Lite Micro

## Customization

To use your own dataset, modify the `path_to_training_set` in `run_colabnas.py` and ensure your dataset follows this structure:

```
your_dataset/
├── class1/
│   ├── image1.jpg
│   └── image2.jpg
└── class2/
    ├── image3.jpg
    └── image4.jpg
```