# Local ColabNAS - Neural Architecture Search for TinyML

A local implementation of ColabNAS that can run on your notebook without Google Colab dependencies. This tool automatically searches for optimal CNN architectures for TinyML devices within specified hardware constraints.

## Features

- **Local Execution**: Runs entirely on your local machine
- **Hardware-Aware**: Considers RAM, Flash, and MACC constraints
- **Automatic Quantization**: Converts models to uint8 for deployment
- **Memory Estimation**: Estimates resource usage without external tools
- **Early Stopping**: Faster convergence with early stopping
- **Flexible Input**: Supports various image sizes and datasets

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook

Open `LocalColabNAS.ipynb` in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook LocalColabNAS.ipynb
```

### 3. Configure Your Target Device

Edit the constraints in the notebook:

```python
# Example: STM32L412KBU3
peak_RAM_upper_bound = 40 * 1024      # 40 KB
Flash_upper_bound = 128 * 1024        # 128 KB  
MACC_upper_bound = 273 * 10000        # CoreMark * 1e4
input_shape = (64, 64, 3)             # Input image size
```

### 4. Run the Search

The notebook will automatically:
1. Download a sample dataset (or use your own)
2. Initialize the NAS system
3. Search for optimal architectures
4. Generate a quantized TFLite model
5. Test the model performance

## Architecture

The system uses a progressive search strategy:

1. **Model Generation**: Creates CNN architectures with k kernels and c cells
2. **Constraint Checking**: Validates RAM, Flash, and MACC requirements
3. **Training**: Trains promising architectures with early stopping
4. **Quantization**: Converts to uint8 TFLite format
5. **Selection**: Chooses the best performing feasible architecture

## Model Structure

Generated models follow this pattern:
- Input layer
- First convolutional layer (k kernels)
- c additional cells with:
  - MaxPooling2D
  - Conv2D (increasing channels)
  - BatchNormalization
  - ReLU activation
- GlobalAveragePooling2D
- Dense classifier layers
- Softmax output

## Hardware Constraints

The system considers three main constraints:

1. **RAM**: Peak memory usage during inference
2. **Flash**: Model storage size
3. **MACC**: Multiply-accumulate operations (computational complexity)

## Customization

### Using Your Own Dataset

Replace the dataset loading section with your data:

```python
# Your dataset should be organized as:
# dataset_folder/
#   ├── class1/
#   │   ├── image1.jpg
#   │   └── image2.jpg
#   └── class2/
#       ├── image3.jpg
#       └── image4.jpg

data_dir = Path("path/to/your/dataset")
```

### Adjusting Search Parameters

Modify the LocalColabNAS initialization:

```python
nas = LocalColabNAS(
    max_RAM=your_ram_limit,
    max_Flash=your_flash_limit,
    max_MACC=your_macc_limit,
    path_to_training_set=str(data_dir),
    val_split=0.2,           # Validation split
    cache=False,             # Cache dataset in memory
    input_shape=(64, 64, 3), # Input image size
    save_path=str(save_path)
)
```

### Training Parameters

Adjust training settings in `local_colabnas.py`:

```python
self.learning_rate = 1e-3
self.batch_size = 32
self.epochs = 50
```

## Output

The system generates:
- **Quantized TFLite model**: Ready for deployment
- **Performance metrics**: Accuracy, model size, resource usage
- **Architecture details**: k, c parameters and constraints satisfaction

## Deployment

The generated `.tflite` model can be deployed using:
- **STM32**: STM32Cube.AI
- **Arduino**: TensorFlow Lite Micro
- **Other MCUs**: TensorFlow Lite Micro framework

## Differences from Original ColabNAS

1. **No STM32 Tools Dependency**: Uses estimation instead of stm32tflm
2. **Local Execution**: No Google Colab requirements
3. **Reduced Resource Usage**: Optimized for local machines
4. **Enhanced Error Handling**: Better error recovery and reporting
5. **Flexible Configuration**: More customization options

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in the LocalColabNAS class
- Set `cache=False` to avoid caching datasets
- Use smaller input image sizes

### No Feasible Architecture Found
- Increase hardware constraints (RAM, Flash, MACC)
- Reduce input image size
- Use fewer classes in your dataset

### Training Too Slow
- Reduce `epochs` in the LocalColabNAS class
- Use GPU acceleration if available
- Reduce dataset size for initial testing

## Example Results

Typical results for flower classification (5 classes, 64x64 input):
- **Architecture**: k=4, c=2
- **Accuracy**: ~85%
- **Model Size**: ~15 KB
- **RAM Usage**: ~25 KB
- **Search Time**: 30-60 minutes

## License

This implementation is based on the original ColabNAS research. Please cite the original paper if you use this in academic work.

## Contributing

Feel free to submit issues and enhancement requests!