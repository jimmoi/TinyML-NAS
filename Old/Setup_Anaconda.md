# Anaconda Setup Guide for LocalColabNAS

This guide will help you set up Anaconda environment for running LocalColabNAS on your local machine.

## Prerequisites

- Windows 10/11, macOS, or Linux
- At least 8GB RAM (16GB recommended)
- 5GB free disk space
- Internet connection for downloading packages

## Step 1: Install Anaconda

### Windows
1. Download Anaconda from https://www.anaconda.com/download
2. Run the installer as administrator
3. Choose "Add Anaconda to PATH" during installation
4. Complete installation and restart your computer

### macOS
```bash
# Download and install via terminal
curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-MacOSX-x86_64.sh
bash Anaconda3-2023.09-0-MacOSX-x86_64.sh
```

### Linux
```bash
# Download and install
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```

## Step 2: Create Environment

Open Anaconda Prompt (Windows) or Terminal (macOS/Linux):

```bash
# Create new environment with Python 3.9
conda create -n tinyml python=3.9 -y

# Activate environment
conda activate tinyml
```

## Step 3: Install Core Packages

```bash
# Install TensorFlow and dependencies
conda install -c conda-forge tensorflow=2.13 -y

# Install additional packages
conda install -c conda-forge numpy matplotlib jupyter ipykernel -y

# Install via pip for latest versions
pip install tensorflow-model-optimization pathlib2
```

## Step 4: GPU Support (Optional)

### For NVIDIA GPU:
```bash
# Install CUDA toolkit
conda install -c conda-forge cudatoolkit=11.8 -y
conda install -c conda-forge cudnn=8.6 -y

# Verify GPU detection
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

### For AMD GPU (ROCm):
```bash
# Install ROCm support
pip install tensorflow-rocm
```

## Step 5: Register Kernel

```bash
# Register environment with Jupyter
python -m ipykernel install --user --name tinyml --display-name "TinyML (Python 3.9)"
```

## Step 6: Verify Installation

Create test file `test_setup.py`:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ NumPy: {np.__version__}")
print(f"✅ Python: {tf.version.VERSION}")
print(f"✅ GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Test basic operations
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.constant([[1.0, 1.0], [0.0, 1.0]])
z = tf.matmul(x, y)
print(f"✅ TensorFlow operations working: {z.numpy()}")
```

Run test:
```bash
python test_setup.py
```

## Step 7: Launch Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook

# Or Jupyter Lab
jupyter lab
```

## Quick Commands Reference

```bash
# Activate environment
conda activate tinyml

# Deactivate environment
conda deactivate

# List environments
conda env list

# Update packages
conda update --all

# Remove environment (if needed)
conda env remove -n tinyml
```

## Troubleshooting

### Common Issues:

**1. TensorFlow not found:**
```bash
conda activate tinyml
pip install --upgrade tensorflow
```

**2. Jupyter kernel not showing:**
```bash
conda activate tinyml
python -m ipykernel install --user --name tinyml --display-name "TinyML"
```

**3. GPU not detected:**
```bash
# Check CUDA installation
nvidia-smi
# Reinstall CUDA toolkit
conda install -c conda-forge cudatoolkit cudnn
```

**4. Memory errors:**
```bash
# Increase virtual memory or reduce batch size in code
# Set environment variable
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

**5. Package conflicts:**
```bash
# Clean install
conda env remove -n tinyml
conda create -n tinyml python=3.9 -y
conda activate tinyml
pip install tensorflow numpy matplotlib jupyter
```

## Environment Export/Import

### Export current environment:
```bash
conda activate tinyml
conda env export > tinyml_environment.yml
```

### Import environment on another machine:
```bash
conda env create -f tinyml_environment.yml
conda activate tinyml
```

## Performance Tips

1. **Use SSD storage** for faster data loading
2. **Enable GPU** if available for 10x speedup
3. **Increase RAM** allocation in system settings
4. **Close unnecessary applications** during training
5. **Use smaller batch sizes** if memory limited

## Verification Checklist

- [ ] Anaconda installed successfully
- [ ] Environment `tinyml` created and activated
- [ ] TensorFlow 2.10+ installed
- [ ] GPU detected (if available)
- [ ] Jupyter notebook launches
- [ ] Can import all required packages
- [ ] Test script runs without errors

## Next Steps

After successful setup:

1. Navigate to your LocalColabNAS directory
2. Activate the environment: `conda activate tinyml`
3. Launch Jupyter: `jupyter notebook`
4. Open `LocalColabNAS.ipynb`
5. Run all cells to start your first NAS experiment

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all packages are installed correctly
3. Ensure you're using the correct environment
4. Check system requirements are met

Happy Neural Architecture Searching! 🚀