#!/usr/bin/env python3
"""
Main script to run ColabNAS on H100 or local GPU
Optimized for high-performance training
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import our modules
from colabnas_core import ColabNAS
from nas_algorithm import OurNAS
from dataset_utils import download_flower_dataset, check_gpu

def main():
    print("🚀 Starting ColabNAS Training")
    print("=" * 50)
    
    # Check GPU availability
    has_gpu = check_gpu()
    if has_gpu:
        print("✅ GPU detected - using GPU acceleration")
    else:
        print("⚠️  No GPU detected - using CPU (will be slower)")
    
    # Download and prepare dataset
    print("\n📥 Preparing dataset...")
    data_dir = download_flower_dataset()
    print(f"Dataset ready at: {data_dir}")
    
    # Configuration - exact as requested
    input_shape = (50, 50, 3)
    
    # Target: STM32L412KBU3
    # 273 CoreMark, 40 kiB RAM, 128 kiB Flash
    peak_RAM_upper_bound = 40960
    Flash_upper_bound = 131072
    MACC_upper_bound = 2730000  # CoreMark * 1e4
    
    path_to_training_set = str(data_dir)
    val_split = 0.3
    cache = True
    save_path = './results/'
    
    print(f"\n🎯 Target Configuration:")
    print(f"  Input shape: {input_shape}")
    print(f"  RAM limit: {peak_RAM_upper_bound} bytes ({peak_RAM_upper_bound/1024:.0f} KB)")
    print(f"  Flash limit: {Flash_upper_bound} bytes ({Flash_upper_bound/1024:.0f} KB)")
    print(f"  MACC limit: {MACC_upper_bound:,}")
    print(f"  Validation split: {val_split}")
    print(f"  Cache: {cache}")
    
    # Create results directory
    Path(save_path).mkdir(exist_ok=True)
    
    # Initialize ColabNAS
    print(f"\n🔧 Initializing ColabNAS...")
    colabNAS = ColabNAS(
        max_RAM=peak_RAM_upper_bound,
        max_Flash=Flash_upper_bound,
        max_MACC=MACC_upper_bound,
        path_to_training_set=path_to_training_set,
        val_split=val_split,
        cache=cache,
        input_shape=input_shape,
        save_path=save_path
    )
    
    print("✅ ColabNAS initialized successfully!")
    
    # Start the search
    print(f"\n🔍 Starting Neural Architecture Search...")
    print("This may take several hours depending on your hardware.")
    print("Progress will be shown below:")
    print("-" * 50)
    
    try:
        path_to_tflite_model = colabNAS.search(OurNAS)
        
        if path_to_tflite_model and Path(path_to_tflite_model).exists():
            print(f"\n🎉 Search completed successfully!")
            print(f"📁 Best model saved at: {path_to_tflite_model}")
            
            # Get model size
            model_size = Path(path_to_tflite_model).stat().st_size
            print(f"📊 Model size: {model_size/1024:.2f} KB")
            
            print(f"\n✅ Ready for deployment on STM32L412KBU3!")
            
        else:
            print(f"\n❌ No feasible architecture found within the given constraints.")
            print("💡 Consider:")
            print("   - Relaxing the hardware constraints")
            print("   - Using a smaller input image size")
            print("   - Reducing the number of classes")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Search interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error during search: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()