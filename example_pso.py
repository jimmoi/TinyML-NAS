#!/usr/bin/env python3
"""
Example: Using PSO-based NAS for TinyML
This script demonstrates how to use Particle Swarm Optimization
for Neural Architecture Search with your specific requirements.
"""

import os
import sys
from pathlib import Path

# Add module path
sys.path.append(str(Path(__file__).parent / 'module'))

from colabnas_core import ColabNAS
from pso_nas import PSONAS
from dataset_utils import download_flower_dataset, check_gpu

def main():
    print("🔬 PSO-based Neural Architecture Search Example")
    print("=" * 60)
    
    # Check GPU
    has_gpu = check_gpu()
    print(f"GPU Available: {'✅ Yes' if has_gpu else '❌ No (using CPU)'}")
    
    # Download dataset
    print("\n📥 Preparing flower dataset...")
    data_dir = download_flower_dataset()
    
    # PSO-specific configuration
    print("\n🧬 PSO Configuration:")
    print("  - Population size: 10 particles")
    print("  - Max iterations: 20")
    print("  - Search space: k=[2,32], c=[0,5]")
    print("  - Inertia weight: 0.5")
    print("  - Cognitive/Social parameters: 1.5")
    
    # Hardware constraints for STM32L412KBU3
    config = {
        'max_RAM': 40960,      # 40KB
        'max_Flash': 131072,   # 128KB  
        'max_MACC': 2730000,   # 273 CoreMark * 1e4
        'path_to_training_set': str(data_dir),
        'val_split': 0.3,
        'cache': True,
        'input_shape': (50, 50, 3),
        'save_path': './pso_results/'
    }
    
    print(f"\n🎯 Target Hardware: STM32L412KBU3")
    print(f"  RAM: {config['max_RAM']/1024:.0f} KB")
    print(f"  Flash: {config['max_Flash']/1024:.0f} KB") 
    print(f"  MACC: {config['max_MACC']:,}")
    
    # Create results directory
    Path(config['save_path']).mkdir(exist_ok=True)
    
    # Initialize ColabNAS
    print(f"\n🔧 Initializing ColabNAS with PSO...")
    colabnas = ColabNAS(**config)
    
    # Run PSO-based search
    print(f"\n🚀 Starting PSO-based Architecture Search...")
    print("This will explore the architecture space using particle swarm optimization")
    print("-" * 60)
    
    try:
        result_path = colabnas.search(PSONAS)
        
        if result_path and Path(result_path).exists():
            model_size = Path(result_path).stat().st_size
            print(f"\n🎉 PSO Search Completed!")
            print(f"📁 Best model: {result_path}")
            print(f"📊 Model size: {model_size/1024:.2f} KB")
            print(f"✅ Ready for STM32 deployment!")
        else:
            print(f"\n❌ PSO could not find feasible architecture")
            print("💡 Try adjusting PSO parameters or hardware constraints")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ PSO search interrupted")
    except Exception as e:
        print(f"\n💥 Error: {e}")

if __name__ == "__main__":
    main()