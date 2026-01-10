#!/usr/bin/env python3
"""
Example: Using Structured PSO-based NAS
This demonstrates the modular PSO approach with separate components.
"""

import sys
from pathlib import Path

# Add module path
sys.path.append(str(Path(__file__).parent / 'module'))

from colabnas_core import ColabNAS
from pso_nas_structured import ArchitectureSearchSpace, ModelDecoder, NASPsoOptimizer
from dataset_utils import download_flower_dataset, check_gpu

def main():
    print("🧬 Structured PSO-based NAS Example")
    print("=" * 50)
    
    # Check GPU
    has_gpu = check_gpu()
    print(f"GPU: {'✅' if has_gpu else '❌'}")
    
    # Download dataset
    print("\n📥 Preparing dataset...")
    data_dir = download_flower_dataset()
    
    # Configuration
    config = {
        'max_RAM': 40960,
        'max_Flash': 131072,
        'max_MACC': 2730000,
        'path_to_training_set': str(data_dir),
        'val_split': 0.3,
        'cache': True,
        'input_shape': (50, 50, 3),
        'save_path': './structured_pso_results/'
    }
    
    # Create results directory
    Path(config['save_path']).mkdir(exist_ok=True)
    
    # Initialize ColabNAS
    print("\n🔧 Initializing ColabNAS...")
    colabnas = ColabNAS(**config)
    
    # Setup PSO components
    print("\n🧬 Setting up PSO components...")
    search_space = ArchitectureSearchSpace(k_range=(4, 32), c_range=(0, 5))
    decoder = ModelDecoder()
    
    # Create PSO optimizer class
    PSOOptimizer = NASPsoOptimizer.setup(
        search_space=search_space,
        decoder=decoder,
        n_particles=5,
        iterations=10
    )
    
    print("✅ PSO components ready!")
    print(f"  Search space: k=[{search_space.k_min}, {search_space.k_max}], c=[{search_space.c_min}, {search_space.c_max}]")
    print(f"  Particles: {PSOOptimizer.n_particles}")
    print(f"  Iterations: {PSOOptimizer.iterations}")
    
    # Run search
    print("\n🚀 Starting structured PSO search...")
    try:
        result_path = colabnas.search(PSOOptimizer)
        
        if result_path and Path(result_path).exists():
            model_size = Path(result_path).stat().st_size
            print(f"\n🎉 Success!")
            print(f"📁 Model: {result_path}")
            print(f"📊 Size: {model_size/1024:.2f} KB")
        else:
            print(f"\n❌ No feasible architecture found")
            
    except Exception as e:
        print(f"\n💥 Error: {e}")

if __name__ == "__main__":
    main()