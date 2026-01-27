import tensorflow as tf
import numpy as np
from pathlib import Path
import os

class MockSTM32ModelAnalyzer:
    """
    Mock analyzer that estimates memory usage without STM32TFLM executable
    """
    
    def __init__(self, stm32tflm_path=None):
        print("⚠️ Using mock STM32 analyzer (STM32TFLM not available)")
    
    def analyze_model(self, tflite_model_path):
        """Estimate memory usage for TFLite model"""
        model_path = Path(tflite_model_path)
        if not model_path.exists():
            return {'error': f'Model not found at {tflite_model_path}'}
        
        try:
            # Load TFLite model to get basic info
            interpreter = tf.lite.Interpreter(str(model_path))
            interpreter.allocate_tensors()
            
            # Get model size
            model_size = model_path.stat().st_size
            
            # Estimate Flash usage (model size + overhead)
            flash_estimate = int(model_size * 1.2)  # 20% overhead
            
            # Estimate RAM usage based on model complexity
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            
            input_size = np.prod(input_details['shape']) * 4  # 4 bytes per float
            output_size = np.prod(output_details['shape']) * 4
            
            # Rough RAM estimation: input + output + intermediate activations
            ram_estimate = int((input_size + output_size) * 3)
            
            return {
                'flash_bytes': flash_estimate,
                'ram_bytes': ram_estimate,
                'model_size_bytes': model_size,
                'analysis_output': f'Mock analysis for {model_path.name}',
                'errors': None
            }
            
        except Exception as e:
            return {'error': f'Model analysis failed: {str(e)}'}

def analyze_trained_models_mock(models_directory, stm32tflm_path=None):
    """Mock version that works without STM32TFLM"""
    
    analyzer = MockSTM32ModelAnalyzer(stm32tflm_path)
    models_dir = Path(models_directory)
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_directory}")
        return []
    
    # Find all TFLite models
    tflite_models = list(models_dir.glob("*.tflite"))
    
    if not tflite_models:
        print(f"No TFLite models found in {models_directory}")
        return []
    
    print(f"Found {len(tflite_models)} TFLite models")
    print("="*60)
    
    results = []
    
    for model_path in tflite_models:
        print(f"\nAnalyzing: {model_path.name}")
        print("-" * 40)
        
        # Analyze with mock analyzer
        analysis = analyzer.analyze_model(model_path)
        
        if 'error' in analysis:
            print(f"❌ Error: {analysis['error']}")
            continue
        
        # Display results
        flash_kb = analysis['flash_bytes'] / 1024
        ram_kb = analysis['ram_bytes'] / 1024
        model_size_kb = analysis['model_size_bytes'] / 1024
        
        print(f"📊 Model Size: {model_size_kb:.2f} KB")
        print(f"💾 Flash Usage: {flash_kb:.2f} KB (estimated)")
        print(f"🔧 RAM Usage: {ram_kb:.2f} KB (estimated)")
        
        results.append({
            'model_name': model_path.name,
            'flash_bytes': analysis['flash_bytes'],
            'ram_bytes': analysis['ram_bytes'],
            'model_size_bytes': analysis['model_size_bytes']
        })
    
    return results

def compare_with_constraints(results, max_ram_bytes, max_flash_bytes):
    """Compare results with hardware constraints"""
    
    print(f"\n🎯 CONSTRAINT ANALYSIS")
    print(f"Max RAM: {max_ram_bytes/1024:.1f} KB")
    print(f"Max Flash: {max_flash_bytes/1024:.1f} KB")
    print("-" * 40)
    
    feasible_models = []
    
    for result in results:
        model_name = result['model_name']
        flash_usage = result['flash_bytes']
        ram_usage = result['ram_bytes']
        
        flash_ok = flash_usage <= max_flash_bytes
        ram_ok = ram_usage <= max_ram_bytes
        
        status = "✅" if (flash_ok and ram_ok) else "❌"
        
        print(f"{status} {model_name}")
        print(f"   Flash: {flash_usage/1024:.2f} KB ({'OK' if flash_ok else 'EXCEED'})")
        print(f"   RAM: {ram_usage/1024:.2f} KB ({'OK' if ram_ok else 'EXCEED'})")
        
        if flash_ok and ram_ok:
            feasible_models.append(result)
    
    print(f"\n✅ Feasible models: {len(feasible_models)}/{len(results)}")
    return feasible_models

if __name__ == "__main__":
    # Test the mock analyzer
    MODELS_DIR = "./content"
    MAX_RAM = 40960
    MAX_FLASH = 131072
    
    results = analyze_trained_models_mock(MODELS_DIR)
    if results:
        feasible = compare_with_constraints(results, MAX_RAM, MAX_FLASH)