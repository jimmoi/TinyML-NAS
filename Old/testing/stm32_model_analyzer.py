import tensorflow as tf
import numpy as np
import subprocess
import re
from pathlib import Path
import os

class STM32ModelAnalyzer:
    """
    Analyze TFLite models using STM32TFLM for accurate memory usage
    """
    
    def __init__(self, stm32tflm_path):
        self.stm32tflm_path = Path(stm32tflm_path)
        if not self.stm32tflm_path.exists():
            raise FileNotFoundError(f"STM32TFLM not found at {stm32tflm_path}")
    
    def analyze_model(self, tflite_model_path):
        """Analyze TFLite model using STM32TFLM"""
        model_path = Path(tflite_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {tflite_model_path}")
        
        try:
            # Run STM32TFLM analysis
            proc = subprocess.Popen(
                [str(self.stm32tflm_path), str(model_path)], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            outs, errs = proc.communicate(timeout=30)
            
            # Parse output for Flash and RAM usage
            flash_match = re.search(r'Flash.*?(\d+)', outs)
            ram_match = re.search(r'RAM.*?(\d+)', outs)
            
            flash_usage = int(flash_match.group(1)) if flash_match else None
            ram_usage = int(ram_match.group(1)) if ram_match else None
            
            return {
                'flash_bytes': flash_usage,
                'ram_bytes': ram_usage,
                'model_size_bytes': model_path.stat().st_size,
                'analysis_output': outs,
                'errors': errs
            }
            
        except subprocess.TimeoutExpired:
            proc.kill()
            return {'error': 'STM32TFLM analysis timeout'}
        except Exception as e:
            return {'error': f'STM32TFLM analysis failed: {str(e)}'}

def analyze_trained_models(models_directory, stm32tflm_path):
    """Analyze all TFLite models in directory using STM32TFLM"""
    
    analyzer = STM32ModelAnalyzer(stm32tflm_path)
    models_dir = Path(models_directory)
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_directory}")
        return
    
    # Find all TFLite models
    tflite_models = list(models_dir.glob("*.tflite"))
    
    if not tflite_models:
        print(f"No TFLite models found in {models_directory}")
        return
    
    print(f"Found {len(tflite_models)} TFLite models")
    print("="*60)
    
    results = []
    
    for model_path in tflite_models:
        print(f"\nAnalyzing: {model_path.name}")
        print("-" * 40)
        
        # Analyze with STM32TFLM
        analysis = analyzer.analyze_model(model_path)
        
        if 'error' in analysis:
            print(f"❌ Error: {analysis['error']}")
            continue
        
        # Display results
        flash_kb = analysis['flash_bytes'] / 1024 if analysis['flash_bytes'] else 0
        ram_kb = analysis['ram_bytes'] / 1024 if analysis['ram_bytes'] else 0
        model_size_kb = analysis['model_size_bytes'] / 1024
        
        print(f"📊 Model Size: {model_size_kb:.2f} KB")
        print(f"💾 Flash Usage: {flash_kb:.2f} KB ({analysis['flash_bytes']} bytes)")
        print(f"🔧 RAM Usage: {ram_kb:.2f} KB ({analysis['ram_bytes']} bytes)")
        
        # Test model inference
        try:
            interpreter = tf.lite.Interpreter(str(model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            
            print(f"🔍 Input: {input_details['shape']} ({input_details['dtype']})")
            print(f"🎯 Output: {output_details['shape']} ({output_details['dtype']})")
            
            if input_details['dtype'] == tf.uint8:
                scale, zero_point = input_details['quantization']
                print(f"⚖️  Quantization - Scale: {scale:.6f}, Zero point: {zero_point}")
        
        except Exception as e:
            print(f"⚠️  Model loading error: {e}")
        
        results.append({
            'model_name': model_path.name,
            'flash_bytes': analysis['flash_bytes'],
            'ram_bytes': analysis['ram_bytes'],
            'model_size_bytes': analysis['model_size_bytes']
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results:
        total_models = len(results)
        avg_flash = np.mean([r['flash_bytes'] for r in results if r['flash_bytes']])
        avg_ram = np.mean([r['ram_bytes'] for r in results if r['ram_bytes']])
        
        print(f"Total models analyzed: {total_models}")
        print(f"Average Flash usage: {avg_flash/1024:.2f} KB")
        print(f"Average RAM usage: {avg_ram/1024:.2f} KB")
        
        # Find best model (lowest resource usage)
        best_model = min(results, key=lambda x: (x['ram_bytes'] or float('inf')))
        print(f"Most efficient model: {best_model['model_name']}")
        print(f"  Flash: {best_model['flash_bytes']/1024:.2f} KB")
        print(f"  RAM: {best_model['ram_bytes']/1024:.2f} KB")
    
    return results

def compare_with_constraints(results, max_ram_bytes, max_flash_bytes):
    """Compare analysis results with hardware constraints"""
    
    print(f"\n🎯 CONSTRAINT ANALYSIS")
    print(f"Max RAM: {max_ram_bytes/1024:.1f} KB")
    print(f"Max Flash: {max_flash_bytes/1024:.1f} KB")
    print("-" * 40)
    
    feasible_models = []
    
    for result in results:
        model_name = result['model_name']
        flash_usage = result['flash_bytes']
        ram_usage = result['ram_bytes']
        
        flash_ok = flash_usage <= max_flash_bytes if flash_usage else False
        ram_ok = ram_usage <= max_ram_bytes if ram_usage else False
        
        status = "✅" if (flash_ok and ram_ok) else "❌"
        
        print(f"{status} {model_name}")
        print(f"   Flash: {flash_usage/1024:.2f} KB ({'OK' if flash_ok else 'EXCEED'})")
        print(f"   RAM: {ram_usage/1024:.2f} KB ({'OK' if ram_ok else 'EXCEED'})")
        
        if flash_ok and ram_ok:
            feasible_models.append(result)
    
    print(f"\n✅ Feasible models: {len(feasible_models)}/{len(results)}")
    return feasible_models

if __name__ == "__main__":
    # Configuration
    MODELS_DIR = "./content"  # Directory containing TFLite models
    STM32TFLM_PATH = "./stm32tflm"  # Path to STM32TFLM executable
    
    # Hardware constraints (STM32L412KBU3)
    MAX_RAM = 40960    # 40 KB
    MAX_FLASH = 131072 # 128 KB
    
    print("🔍 STM32 Model Analysis Tool")
    print("="*60)
    
    try:
        # Analyze all models
        results = analyze_trained_models(MODELS_DIR, STM32TFLM_PATH)
        
        if results:
            # Compare with constraints
            feasible = compare_with_constraints(results, MAX_RAM, MAX_FLASH)
            
            if feasible:
                print(f"\n🎉 Found {len(feasible)} feasible models for deployment!")
            else:
                print(f"\n⚠️  No models meet the hardware constraints.")
                print("Consider:")
                print("- Reducing input image size")
                print("- Using fewer model parameters")
                print("- Increasing hardware limits")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. STM32TFLM executable is available")
        print("2. Models directory exists")
        print("3. TFLite models are present")