# MCUNet Research Summary

## Topic
MCUNet: Tiny deep learning on IoT devices through joint neural architecture and inference library co-design for microcontroller units.

## Used Method
- Two-stage neural architecture search (TinyNAS)
- Hardware-aware model design
- Inference engine optimization (TinyEngine)
- Memory scheduling optimization
- Quantization and pruning techniques

## Advantages/Negatives of Papers

### Advantages:
- Joint optimization of architecture and inference engine
- Significant memory reduction through scheduling
- Real deployment on commercial MCUs
- Open-source implementation available
- Comprehensive evaluation on multiple datasets

### Negatives:
- Limited to specific MCU platforms
- Complex co-design process
- Requires expertise in both ML and embedded systems

## NAS Code

```python
# MCUNet TinyNAS Implementation
import numpy as np
from typing import List, Tuple

class TinyNAS:
    def __init__(self, memory_constraint=256, latency_constraint=100):
        self.memory_limit = memory_constraint  # KB
        self.latency_limit = latency_constraint  # ms
        self.search_space = self.define_search_space()
    
    def define_search_space(self):
        return {
            'depths': [1, 2, 3, 4],
            'widths': [0.25, 0.5, 0.75, 1.0],
            'kernel_sizes': [3, 5, 7],
            'expansions': [3, 4, 6],
            'skip_connections': [True, False]
        }
    
    def progressive_shrinking(self, supernet):
        """Two-stage progressive shrinking strategy"""
        # Stage 1: Width shrinking
        width_candidates = [1.0, 0.75, 0.5, 0.25]
        best_width_subnets = []
        
        for width in width_candidates:
            subnet = self.extract_subnet(supernet, width_multiplier=width)
            if self.meets_constraints(subnet):
                accuracy = self.evaluate_subnet(subnet)
                best_width_subnets.append((subnet, accuracy))
        
        # Stage 2: Depth shrinking
        best_subnets = []
        for subnet, _ in sorted(best_width_subnets, key=lambda x: x[1], reverse=True)[:3]:
            depth_variants = self.generate_depth_variants(subnet)
            for variant in depth_variants:
                if self.meets_constraints(variant):
                    accuracy = self.evaluate_subnet(variant)
                    best_subnets.append((variant, accuracy))
        
        return max(best_subnets, key=lambda x: x[1])[0]
    
    def meets_constraints(self, subnet):
        """Check if subnet meets hardware constraints"""
        memory_usage = self.estimate_memory_usage(subnet)
        latency = self.estimate_latency(subnet)
        
        return memory_usage <= self.memory_limit and latency <= self.latency_limit
    
    def estimate_memory_usage(self, subnet):
        """Estimate peak memory usage including activations"""
        total_memory = 0
        
        for layer in subnet['layers']:
            # Parameters memory
            if layer['type'] == 'conv':
                params = layer['in_channels'] * layer['out_channels'] * layer['kernel_size']**2
                total_memory += params * 4  # 4 bytes per parameter
            
            # Activation memory (assuming worst case)
            if 'output_shape' in layer:
                h, w, c = layer['output_shape']
                activation_mem = h * w * c * 4  # 4 bytes per activation
                total_memory += activation_mem
        
        return total_memory / 1024  # Convert to KB
    
    def estimate_latency(self, subnet):
        """Estimate inference latency on target MCU"""
        total_ops = 0
        
        for layer in subnet['layers']:
            if layer['type'] == 'conv':
                h, w = layer['input_shape'][:2]
                ops = h * w * layer['in_channels'] * layer['out_channels'] * layer['kernel_size']**2
                total_ops += ops
        
        # Assume 1M ops/ms for target MCU
        return total_ops / 1000000
    
    def search(self):
        """Main search algorithm"""
        # Build supernet
        supernet = self.build_supernet()
        
        # Train supernet with progressive shrinking
        trained_supernet = self.train_supernet(supernet)
        
        # Extract optimal subnet
        optimal_subnet = self.progressive_shrinking(trained_supernet)
        
        return optimal_subnet
    
    def build_supernet(self):
        """Build supernet containing all possible architectures"""
        return {
            'layers': [
                {
                    'type': 'conv',
                    'in_channels': 3,
                    'out_channels': 32,
                    'kernel_size': 3,
                    'stride': 2,
                    'input_shape': (224, 224, 3),
                    'output_shape': (112, 112, 32)
                },
                # MobileNet-like blocks with different configurations
                *self.create_inverted_residual_blocks(),
                {
                    'type': 'global_avg_pool',
                    'input_shape': (7, 7, 320),
                    'output_shape': (1, 1, 320)
                },
                {
                    'type': 'dense',
                    'in_features': 320,
                    'out_features': 1000
                }
            ]
        }
    
    def create_inverted_residual_blocks(self):
        """Create inverted residual blocks for supernet"""
        blocks = []
        in_channels = 32
        
        for stage in range(5):  # 5 stages
            out_channels = min(320, 32 * (2 ** stage))
            stride = 2 if stage > 0 else 1
            
            for depth in range(max(self.search_space['depths'])):
                block = {
                    'type': 'inverted_residual',
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'stride': stride if depth == 0 else 1,
                    'expansion': max(self.search_space['expansions']),
                    'kernel_size': max(self.search_space['kernel_sizes'])
                }
                blocks.append(block)
                in_channels = out_channels
        
        return blocks
```