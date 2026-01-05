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