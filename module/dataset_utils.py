import tensorflow as tf
from pathlib import Path

def download_flower_dataset():
    """Download and prepare the flower dataset"""
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    data_dir = Path(data_dir).with_suffix('')
    
    # Fix dataset structure - point to actual flower_photos folder
    actual_data_dir = data_dir / 'flower_photos'
    if actual_data_dir.exists():
        data_dir = actual_data_dir
    
    return data_dir

def check_gpu():
    """Check GPU availability and display information"""
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))
    
    if gpu_devices:
        print("Found GPU(s):", gpu_devices)
        # Set memory growth to avoid allocating all GPU memory at once
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU devices found. TensorFlow is using the CPU.")
    
    return len(gpu_devices) > 0