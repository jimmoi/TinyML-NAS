from tensorflow import keras
from pathlib import Path
import tensorflow as tf
from experiment_manager import Manager
from Optimizer import *
from my_util import load_dataset, test_tflite_model
import sys
import json
import pickle
import matplotlib.pyplot as plt
    
def main():
    
    import os
    import tensorflow as tf
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))

    if gpu_devices:
        print("Found GPU(s):", gpu_devices)
    else:
        print("No GPU devices found. TensorFlow is likely using the CPU.")
    
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    data_dir = Path(data_dir).with_suffix('')
    
    manager = Manager(path_to_training_set=data_dir, experiment_name="vanillaNAS_fullyCNN")
    nas = manager.setup_nas()

    # search_output = nas.search(PSO_NAS.setup(search_space, decoder))
    search_output = nas.search(VanillaCNN_NAS)
    
    _, test_ds = nas.get_data()
    tflite_accuracy = test_tflite_model(search_output['path_to_best_architecture'], test_ds)
    search_output["tflite_accuracy"] = round(tflite_accuracy, 4)
    
    manager.visualize(search_output)
    
    try:
        with open(manager.experiment_dir / "search_output.pkl", "wb") as f:
            pickle.dump(search_output, f)
    except Exception as e:
        print(f"Error: {e}")
    

if __name__ == "__main__":
    main()
