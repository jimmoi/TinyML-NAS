from tensorflow import keras
from pathlib import Path
import tensorflow as tf
from experiment_manager import Manager, all_experiment_dir
from Optimizer import *
from Optimizer.MCUNetV2 import MobileNetV2_RD_FixedV2
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
    
    # dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    # data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    # data_dir = Path(data_dir).with_suffix('')
    
    datasets_dir = Path("Datasets")
    for data_dir in [datasets_dir / "Flowers-4"] : #for data_dir in datasets_dir.iterdir() :
        if data_dir.is_dir():
            experiment_dir = all_experiment_dir / data_dir.name
            experiment_dir.mkdir(parents=False, exist_ok=True)
            
            train_path_dir = data_dir / "train"
            test_path_dir = data_dir / "test"

            input_shape = (50, 50, 3)
            
            manager = Manager(train_path_dir, experiment_dir=experiment_dir, experiment_name="Test_de1", input_shape=input_shape)
            nas = manager.setup_nas()

            # search_output = nas.search(PSO_NAS.setup(search_space, decoder))
            search_output = nas.search(Vanilla_NAS)
            
            test_ds = tf.keras.utils.image_dataset_from_directory(
                directory= test_path_dir,
                labels='inferred',
                label_mode='categorical',
                color_mode='rgb',
                batch_size=1,
                image_size=input_shape[:2],
                shuffle=True
            )

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
