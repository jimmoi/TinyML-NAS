from tensorflow import keras
from pathlib import Path
import tensorflow as tf
from experiment_manager import Manager, all_experiment_dir
from Optimizer import *
from my_util import load_train_dataset, prepare_nas_datasets
import sys
import json
import pickle
import matplotlib.pyplot as plt
import os
    
def main():
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))

    if gpu_devices:
        print("Found GPU(s):", gpu_devices)
    else:
        print("No GPU devices found. TensorFlow is likely using the CPU.")
    
    datasets_dir = Path("Datasets")
    for data_dir in datasets_dir.iterdir() :
        if data_dir.is_dir():
            experiment_dir = all_experiment_dir / data_dir.name
            experiment_dir.mkdir(parents=False, exist_ok=True)
            
            train_path_dir = data_dir / "train"
            test_path_dir = data_dir / "test"

            input_shape = (50, 50, 3)
            
            ## load data
            num_classes = len(next(os.walk(train_path_dir))[1])
            train_ds, validation_ds = load_train_dataset(input_shape, train_path_dir, val_split=0.3, batch_size=128, cache=True)
            test_ds = tf.keras.utils.image_dataset_from_directory(
                    directory= test_path_dir,
                    labels='inferred',
                    label_mode='categorical',
                    color_mode='rgb',
                    batch_size=1,
                    image_size=input_shape[:2],
                    shuffle=True
                )
            
            try:
                data = {
                    "train": train_ds,
                    "validation": validation_ds,
                    "test": test_ds,
                    "transform": prepare_nas_datasets
                }
                manager = Manager(data, num_classes, experiment_dir=experiment_dir, experiment_name="WTF", input_shape=input_shape)
                nas = manager.setup_nas()

                search_output = nas.search(Tiger_NAS)
          
                manager.visualize(search_output)
                
                try:
                    with open(manager.experiment_dir / "search_output.pkl", "wb") as f:
                        pickle.dump(search_output, f)
                except Exception as e:
                    print(f"Error: {e}")
                    
            except FileExistsError as e:
                print(e)
                continue
    

if __name__ == "__main__":
    main()
