from tensorflow import keras
from pathlib import Path
import tensorflow as tf
from experiment_manager import Manager
from Optimizer import *
from my_util import load_train_dataset, test_tflite_model
import sys
import json
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re

def load_data(data_dir, input_shape=(50, 50, 3), batch_size=128, val_split=0.3, cache=True):
    train_path_dir = data_dir / "train"
    test_path_dir = data_dir / "test"

    num_classes = len(next(os.walk(train_path_dir))[1])
    train_ds, validation_ds = load_train_dataset(input_shape, train_path_dir, val_split=val_split, batch_size=batch_size, cache=cache)
    test_ds = tf.keras.utils.image_dataset_from_directory(
            directory= test_path_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=1,
            image_size=input_shape[:2],
            shuffle=True
        )
    return train_ds, validation_ds, test_ds, input_shape, num_classes

def write_model_log(model_log, file_path, model_names):
    
    text = ""
    
    # make column name
    columns = ["decision_variable",
               f"{model_names[0]}_best_acc", 
               f"{model_names[1]}_best_acc", 
               f"{model_names[0]}_tflite_acc", 
               f"{model_names[1]}_tflite_acc", 
               f"{model_names[0]}_macs", 
               f"{model_names[1]}_macs", 
               f"{model_names[0]}_flash", 
               f"{model_names[1]}_flash", 
               f"{model_names[0]}_peak_ram", 
               f"{model_names[1]}_peak_ram"]
    column_name = ",".join(columns)
    text += column_name + "\n"
    
    # make row
    for decision_variable, models in model_log.items():
        #each decision variable
        row_text = f"{decision_variable},"
        
        row_text += f"{models[model_names[0]]['best_acc']}," if models[model_names[0]]['best_acc'] is not None else "Null"
        row_text += f"{models[model_names[1]]['best_acc']}," if models[model_names[1]]['best_acc'] is not None else "Null"
        row_text += f"{models[model_names[0]]['tflite_acc']}," if models[model_names[0]]['tflite_acc'] is not None else "Null"
        row_text += f"{models[model_names[1]]['tflite_acc']}," if models[model_names[1]]['tflite_acc'] is not None else "Null"
        row_text += f"{models[model_names[0]]['mac_count']}," if models[model_names[0]]['mac_count'] is not None else "Null"
        row_text += f"{models[model_names[1]]['mac_count']}," if models[model_names[1]]['mac_count'] is not None else "Null"
        row_text += f"{models[model_names[0]]['flash']}," if models[model_names[0]]['flash'] is not None else "Null"
        row_text += f"{models[model_names[1]]['flash']}," if models[model_names[1]]['flash'] is not None else "Null"
        row_text += f"{models[model_names[0]]['peak_ram']}," if models[model_names[0]]['peak_ram'] is not None else "Null"
        row_text += f"{models[model_names[1]]['peak_ram']}," if models[model_names[1]]['peak_ram'] is not None else "Null"
        
        text += row_text + "\n"

    with open(file_path, 'w') as f:
        f.write(text)

def quantize_model(train_ds, model_file, tflite_model_file):
    def representative_dataset():
            count = 0
            for images, labels in train_ds:
                for i in range(images.shape[0]):
                    if count >= 150:
                        return
                # Ensure the data matches the model's expected input shape and type
                yield [tf.dtypes.cast(images[i:i+1], tf.float32)]
                count += 1

    model = tf.keras.models.load_model(model_file)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 1. Standard optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # 2. ENFORCE Integer-only (Crucial for STM32)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # 3. Explicitly set input/output to UINT8 for the hardware interface
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # 4. Mandatory for some TFLite versions to ensure full quantization
    # This prevents the "fully_quantize: 0" status you saw earlier
    converter._experimental_new_quantizer = True 

    tflite_quant_model = converter.convert()

    with open(tflite_model_file, 'wb') as f:
        f.write(tflite_quant_model)

def evaluate_flash_and_peak_RAM_occupancy(stm32_path, tflite_model_path) :
    #evaluate its peak RAM occupancy and its Flash occupancy using STMicroelectronics' X-CUBE-AI
    proc = subprocess.Popen([stm32_path, tflite_model_path], stdout=subprocess.PIPE)
    try:
        outs, errs = proc.communicate(timeout=15)
        Flash, RAM = re.findall(r'\d+', str(outs))
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        print("stm32tflm error")
        exit()

    return int(Flash), int(RAM)

def plot(log, model_names, file_path):
    def plot_loss(ax, hist, model_name):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.plot(hist['train_losses'], label="train_losses")
        ax.plot(hist['val_losses'], label="val_losses")
        ax.set_title(model_name)
        ax.legend()
        
    def plot_acc(ax, hist, model_name):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.plot(hist['train_accs'], label="train_accs")
        ax.plot(hist['val_accs'], label="val_accs")
        ax.set_title(model_name)
        ax.legend()
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    plot_loss(ax[0,0], log[model_names[0]], model_names[0])
    plot_loss(ax[0,1], log[model_names[1]], model_names[1])
    plot_acc(ax[1,0], log[model_names[0]], model_names[0])
    plot_acc(ax[1,1], log[model_names[1]], model_names[1])
    
    plt.savefig(file_path)
    plt.close()

def main():
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))

    if gpu_devices:
        print("Found GPU(s):", gpu_devices)
    else:
        print("No GPU devices found. TensorFlow is likely using the CPU.")
        
        
    all_experiment_dir = Path('Experiments_Compare_Model')
    all_experiment_dir.mkdir(exist_ok=True)
        
    ## Experiment Settings
    epochs = 100
    learning_rate = 1e-3
    k_list = [4, 8]
    c_list = [0, 1, 2, 3, 4]
    model_list = {'vanillaNAS_dense' : Vanilla_NAS, 
                  'JimmyNAS_I_fullyCNN' : Jimmy_NAS_I}
    
    stm32_path = Path("stm32tflm.exe")
    
    datasets_dir = Path("Datasets")
    for data_dir in datasets_dir.iterdir() :
        if data_dir.is_dir() and data_dir.name in ["Animals-3"]:
            experiment_dir = all_experiment_dir / data_dir.name
            experiment_dir.mkdir(parents=False, exist_ok=True)
            
            # model log
            model_log_file = experiment_dir / "model_log.csv"
            model_log = {}
            
            # load dataset
            train_ds, validation_ds, test_ds, input_shape, num_classes = load_data(data_dir, input_shape=(50, 50, 3), batch_size=128, val_split=0.3, cache=True)
            
            for k in k_list:
                for c in c_list:
                    decision_variable_dir = experiment_dir / Path(f"k_{k}_c_{c}")
                    decision_variable_dir.mkdir(parents=False, exist_ok=True)
                    
                    model_log[f"k_{k}_c_{c}"] = {}
                    temp_log = model_log[f"k_{k}_c_{c}"]
                    
                    for model_name, ModelClass in model_list.items():
                            
                        temp_log[model_name] = {
                            "train_loss":None,
                            "val_loss":None,
                            "best_acc":None,
                            "tflite_acc":None,
                        }
                        
                        # full model
                        model_file = decision_variable_dir / (model_name + ".h5")
                        model, mac_count, cell_limit_status = ModelClass.create_model_static(k, c, input_shape, num_classes, learning_rate)
                        
                        if cell_limit_status:
                            temp_log[model_name]["best_acc"] = None
                            temp_log[model_name]["train_losses"] = None
                            temp_log[model_name]["val_losses"] = None
                            temp_log[model_name]["tflite_acc"] = None
                            continue
                        
                        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                            str(model_file), monitor='val_accuracy',
                            verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
                        
                        hist = model.fit(train_ds, epochs=epochs - 1, validation_data=validation_ds, validation_freq=1, callbacks=[checkpoint])
                        temp_log[model_name]["best_acc"] = np.around(np.amax(hist.history['val_accuracy']), decimals=3)
                        temp_log[model_name]["train_losses"] = np.around(hist.history['loss'], 6)
                        temp_log[model_name]["val_losses"] = np.around(hist.history['val_loss'], 6)
                        temp_log[model_name]["train_accs"] = np.around(hist.history['accuracy'], 6)
                        temp_log[model_name]["val_accs"] = np.around(hist.history['val_accuracy'], 6)
                        temp_log[model_name]["mac_count"] = mac_count
                        
                        
                        # tflite model
                        tflite_model_file = decision_variable_dir / (model_name + ".tflite")
                        quantize_model(train_ds, model_file, tflite_model_file)
                        tflite_accuracy = test_tflite_model(str(tflite_model_file), test_ds)
                        temp_log[model_name]["tflite_acc"] = np.around(tflite_accuracy, decimals=3)
                        
                        flash, peak_ram = evaluate_flash_and_peak_RAM_occupancy(stm32_path, str(tflite_model_file))
                        temp_log[model_name]["flash"] = flash
                        temp_log[model_name]["peak_ram"] = peak_ram
                        
                    # plot loss
                    plot(temp_log, list(model_list.keys()), decision_variable_dir / f"loss.png")
                            
            write_model_log(model_log, model_log_file, list(model_list.keys()))    
                        
            
            

if __name__ == "__main__":
    main()