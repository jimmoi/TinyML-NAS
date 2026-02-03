from pathlib import Path
import tensorflow as tf
import numpy as np
import subprocess
import shutil
import re
import os
import time
import io
from my_util import test_tflite_model

class ColabNAS :
    architecture_name = 'resulting_architecture'
    def __init__(
        self, 
        max_RAM, 
        max_Flash, 
        max_MACC, 
        data,
        num_classes,
        input_shape,
        save_path='.', 
        path_to_stm32tflm='stm32tflm.exe'
        ) :
        
        self.learning_rate = 1e-3
        self.epochs = 100 #minimum 2

        self.max_MACC = max_MACC
        self.max_Flash = max_Flash
        self.max_RAM = max_RAM
        self.train_ds = data['train']
        self.validation_ds = data['validation']
        self.test_ds = data['test']
        self.transform = data.get('transform')
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.save_path = Path(save_path)

        self.path_to_trained_models = self.save_path / "trained_models"
        self.path_to_trained_models.mkdir(parents=True, exist_ok=True)

        self.path_to_stm32tflm = Path(path_to_stm32tflm)

    def quantize_model_uint8(self, model_name):
        def representative_dataset():
            count = 0
            for images, labels in self.train_ds:
                for i in range(images.shape[0]):
                    if count >= 150:
                        return
                # Ensure the data matches the model's expected input shape and type
                yield [tf.dtypes.cast(images[i:i+1], tf.float32)]
                count += 1

        model = tf.keras.models.load_model(self.path_to_trained_models / f"{model_name}.h5")
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

        with open(self.path_to_trained_models / f"{model_name}.tflite", 'wb') as f:
            f.write(tflite_quant_model)

        (self.path_to_trained_models / f"{model_name}.h5").unlink()

    def evaluate_flash_and_peak_RAM_occupancy(self, model_name) :
        #quantize model to evaluate its peak RAM occupancy and its Flash occupancy
        self.quantize_model_uint8(model_name)

        #evaluate its peak RAM occupancy and its Flash occupancy using STMicroelectronics' X-CUBE-AI
        proc = subprocess.Popen([self.path_to_stm32tflm, self.path_to_trained_models / f"{model_name}.tflite"], stdout=subprocess.PIPE)
        try:
            outs, errs = proc.communicate(timeout=15)
            Flash, RAM = re.findall(r'\d+', str(outs))
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
            print("stm32tflm error")
            exit()

        return int(Flash), int(RAM)

    def evaluate_model(self, model, MACC, number_of_cells_limited, model_name) :
        # Re-map the labels to match the grid
        if self.transform:
            # Get the output shape from your generated NAS model
            output_shape = model.output_shape # e.g., (None, 7, 7, 10)
            h, w = output_shape[1], output_shape[2]
            train_ds_mapped, val_ds_mapped = self.transform(self.train_ds, self.validation_ds, patch_size=(h, w))
        else:
            train_ds_mapped = self.train_ds
            val_ds_mapped = self.validation_ds
        
        print(f"\n{model_name}\n")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            str(self.path_to_trained_models / f"{model_name}.h5"), monitor='val_accuracy',
            verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
        #One epoch of training must be done before quantization, which is needed to evaluate RAM and Flash occupancy
        model.fit(train_ds_mapped, 
                  epochs=1, 
                  validation_data=val_ds_mapped, 
                  validation_freq=1)
        model.save(self.path_to_trained_models / f"{model_name}.h5")
        Flash, RAM = self.evaluate_flash_and_peak_RAM_occupancy(model_name)
        print(f"\nRAM: {RAM},\t Flash: {Flash},\t MACC: {MACC}\n")
        if MACC <= self.max_MACC and Flash <= self.max_Flash and RAM <= self.max_RAM and not number_of_cells_limited :
            hist = model.fit(train_ds_mapped, epochs=self.epochs - 1, validation_data=val_ds_mapped, validation_freq=1, callbacks=[checkpoint])
            self.quantize_model_uint8(model_name)
            
            stringio = io.StringIO()
            model.summary(print_fn=lambda x: stringio.write(x + '\n'))
            model_summary = stringio.getvalue()
            return {'RAM': RAM,
                    'Flash': Flash,
                    'MACC': MACC,
                    'model_architecture':model_summary,
                    'max_val_acc':
                    np.around(np.amax(hist.history['val_accuracy']), decimals=3)
                    if 'hist' in locals() else -3,
                    'final_train_loss': np.around(hist.history['loss'], 6),
                    'final_val_loss': np.around(hist.history['val_loss'], 6),
                    'output_shape': (h, w)
                    }
        else :
            return {'max_val_acc':0}

    def search(self, NAS):
        
        search_output = {
            "time":None,
            "iterations_accuracy":None,
            "RAM":None,
            "Flash":None,
            "MACC":None,
            "val_accuracy":None,
            "decision_variables":None,
            "model_architecture":None,
            "path_to_best_architecture":None,
            "val_losses":None,
            "train_losses":None,
            "output_shape":None
            
        }
        
        nas = NAS(
            evaluate_model_fnc = self.evaluate_model, 
            input_shape = self.input_shape, 
            num_classes = self.num_classes, 
            learning_rate = self.learning_rate
        )
        resulting_architecture_dict, take_time, iterations_accuracy = nas.search()

        if (resulting_architecture_dict['max_val_acc'] > 0) :
            resulting_architecture_name = f"k_{resulting_architecture_dict['k']}_c_{resulting_architecture_dict['c']}.tflite"
            path_to_resulting_architecture = self.save_path / f"resulting_architecture_{resulting_architecture_name}"
            (self.path_to_trained_models / f"{resulting_architecture_name}").rename(path_to_resulting_architecture)
            shutil.rmtree(self.path_to_trained_models)
            
            if self.transform:
                self.test_ds = self.transform(self.test_ds, patch_size=resulting_architecture_dict['output_shape'])
            tflite_accuracy = test_tflite_model(path_to_resulting_architecture, self.test_ds)
            
            print(f"\nResulting architecture: {resulting_architecture_dict}\n")
            search_output["time"] = str(take_time)
            search_output["iterations_accuracy"] = iterations_accuracy
            search_output["RAM"] = resulting_architecture_dict['RAM']
            search_output["Flash"] = resulting_architecture_dict['Flash']
            search_output["MACC"] = resulting_architecture_dict['MACC']
            search_output["val_accuracy"] = resulting_architecture_dict['max_val_acc']
            search_output["decision_variables"] = {'k':resulting_architecture_dict['k'], 'c':resulting_architecture_dict['c']}
            search_output["model_architecture"] = resulting_architecture_dict['model_architecture']
            search_output["path_to_best_architecture"] = str(path_to_resulting_architecture)
            search_output["val_losses"] = resulting_architecture_dict['final_val_loss']
            search_output["train_losses"] = resulting_architecture_dict['final_train_loss']
            search_output["tflite_accuracy"] = round(tflite_accuracy, 4)
            
            return search_output
        else :
            print(f"\nNo feasible architecture found\n")
        print(f"Elapsed time (search): {take_time}\n")
        
        return None