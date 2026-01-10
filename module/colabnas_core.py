from tensorflow import keras
from pathlib import Path
import tensorflow as tf
import numpy as np
import subprocess
import datetime
import shutil
import glob
import re
import os

class ColabNAS:
    architecture_name = 'resulting_architecture'
    
    def __init__(self, max_RAM, max_Flash, max_MACC, path_to_training_set, val_split, 
                 cache=False, input_shape=(50,50,3), save_path='.', path_to_stm32tflm='stm32tflm.exe'):
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.epochs = 100  # minimum 2

        self.max_MACC = max_MACC
        self.max_Flash = max_Flash
        self.max_RAM = max_RAM
        self.path_to_training_set = path_to_training_set
        self.num_classes = len(next(os.walk(path_to_training_set))[1])
        self.val_split = val_split
        self.cache = cache
        self.input_shape = input_shape
        self.save_path = Path(save_path)

        self.path_to_trained_models = self.save_path / "trained_models"
        self.path_to_trained_models.mkdir(parents=True, exist_ok=True)

        self.path_to_stm32tflm = Path(path_to_stm32tflm)

        self.load_training_set()

    def load_training_set(self):
        color_mode = 'rgb' if self.input_shape[2] == 3 else 'grayscale'

        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.path_to_training_set,
            labels='inferred',
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=self.batch_size,
            image_size=self.input_shape[0:2],
            shuffle=True,
            seed=11,
            validation_split=self.val_split,
            subset='training'
        )

        validation_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.path_to_training_set,
            labels='inferred',
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=self.batch_size,
            image_size=self.input_shape[0:2],
            shuffle=True,
            seed=11,
            validation_split=self.val_split,
            subset='validation'
        )

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2, fill_mode='constant', interpolation='bilinear'),
        ])

        if self.cache:
            self.train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                       num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            self.validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            self.train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                       num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
            self.validation_ds = validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    def get_data(self):
        return self.train_ds, self.validation_ds

    def quantize_model_uint8(self, model_name):
        def representative_dataset():
            count = 0
            for images, labels in self.train_ds:
                for i in range(images.shape[0]):
                    if count >= 150:
                        return
                    yield [tf.dtypes.cast(images[i:i+1], tf.float32)]
                    count += 1

        model = tf.keras.models.load_model(self.path_to_trained_models / f"{model_name}.h5")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_types = [tf.int8]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_quant_model = converter.convert()

        with open(self.path_to_trained_models / f"{model_name}.tflite", 'wb') as f:
            f.write(tflite_quant_model)

        (self.path_to_trained_models / f"{model_name}.h5").unlink()

    def evaluate_flash_and_peak_RAM_occupancy(self, model_name):
        self.quantize_model_uint8(model_name)

        proc = subprocess.Popen([self.path_to_stm32tflm, self.path_to_trained_models / f"{model_name}.tflite"], 
                               stdout=subprocess.PIPE)
        try:
            outs, errs = proc.communicate(timeout=15)
            Flash, RAM = re.findall(r'\d+', str(outs))
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
            print("stm32tflm error")
            exit()

        return int(Flash), int(RAM)

    def evaluate_model(self, model, MACC, number_of_cells_limited, model_name):
        print(f"\n{model_name}\n")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            str(self.path_to_trained_models / f"{model_name}.h5"), monitor='val_accuracy',
            verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
        
        model.fit(self.train_ds, epochs=1, validation_data=self.validation_ds, validation_freq=1)
        model.save(self.path_to_trained_models / f"{model_name}.h5")
        Flash, RAM = self.evaluate_flash_and_peak_RAM_occupancy(model_name)
        print(f"\nRAM: {RAM},\t Flash: {Flash},\t MACC: {MACC}\n")
        
        if MACC <= self.max_MACC and Flash <= self.max_Flash and RAM <= self.max_RAM and not number_of_cells_limited:
            hist = model.fit(self.train_ds, epochs=self.epochs - 1, validation_data=self.validation_ds, 
                           validation_freq=1, callbacks=[checkpoint])
            self.quantize_model_uint8(model_name)
            return {'RAM': RAM,
                    'Flash': Flash,
                    'MACC': MACC,
                    'max_val_acc': np.around(np.amax(hist.history['val_accuracy']), decimals=3)}
        else:
            return {'max_val_acc': 0}

    def search(self, NAS):
        nas = NAS(
            evaluate_model_fnc=self.evaluate_model, 
            input_shape=self.input_shape, 
            num_classes=self.num_classes, 
            learning_rate=self.learning_rate
        )
        resulting_architecture, take_time = nas.search()

        if resulting_architecture['max_val_acc'] > 0:
            resulting_architecture_name = f"k_{resulting_architecture['k']}_c_{resulting_architecture['c']}.tflite"
            self.path_to_resulting_architecture = self.save_path / f"resulting_architecture_{resulting_architecture_name}"
            (self.path_to_trained_models / f"{resulting_architecture_name}").rename(self.path_to_resulting_architecture)
            shutil.rmtree(self.path_to_trained_models)
            print(f"\nResulting architecture: {resulting_architecture}\n")
        else:
            print(f"\nNo feasible architecture found\n")
        print(f"Elapsed time (search): {take_time}\n")

        return self.path_to_resulting_architecture