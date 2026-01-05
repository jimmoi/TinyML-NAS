import tensorflow as tf
import numpy as np
import datetime
import shutil
import os
from pathlib import Path
import subprocess
import re

class LocalColabNAS:
    """
    Local version of ColabNAS for Neural Architecture Search on TinyML devices
    Adapted to run on local machines without Google Colab dependencies
    """
    
    def __init__(self, max_RAM, max_Flash, max_MACC, path_to_training_set, 
                 val_split=0.2, cache=False, input_shape=(50,50,3), 
                 save_path='.', stm32tflm_path=None):
        
        self.learning_rate = 1e-3
        self.batch_size = 32  # Reduced for local machines
        self.epochs = 50      # Reduced for faster local execution
        
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
        
        self.stm32tflm_path = stm32tflm_path
        
        print(f"Initializing LocalColabNAS with {self.num_classes} classes")
        self.load_training_set()
    
    def Model(self, k, c):
        """Create CNN model with k kernels and c cells"""
        kernel_size = (3,3)
        pool_size = (2,2)
        pool_strides = (2,2)
        
        number_of_cells_limited = False
        number_of_mac = 0
        
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Convolutional base
        n = int(k)
        multiplier = 2
        
        # First convolutional layer
        c_in = self.input_shape[2]
        x = tf.keras.layers.Conv2D(n, kernel_size, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        number_of_mac += (c_in * kernel_size[0] * kernel_size[1] * 
                         x.shape[1] * x.shape[2] * x.shape[3])
        
        # Adding cells
        for i in range(1, c + 1):
            if x.shape[1] <= 1 or x.shape[2] <= 1:
                number_of_cells_limited = True
                break
            
            n = int(np.ceil(n * multiplier))
            multiplier = multiplier - 2**-i
            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, 
                                           strides=pool_strides, 
                                           padding='valid')(x)
            c_in = x.shape[3]
            x = tf.keras.layers.Conv2D(n, kernel_size, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            number_of_mac += (c_in * kernel_size[0] * kernel_size[1] * 
                             x.shape[1] * x.shape[2] * x.shape[3])
        
        # Classifier
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        input_shape = x.shape[1]
        x = tf.keras.layers.Dense(n)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        number_of_mac += (input_shape * x.shape[1])
        x = tf.keras.layers.Dense(self.num_classes)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Softmax()(x)
        number_of_mac += (x.shape[1] * outputs.shape[1])
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model, number_of_mac, number_of_cells_limited
    
    def load_training_set(self):
        """Load and preprocess training dataset"""
        color_mode = 'rgb' if self.input_shape[2] == 3 else 'grayscale'
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.path_to_training_set,
            labels='inferred',
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=self.batch_size,
            image_size=self.input_shape[0:2],
            shuffle=True,
            seed=42,
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
            seed=42,
            validation_split=self.val_split,
            subset='validation'
        )
        
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        
        # Apply preprocessing
        if self.cache:
            self.train_ds = train_ds.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            self.validation_ds = validation_ds.cache().prefetch(
                buffer_size=tf.data.AUTOTUNE)
        else:
            self.train_ds = train_ds.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(buffer_size=tf.data.AUTOTUNE)
            self.validation_ds = validation_ds.prefetch(
                buffer_size=tf.data.AUTOTUNE)
    
    def quantize_model_uint8(self):
        """Quantize model to uint8 for deployment"""
        def representative_dataset():
            for data in self.train_ds.rebatch(1).take(100):
                yield [tf.dtypes.cast(data[0], tf.float32)]
        
        model = tf.keras.models.load_model(
            self.path_to_trained_models / f"{self.model_name}.h5")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_quant_model = converter.convert()
        
        with open(self.path_to_trained_models / f"{self.model_name}.tflite", 'wb') as f:
            f.write(tflite_quant_model)
        
        # Clean up h5 file
        (self.path_to_trained_models / f"{self.model_name}.h5").unlink()
    
    def estimate_memory_usage(self, model):
        """Estimate Flash and RAM usage without STM32 tools"""
        # Get model size (Flash estimation)
        model.save(self.path_to_trained_models / f"{self.model_name}.h5")
        flash_size = os.path.getsize(self.path_to_trained_models / f"{self.model_name}.h5")
        
        # Estimate RAM (simplified calculation)
        total_params = model.count_params()
        # Rough estimation: 4 bytes per parameter + activation memory
        ram_estimate = total_params * 4 + np.prod(self.input_shape) * 4
        
        return flash_size, int(ram_estimate)
    
    def evaluate_model_process(self, k, c):
        """Evaluate a single model configuration"""
        if k <= 0:
            return {'k': 'unfeasible', 'c': c, 'max_val_acc': -3}
        
        self.model_name = f"k_{k}_c_{c}"
        print(f"\n{self.model_name}\n")
        
        try:
            model, MACC, number_of_cells_limited = self.Model(k, c)
            
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10, restore_best_weights=True)
            
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                str(self.path_to_trained_models / f"{self.model_name}.h5"),
                monitor='val_accuracy', save_best_only=True, verbose=1)
            
            # Train for one epoch to get initial weights
            model.fit(self.train_ds, epochs=1, validation_data=self.validation_ds)
            
            # Estimate memory usage
            Flash, RAM = self.estimate_memory_usage(model)
            print(f"\nRAM: {RAM}, Flash: {Flash}, MACC: {MACC}\n")
            
            # Check constraints
            if (MACC <= self.max_MACC and Flash <= self.max_Flash and 
                RAM <= self.max_RAM and not number_of_cells_limited):
                
                # Continue training
                hist = model.fit(
                    self.train_ds, 
                    epochs=self.epochs - 1,
                    validation_data=self.validation_ds,
                    callbacks=[checkpoint, early_stopping],
                    verbose=1
                )
                
                self.quantize_model_uint8()
                max_val_acc = np.max(hist.history['val_accuracy'])
            else:
                max_val_acc = -3
            
            return {
                'k': k,
                'c': c if not number_of_cells_limited else "Not feasible",
                'RAM': RAM if RAM <= self.max_RAM else "Outside the upper bound",
                'Flash': Flash if Flash <= self.max_Flash else "Outside the upper bound",
                'MACC': MACC if MACC <= self.max_MACC else "Outside the upper bound",
                'max_val_acc': np.around(max_val_acc, decimals=3) if max_val_acc > 0 else -3
            }
            
        except Exception as e:
            print(f"Error evaluating model k={k}, c={c}: {e}")
            return {'k': k, 'c': c, 'max_val_acc': -3}
    
    def explore_num_cells(self, k):
        """Explore number of cells for given k"""
        previous_architecture = {'k': -1, 'c': -1, 'max_val_acc': -2}
        current_architecture = {'k': -1, 'c': -1, 'max_val_acc': -1}
        c = -1
        k = int(k)
        
        while current_architecture['max_val_acc'] > previous_architecture['max_val_acc']:
            previous_architecture = current_architecture
            c = c + 1
            self.model_counter += 1
            current_architecture = self.evaluate_model_process(k, c)
            print(f"\n\n\n{current_architecture}\n\n\n")
            
        return previous_architecture
    
    def search(self):
        """Main NAS search algorithm"""
        self.model_counter = 0
        epsilon = 0.005
        k0 = 4
        
        start = datetime.datetime.now()
        print(f"Starting NAS search at {start}")
        
        k = k0
        previous_architecture = self.explore_num_cells(k)
        k = 2 * k
        current_architecture = self.explore_num_cells(k)
        
        if current_architecture['max_val_acc'] > previous_architecture['max_val_acc']:
            previous_architecture = current_architecture
            k = 2 * k
            current_architecture = self.explore_num_cells(k)
            while (current_architecture['max_val_acc'] > 
                   previous_architecture['max_val_acc'] + epsilon):
                previous_architecture = current_architecture
                k = 2 * k
                current_architecture = self.explore_num_cells(k)
        else:
            k = k0 / 2
            current_architecture = self.explore_num_cells(k)
            while (current_architecture['max_val_acc'] >= 
                   previous_architecture['max_val_acc']):
                previous_architecture = current_architecture
                k = k / 2
                current_architecture = self.explore_num_cells(k)
        
        resulting_architecture = previous_architecture
        end = datetime.datetime.now()
        
        if resulting_architecture['max_val_acc'] > 0:
            resulting_architecture_name = (f"k_{resulting_architecture['k']}_"
                                         f"c_{resulting_architecture['c']}.tflite")
            self.path_to_resulting_architecture = (self.save_path / 
                                                 f"resulting_architecture_{resulting_architecture_name}")
            
            source_path = self.path_to_trained_models / resulting_architecture_name
            if source_path.exists():
                source_path.rename(self.path_to_resulting_architecture)
            
            print(f"\nResulting architecture: {resulting_architecture}\n")
            print(f"Model saved to: {self.path_to_resulting_architecture}")
        else:
            print("\nNo feasible architecture found\n")
            self.path_to_resulting_architecture = None
        
        print(f"Elapsed time (search): {end-start}\n")
        print(f"Total models evaluated: {self.model_counter}")
        
        # Cleanup
        if self.path_to_trained_models.exists():
            shutil.rmtree(self.path_to_trained_models)
        
        return self.path_to_resulting_architecture

def test_tflite_model(model_path, test_ds):
    """Test a TFLite model"""
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    
    output = interpreter.get_output_details()[0]
    input_details = interpreter.get_input_details()[0]
    
    correct = 0
    total = 0
    
    for batch in test_ds:
        images, labels = batch[0], batch[1]
        for i in range(images.shape[0]):
            image = images[i:i+1]
            label = labels[i]
            
            # Handle quantization
            if input_details['dtype'] == tf.uint8:
                input_scale, input_zero_point = input_details["quantization"]
                image = image / input_scale + input_zero_point
            
            input_data = tf.cast(image, input_details['dtype'])
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()
            
            prediction = interpreter.get_tensor(output['index'])
            
            if label.numpy().argmax() == prediction.argmax():
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nTFLite model test accuracy: {accuracy:.4f}")
    return accuracy