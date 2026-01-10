#!/usr/bin/env python3
"""
Complete PSO-based Neural Architecture Search with consistent random seed
"""

import os
import sys
import numpy as np
import tensorflow as tf
import datetime
import random
from pathlib import Path
from tensorflow import keras

# Set consistent random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class ColabNAS:
    def __init__(self, max_RAM, max_Flash, max_MACC, path_to_training_set, val_split, 
                 cache=False, input_shape=(50,50,3), save_path='.', path_to_stm32tflm='stm32tflm.exe'):
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.epochs = 100
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
            directory=self.path_to_training_set, labels='inferred', label_mode='categorical',
            color_mode=color_mode, batch_size=self.batch_size, image_size=self.input_shape[0:2],
            shuffle=True, seed=RANDOM_SEED, validation_split=self.val_split, subset='training'
        )
        
        validation_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.path_to_training_set, labels='inferred', label_mode='categorical',
            color_mode=color_mode, batch_size=self.batch_size, image_size=self.input_shape[0:2],
            shuffle=True, seed=RANDOM_SEED, validation_split=self.val_split, subset='validation'
        )
        
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
            tf.keras.layers.RandomRotation(0.2, fill_mode='constant', interpolation='bilinear', seed=RANDOM_SEED),
        ])
        
        if self.cache:
            self.train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                       num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            self.validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            self.train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                       num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
            self.validation_ds = validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

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
        # Mock implementation for systems without stm32tflm
        model_path = self.path_to_trained_models / f"{model_name}.tflite"
        model_size = model_path.stat().st_size
        Flash = model_size
        RAM = int(model_size * 0.5)  # Estimate RAM as 50% of Flash
        return Flash, RAM

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
            return {'RAM': RAM, 'Flash': Flash, 'MACC': MACC,
                    'max_val_acc': np.around(np.amax(hist.history['val_accuracy']), decimals=3)}
        else:
            return {'max_val_acc': 0}

    def search(self, NAS):
        nas = NAS(evaluate_model_fnc=self.evaluate_model, input_shape=self.input_shape, 
                  num_classes=self.num_classes, learning_rate=self.learning_rate)
        resulting_architecture, take_time = nas.search()

        if resulting_architecture['max_val_acc'] > 0:
            resulting_architecture_name = f"k_{resulting_architecture['k']}_c_{resulting_architecture['c']}.tflite"
            self.path_to_resulting_architecture = self.save_path / f"resulting_architecture_{resulting_architecture_name}"
            (self.path_to_trained_models / f"{resulting_architecture_name}").rename(self.path_to_resulting_architecture)
            print(f"\nResulting architecture: {resulting_architecture}\n")
        else:
            print(f"\nNo feasible architecture found\n")
        print(f"Elapsed time (search): {take_time}\n")
        return getattr(self, 'path_to_resulting_architecture', None)

class ArchitectureSearchSpace:
    def __init__(self, k_range=(4, 128), c_range=(0, 10)):
        self.k_min, self.k_max = k_range
        self.c_min, self.c_max = c_range

    def clamp(self, k, c):
        return np.clip(k, self.k_min, self.k_max), np.clip(c, self.c_min, self.c_max)

class ModelDecoder:
    def __init__(self):
        self.input_shape = None
        self.num_classes = None
        self.learning_rate = None

    def decode_and_build(self, k, c):
        k, c = int(k), int(c)
        kernel_size = (3, 3)
        pool_size = (2, 2)
        number_of_mac = 0
        number_of_cells_limited = False
        
        inputs = keras.Input(shape=self.input_shape)
        n = k
        multiplier = 2
        
        c_in = self.input_shape[2]
        x = keras.layers.Conv2D(n, kernel_size, padding='same', kernel_initializer='glorot_uniform')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        number_of_mac += (c_in * np.prod(kernel_size) * x.shape[1] * x.shape[2] * x.shape[3])

        for i in range(1, c + 1):
            if x.shape[1] <= 1 or x.shape[2] <= 1:
                number_of_cells_limited = True
                break
            
            n = int(np.ceil(n * multiplier))
            multiplier -= 2**-i
            x = keras.layers.MaxPooling2D(pool_size=pool_size, strides=(2,2), padding='valid')(x)
            c_in = x.shape[3]
            x = keras.layers.Conv2D(n, kernel_size, padding='same', kernel_initializer='glorot_uniform')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            number_of_mac += (c_in * np.prod(kernel_size) * x.shape[1] * x.shape[2] * x.shape[3])

        x = keras.layers.GlobalAveragePooling2D()(x)
        feat_dim = x.shape[1]
        x = keras.layers.Dense(n, kernel_initializer='glorot_uniform')(x)
        number_of_mac += (feat_dim * n)
        x = keras.layers.Dense(self.num_classes, kernel_initializer='glorot_uniform')(x)
        outputs = keras.layers.Softmax()(x)
        number_of_mac += (n * self.num_classes)

        model = keras.Model(inputs=inputs, outputs=outputs)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model, number_of_mac, number_of_cells_limited

class NASPsoOptimizer:
    def __init__(self, evaluate_model_fnc, input_shape, num_classes, learning_rate):
        self.evaluate_model_fnc = evaluate_model_fnc
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_name = ""
        self.setup_decoder()
        
    def setup_decoder(self):
        self.decoder.input_shape = self.input_shape
        self.decoder.num_classes = self.num_classes
        self.decoder.learning_rate = self.learning_rate

    def search(self):
        particles = np.array([
            [np.random.uniform(self.space.k_min, self.space.k_max), 
             np.random.uniform(self.space.c_min, self.space.c_max)] 
            for _ in range(self.n_particles)
        ])
        velocities = np.zeros((self.n_particles, 2))
        p_best = np.copy(particles)
        p_best_scores = np.full(self.n_particles, -1.0)
        g_best = None
        g_best_score = -1.0
        w, c1, c2 = 0.5, 1.5, 1.5
        start = datetime.datetime.now()
        results_best = {}
        
        for it in range(self.iterations):
            print(f"==================== iteration {it} ====================")
            for i in range(self.n_particles):
                k, c = particles[i]
                model, macc, limited = self.decoder.decode_and_build(k, c)
                
                if limited:
                    score = 0
                else:
                    self.model_name = f"k_{int(k)}_c_{int(c)}"
                    results = self.evaluate_model_fnc(model, macc, limited, self.model_name)
                    score = results['max_val_acc']

                if score > p_best_scores[i]:
                    p_best_scores[i] = score
                    p_best[i] = particles[i]

                if score > g_best_score:
                    results_best = results
                    g_best_score = score
                    g_best = np.copy(particles[i])

            if g_best is None:
                print("Warning: No valid architecture found in this iteration. Re-randomizing velocities...")
                velocities = np.random.uniform(-1, 1, size=velocities.shape)
            else:
                for i in range(self.n_particles):
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[i] = (w * velocities[i] + 
                                    c1 * r1 * (p_best[i] - particles[i]) + 
                                    c2 * r2 * (g_best - particles[i]))
                    particles[i] += velocities[i]
                    particles[i][0], particles[i][1] = self.space.clamp(particles[i][0], particles[i][1])

                print(f"Iteration {it}: Global Best Score = {g_best_score:.4f} at k={int(g_best[0])}, c={int(g_best[1])}")

        results_best["k"] = int(g_best[0])
        results_best["c"] = int(g_best[1])
        end = datetime.datetime.now()
        return results_best, end-start
    
    @classmethod
    def setup(cls, search_space, decoder, n_particles=5, iterations=10):
        cls.n_particles = n_particles
        cls.iterations = iterations
        cls.space = search_space
        cls.decoder = decoder
        return cls

def download_flower_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    return Path(data_dir).with_suffix('')

def main():
    print("🧬 Complete PSO-based NAS with Consistent Random Seed")
    print("=" * 60)
    
    # Download dataset
    data_dir = download_flower_dataset()
    
    # Configuration
    config = {
        'max_RAM': 40960,
        'max_Flash': 131072,
        'max_MACC': 2730000,
        'path_to_training_set': str(data_dir),
        'val_split': 0.3,
        'cache': True,
        'input_shape': (50, 50, 3),
        'save_path': './pso_results/'
    }
    
    Path(config['save_path']).mkdir(exist_ok=True)
    
    # Initialize ColabNAS
    colabnas = ColabNAS(**config)
    
    # Setup PSO components
    search_space = ArchitectureSearchSpace(k_range=(4, 32), c_range=(0, 5))
    decoder = ModelDecoder()
    PSOOptimizer = NASPsoOptimizer.setup(search_space=search_space, decoder=decoder, 
                                        n_particles=5, iterations=10)
    
    # Run search
    print(f"\n🚀 Starting PSO search with seed {RANDOM_SEED}...")
    result_path = colabnas.search(PSOOptimizer)
    
    if result_path and Path(result_path).exists():
        print(f"✅ Success! Model saved at: {result_path}")
    else:
        print("❌ No feasible architecture found")

if __name__ == "__main__":
    main()