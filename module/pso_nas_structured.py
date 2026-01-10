import numpy as np
import datetime
from tensorflow import keras
import tensorflow as tf

# --- 1. SEARCH SPACE & ENCODING ---
# We represent an architecture as a vector: [k, c]
# k: Initial number of filters (Continuous, then rounded to int)
# c: Number of additional cells (Continuous, then rounded to int)

class ArchitectureSearchSpace:
    def __init__(self, k_range=(4, 128), c_range=(0, 10)):
        self.k_min, self.k_max = k_range
        self.c_min, self.c_max = c_range

    def clamp(self, k, c):
        """Ensures particles stay within the defined search space."""
        return np.clip(k, self.k_min, self.k_max), np.clip(c, self.c_min, self.c_max)

# --- 2. DECODER (Model Creator) ---
class ModelDecoder:
    def __init__(self):
        self.input_shape = None
        self.num_classes = None
        self.learning_rate = None

    def decode_and_build(self, k, c):
        """Transforms PSO coordinates into a Keras model + MAC count."""
        k, c = int(k), int(c)
        kernel_size = (3, 3)
        pool_size = (2, 2)
        
        number_of_mac = 0
        number_of_cells_limited = False
        
        inputs = keras.Input(shape=self.input_shape)
        n = k
        multiplier = 2
        
        # First Layer
        c_in = self.input_shape[2]
        x = keras.layers.Conv2D(n, kernel_size, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        # Simplified MAC calculation for clarity
        number_of_mac += (c_in * np.prod(kernel_size) * x.shape[1] * x.shape[2] * x.shape[3])

        # Adding Cells
        for i in range(1, c + 1):
            if x.shape[1] <= 1 or x.shape[2] <= 1:
                number_of_cells_limited = True
                break
            
            n = int(np.ceil(n * multiplier))
            multiplier -= 2**-i
            x = keras.layers.MaxPooling2D(pool_size=pool_size, strides=(2,2), padding='valid')(x)
            
            c_in = x.shape[3]
            x = keras.layers.Conv2D(n, kernel_size, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            number_of_mac += (c_in * np.prod(kernel_size) * x.shape[1] * x.shape[2] * x.shape[3])

        # Classifier
        x = keras.layers.GlobalAveragePooling2D()(x)
        feat_dim = x.shape[1]
        x = keras.layers.Dense(n)(x)
        number_of_mac += (feat_dim * n)
        x = keras.layers.Dense(self.num_classes)(x)
        outputs = keras.layers.Softmax()(x)
        number_of_mac += (n * self.num_classes)

        model = keras.Model(inputs=inputs, outputs=outputs)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model, number_of_mac, number_of_cells_limited

# --- 3. EVALUATOR & CONSTRAINTS ---
class NASPsoOptimizer:
    def __init__(self, evaluate_model_fnc, input_shape, num_classes, learning_rate):
        self.evaluate_model_fnc = evaluate_model_fnc # External fnc that returns {'max_val_acc': float}
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
        # Initialize particles [k, c] and velocities
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

        w, c1, c2 = 0.5, 1.5, 1.5 # Hyperparameters for PSO

        start = datetime.datetime.now()
        
        results_best = {}
        for it in range(self.iterations):
            print(f"==================== iteration {it} ====================")
            for i in range(self.n_particles):
                k, c = particles[i]
                
                # Build and Evaluate
                model, macc, limited = self.decoder.decode_and_build(k, c)
                
                # Constraint Check
                if limited:
                    score = 0 # Penalty for invalid architectures
                else:
                    self.model_name = f"k_{int(k)}_c_{int(c)}"
                    results = self.evaluate_model_fnc(model, macc, limited, self.model_name)
                    score = results['max_val_acc']

                # Update Personal Best
                if score > p_best_scores[i]:
                    p_best_scores[i] = score
                    p_best[i] = particles[i]

                # Update Global Best
                if score > g_best_score:
                    results_best = results
                    g_best_score = score
                    g_best = np.copy(particles[i])

            # Update Velocities and Positions
            if g_best is None:
                # Option A: If no valid architecture was found, re-randomize or skip update
                print("Warning: No valid architecture found in this iteration. Re-randomizing velocities...")
                velocities = np.random.uniform(-1, 1, size=velocities.shape)
            else:
                # Standard PSO Update logic
                for i in range(self.n_particles):
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[i] = (w * velocities[i] + 
                                    c1 * r1 * (p_best[i] - particles[i]) + 
                                    c2 * r2 * (g_best - particles[i])) # No longer crashes
                    
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