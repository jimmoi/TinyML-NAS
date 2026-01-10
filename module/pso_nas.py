import numpy as np
import datetime
from tensorflow import keras
import tensorflow as tf

class PSONAS:
    """Particle Swarm Optimization based Neural Architecture Search"""
    
    def __init__(self, evaluate_model_fnc, input_shape, num_classes, learning_rate):
        self.evaluate_model_fnc = evaluate_model_fnc
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_counter = 0
        
        # PSO parameters
        self.num_particles = 10
        self.max_iterations = 20
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive parameter
        self.c2 = 1.5  # social parameter
        
        # Architecture bounds
        self.k_min, self.k_max = 2, 32
        self.c_min, self.c_max = 0, 5
        
    def create_model(self, k, c):
        """Create CNN model with given parameters"""
        kernel_size = (3, 3)
        pool_size = (2, 2)
        pool_strides = (2, 2)
        
        number_of_cells_limited = False
        number_of_mac = 0
        
        inputs = keras.Input(shape=self.input_shape)
        
        # First convolutional layer
        n = int(k)
        multiplier = 2
        c_in = self.input_shape[2]
        x = keras.layers.Conv2D(n, kernel_size, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        number_of_mac += (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])
        
        # Adding cells
        for i in range(1, c + 1):
            if x.shape[1] <= 1 or x.shape[2] <= 1:
                number_of_cells_limited = True
                break
            n = int(np.ceil(n * multiplier))
            multiplier = multiplier - 2**-i
            x = keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='valid')(x)
            c_in = x.shape[3]
            x = keras.layers.Conv2D(n, kernel_size, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            number_of_mac += (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])
        
        # Classifier
        x = keras.layers.GlobalAveragePooling2D()(x)
        input_shape = x.shape[1]
        x = keras.layers.Dense(n)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        number_of_mac += (input_shape * x.shape[1])
        x = keras.layers.Dense(self.num_classes)(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Softmax()(x)
        number_of_mac += (x.shape[1] * outputs.shape[1])
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model, number_of_mac, number_of_cells_limited
    
    def evaluate_particle(self, k, c):
        """Evaluate a single particle (architecture)"""
        if k <= 0:
            return {'k': 'unfeasible', 'c': c, 'max_val_acc': -3}
        
        self.model_counter += 1
        model_name = f"pso_k_{int(k)}_c_{c}"
        model, MACC, number_of_cells_limited = self.create_model(k, c)
        result = self.evaluate_model_fnc(model, MACC, number_of_cells_limited, model_name)
        result["k"] = int(k)
        result["c"] = c if not number_of_cells_limited else "Not feasible"
        return result
    
    def search(self):
        """Main PSO search algorithm"""
        start = datetime.datetime.now()
        
        # Initialize particles
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        for _ in range(self.num_particles):
            # Random initialization
            k = np.random.uniform(self.k_min, self.k_max)
            c = np.random.randint(self.c_min, self.c_max + 1)
            particles.append([k, c])
            
            # Random velocity
            vk = np.random.uniform(-2, 2)
            vc = np.random.randint(-1, 2)
            velocities.append([vk, vc])
            
            # Evaluate initial particle
            result = self.evaluate_particle(k, c)
            personal_best.append([k, c])
            personal_best_fitness.append(result['max_val_acc'])
            print(f"Particle {len(particles)}: {result}")
        
        # Find global best
        global_best_idx = np.argmax(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        print(f"Initial global best: k={global_best[0]:.2f}, c={global_best[1]}, fitness={global_best_fitness:.3f}")
        
        # PSO iterations
        for iteration in range(self.max_iterations):
            print(f"\nPSO Iteration {iteration + 1}/{self.max_iterations}")
            
            for i in range(self.num_particles):
                # Update velocity
                r1, r2 = np.random.random(2)
                
                # Velocity update for k
                velocities[i][0] = (self.w * velocities[i][0] + 
                                   self.c1 * r1 * (personal_best[i][0] - particles[i][0]) +
                                   self.c2 * r2 * (global_best[0] - particles[i][0]))
                
                # Velocity update for c
                velocities[i][1] = (self.w * velocities[i][1] + 
                                   self.c1 * r1 * (personal_best[i][1] - particles[i][1]) +
                                   self.c2 * r2 * (global_best[1] - particles[i][1]))
                
                # Update position
                particles[i][0] += velocities[i][0]
                particles[i][1] += velocities[i][1]
                
                # Apply bounds
                particles[i][0] = np.clip(particles[i][0], self.k_min, self.k_max)
                particles[i][1] = int(np.clip(particles[i][1], self.c_min, self.c_max))
                
                # Evaluate new position
                result = self.evaluate_particle(particles[i][0], particles[i][1])
                fitness = result['max_val_acc']
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    # Update global best
                    if fitness > global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
                        print(f"New global best: k={global_best[0]:.2f}, c={global_best[1]}, fitness={global_best_fitness:.3f}")
                
                print(f"Particle {i+1}: k={particles[i][0]:.2f}, c={particles[i][1]}, fitness={fitness:.3f}")
        
        # Return best architecture
        best_result = self.evaluate_particle(global_best[0], global_best[1])
        end = datetime.datetime.now()
        
        print(f"\nPSO completed. Best architecture: k={int(global_best[0])}, c={global_best[1]}, accuracy={global_best_fitness:.3f}")
        
        return best_result, end - start