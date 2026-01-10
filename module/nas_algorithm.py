from tensorflow import keras
import tensorflow as tf
import numpy as np
import datetime

class OurNAS:
    architecture_name = 'resulting_architecture'
    
    def __init__(self, evaluate_model_fnc, input_shape, num_classes, learning_rate):
        self.evaluate_model_fnc = evaluate_model_fnc
        self.model_count = 0
        self.model_name = ""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def create_model(self, k, c):
        kernel_size = (3,3)
        pool_size = (2,2)
        pool_strides = (2,2)

        number_of_cells_limited = False
        number_of_mac = 0

        inputs = keras.Input(shape=self.input_shape)

        # Convolutional base
        n = int(k)
        multiplier = 2

        # First convolutional layer
        c_in = self.input_shape[2]
        x = keras.layers.Conv2D(n, kernel_size, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        number_of_mac = number_of_mac + (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])

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
            number_of_mac = number_of_mac + (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])

        # Classifier
        x = keras.layers.GlobalAveragePooling2D()(x)
        input_shape = x.shape[1]
        x = keras.layers.Dense(n)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        number_of_mac = number_of_mac + (input_shape * x.shape[1])
        x = keras.layers.Dense(self.num_classes)(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Softmax()(x)
        number_of_mac = number_of_mac + (x.shape[1] * outputs.shape[1])

        model = keras.Model(inputs=inputs, outputs=outputs)

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

        model.summary()

        return model, number_of_mac, number_of_cells_limited

    def search(self):
        self.model_counter = 0
        epsilon = 0.005
        k0 = 4

        start = datetime.datetime.now()

        k = k0
        previous_architecture = self.explore_num_cells(k)
        k = 2 * k
        current_architecture = self.explore_num_cells(k)

        if current_architecture['max_val_acc'] > previous_architecture['max_val_acc']:
            previous_architecture = current_architecture
            k = 2 * k
            current_architecture = self.explore_num_cells(k)
            while current_architecture['max_val_acc'] > previous_architecture['max_val_acc'] + epsilon:
                previous_architecture = current_architecture
                k = 2 * k
                current_architecture = self.explore_num_cells(k)
        else:
            k = k0 / 2
            current_architecture = self.explore_num_cells(k)
            while current_architecture['max_val_acc'] >= previous_architecture['max_val_acc']:
                previous_architecture = current_architecture
                k = k / 2
                current_architecture = self.explore_num_cells(k)

        resulting_architecture = previous_architecture
        end = datetime.datetime.now()

        return resulting_architecture, end-start

    def explore_num_cells(self, k):
        previous_architecture = {'k': -1, 'c': -1, 'max_val_acc': -2}
        current_architecture = {'k': -1, 'c': -1, 'max_val_acc': -1}
        c = -1
        k = int(k)

        while current_architecture['max_val_acc'] > previous_architecture['max_val_acc']:
            previous_architecture = current_architecture
            c = c + 1
            self.model_counter = self.model_counter + 1
            current_architecture = self.evaluate_model_process(k, c)
            print(f"\n\n\n{current_architecture}\n\n\n")
        return previous_architecture

    def evaluate_model_process(self, k, c):
        if k > 0:
            self.model_name = f"k_{k}_c_{c}"
            model, MACC, number_of_cells_limited = self.create_model(k, c)
            result_property_dict = self.evaluate_model_fnc(model, MACC, number_of_cells_limited, self.model_name)
            result_property_dict["k"] = k
            result_property_dict["c"] = c if not number_of_cells_limited else "Not feasible"
            return result_property_dict
        else:
            return {'k': 'unfeasible', 'c': c, 'max_val_acc': -3}