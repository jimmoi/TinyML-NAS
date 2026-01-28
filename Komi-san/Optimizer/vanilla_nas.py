from tensorflow import keras
import tensorflow as tf
import numpy as np
import datetime
from .abc_nas import ABC_NAS

class Vanilla_NAS(ABC_NAS):
  architecture_name = 'resulting_architecture'
  def __init__(self, evaluate_model_fnc, input_shape, num_classes, learning_rate):
    super().__init__(evaluate_model_fnc, input_shape, num_classes, learning_rate)
    self.model_count = 0

  def create_model(self, k, c):
    kernel_size = (3,3)
    pool_size = (2,2)
    pool_strides = (2,2)

    number_of_cells_limited = False
    number_of_mac = 0

    inputs = keras.Input(shape=self.input_shape)

    #convolutional base
    n = int(k)
    multiplier = 2

    #first convolutional layer
    c_in = self.input_shape[2]
    x = keras.layers.Conv2D(n, kernel_size, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    number_of_mac = number_of_mac + (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])

    #adding cells
    for i in range(1, c + 1) :
        if x.shape[1] <= 1 or x.shape[2] <= 1 :
            number_of_cells_limited = True
            break;
        n = int(np.ceil(n * multiplier))
        multiplier = multiplier - 2**-i
        x = keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='valid')(x)
        c_in = x.shape[3]
        x = keras.layers.Conv2D(n, kernel_size, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        number_of_mac = number_of_mac + (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])

    #classifier
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

    if (current_architecture['max_val_acc'] > previous_architecture['max_val_acc']) :
        self.iterations_accuracy.append(current_architecture['max_val_acc']) 
        previous_architecture = current_architecture
        k = 2 * k
        current_architecture = self.explore_num_cells(k)
        while(current_architecture['max_val_acc'] > previous_architecture['max_val_acc'] + epsilon) :
            self.iterations_accuracy.append(current_architecture['max_val_acc']) 
            previous_architecture = current_architecture
            k = 2 * k
            current_architecture = self.explore_num_cells(k)
    else :
        k = k0 / 2
        self.iterations_accuracy.append(previous_architecture['max_val_acc']) 
        current_architecture = self.explore_num_cells(k)
        while(current_architecture['max_val_acc'] >= previous_architecture['max_val_acc']) :
            self.iterations_accuracy.append(current_architecture['max_val_acc']) 
            previous_architecture = current_architecture
            k = k / 2
            current_architecture = self.explore_num_cells(k)

    resulting_architecture_dict = previous_architecture
    end = datetime.datetime.now()

    return resulting_architecture_dict, end-start, self.iterations_accuracy

  def explore_num_cells(self, k) :
      previous_architecture = {'k': -1, 'c': -1, 'max_val_acc': -2}
      current_architecture = {'k': -1, 'c': -1, 'max_val_acc': -1}
      c = -1
      k = int(k)

      while(current_architecture['max_val_acc'] > previous_architecture['max_val_acc']) :
          previous_architecture = current_architecture
          c = c + 1
          self.model_counter = self.model_counter + 1
          current_architecture = self.evaluate_model_process(k, c)
          self.iterations_accuracy.append(current_architecture['max_val_acc'])
          print(f"\n\n\n{current_architecture}\n\n\n")
      return previous_architecture

  def evaluate_model_process(self, k, c):
    if k > 0 :
      self.model_name = f"k_{k}_c_{c}"
      model, MACC, number_of_cells_limited = self.create_model(k, c)
      result_property_dict = self.evaluate_model_fnc(model, MACC, number_of_cells_limited, self.model_name)
      result_property_dict["k"] = k
      result_property_dict["c"] = c if not number_of_cells_limited else "Not feasible"
      return result_property_dict
    else :
      return{'k': 'unfeasible', 'c': c, 'max_val_acc': -3}
    
class VanillaCNN_NAS(Vanilla_NAS):
  architecture_name = 'resulting_architecture'
  def __init__(self, evaluate_model_fnc, input_shape, num_classes, learning_rate):
    super().__init__(evaluate_model_fnc, input_shape, num_classes, learning_rate)

  def create_model(self, k, c):
    kernel_size = (3,3)
    pool_size = (2,2)
    pool_strides = (2,2)

    number_of_cells_limited = False
    number_of_mac = 0

    inputs = keras.Input(shape=self.input_shape)

    #convolutional base
    n = int(k)
    multiplier = 2

    #first convolutional layer
    c_in = self.input_shape[2]
    x = keras.layers.Conv2D(n, kernel_size, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    number_of_mac = number_of_mac + (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])

    #adding cells
    for i in range(1, c + 1) :
        if x.shape[1] <= 1 or x.shape[2] <= 1 :
            number_of_cells_limited = True
            break;
        n = int(np.ceil(n * multiplier))
        multiplier = multiplier - 2**-i
        x = keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='valid')(x)
        c_in = x.shape[3]
        x = keras.layers.Conv2D(n, kernel_size, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        number_of_mac = number_of_mac + (c_in * kernel_size[0] * kernel_size[1] * x.shape[1] * x.shape[2] * x.shape[3])
    
    # --- Fully Convolutional Classifier ---
    # Instead of Global Pooling, we ensure spatial dimensions are 1x1 
    # using a final GlobalAveragePooling then expanding, or just GlobalAveragePooling
    # To keep it "Purely Convolutional", we use a global pool followed by 1x1 convolutions
    x = keras.layers.GlobalAveragePooling2D()(x) 
    # Reshape to (1, 1, filters) so we can apply Conv2D
    x = keras.layers.Reshape((1, 1, x.shape[1]))(x)

    # Convolutional replacement for Dense(n)
    c_in = x.shape[3]
    x = keras.layers.Conv2D(filters=self.num_classes, kernel_size=(1, 1), padding='same')(x)
    # x = keras.layers.ReLU()(x)
    number_of_mac += (c_in * 1 * 1 * x.shape[1] * x.shape[2] * x.shape[3])

    # Flatten the (1, 1, num_classes) output to (num_classes,) for the loss function
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Softmax()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model, number_of_mac, number_of_cells_limited
