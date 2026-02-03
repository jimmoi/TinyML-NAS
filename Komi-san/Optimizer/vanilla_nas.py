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
    c_in = x.shape[3]
    x = keras.layers.Conv2D(filters=self.num_classes, kernel_size=(1, 1), padding='same')(x)
    number_of_mac += (c_in * 1 * 1 * x.shape[1] * x.shape[2] * x.shape[3])
    
    x = keras.layers.GlobalAveragePooling2D()(x) 
    outputs = keras.layers.Softmax()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model, number_of_mac, number_of_cells_limited

class Mametoyas_NAS(Vanilla_NAS):
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
    current_h = x.shape[1]
    current_w = x.shape[2]
    c_in = x.shape[3]

    # ใช้ Conv2D ที่มี Kernel Size เท่ากับขนาดภาพที่เหลืออยู่ (Global Convolution)
    # เพื่อลดมิติจาก (H, W, C) ให้เหลือ (1, 1, num_classes)
    x = keras.layers.Conv2D(filters=self.num_classes, 
                            kernel_size=(current_h, current_w), 
                            padding='valid')(x)
    
    # คำนวณ MAC: (Cin * K_h * K_w * 1 * 1 * Cout)
    number_of_mac += (c_in * current_h * current_w * 1 * 1 * self.num_classes)

    # บีบมิติที่เหลือ (1, 1, num_classes) ออกให้เหลือเพียง (num_classes,)
    x = keras.layers.Reshape((self.num_classes,))(x)
    
    outputs = keras.layers.Softmax()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model, number_of_mac, number_of_cells_limited
  
class X_NAS(Vanilla_NAS):
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
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape((1, 1, x.shape[1]))(x)
    
    c_in = x.shape[3]
    x = keras.layers.Conv2D(filters=self.num_classes, kernel_size=(1, 1), padding='same')(x)
    number_of_mac += (c_in * 1 * 1 * x.shape[1] * x.shape[2] * x.shape[3])
    
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Softmax()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model, number_of_mac, number_of_cells_limited

class Jimmy_NAS(Vanilla_NAS):
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
    
    # --- New Classifier Logic: Force down to 1x1 ---
    # If the loop finished but we aren't at 1x1 yet, keep pooling/convolving
    while x.shape[1] > 1 or x.shape[2] > 1:
        # Use a valid pool to shrink or a stride-2 conv
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        
        # If the pooling made it 0 or negative (edge cases), break
        if x.shape[1] == 0: break 
        
        # Add a 1x1 or 3x3 conv to keep the pattern
        c_in = x.shape[3]
        x = keras.layers.Conv2D(n, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        number_of_mac += (c_in * 3 * 3 * x.shape[1] * x.shape[2] * x.shape[3])

    # --- Final Classification Layer (1x1 Conv to num_classes) ---
    c_in = x.shape[3]
    x = keras.layers.Conv2D(filters=self.num_classes, kernel_size=(1, 1), padding='same')(x)
    number_of_mac += (c_in * 1 * 1 * x.shape[1] * x.shape[2] * x.shape[3])

    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Softmax()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model, number_of_mac, number_of_cells_limited
  
class Real_Hyper_FullyCNN_NAS(Vanilla_NAS):
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
    x = keras.layers.Conv2D(filters=self.num_classes, kernel_size=(1, 1), padding='same')(x)
    outputs = keras.layers.Softmax(axis=-1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model, number_of_mac, number_of_cells_limited

class Mametoyas_TinyCNNNASS(Vanilla_NAS):

    architecture_name = 'resulting_architecture'

    def __init__(self, evaluate_model_fnc, input_shape, num_classes, learning_rate):
        super().__init__(evaluate_model_fnc, input_shape, num_classes, learning_rate)

    def create_model(self, k, c):

        kernel_size = (3, 3)
        number_of_cells_limited = False
        number_of_mac = 0

        inputs = keras.Input(shape=self.input_shape)

        # ===== Stem (Depthwise Separable) =====
        n = int(k)
        c_in = self.input_shape[2]

        x = keras.layers.DepthwiseConv2D(
            kernel_size,
            padding='same',
            use_bias=False
        )(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        # MAC depthwise
        number_of_mac += (
            c_in * kernel_size[0] * kernel_size[1] *
            x.shape[1] * x.shape[2]
        )

        x = keras.layers.Conv2D(
            n,
            kernel_size=(1, 1),
            padding='same',
            use_bias=False
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        # MAC pointwise
        number_of_mac += (
            c_in * x.shape[1] * x.shape[2] * x.shape[3]
        )

        # ===== NAS Cells =====
        for i in range(1, c + 1):

            if x.shape[1] <= 1 or x.shape[2] <= 1:
                number_of_cells_limited = True
                break

            n = int(np.ceil(n * 1.5))   # โตช้ากว่า Vanilla
            c_in = x.shape[3]

            # Patch downsampling
            x = keras.layers.DepthwiseConv2D(
                kernel_size,
                strides=2,
                padding='same',
                use_bias=False
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

            number_of_mac += (
                c_in * kernel_size[0] * kernel_size[1] *
                x.shape[1] * x.shape[2]
            )

            x = keras.layers.Conv2D(
                n,
                kernel_size=(1, 1),
                padding='same',
                use_bias=False
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

            number_of_mac += (
                c_in * x.shape[1] * x.shape[2] * x.shape[3]
            )

        # ===== Ultra-light Classifier =====
        c_in = x.shape[3]

        x = keras.layers.Conv2D(
            self.num_classes,
            kernel_size=(1, 1),
            padding='same',
            use_bias=False
        )(x)

        number_of_mac += (
            c_in * x.shape[1] * x.shape[2] * x.shape[3]
        )

        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Softmax()(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()

        return model, number_of_mac, number_of_cells_limited