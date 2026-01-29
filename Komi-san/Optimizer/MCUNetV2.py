from tensorflow import keras
import tensorflow as tf
import numpy as np
from Optimizer import Vanilla_NAS

class MCUNetV2_Backbone_NAS(Vanilla_NAS):
    architecture_name = 'mcunetv2_backbone_diverse'

    def inverted_residual_block(self, inputs, filters, stride, expansion_factor=6, kernel_size=3):
        """
        Block ที่รับค่า kernel_size ได้ (3, 5, หรือ 7)
        """
        in_channels = inputs.shape[-1]
        expanded_channels = in_channels * expansion_factor
        macs = 0
        x = inputs

        # 1. Expansion Phase (1x1)
        if expansion_factor != 1:
            x = keras.layers.Conv2D(expanded_channels, kernel_size=(1, 1), padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU(max_value=6.0)(x)
            macs += (in_channels * expanded_channels * x.shape[1] * x.shape[2])

        # 2. Depthwise Convolution (Variable Kernel Size!)
        # ใช้ kernel_size ที่ส่งเข้ามา
        x = keras.layers.DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), 
                                         strides=stride, 
                                         padding='same', 
                                         use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(max_value=6.0)(x)
        
        # MACs: Cexp * K * K * H * W
        macs += (expanded_channels * kernel_size * kernel_size * x.shape[1] * x.shape[2])

        # 3. Projection Phase (Linear)
        x = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        macs += (expanded_channels * filters * x.shape[1] * x.shape[2])

        # 4. Skip Connection
        if stride == 1 and in_channels == filters:
            # เช็คด้วยว่าขนาดภาพเท่ากันไหม (ปกติ padding='same' จะเท่ากันถ้า stride=1)
            if inputs.shape[1] == x.shape[1] and inputs.shape[2] == x.shape[2]:
                x = keras.layers.Add()([inputs, x])

        return x, macs

    def create_model(self, k, c):
        number_of_cells_limited = False
        number_of_mac = 0
        inputs = keras.Input(shape=self.input_shape)

        # --- Stem Layer (Fix 3x3 ตามมาตรฐาน) ---
        n = int(k)
        x = keras.layers.Conv2D(n, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(max_value=6.0)(x)
        
        c_in = self.input_shape[2]
        number_of_mac += (c_in * 3 * 3 * x.shape[1] * x.shape[2] * n)

        # --- Backbone Building ---
        multiplier = 1.5
        
        # ตัวเลือก Kernel Size ตาม Paper MCUNetV2 [3, 5, 7]
        possible_kernels = [3, 5, 7] 

        for i in range(1, c + 1):
            if x.shape[1] <= 1 or x.shape[2] <= 1:
                number_of_cells_limited = True
                break

            n = int(np.ceil(n * multiplier))
            
            # --- จุดสำคัญ: สุ่มเลือก Kernel Size ---
            # การสุ่มนี้จำลอง "Search Space" ที่ NAS สามารถเลือกได้
            # หมายเหตุ: ใน NAS ขั้นสูง ตัวแปรนี้จะถูกกำหนดโดย Algorithm (เช่น GA/PSO) 
            # แต่ใน Vanilla NAS นี้เราใช้การสุ่มเพื่อให้เกิดความหลากหลายในแต่ละโมเดลที่สร้าง
            current_k = np.random.choice(possible_kernels)
            
            # เรียกใช้ Block ใหม่พร้อมส่ง current_k เข้าไป
            x, block_macs = self.inverted_residual_block(x, filters=n, stride=2, expansion_factor=6, kernel_size=current_k)
            
            number_of_mac += block_macs

        # --- Classifier Head ---
        x = keras.layers.GlobalAveragePooling2D()(x)
        current_dim = x.shape[1]
        x = keras.layers.Dense(self.num_classes)(x)
        outputs = keras.layers.Softmax()(x)
        
        number_of_mac += (current_dim * self.num_classes)

        model = keras.Model(inputs=inputs, outputs=outputs)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model, number_of_mac, number_of_cells_limited