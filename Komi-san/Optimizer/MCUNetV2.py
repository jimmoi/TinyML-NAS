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

class MobileNetV2_RD_Fixed(Vanilla_NAS):
    architecture_name = 'mobilenetv2_rd_fixed'

    def inverted_residual_block(self, inputs, filters, stride, expansion_factor, kernel_size):
        """
        Standard Inverted Residual Block
        รับค่า kernel_size เพื่อปรับ RF (Receptive Field) ได้
        """
        in_channels = inputs.shape[-1]
        expanded_channels = in_channels * expansion_factor
        macs = 0
        x = inputs

        # --- 1. Expansion Phase (1x1) ---
        # MobileNetV2 ปกติถ้า expansion=1 จะข้ามขั้นตอนนี้ (เช่น block แรก)
        if expansion_factor != 1:
            x = keras.layers.Conv2D(expanded_channels, kernel_size=(1, 1), padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU(max_value=6.0)(x)
            macs += (in_channels * expanded_channels * x.shape[1] * x.shape[2])
        else:
            expanded_channels = in_channels # ถ้าไม่ขยาย ก็ใช้ channel เท่าเดิม

        # --- 2. Depthwise Convolution (Configurable Kernel Size) ---
        # นี่คือจุดที่ทำตามทฤษฎี: รับ kernel_size เข้ามา (3, 5, หรือ 7)
        x = keras.layers.DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), 
                                         strides=stride, 
                                         padding='same', 
                                         use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(max_value=6.0)(x)
        
        # MACs calculation
        macs += (expanded_channels * kernel_size * kernel_size * x.shape[1] * x.shape[2])

        # --- 3. Projection Phase (Linear - 1x1) ---
        x = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        # Note: MobileNetV2 ไม่มี activation ต่อท้าย projection (Linear Bottleneck)
        macs += (expanded_channels * filters * x.shape[1] * x.shape[2])

        # --- 4. Skip Connection ---
        # เงื่อนไข: Stride ต้องเป็น 1 และ Channel เข้า-ออกต้องเท่ากัน
        if stride == 1 and in_channels == filters:
            if inputs.shape[1] == x.shape[1] and inputs.shape[2] == x.shape[2]:
                x = keras.layers.Add()([inputs, x])

        return x, macs

    def create_model(self, k, c):
        """
        k: จำนวน filter เริ่มต้น
        c: จำนวน block (ความลึก)
        """
        number_of_cells_limited = False
        number_of_mac = 0
        inputs = keras.Input(shape=self.input_shape)

        # --- Stem Layer (Standard 3x3) ---
        # เลเยอร์แรกสุดมักใช้ 3x3 เสมอเพื่อดึง Feature พื้นฐาน
        n = int(k)
        x = keras.layers.Conv2D(n, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(max_value=6.0)(x)
        
        c_in = self.input_shape[2]
        number_of_mac += (c_in * 3 * 3 * x.shape[1] * x.shape[2] * n)

        # --- Backbone Building (Fixed RD Strategy) ---
        # ทฤษฎี RD: Layer แรกๆ ใช้ Kernel เล็ก (3x3) -> Layer หลังๆ ใช้ Kernel ใหญ่ (5x5, 7x7)
        
        multiplier = 1.5 
        
        for i in range(1, c + 1):
            if x.shape[1] <= 1 or x.shape[2] <= 1:
                number_of_cells_limited = True
                break

            # คำนวณความกว้าง (Channels)
            n = int(np.ceil(n * multiplier))
            
            # --- กำหนด Kernel Size ตามลำดับชั้น (Deterministic) ---
            # ไม่ใช้การสุ่ม (Random) แต่ใช้ Logic แบ่งช่วง
            if i <= c // 3:
                # ช่วงแรก (33% แรก): ใช้ Kernel 3x3 (Narrow RF)
                # เพื่อลด Overlap และลด MACs ในช่วงที่ Resolution ยังใหญ่
                current_k = 3
            elif i <= (2 * c) // 3:
                # ช่วงกลาง (33% - 66%): ใช้ Kernel 5x5 (Medium RF)
                current_k = 5
            else:
                # ช่วงท้าย (33% สุดท้าย): ใช้ Kernel 7x7 (Wide RF)
                # เพื่อเพิ่มความแม่นยำ (Accuracy) ในช่วงที่ Resolution เล็กแล้ว
                current_k = 7
            
            # สร้าง Block
            # หมายเหตุ: ในโค้ดตัวอย่างเดิมมีการลดขนาดภาพ (stride=2) ทุกรอบลูป
            # ซึ่งจะทำให้ภาพเล็กลงเร็วมาก ถ้าต้องการแบบ MobileNet เป๊ะๆ อาจต้องปรับ stride=1 บางจังหวะ
            # แต่เพื่อคงโครงสร้างเดิมไว้ ผมจะใช้ stride=2 ตามเดิม
            x, block_macs = self.inverted_residual_block(
                x, 
                filters=n, 
                stride=2, 
                expansion_factor=6, 
                kernel_size=current_k
            )
            
            number_of_mac += block_macs

        # --- Classifier Head ---
        x = keras.layers.GlobalAveragePooling2D()(x)
        current_dim = x.shape[1]
        # เพิ่ม Dropout เล็กน้อยตามสไตล์ MobileNetV2 สมัยใหม่ (optional)
        # x = keras.layers.Dropout(0.2)(x) 
        
        x = keras.layers.Dense(self.num_classes)(x)
        outputs = keras.layers.Softmax()(x)
        
        number_of_mac += (current_dim * self.num_classes)

        model = keras.Model(inputs=inputs, outputs=outputs)
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        # model.summary() # uncomment ถ้าอยากดูโครงสร้าง

        return model, number_of_mac, number_of_cells_limited
    
class MobileNetV2_RD_FixedV2(Vanilla_NAS):
    architecture_name = 'mobilenetv2_rd_final'

    def inverted_residual_block(self, inputs, filters, stride, expansion_factor, kernel_size):
        """
        Standard Inverted Residual Block
        """
        in_channels = inputs.shape[-1]
        expanded_channels = in_channels * expansion_factor
        macs = 0
        x = inputs

        # --- 1. Expansion Phase (1x1) ---
        if expansion_factor != 1:
            x = keras.layers.Conv2D(expanded_channels, kernel_size=(1, 1), padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU(max_value=6.0)(x)
            macs += (in_channels * expanded_channels * x.shape[1] * x.shape[2])
        else:
            expanded_channels = in_channels

        # --- 2. Depthwise Convolution (Variable Kernel Size) ---
        x = keras.layers.DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), 
                                         strides=stride, 
                                         padding='same', 
                                         use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(max_value=6.0)(x)
        macs += (expanded_channels * kernel_size * kernel_size * x.shape[1] * x.shape[2])

        # --- 3. Projection Phase (Linear 1x1) ---
        x = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        macs += (expanded_channels * filters * x.shape[1] * x.shape[2])

        # --- 4. Skip Connection ---
        if stride == 1 and in_channels == filters:
            if inputs.shape[1] == x.shape[1] and inputs.shape[2] == x.shape[2]:
                x = keras.layers.Add()([inputs, x])

        return x, macs

    def create_model(self, k, c):
        """
        Final Version:
        - Dynamic Head: ปรับขนาดหัวตามความละเอียดภาพ (แก้ RAM ระเบิด)
        - RD Strategy: Kernel 3->5->7
        - Delayed Downsampling: รักษาภาพ 25x25 ในช่วงแรก
        """
        number_of_cells_limited = False
        number_of_mac = 0
        inputs = keras.Input(shape=self.input_shape)

        # --- Stem Layer (ลดภาพ 50 -> 25) ---
        n = int(k)
        x = keras.layers.Conv2D(n, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(max_value=6.0)(x)
        
        c_in = self.input_shape[2]
        number_of_mac += (c_in * 3 * 3 * x.shape[1] * x.shape[2] * n)

        # --- Stage Configuration ---
        # แบ่งจำนวนบล็อก c ให้กระจายลง 3 Stages
        stage_depths = [c // 3] * 3
        for i in range(c % 3): stage_depths[i] += 1

        stage_specs = [
            # Stage 1: Kernel 3, Exp 3 (Exp น้อยเพื่อประหยัด RAM เพราะภาพยังใหญ่)
            (3, 3), 
            # Stage 2: Kernel 5, Exp 6 (เริ่มจัดเต็ม)
            (5, 6), 
            # Stage 3: Kernel 7, Exp 6 (เน้นแม่นช่วงท้าย)
            (7, 6)  
        ]
        
        channel_multiplier = 1.5

        # --- Build Blocks ---
        for stage_idx, num_blocks in enumerate(stage_depths):
            if num_blocks == 0: continue
            
            kernel_size, expansion = stage_specs[stage_idx]
            
            for b in range(num_blocks):
                # Logic: จะลดขนาดภาพก็ต่อเมื่อขึ้น Stage ใหม่ (ที่ไม่ใช่ Stage แรก)
                # และภาพต้องยังมีขนาดใหญ่พอ (>8px)
                is_first_block_of_later_stage = (b == 0) and (stage_idx > 0)
                should_downsample = is_first_block_of_later_stage and (x.shape[1] > 8)
                
                current_stride = 2 if should_downsample else 1
                
                # ถ้าลดภาพ -> เพิ่ม Channel
                if should_downsample:
                    n = int(np.ceil(n * channel_multiplier))
                
                # Check Constraints
                if x.shape[1] <= 1:
                    number_of_cells_limited = True
                    break

                x, block_macs = self.inverted_residual_block(
                    x, 
                    filters=n, 
                    stride=current_stride, 
                    expansion_factor=expansion, 
                    kernel_size=kernel_size
                )
                number_of_mac += block_macs

        # --- DYNAMIC CLASSIFIER HEAD (The Critical Fix) ---
        # เลือกขนาด Head ตามความละเอียดภาพที่เหลืออยู่ เพื่อไม่ให้ RAM แตก
        current_res = x.shape[1]
        
        if current_res >= 25:
            # ภาพใหญ่: ใช้ Head เล็ก (ประหยัด RAM)
            target_head_ch = max(32, int(n * 2))
        elif current_res >= 13:
            # ภาพกลาง: ใช้ Head กลาง
            target_head_ch = max(128, int(n * 4))
        else:
            # ภาพเล็ก (<=7): จัดเต็ม Massive Head (เผา Flash ที่เหลือเพื่อ Accuracy)
            target_head_ch = max(256, int(n * 8))

        # 1x1 Conv Head
        x = keras.layers.Conv2D(target_head_ch, (1, 1), use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(max_value=6.0)(x)
        number_of_mac += (n * target_head_ch * x.shape[1] * x.shape[2])

        # Global Pooling & Output
        x = keras.layers.GlobalAveragePooling2D()(x)
        current_dim = x.shape[1]
        
        x = keras.layers.Dropout(0.2)(x) # Dropout กัน Overfit
        
        x = keras.layers.Dense(self.num_classes)(x)
        outputs = keras.layers.Softmax()(x)
        
        number_of_mac += (current_dim * self.num_classes)

        model = keras.Model(inputs=inputs, outputs=outputs)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model, number_of_mac, number_of_cells_limited