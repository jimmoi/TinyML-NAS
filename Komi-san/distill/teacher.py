import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def wide_resnet_block(x, filters, stride, dropout_rate=0.3):
    shortcut = x
    
    # First convolution
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
              kernel_regularizer=regularizers.l2(0.0005))(x)
    
    x = layers.Dropout(dropout_rate)(x)
    
    # Second convolution
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same',
              kernel_regularizer=regularizers.l2(0.0005))(x)
    
    # Projection shortcut if dimensions change
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
              kernel_regularizer=regularizers.l2(0.0005))(shortcut)
        
    return layers.add([x, shortcut])

def build_wide_resnet(input_shape, depth, k, num_classes):
    """
    depth: total layers (e.g., 16 or 28)
    k: width factor (e.g., 8 or 10)
    """
    
    n = (depth - 4) // 6
    inputs = layers.Input(shape=input_shape)
    
    # MODIFICATION FOR 50x50: Small kernel, stride 1, no maxpool
    x = layers.Conv2D(16, 3, strides=1, padding='same',
              kernel_regularizer=regularizers.l2(0.0005))(inputs)
    
    # Stages
    for stride in [1, 2, 2]:
        filters = 16 * k if stride == 1 else (32 * k if stride == 2 else 64 * k)
        x = wide_resnet_block(x, filters, stride)
        for _ in range(n-1):
            x = wide_resnet_block(x, filters, 1)
            
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def build_mobilenetv2(input_shape, num_classes):
    # Load base model without the top classification layer
    # We use MobileNetV2 because it's excellent at small resolutions
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Fine-tuning Strategy: 
    # Initially, we freeze the base_model to train only the new head
    base_model.trainable = True 
    
    # Optional: If overfitting is still high, freeze the first 100 layers:
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_efficientnetb0(input_shape, num_classes):
    # 1. Load the base model
    # Note: EfficientNetB0 expects input pixels in range [0, 255]
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # 2. Fine-tuning strategy: Unfreeze the top layers
    # We unfreeze everything, but we will use a small learning rate (0.0001)
    base_model.trainable = True

    # 3. Create the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        # Add a high dropout rate to force generalization
        layers.Dropout(0.4),
        # Final classification layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model