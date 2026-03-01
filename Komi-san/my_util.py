import tensorflow as tf
import numpy as np
import os

def split_dataset(path_to_dataset, test_split=0.2) :
    directories = [e for e in os.scandir(path_to_dataset) if e.is_dir()]

    for directory in directories :
        train_directory = path_to_dataset + '/train/' + directory.name
        test_directory = path_to_dataset + '/test/' + directory.name
        os.makedirs(train_directory)
        os.makedirs(test_directory)

        files = [e for e in os.scandir(directory) if os.path.isfile(e)]
        treshold = len(files) * (1 - test_split)
        count = 0

        for f in files :
            if count < treshold :
                os.rename(f.path, f"{train_directory}/{f.name}")
            else :
                os.rename(f.path, f"{test_directory}/{f.name}")
            count = count + 1

        os.rmdir(directory.path)

def load_train_dataset(input_shape, path_to_training_set, val_split, batch_size, cache):
    color_mode = 'rgb' if input_shape[2] == 3 else 'grayscale'
    
    # 1. Load raw data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=path_to_training_set,
        labels='inferred',
        label_mode='categorical',
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=input_shape[0:2],
        shuffle=True,
        seed=11,
        validation_split=val_split,
        subset='training',
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory=path_to_training_set,
        labels='inferred',
        label_mode='categorical',
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=input_shape[0:2],
        shuffle=True,
        seed=11,
        validation_split=val_split,
        subset='validation',
    )

    if cache:
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, validation_ds

def load_train_dataset_mobilenet(input_shape, path_to_training_set, val_split, batch_size, cache):
    def get_augmentation_layer():
        return tf.keras.Sequential([
            # MobileNetV2 expects pixels in range [-1, 1]
            # This layer converts 0-255 to -1 to 1 automatically
            tf.keras.layers.Rescaling(1./127.5, offset=-1), 
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])

    color_mode = 'rgb' if input_shape[2] == 3 else 'grayscale'

    # 1. Load raw data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=path_to_training_set,
        labels='inferred',
        label_mode='categorical',
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=input_shape[0:2],
        shuffle=True,
        seed=11,
        validation_split=val_split,
        subset='training',
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory=path_to_training_set,
        labels='inferred',
        label_mode='categorical',
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=input_shape[0:2],
        shuffle=True,
        seed=11,
        validation_split=val_split,
        subset='validation',
    )

    # 2. Add Rescaling to Validation too! 
    # (Validation doesn't get RandomFlip, but it MUST be rescaled to [-1, 1])
    rescale_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    validation_ds = validation_ds.map(lambda x, y: (rescale_layer(x), y))

    # 3. Add Augmentation + Rescaling to Training
    data_augmentation = get_augmentation_layer()
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache:
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, validation_ds

def load_train_dataset_efficientnet(input_shape, path_to_training_set, val_split, batch_size, cache):
    # EfficientNet handles rescaling internally, 
    # so we ONLY do the geometric augmentations.
    def get_augmentation_layer():
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
        ])

    color_mode = 'rgb' if input_shape[2] == 3 else 'grayscale'

    # 1. Load raw data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=path_to_training_set,
        labels='inferred',
        label_mode='categorical',
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=input_shape[0:2],
        shuffle=True,
        seed=11,
        validation_split=val_split,
        subset='training',
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory=path_to_training_set,
        labels='inferred',
        label_mode='categorical',
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=input_shape[0:2],
        shuffle=True,
        seed=11,
        validation_split=val_split,
        subset='validation',
    )
    

    # 3. Add Augmentation + Rescaling to Training
    data_augmentation = get_augmentation_layer()
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache:
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, validation_ds

def load_train_dataset_resnet(input_shape, path_to_training_set, val_split, batch_size, cache):
    
    # Define the preprocessing function for ResNet50
    # This handles the BGR conversion and mean subtraction
    def preprocess_resnet(image, label):
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label

    def get_augmentation_layer():
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])

    color_mode = 'rgb' if input_shape[2] == 3 else 'grayscale'

    # 1. Load raw data (pixels are 0-255 here)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=path_to_training_set,
        labels='inferred',
        label_mode='categorical',
        image_size=input_shape[0:2],
        batch_size=batch_size,
        seed=11,
        validation_split=val_split,
        subset='training',
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory=path_to_training_set,
        labels='inferred',
        label_mode='categorical',
        image_size=input_shape[0:2],
        batch_size=batch_size,
        seed=11,
        validation_split=val_split,
        subset='validation',
    )

    # 2. Apply ResNet50 Preprocessing to Validation
    validation_ds = validation_ds.map(preprocess_resnet, num_parallel_calls=tf.data.AUTOTUNE)

    # 3. Apply Augmentation THEN ResNet50 Preprocessing to Training
    data_augmentation = get_augmentation_layer()
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Final step: Preprocess the augmented images
    train_ds = train_ds.map(preprocess_resnet, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, validation_ds

def load_test_dataset(input_shape, path_to_test_set, batch_size, cache) :
    color_mode = 'rgb' if input_shape[2] == 3 else 'grayscale'
    test_ds = tf.keras.utils.image_dataset_from_directory(
            directory= path_to_test_set,
            labels='inferred',
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=input_shape[:2],
            shuffle=True
        )
    
    if cache:
        test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return test_ds

def load_test_dataset_mobilenet(input_shape, path_to_test_set, batch_size, cache) :
    color_mode = 'rgb' if input_shape[2] == 3 else 'grayscale'
    test_ds = tf.keras.utils.image_dataset_from_directory(
            directory= path_to_test_set,
            labels='inferred',
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=input_shape[:2],
            shuffle=True
        )
    
    rescale_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    test_ds = test_ds.map(lambda x, y: (rescale_layer(x), y))
    
    if cache:
        test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return test_ds

def load_test_dataset_resnet(input_shape, path_to_test_set, batch_size, cache):
    # 1. Define the specific ResNet50 preprocessing
    def preprocess_resnet(image, label):
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label

    color_mode = 'rgb' if input_shape[2] == 3 else 'grayscale'
    
    # 2. Load raw data
    test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=path_to_test_set,
            labels='inferred',
            label_mode='categorical',
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=input_shape[:2],
            shuffle=False # Usually False for testing/evaluation
        )
    
    # 3. Apply ResNet50 Preprocessing
    test_ds = test_ds.map(preprocess_resnet, num_parallel_calls=tf.data.AUTOTUNE)
    
    if cache:
        test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return test_ds

def test_tflite_model(path_to_resulting_architecture, test_ds) :
    interpreter = tf.lite.Interpreter(path_to_resulting_architecture)
    interpreter.allocate_tensors()

    output_details = interpreter.get_output_details()[0]  # Model has single output.
    input_details = interpreter.get_input_details()[0]  # Model has single input.

    correct = 0
    wrong = 0

    for image, label in test_ds :
        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == tf.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            image = image / input_scale + input_zero_point
        input_data = tf.dtypes.cast(image, tf.uint8)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        if label.numpy().argmax() == interpreter.get_tensor(output_details['index']).argmax() :
            correct = correct + 1
        else :
            wrong = wrong + 1
    print(f"\nTflite model test accuracy: {correct/(correct+wrong)}")
    return correct/(correct+wrong)

def prepare_nas_datasets(*ds, patch_size):
    target_height, target_width = patch_size
    def transform_labels(images, labels):
        # labels shape: (batch, num_classes)
        # We want: (batch, target_height, target_width, num_classes)
        
        # 1. Add spatial dimensions: (batch, 1, 1, num_classes)
        labels = tf.reshape(labels, (-1, 1, 1, labels.shape[-1]))
        
        # 2. Tile (broadcast) to the required output shape
        labels = tf.tile(labels, [1, target_height, target_width, 1])
        
        return images, labels
    
    return [f.map(transform_labels) for f in ds]

if __name__ == '__main__' :
    for dir_name in [
        "Animals-3", 
        "Flowers-4", 
        # "Human"
        ] :
        split_dataset(f'Datasets\{dir_name}', 0.2)