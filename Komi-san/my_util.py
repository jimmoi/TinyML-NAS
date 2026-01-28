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

def load_dataset(path_to_training_set, path_to_test_set, batch_size, validation_split, input_shape) :
    num_classes = len(next(os.walk(path_to_training_set))[1])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory= path_to_training_set,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size= batch_size,
        image_size=(input_shape[0], input_shape[1]),
        shuffle=True,
        seed=11,
        validation_split=validation_split,
        subset='training'
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory= path_to_training_set,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size= batch_size,
        image_size=(input_shape[0], input_shape[1]),
        shuffle=True,
        seed=11,
        validation_split=validation_split,
        subset='validation'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory= path_to_test_set,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size= batch_size,
        image_size=(input_shape[0], input_shape[1]),
        shuffle=True,
        seed=11
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, validation_ds, test_ds, num_classes

def test_tflite_model(path_to_resulting_architecture, test_ds):
    # Convert the Path object to a string
    interpreter = tf.lite.Interpreter(model_path=str(path_to_resulting_architecture))
    interpreter.allocate_tensors()

    output = interpreter.get_output_details()[0]  # Model has single output.
    input = interpreter.get_input_details()[0]  # Model has single input.
    input_dtype = input['dtype']

    correct = 0
    wrong = 0

    for images_batch, labels_batch in test_ds:
        for image, label in zip(images_batch, labels_batch):
            # Check if the input type is quantized, then rescale input data to uint8
            if input_dtype == np.uint8 or input_dtype == tf.uint8:
                input_scale, input_zero_point = input["quantization"]
                image = image / input_scale + input_zero_point
            input_data = np.expand_dims(image.numpy().astype(input_dtype), axis=0)
            interpreter.set_tensor(input['index'], input_data)
            interpreter.invoke()
            if label.numpy().argmax() == interpreter.get_tensor(output['index']).argmax() :
                correct = correct + 1
            else :
                wrong = wrong + 1
    print(f"\nTflite model test accuracy: {correct/(correct+wrong)}")
    
    return correct/(correct+wrong)  

if __name__ == '__main__' :
    for dir_name in [
        "Animals-3", 
        "Flowers-4", 
        # "Human"
        ] :
        split_dataset(f'Datasets\{dir_name}', 0.2)