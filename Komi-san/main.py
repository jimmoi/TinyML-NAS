from tensorflow import keras
from pathlib import Path
import tensorflow as tf
import numpy as np
from experiment_manager import Manager
from Optimizer import *
import sys
import json
import pickle


# search_space = ArchitectureSearchSpace(k_range=(2, 10), c_range=(1, 5))
# decoder = ModelDecoder()

# # Pass your existing evaluation logic here
# colabNAS = ColabNAS(peak_RAM_upper_bound, Flash_upper_bound, MACC_upper_bound, path_to_training_set, val_split, cache, input_shape, save_path=save_path)
# #search
# path_to_tflite_model = colabNAS.search(NASPsoOptimizer.setup(search_space, decoder))

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

def main():
    
    import os
    import tensorflow as tf
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))

    if gpu_devices:
        print("Found GPU(s):", gpu_devices)
    else:
        print("No GPU devices found. TensorFlow is likely using the CPU.")
    
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    data_dir = Path(data_dir).with_suffix('')
    
    manager = Manager(path_to_training_set=data_dir, experiment_name="Test_PSO_full_cov_particle_10_iter_20")
    nas = manager.setup_nas()

    #search
    search_space = ArchitectureSearchSpace(k_range=(2, 10), c_range=(1, 5))
    decoder = ModelDecoder2()
    search_output = nas.search(PSO_NAS.setup(search_space, decoder))
    
    
    
    try:
        with open(manager.experiment_dir / "search_output.json", "w") as f:
            json.dump(search_output, f, indent=4)
    except Exception as e:
        try:
            with open(manager.experiment_dir / "search_output.pkl", "wb") as f:
                pickle.dump(search_output, f)
        except Exception as e:
            print(f"Error: {e}")
    
    _, test_ds = nas.get_data()
    test_tflite_model(search_output['path_to_best_architecture'], test_ds)

if __name__ == "__main__":
    main()
