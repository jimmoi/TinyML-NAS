from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from Optimizer import *
import subprocess
import re
from ColabNAS import ColabNAS
import seaborn as sns

RANDOM_SEED = 42

def load_image_paths(data_dir):
    # imitate tensorflow.keras.utils.image_dataset_from_directory
    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()]) 
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    image_paths = []
    labels = []
    for class_name in class_names:
        for img_path in (data_dir / class_name).glob("*"):
            image_paths.append(str(img_path))
            labels.append(class_to_idx[class_name])

    return np.array(image_paths), np.array(labels), class_names

def make_dataset(paths, labels, input_shape, num_classes, batch_size, shuffle=True):
    def parse_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, input_shape[:2])
        label = tf.one_hot(label, num_classes)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def load_data_kfold(data_dir, input_shape, batch_size, k_folds=5, cache=True ):
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    # Load train paths
    image_paths, labels, class_names = load_image_paths(train_dir)
    num_classes = len(class_names)

    # Test dataset (unchanged)
    test_ds = tf.keras.utils.image_dataset_from_directory(
            directory= test_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='rgb',
            batch_size=1,
            image_size=input_shape[:2],
            shuffle=True
        )
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        train_ds = make_dataset(image_paths[train_idx], labels[train_idx],
                                input_shape, num_classes, batch_size, shuffle=True)
        val_ds = make_dataset(image_paths[val_idx], labels[val_idx],
                                input_shape, num_classes, batch_size, shuffle=False)

        if cache:
            train_ds = train_ds.cache()
            val_ds = val_ds.cache()
        folds.append((train_ds, val_ds))

    return folds, test_ds, num_classes

def write_compare_model_log(model_log, file_path, model_names):
    text = ""
    
    # make column name
    columns = ["k-fold"]
    
    # extract metric names
    first_kfold = next(iter(model_log.values()))
    first_model = next(iter(first_kfold.values()))
    metrices_name = list(first_model.keys()) 
    
    for metric in metrices_name:
        if metric in ["train_losses", "val_losses"]:
            continue
        columns.append(f"{model_names[0]}_{metric}")
        columns.append(f"{model_names[1]}_{metric}")
    
    column_name = ",".join(columns)
    text += column_name + "\n"
    
    # make row
    for kfold_idx, models in model_log.items():
        row_text = []
        row_text.append(kfold_idx)
        
        for metric in metrices_name:
            if metric in ["train_losses", "val_losses"]:
                continue
            row_text.append(str(models[model_names[0]][metric]) if models[model_names[0]][metric] is not None else "Null")
            row_text.append(str(models[model_names[1]][metric]) if models[model_names[1]][metric] is not None else "Null")
        text += ",".join(row_text) + "\n"
    with open(file_path, 'w') as f:
        f.write(text)

def write_individual_model_log(search_output, file_path):
    text = ""
    text = "Time: " + str(search_output['time']) + "\n\n"
    text += "Decision Variable: " + str(search_output['decision_variables']) + "\n\n"
    text += "Tflite Accuracy: " + str(search_output['tflite_accuracy']) + "\n\n"
    text += "Classification Report: \n" + search_output['classification_report'] + "\n\n"
    text += "Architecture: \n" + search_output['model_architecture'] + "\n\n"
    
    print(text, file=open(file_path, "w"))

def plot_losses(log, model_names, file_path):
    def plot_loss(ax, hist, model_name):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.plot(hist['train_losses'], label="train_losses")
        ax.plot(hist['val_losses'], label="val_losses")
        ax.set_title(model_name)
        ax.legend()
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    plot_loss(ax[0], log[model_names[0]], model_names[0])
    plot_loss(ax[1], log[model_names[1]], model_names[1])
    
    plt.savefig(file_path)
    plt.close()

def plot_confusion_matrix(cm, file_path):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('TFLite Model Confusion Matrix')
    plt.savefig(file_path)
    plt.close()

def visualize_dataset_sample(dataset, fold_num, num_images=9):
    plt.figure(figsize=(10, 10))
    # Take one batch from the dataset
    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            # Rescale back to 0-255 for display if necessary, 
            # but your make_dataset scales to [0,1], which plt handles fine.
            plt.imshow(images[i].numpy())
            # Convert one-hot back to index for display
            plt.title(f"Class: {np.argmax(labels[i])}")
            plt.axis("off")
    
    plt.suptitle(f"Sample from Fold {fold_num} Training Set")
    plt.show()

def test_tflite_model(tflite_model_file, test_ds):
    interpreter = tf.lite.Interpreter(tflite_model_file)
    interpreter.allocate_tensors()

    output_details = interpreter.get_output_details()[0]  # Model has single output.
    input_details = interpreter.get_input_details()[0]  # Model has single input.

    all_true_labels = []
    all_predicted_labels = []

    for image, label in test_ds:
        # Quantization handling
        if input_details['dtype'] == tf.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            image = image / input_scale + input_zero_point
            input_data = tf.dtypes.cast(image, tf.uint8)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        
        # Get prediction
        predicted_label = interpreter.get_tensor(output_details['index']).argmax()
        true_label = label.numpy().argmax()

        all_true_labels.append(true_label)
        all_predicted_labels.append(predicted_label)

    # --- Metrics Calculation using sklearn ---
    
    # Calculate precision, recall, and f1 (macro averaged for multi-class)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, 
        all_predicted_labels, 
        average='macro'
    )
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    
    result_metrices = {
        "tflite_acc": np.around(accuracy, decimals=3),
        "tflite_precision": np.around(precision, decimals=3),
        "tflite_recall": np.around(recall, decimals=3),
        "tflite_f1": np.around(f1, decimals=3),
    }
    
    # Generate the full text report
    report = classification_report(all_true_labels, all_predicted_labels)
    
    # Generate the confusion matrix
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    
    return result_metrices, report, cm

def main():
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))

    if gpu_devices:
        print("Found GPU(s):", gpu_devices)
    else:
        print("No GPU devices found. TensorFlow is likely using the CPU.")
        
        
    all_experiment_dir = Path('Experiments_KFolds')
    all_experiment_dir.mkdir(exist_ok=True)
        
    ## Experiment Settings
    # STM32 settings
    stm32_path = Path("stm32tflm.exe")
    peak_RAM_upper_bound = 40960
    Flash_upper_bound = 131072
    MACC_upper_bound = 2730000
    input_shape = (50,50,3)
    
    # dataset settings
    batch_size = 128
    
    # Models
    model_list = {'vanillaNAS_dense' : Vanilla_NAS, 
                    'JimmyNAS_I_fullyCNN' : Jimmy_NAS_I}
    
    
    datasets_dir = Path("Datasets")
    for data_dir in datasets_dir.iterdir():
        if data_dir.is_dir() and data_dir.name == "Flowers-4":
            experiment_dir = all_experiment_dir / data_dir.name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            model_log_file = experiment_dir / "models_log.csv"
            model_log = {}
            
            # 1. LOAD DATA USING YOUR KFOLD FUNCTION
            folds, test_ds, num_classes = load_data_kfold(data_dir, input_shape=input_shape, batch_size=batch_size, k_folds=5)

            # 2. ITERATE THROUGH FOLDS
            for fold_idx, (train_ds, validation_ds) in enumerate(folds):
                fold_dir = experiment_dir / f"fold_{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"\n--- Starting Fold {fold_idx + 1} ---")
                # # Visualize the current fold's data
                # visualize_dataset_sample(train_ds, fold_idx)
                
                model_log["fold_" + str(fold_idx)] = {}
                temp_log = model_log["fold_" + str(fold_idx)]
                
                data = {
                    "train": train_ds,
                    "validation": validation_ds,
                    "test": test_ds,
                }
                        
                for model_name, ModelClass in model_list.items():
                    model_dir = fold_dir / model_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    
                    print(f"\n--- Starting {model_name} ---")
                    
                    
                    temp_log[model_name] = {
                        "param_count":None,
                        "train_losses":None,
                        "val_losses":None,
                        "best_acc":None,
                        "mac_count":None,
                        "flash":None,
                        "peak_ram":None,
                        "tflite_acc":None,
                        "tflite_precision":None,
                        "tflite_recall":None,
                        "tflite_f1":None,
                    }
                    
                    nas = ColabNAS(peak_RAM_upper_bound, Flash_upper_bound, MACC_upper_bound, data, num_classes, input_shape, model_dir)
                    search_output = nas.search(ModelClass)
                    
                    if search_output is None:
                        continue

                    temp_log[model_name]["param_count"] = search_output["param_count"]
                    temp_log[model_name]["train_losses"] = search_output["train_losses"]
                    temp_log[model_name]["val_losses"] = search_output["val_losses"]
                    temp_log[model_name]["best_acc"] = search_output["val_accuracy"]
                    temp_log[model_name]["mac_count"] = search_output["MACC"]
                    temp_log[model_name]["flash"] = search_output["Flash"]
                    temp_log[model_name]["peak_ram"] = search_output["RAM"]
                    
                    # tflite model
                    tflite_model_file = search_output["path_to_best_architecture"]
                    result_metrices, report, cm = test_tflite_model(str(tflite_model_file), test_ds)
                    temp_log[model_name].update(result_metrices)
                    search_output['classification_report'] = report
                    
                    
                    write_individual_model_log(search_output, model_dir / "model_log.txt")
                    
                    plot_confusion_matrix(cm, model_dir / "cm.png")
                    
                # plot loss
                plot_losses(temp_log, list(model_list.keys()), fold_dir / f"loss.png")
                            
            write_compare_model_log(model_log, model_log_file, list(model_list.keys()))    

if __name__ == "__main__":
    main()
