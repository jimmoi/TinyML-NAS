import tensorflow as tf
from pathlib import Path
from experiment_manager import all_experiment_dir
from my_util import *
from distill import *
import os
import pickle

def main():
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))

    if gpu_devices:
        print("Found GPU(s):", gpu_devices)
    else:
        print("No GPU devices found. TensorFlow is likely using the CPU.")
    
    datasets_dir = Path("Datasets")
    for data_dir in datasets_dir.iterdir() :
        if data_dir.is_dir(): 
            experiment_dir = all_experiment_dir / data_dir.name / "teacher_model"
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            train_path_dir = data_dir / "train"
            test_path_dir = data_dir / "test"
            
            checkpoint_path = experiment_dir / "best_teacher_weights.h5"

            input_shape = (50, 50, 3)
            epochs = 50
            
            ## load data
            num_classes = len(next(os.walk(train_path_dir))[1])
            train_ds, validation_ds = load_train_dataset_efficientnet(input_shape, train_path_dir, val_split=0.3, batch_size=128, cache=True)
            test_ds = load_test_dataset(input_shape, test_path_dir, batch_size=1, cache=True)
            
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
            
            lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2, 
                patience=5, 
                min_lr=0.00001
            )
            
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True
            )
            
            ## teacher model
            # model = build_wide_resnet(input_shape, depth=4, k=8, num_classes=num_classes)
            model = build_efficientnetb0(input_shape, num_classes)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()
            hist = model.fit(
                train_ds, 
                epochs=epochs, 
                validation_data=validation_ds,
                callbacks=[checkpoint_callback, lr_callback, early_stop]
            )
            
            model.load_weights(str(checkpoint_path))
            
            ## test model
            test_loss, test_acc = model.evaluate(test_ds)
            print(f"Test accuracy: {test_acc}")

            ## save teacher knowledge
            if test_acc > 0.9:
                model.save(experiment_dir / f"teacher_model_{epochs}.h5")
                save_teacher_knowledge(model, train_ds, experiment_dir / f"teacher_knowledge_train_{epochs}.npz")
                save_teacher_knowledge(model, test_ds, experiment_dir / f"teacher_knowledge_test_{epochs}.npz")

                ## save history
                with open(experiment_dir / f"history_{epochs}.pkl", "wb") as f:
                    pickle.dump(hist.history, f)
            else:
                print(f"Test accuracy: {test_acc} is not enough to save teacher knowledge")
                print("Stupid Fucking Human")


if __name__ == "__main__":
    main()
    
