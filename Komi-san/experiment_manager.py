from ColabNAS import ColabNAS
from pathlib import Path
import matplotlib.pyplot as plt

all_experiment_dir = Path('Experiments')
all_experiment_dir.mkdir(exist_ok=True)

class Manager:
    def __init__(self, 
                 train_ds,
                 validation_ds,
                 num_classes,
                 experiment_dir,
                 peak_RAM_upper_bound = 40960, 
                 Flash_upper_bound = 131072, 
                 MACC_upper_bound = 2730000, 
                 input_shape = (50,50,3), 
                 experiment_name = ''
                 ):
        self.peak_RAM_upper_bound = peak_RAM_upper_bound
        self.Flash_upper_bound = Flash_upper_bound
        self.MACC_upper_bound = MACC_upper_bound
        self.train_ds = train_ds
        self.validation_ds = validation_ds
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        
    def create_experiment_dir(self):
        self.experiment_dir = self.experiment_dir / Path(self.experiment_name)
        try:
            self.experiment_dir.mkdir(exist_ok=False)
        except FileExistsError:
            raise FileExistsError(f"Experiment directory '{self.experiment_dir}' already exists. Please choose a different experiment name.")
        
    def setup_nas(self):
        self.create_experiment_dir()
        nas = ColabNAS(self.peak_RAM_upper_bound, self.Flash_upper_bound, self.MACC_upper_bound, self.train_ds, self.validation_ds, self.num_classes, self.input_shape, self.experiment_dir)
        return nas
    
    def visualize(self, search_output):
        # Plot Section
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].set_title("Loss")
        ax[0].plot(search_output['val_losses'], label="Validation Loss")
        ax[0].plot(search_output['train_losses'], label="Training Loss")
        
        ax[1].set_title("Iterations Validation Accuracy")
        ax[1].plot(search_output['iterations_accuracy'])
        
        plt.savefig(self.experiment_dir / "loss.png")
        
        # Write text file Section
        
        text = "Time: " + str(search_output['time']) + "\n\n"
        text += "Peak RAM: " + str(search_output['RAM']) + "\n\n"
        text += "Flash: " + str(search_output['Flash']) + "\n\n"
        text += "MACC: " + str(search_output['MACC']) + "\n\n"
        text += "Model Validation Accuracy: " + str(search_output['val_accuracy']) + "\n\n"
        text += "Decision Variable: " + str(search_output['decision_variables']) + "\n\n"
        text += "tflite Validation Accuracy: " + str(search_output['tflite_accuracy']) + "\n\n\n\n\n"
        text += "Architecture: " + str(search_output['model_architecture']) + "\n\n"
        
        print(text, file=open(self.experiment_dir / "results.txt", "w"))
        
        