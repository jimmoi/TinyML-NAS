from ColabNAS import ColabNAS
from pathlib import Path
import matplotlib.pyplot as plt

all_experiment_dir = Path('Experiments')
all_experiment_dir.mkdir(exist_ok=True)

class Manager:
    def __init__(self, 
                 peak_RAM_upper_bound = 40960, 
                 Flash_upper_bound = 131072, 
                 MACC_upper_bound = 2730000, 
                 path_to_training_set = '', 
                 val_split = 0.3, 
                 cache = True, 
                 input_shape = (50,50,3), 
                 experiment_name = ''
                 ):
        
        
        self.peak_RAM_upper_bound = peak_RAM_upper_bound
        self.Flash_upper_bound = Flash_upper_bound
        self.MACC_upper_bound = MACC_upper_bound
        self.path_to_training_set = path_to_training_set
        self.val_split = val_split
        self.cache = cache
        self.input_shape = input_shape
        self.experiment_name = experiment_name
        self.experiment_dir = None
        
        
    def create_experiment_dir(self):
        self.experiment_dir = all_experiment_dir / Path(self.experiment_name)
        try:
            self.experiment_dir.mkdir(exist_ok=False)
        except FileExistsError:
            print(f"Experiment directory '{self.experiment_name}' already exists. Please choose a different experiment name.")
            exit()
        
    def setup_nas(self):
        self.create_experiment_dir()
        nas = ColabNAS(self.peak_RAM_upper_bound, self.Flash_upper_bound, self.MACC_upper_bound, self.path_to_training_set, self.val_split, self.cache, self.input_shape, save_path=self.experiment_dir)
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
        
        