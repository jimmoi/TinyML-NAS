from ColabNAS import ColabNAS
from pathlib import Path

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
        self.experiment_dir.mkdir(exist_ok=True)
        
    def setup_nas(self):
        self.create_experiment_dir()
        nas = ColabNAS(self.peak_RAM_upper_bound, self.Flash_upper_bound, self.MACC_upper_bound, self.path_to_training_set, self.val_split, self.cache, self.input_shape, save_path=self.experiment_dir)
        return nas