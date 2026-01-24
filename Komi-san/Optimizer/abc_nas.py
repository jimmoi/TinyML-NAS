import abc

class ABC_NAS(abc.ABC):
    def __init__(self, evaluate_model_fnc, input_shape, num_classes, learning_rate):
        self.evaluate_model_fnc = evaluate_model_fnc # External fnc that returns {'max_val_acc': float}
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_name = ""
    
    @abc.abstractmethod
    def search(self):
        pass