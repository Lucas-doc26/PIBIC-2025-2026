import os
import pathlib

class PathManager:
    def __init__(self, dataset_path=None):
        self.base_path = pathlib.Path(__file__).resolve().parent.parent
        if dataset_path is None:
            self.dataset_path = os.path.join(self.base_path, 'dataset')
        else:
            self.dataset_path = pathlib.Path(dataset_path).resolve()
        
        self.create_important_directories()

    def get_base_path(self):
        return str(self.base_path)
    
    def get_dataset_path(self):
        return str(self.dataset_path)
    
    def create_important_directories(self):
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
    