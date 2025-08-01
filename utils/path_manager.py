import os
import pathlib
import random
import matplotlib.image as mpimg

class PathManager:
    def __init__(self, dataset_path=None):
        self.base_path = pathlib.Path(__file__).resolve().parent.parent #Pega o caminho inteiro
        self.images_path = os.path.join(self.base_path, 'Images')
        self.models_path = os.path.join(self.base_path, 'Models')

        if dataset_path is None:
            self.dataset_path = os.path.join(self.base_path, 'dataset')
        else:
            self.dataset_path = pathlib.Path(dataset_path).resolve()
        
        self.create_important_directories()

    def get_base_path(self):
        return str(self.base_path)
    
    def get_images_path(self, dirs=False):
        if dirs == False:
            return str(self.images_path)
        else:
            for root, dirs, files in os.walk(self.images_path):
                for dir in dirs:
                    print(os.path.join(root, dir))
    
    def get_dataset_path(self):
        return str(self.dataset_path)
    
    def create_important_directories(self):
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

    def models_managment(self):
        pass
    