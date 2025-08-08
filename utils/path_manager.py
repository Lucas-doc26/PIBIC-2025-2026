import os
import pathlib
import random
import matplotlib.image as mpimg

class PathManager:
    def __init__(self, dataset_path=None):
        self.base_path = pathlib.Path(__file__).resolve().parent.parent #Pega o caminho inteiro
        self.models_path = os.path.join(self.base_path, 'Models')
        self.logs_path = os.path.join(self.base_path, 'Models', 'Logs')

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

    def get_logs_path(self, name:str):
        return os.path.join(self.models_path, name, 'Logs')

    def create_folders_for_model(self, name:str):
        dir = os.path.join(self.models_path, name)
        os.makedirs(dir, exist_ok=True)
        sub_dirs = [
            os.path.join(dir, 'Logs'),
            os.path.join(dir, 'Autoencoders', 'Structure'),
            os.path.join(dir, 'Autoencoders', 'Weights'),
            os.path.join(dir, 'Classifiers', 'Structure'),
            os.path.join(dir, 'Classifiers', 'Weights'),
            os.path.join(dir, 'Classifiers', 'Results', 'Numpy'),
            os.path.join(dir, 'Classifiers', 'Results', 'Accuracies', 'Table'),
            os.path.join(dir, 'Classifiers', 'Results', 'Accuracies', 'Graphics'),
            os.path.join(dir, 'Classifiers', 'Results', 'Accuracies', 'Confusion-Matrix'),
            os.path.join(dir, 'Images', 'Plots', 'Autoencoders'),
            os.path.join(dir, 'Images', 'Plots', 'Classifiers-errors'),
            os.path.join(dir, 'Images', 'Graphics', 'Histories'),
        ]

        for sub in sub_dirs:
            os.makedirs(sub, exist_ok=True)

        print(f"Estrutura para Modelo '{name}' criada com sucesso!")

    def return_paths_autoencoder(self, name:str='Mark01', type='Autoencoder', base:str=None, loss:str=None):
        if loss:
            autoencoder_name = f'{type}-{base}-{loss}'
        else:
            autoencoder_name = f'{type}-{base}'
        
        model = pathlib.Path(os.path.join(self.models_path, name, 'Autoencoders/Structure', type+'.keras'))
        weights = pathlib.Path(os.path.join(self.models_path, name, 'Autoencoders/Weights', autoencoder_name+'.weights.h5'))
        history = pathlib.Path(os.path.join(self.models_path, name, 'Images', 'Graphics', 'Histories'))
        plot = pathlib.Path(os.path.join(self.models_path, name, 'Images', 'Plots', 'Autoencoders'))
        return model, weights, history, plot

    def return_paths_classifier(self, name:str='Mark01', type:str='Autoencoder',base:str=None):
        classifier_name = f'{type}-{base}'
        
        model = pathlib.Path(os.path.join(self.models_path, name, 'Classifiers/Structure', type+'.keras'))
        weights = pathlib.Path(os.path.join(self.models_path, name, 'Classifiers/Weights', classifier_name+'.weights.h5'))
        history = pathlib.Path(os.path.join(self.models_path, name, 'Images', 'Graphics', 'Histories'))
        accs = pathlib.Path(os.path.join(self.models_path, name, 'Classifiers', 'Results', 'Accuracies'))
        return model, weights, history, accs

    def models_managment(self):
        pass
