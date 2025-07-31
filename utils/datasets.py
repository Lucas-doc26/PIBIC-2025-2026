import utils.path_manager as pm
import pandas as pd
import os
import random

SEED = 42
random.seed(SEED)

class DatasetManager:
    def __init__(self, dataset_path=None):
        self.path_manager = pm.PathManager(dataset_path)
        self.dataset_path = self.path_manager.get_dataset_path()
        self.csv_dir = os.path.join(self.path_manager.get_base_path(), 'CSV')
        os.makedirs(self.csv_dir, exist_ok=True)
        self.create_important_directories()

    def create_important_directories(self, 
        directories=['PKLot', 'Kyoto', 'CNR', 
                     'PUC', 'UFPR04', 'UFPR05', 
                     'camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']):
        """
        Cria diretórios importantes para o gerenciamento dos datasets
        """
        for dir in directories:
            dir_path = os.path.join(self.csv_dir, dir)
            os.makedirs(dir_path, exist_ok=True)

    def create_csvs(self):
        """
        Cria arquivos CSV com os caminhos das imagens e seus rótulos
        """
        CNRDataset(self.dataset_path).create_csv_cnr()

class CNRDataset(DatasetManager):
    def __init__(self, dataset_path=None):
        super().__init__(dataset_path)
        self.dataset_name = "CNR-EXT-Patches-150x150"

    def create_csv_cnr(self):
        """
        Cria um arquivo CSV com os caminhos das imagens e seus rótulos
        """
        cnr_dir = os.path.join(self.path_manager.get_dataset_path(), self.dataset_name)
        all_cnr = os.path.join(cnr_dir, 'LABELS', 'all.txt')

        df = pd.read_csv(all_cnr, sep=' ', header=None, names=['path_image', 'class'])

        path_split = df["path_image"].str.split("/", expand=True)

        df["weather"] = path_split[0]
        df["date"] = path_split[1]
        df["camera"] = path_split[2]

        self.create_cameras(df)
        
        df.to_csv(os.path.join(self.csv_dir,'CNR/CNR.csv'), index=False, columns=["path_image", "class"])

    def create_cameras(self, df):
        """
        Cria o csv para cada câmera
        """
        cameras = df["camera"].unique().tolist()
        for cam in cameras:
            df_cam = df[df["camera"] == cam]
            df_cam.to_csv(os.path.join(self.csv_dir,f"{cam}", f"{cam}.csv"), index=False, columns=["path_image", "class"])

