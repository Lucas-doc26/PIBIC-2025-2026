import utils.path_manager as pm
import pandas as pd
import os
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

map_classes = {"Occupied": 1, "Empty": 0}

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
        PKLotDataset(self.dataset_path).create_csv_PKLot()
        shuffle_dataset(self.csv_dir)


    def _return_days(self, days, dict_days_values, n_images, n_days): 

        days_selected = days[:n_days]

        base_images_per_day_total = n_images // n_days  # total de imagens por dia (soma das 2 classes)
        # Garante que seja par (para dividir igualmente)
        if base_images_per_day_total % 2 != 0:
            base_images_per_day_total -= 1

        remainder = n_images - (base_images_per_day_total * n_days)  # quantas imagens sobraram

        while True:
            adjust = False
            days_selected = days[:n_days]

            # Distribui o excesso em múltiplos de 2 (pra manter par) para os primeiros dias
            images_per_day_total_list = [
                base_images_per_day_total + 2 if i < remainder // 2 else base_images_per_day_total 
                for i in range(n_days)
            ]

            # Agora verifica se cada dia tem pelo menos metade das imagens disponíveis por classe
            for day, total_images_day in zip(days_selected, images_per_day_total_list):
                half = total_images_day // 2
                values = dict_days_values[day]  
                if values[0] < half or values[1] < half:
                    n_days += 1  # tenta pegar mais um dia para suprir a demanda
                    adjust = True
                    break

            if not adjust:
                break

        days_left = sorted(list(set(days) - set(days_selected)))
        
        return days_selected, days_left, images_per_day_total_list

    def _create_df_per_days(self, df, days, images_per_day=None):
        df_final = pd.DataFrame(columns=['path_image', 'class'])

        if images_per_day is None:
            for day in days:
                df_day = df[df['day'] == day]
                df_final = pd.concat([df_final, df_day], ignore_index=True)
        else:
            for day, n_images in zip(days, images_per_day):
                df_day = df[df['day'] == day]
                empty = df_day[df_day['class'] == 0]
                occupied = df_day[df_day['class'] == 1]

                half = n_images // 2
                half_plus = half + (n_images % 2)  # caso seja ímpar, a classe 0 recebe a imagem a mais

                empty_sample = empty.sample(n=min(half_plus, len(empty)), random_state=SEED, replace=(len(empty) < half_plus))
                occupied_sample = occupied.sample(n=min(half, len(occupied)), random_state=SEED, replace=(len(occupied) < half))

                df_final = pd.concat([df_final, empty_sample, occupied_sample], ignore_index=True)

        return df_final

    def split_train_valid_test(self, df, n_days_train, n_days_valid, n_images_train, n_images_valid):
        days = sorted(df['day'].unique().tolist())

        list_df_per_day = [df[df['day'] == day] for day in days]
        number_of_images_per_day = {
            #pegando o dia
            day_df['day'].iloc[0]: (
                len(day_df[day_df['class'] == 0]), #conta a quantidade de cada classe
                len(day_df[day_df['class'] == 1])
            )
            for day_df in list_df_per_day
        } #retorno um dic -> 2023-10-19: (678, 123), 2023-10-20: (876, 456)

        days_to_train, days_left, images_per_class_train = self._return_days(days, number_of_images_per_day, n_images_train, n_days_train)

        days_to_valid, days_to_test, images_per_class_valid = self._return_days(days_left, number_of_images_per_day, n_images_valid, n_days_valid)

        df_train = self._create_df_per_days(df, days_to_train, images_per_class_train)
        df_valid = self._create_df_per_days(df, days_to_valid, images_per_class_valid)
        df_test = self._create_df_per_days(df, days_to_test, None)

        return df_train, df_valid, df_test

class CNRDataset(DatasetManager):
    def __init__(self, dataset_path=None):
        super().__init__(dataset_path)
        self.dataset_name = "CNR-EXT-Patches-150x150"

    def create_csv_cnr(self, days_train=5, days_valid=1, n_images_train=1024, n_images_valid=64):
        """
        Cria um arquivo CSV com os caminhos das imagens e seus rótulos
        """
        cnr_dir = os.path.join(self.path_manager.get_dataset_path(), self.dataset_name)
        all_cnr = os.path.join(cnr_dir, 'LABELS', 'all.txt')

        df = pd.read_csv(all_cnr, sep=' ', header=None, names=['path_image', 'class'])

        path_split = df["path_image"].str.split("/", expand=True)

        df["weather"] = path_split[0]
        df["day"] = path_split[1]
        df["camera"] = path_split[2]

        self.create_cameras(df)
        train, valid, test = self.split_train_valid_test(df, days_train, days_valid, n_images_train, n_images_valid)

        df.to_csv(os.path.join(self.csv_dir,'CNR/CNR.csv'), index=False, columns=["path_image", "class"])
        train.to_csv(os.path.join(self.csv_dir,'CNR/CNR_train.csv'), index=False, columns=["path_image", "class"])
        valid.to_csv(os.path.join(self.csv_dir,'CNR/CNR_valid.csv'), index=False, columns=["path_image", "class"])
        test.to_csv(os.path.join(self.csv_dir,'CNR/CNR_test.csv'), index=False, columns=["path_image", "class"])

    def create_cameras(self, df, days_train=5, days_valid=1, n_images_train=1024, n_images_valid=64):
        """
        Cria o csv para cada câmera
        """
        cameras = df["camera"].unique().tolist()
        for cam in sorted(cameras):
            df_cam = df[df["camera"] == cam]

            train, valid, test = self.split_train_valid_test(df, days_train, days_valid, n_images_train, n_images_valid)
            train.to_csv(os.path.join(self.csv_dir,f"{cam}", f"{cam}_train.csv"), index=False, columns=["path_image", "class"])
            valid.to_csv(os.path.join(self.csv_dir,f"{cam}", f"{cam}_valid.csv"), index=False, columns=["path_image", "class"])
            test.to_csv(os.path.join(self.csv_dir,f"{cam}", f"{cam}_test.csv"), index=False, columns=["path_image", "class"])

            df_cam.to_csv(os.path.join(self.csv_dir,f"{cam}", f"{cam}.csv"), index=False, columns=["path_image", "class"])

class PKLotDataset(DatasetManager):
    def __init__(self, dataset_path=None):
        super().__init__(dataset_path)
        self.dataset_name = 'PKLot/PKLotSegmented'
        self.universities = ["PUC", "UFPR04", "UFPR05"]
        self.weathers = ["Cloudy", "Sunny", "Rainy"]

    def create_csv_PKLot(self, days_train=5, days_valid=1, n_images_train=1024, n_images_valid=64):
        """
        Cria o arquivo CSV da PKLot
        """
        data = []
        for university in self.universities:
            for weather in self.weathers:
                days_dir = os.path.join(self.dataset_path, self.dataset_name, university, weather)
                for day in os.listdir(days_dir):
                    day_dir = os.path.join(days_dir, day)
                    for label in ['Empty', 'Occupied']:
                        dir_imgs = os.path.join(day_dir, label)
                        if os.path.isdir(dir_imgs):
                            path_imgs = os.listdir(dir_imgs)
                            data.extend([[university, weather, day, os.path.join(day_dir, label, img), label] 
                                         for img in path_imgs])
        df = pd.DataFrame(data=data, columns=['university', 'weather', 'day', 'path_image', 'class'])
        df['class'] = df['class'].map(map_classes)

        self.create_universities(df)
        train, valid, test = self.split_train_valid_test(df, days_train, days_valid, n_images_train, n_images_valid)

        df.to_csv(os.path.join(self.csv_dir,'PKLot/PKLot.csv'), index=False, columns=["path_image", "class"])
        train.to_csv(os.path.join(self.csv_dir,'PKLot/PKLot_train.csv'), index=False, columns=["path_image", "class"])
        valid.to_csv(os.path.join(self.csv_dir,'PKLot/PKLot_valid.csv'), index=False, columns=["path_image", "class"])
        test.to_csv(os.path.join(self.csv_dir,'PKLot/PKLot_test.csv'), index=False, columns=["path_image", "class"])

    def create_universities(self, df, days_train=5, days_valid=1, n_images_train=1024, n_images_valid=64):
        """
        Cria o arquivo csv de cada uma das faculdades
        """
        for university in self.universities:
            df_university = df[df["university"] == university]
            train, valid, test = self.split_train_valid_test(df, days_train, days_valid, n_images_train, n_images_valid)

            df_university.to_csv(os.path.join(self.csv_dir, f"{university}/{university}.csv" ), columns=['path_image', 'class'], index=False)
            train.to_csv(os.path.join(self.csv_dir,f"{university}/{university}_train.csv"), index=False, columns=["path_image", "class"])
            valid.to_csv(os.path.join(self.csv_dir,f"{university}/{university}_valid.csv"), index=False, columns=["path_image", "class"])
            test.to_csv(os.path.join(self.csv_dir,f"{university}/{university}_test.csv"), index=False, columns=["path_image", "class"])


class KyotoDataset(DatasetManager):
    def __init__(self, dataset_path=None):
        super().__init__(dataset_path)

def shuffle_dataset(path_datasets):
    for root, dir, csv in os.walk(path_datasets):
         for fname in csv:
            if fname.lower().endswith('.csv'):
                csv_path = os.path.join(root, fname)
                # lê, embaralha, reseta índice e grava de volta sem índice extra
                df = pd.read_csv(csv_path)
                df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
                df_shuffled.to_csv(csv_path, index=False)
                print(f'Embaralhado: {csv_path} ({len(df_shuffled)} linhas)')