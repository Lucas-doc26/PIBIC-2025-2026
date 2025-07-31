import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default=None, help="Caminho para download dos datasets")
args = parser.parse_args()

#Baixa os datasets necessários
PathManager = utils.PathManager(dataset_path=args.dataset_path)
Downloader = utils.download_datasets.Downloader(dataset_path=PathManager.get_dataset_path())
DatasetManager = utils.datasets.DatasetManager(dataset_path=PathManager.get_dataset_path())
DatasetManager.create_csvs()