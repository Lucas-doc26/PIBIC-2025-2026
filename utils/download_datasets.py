import zipfile
import os
import tarfile
import urllib.request
import shutil
from pathlib import Path
import requests
from utils.path_manager import PathManager

class Downloader():
    def __init__(self, dataset_path=None, 
                 important_directories={'PKLot': "http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz", 
                                        'Kyoto': None, 
                                        'CNR-EXT-Patches-150x150' : "https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip"}):

        self.path_manager = PathManager(dataset_path)
        directories = [dir for dir in os.listdir(self.path_manager.get_dataset_path())]
        for dir in important_directories:
            if dir not in directories:
                print(f"Diretório {dir} não encontrado. Baixando...")
                if dir == 'Kyoto':
                    self.download_and_extract_kyoto()
                else:
                    self.download_and_extract(important_directories[dir])

        # deleta os ZIPs
        zip_files = list(Path(dataset_path).glob("*.zip")) + list(Path(dataset_path).glob("*.tar.gz"))
        if zip_files:
            self.exclude_zip_files()

    def download_and_extract(self, url):
        """
        Função para baixar e extrair um dataset.tar.gz 
        """
        name = url.split('/')[-1]

        print(f"Baixando arquivo {name}...")
        urllib.request.urlretrieve(url, name)

        print(f"Extraindo arquivo {name}...")
        with tarfile.open(name, 'r:gz') as tar:
            tar.extractall(path=self.path_manager.get_dataset_path())

        print(f"Arquivo {name} extraído com sucesso!")

        return name

    def download_and_extract_kyoto(self):
        """
        Função para baixar e extrair o dataset Kyoto 
        """

        kyoto_path = os.path.join(self.path_manager.get_dataset_path(), "Kyoto")

        url = "https://github.com/eizaburo-doi/kyoto_natim/archive/refs/heads/master.zip"
        print("Baixando arquivo ZIP...")

        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Erro ao baixar o arquivo. Status code: {response.status_code}")
        
        # Cria um arq temporário para o ZIP
        zip_path = os.path.join(self.path_manager.get_dataset_path(), "kyoto_temp.zip")
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print(f"Arquivo ZIP salvo em: {zip_path}")
        
        # Extrair o arquivo ZIP
        print("Extraindo arquivo ZIP...")
        temp_extract_path = os.path.join(self.path_manager.get_dataset_path(), "temp_extract")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)
        
        # Listar todos os arquivos extraídos para debug
        print("\nConteúdo extraído:")
        for root, dirs, files in os.walk(temp_extract_path):
            print(f"\nDiretório: {root}")
            for name in files:
                print(f"- {name}")
        
        # Tentar diferentes possíveis caminhos para a pasta thumb
        possible_paths = [
            temp_extract_path / "kyoto_natim-master/kyoto_natim-master/thumb",
            temp_extract_path / "kyoto_natim-master/thumb",
            temp_extract_path / "thumb"
        ]
        
        thumb_path = None
        for p in possible_paths:
            if p.exists():
                thumb_path = p
                print(f"\nPasta thumb encontrada em: {thumb_path}")
                break
        
        if not thumb_path:
            raise Exception("Pasta thumb não encontrada nos caminhos esperados")
        
        # Copiar todas as imagens para a pasta Kyoto
        print("\nCopiando imagens...")
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
        images_copied = 0
        
        for file in thumb_path.glob("*"):
            print(f"Encontrado arquivo: {file.name}")
            if file.suffix.lower() in image_extensions:
                shutil.copy2(file, kyoto_path)
                print(f"Copiado: {file.name}")
                images_copied += 1
        
        # Limpar arquivos temporários
        print("\nLimpando arquivos temporários...")
        os.remove(zip_path)
        shutil.rmtree(temp_extract_path)
        
        if images_copied == 0:
            print("\nNenhuma imagem foi encontrada para copiar!")
        else:
            print(f"\nProcesso concluído! {images_copied} imagens foram copiadas para a pasta Kyoto")
    
    def exclude_zip_files(self):
        """
        Função para excluir arquivos ZIP na pasta do dataset
        """
        dataset_path = self.path_manager.get_dataset_path()
        zip_files = list(Path(dataset_path).glob("*.zip"))
        
        if not zip_files:
            print("Nenhum arquivo ZIP encontrado para excluir.")
            return
        
        for zip_file in zip_files:
            try:
                os.remove(zip_file)
                print(f"Arquivo ZIP excluído: {zip_file}")
            except Exception as e:
                print(f"Erro ao excluir {zip_file}: {e}")