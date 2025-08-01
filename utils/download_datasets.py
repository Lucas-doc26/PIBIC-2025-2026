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
            self.exclude_dataset_files()

    def download_and_extract(self, url):
        """
        Função para baixar e extrair um dataset.tar.gz 
        """
        name = url.split('/')[-1]

        dest_dir = self.path_manager.get_dataset_path()
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, name)

        print(f"Baixando arquivo {name} em {dest_dir}...")
        urllib.request.urlretrieve(url, dest_path)

        print(f"Extraindo arquivo {name}...")

        folder_name = name.replace('.tar.gz', '').replace('.zip', '')
        if tarfile.is_tarfile(dest_path):
            with tarfile.open(dest_path, 'r:*') as tar:
                tar.extractall(path=os.path.join(self.path_manager.get_dataset_path()))
        else:
            with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(path=os.path.join(self.path_manager.get_dataset_path(), folder_name))

        print(f"Arquivo {dest_path} extraído com sucesso!")

        return dest_path

    def download_and_extract_kyoto(self):
        """
        Função para baixar e extrair o dataset Kyoto
        """
        # Criar pasta Kyoto dentro do caminho especificado
        path = self.path_manager.get_dataset_path()

        kyoto_path = Path(path) / "Kyoto"
        kyoto_path.mkdir(exist_ok=True)
        print(f"Pasta Kyoto criada em: {kyoto_path.absolute()}")
        
        # Baixar o arquivo ZIP
        url = "https://github.com/eizaburo-doi/kyoto_natim/archive/refs/heads/master.zip"
        print("Baixando arquivo ZIP...")
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Erro ao baixar o arquivo. Status code: {response.status_code}")
        
        # Salvar o arquivo ZIP temporariamente no caminho especificado
        zip_path = Path(path) / "kyoto_temp.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print(f"Arquivo ZIP salvo em: {zip_path}")
        
        # Extrair o arquivo ZIP
        print("Extraindo arquivo ZIP...")
        temp_extract_path = Path(path) / "temp_extract"
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
    
    def exclude_dataset_files(self):
        """
        Função para excluir arquivos ZIP na pasta do dataset
        """
        dataset_path = self.path_manager.get_dataset_path()
        #ZIP e TAR.GZ
        zip_files = list(Path(dataset_path).glob("*.zip"))
        tar_gz_files = list(Path(dataset_path).glob("*.tar.gz"))

        compressed_files = zip_files + tar_gz_files

        if not compressed_files:
            print("Nenhum arquivo ZIP ou TAR.GZ encontrado para excluir.")
            return

        for file in compressed_files:
            try:
                os.remove(file)
                print(f"Arquivo excluído: {file}")
            except Exception as e:
                print(f"Erro ao excluir {file}: {e}")