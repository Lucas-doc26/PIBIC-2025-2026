import random
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from utils.preprocessing import normalize_img
from utils.image_metrics import calculate_ssim

import tensorflow as tf
import tensorflow.image

def return_random_image_from_PKLot(dataset_path, label, SEED=42):
    """
    Retorna uma imagem aleatória do dataset PKLot para testes
    """
    random.seed(SEED)
    path_imgs = os.path.join(dataset_path, 'PKLot/PKLotSegmented/PUC/Sunny/2012-09-11', label)
    target_img = random.sample(os.listdir(path_imgs), 1)
    img = mpimg.imread(os.path.join(path_imgs, target_img[0]))
    return img/255., label

def plot_img(img, label=None, savepath:Path=None):
    plt.cla()
    plt.imshow(img)
    plt.axis("off")
    if label is not None:
        plt.title(label)
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

def plot_autoencoder(x_test, Autoencoder, width=128, height=128, caminho_para_salvar=None, nome_autoencoder='Kyoto'):
    def normalize(image):
        image = np.clip(image, 0, 1)  # Garante que a imagem esteja no intervalo [0, 1]
        return (image - image.min()) / (image.max() - image.min()) if image.max() != image.min() else image

    plt.figure(figsize=(16, 8))
    for i in range(8):
        # Imagem original
        plt.subplot(2, 8, i + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        plt.axis("off")

        # Predição e normalização
        pred = Autoencoder.predict(x_test[i].reshape((1,width, height,3)))
        pred_img = normalize(pred[0])

        plt.subplot(2, 8, i + 8 + 1)
        plt.imshow(pred_img)

        del pred_img, pred

        plt.title(f"{calculate_ssim(x_test[i], pred_img)}")
        plt.axis("off")
    
    if caminho_para_salvar != None:
        save_path = os.path.join(caminho_para_salvar, f'Autoencoder-{nome_autoencoder}.png')
        plt.savefig(save_path)
    
    plt.show()

def plot_autoencoder_with_ssim(x_test, autoencoder, width=128, height=128, save_path=None):
    n = min(8, len(x_test))
    # preditos em batch
    batch = x_test[:n].reshape((n, width, height, 3))
    preds = autoencoder.predict(batch, verbose=0)

    plt.figure(figsize=(16, 6))
    for i in range(n):
        # original
        plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        plt.axis("off")

        # reconstrução
        recon = np.clip(preds[i], 0, 1)
        plt.subplot(2, n, n + i + 1)
        plt.imshow(recon)
        # título com SSIM
        s = calculate_ssim(
            tf.convert_to_tensor(x_test[i][None], dtype=tf.float32),
            tf.convert_to_tensor(recon[None], dtype=tf.float32)
        ).numpy()
        plt.title(f"SSIM: {s}")
        plt.axis("off")

    plt.tight_layout()
    if save_path != None:
        plt.savefig(save_path)
    #plt.show()
