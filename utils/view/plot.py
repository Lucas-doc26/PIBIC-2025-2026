import random
import os
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, accuracy_score
from utils.preprocessing import normalize_img
from utils.image_metrics import calculate_ssim


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

    n = min(8, len(x_test))
    # preditos em batch
    batch = x_test[:n]#.reshape((n, width, height, 3))
    preds = Autoencoder.predict(batch, verbose=0)

    plt.figure(figsize=(16, 6))
    for i in range(8):
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
    
    if caminho_para_salvar != None:
        save_path = os.path.join(caminho_para_salvar, f'Autoencoder-{nome_autoencoder}.png')
        plt.savefig(save_path)
    
def plot_autoencoder_with_ssim(x_test, autoencoder, width=128, height=128, save_path=None):
    def normalize(image):
        image = np.clip(image, 0, 1)  # Garante que a imagem esteja no intervalo [0, 1]
        return (image - image.min()) / (image.max() - image.min()) if image.max() != image.min() else image

    item = next(x_test)

    # Verifica se o test está vindo com 
    if isinstance(item, tuple) and len(item) == 2:
        batch, label = item
        if np.array_equal(label[0], batch[0]):
            label = None
    else:
        batch = item
        label = None

    n = min(8, batch.shape[0])
    batch = batch[:n]

    plt.figure(figsize=(16, 8))
    for i in range(n):
        plt.subplot(2, 8, i + 1)
        plt.imshow(batch[i])
        plt.title("Original")
        plt.axis("off")

        if label is not None:
            pred = autoencoder.predict([batch[i].reshape((1, width, height, 3)), np.array([[label[i]]])])
        else:
            pred = autoencoder.predict(batch[i].reshape((1, width, height, 3)))

        pred_img = normalize(pred[0])

        plt.subplot(2, 8, i + 8 + 1)
        plt.imshow(pred_img)
        plt.title(f"SSIM: {calculate_ssim(batch[i], pred_img):.3f}")
        plt.axis("off")

        del pred_img, pred

    if save_path is not None:
        plt.savefig(save_path)

def plot_history(history, type='Classifier', save_fig=None):
    # Função auxiliar pra evitar erro com chave inexistente
    def get_hist(key):
        return history.history.get(key, [])

    if type == 'Classifier':
        loss = get_hist('loss')
        val_loss = get_hist('val_loss')
        accuracy = get_hist('accuracy')
        val_accuracy = get_hist('val_accuracy')
        epochs = range(len(loss))

        plt.figure(figsize=(15, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        plt.xticks(epochs)
        plt.legend()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label='accuracy')
        plt.plot(epochs, val_accuracy, label='val_accuracy')
        plt.title('Accuracy')
        plt.xticks(epochs)
        plt.legend()

    elif type == 'Autoencoder':
        loss = get_hist('loss')
        val_loss = get_hist('val_loss')
        epochs = range(len(loss))

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss, label='loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        plt.xticks(epochs)
        plt.legend()

    else:  # CVAE ou CCVAE
        loss = get_hist('loss')
        val_loss = get_hist('val_loss')
        kl_loss = get_hist('kl_loss')
        val_kl_loss = get_hist('val_kl_loss')
        reconstruction_loss = get_hist('reconstruction_loss')
        val_reconstruction_loss = get_hist('val_reconstruction_loss')
        epochs = range(len(loss))

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, loss, label='loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        plt.xticks(epochs)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, kl_loss, label='kl_loss')
        plt.plot(epochs, val_kl_loss, label='val_kl_loss')
        plt.title('KL Loss')
        plt.xticks(epochs)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, reconstruction_loss, label='reconstruction_loss')
        plt.plot(epochs, val_reconstruction_loss, label='val_reconstruction_loss')
        plt.title('Reconstruction Loss')
        plt.xticks(epochs)
        plt.legend()

    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')

    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels=['Empty', 'Occupied'], legend:str=None , save_path=None):
    """
    Plota uma matriz de confusão.

    Args:
        y_true: Array numpy com os rótulos verdadeiros
        y_pred: Array numpy com as previsões do modelo
        labels: Lista de rótulos das classes
        title: Título da figura (opcional)
        save_path: Caminho para salvar a figura (opcional)
    """
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    accuracy_text = f"Accuracy: {accuracy * 100:.2f}%"
    plt.title(f"{accuracy_text}")
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if legend:
        if isinstance(legend, str):
            legend = [legend]
        patches = [mpatches.Patch(color='lightblue', label=text) for text in legend]
        plt.legend(handles=patches, loc='lower right', fontsize=10, frameon=True)

    # Salvar ou exibir a figura
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()