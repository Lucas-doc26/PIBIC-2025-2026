import tensorflow as tf
import tensorflow.image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import albumentations as A
import cv2
import pandas as pd

def normalize_img(img):
    return img/255.

def preprocessing_dataframe(path_csv: str, autoencoder: bool = False, data_algumentantation:bool = True, input_shape:int=(64,64)):
    """
    Ao passar um dataFrame .csv, ele irá retornar o gerador e dataframe
    
    Parâmetros:
        caminho (str): Caminho para o arquivo CSV.
        autoencoder (bool): Se True, prepara os dados para um autoencoder (class_mode='input').
                            Se False, prepara os dados para classificação binária (class_mode='binary').
        data_algumentation (bool): Se True, faz o aumento dos dados .

    Retorna:
        Gerador, dataframe
    """

    dataframe = pd.read_csv(path_csv)
    batch_size = 16

    datagen = ImageDataGenerator(preprocessing_function=albumentations_tf if data_algumentantation else normalize_img)

    if len(dataframe.columns) > 1:
        dataframe['class'] = dataframe['class'].astype(str)

    
    class_mode = None if autoencoder else 'sparse'

    generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='path_image',
        y_col=None if autoencoder else 'class',
        target_size=(input_shape),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )

    print("Imagens totais:", generator.samples)
    
    return generator, dataframe

def albumentations_tf(img, transform):
    if isinstance(img, tf.Tensor): img = img.numpy()
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    out = transform(image=img)['image']
    return out.astype(np.float32) / 255.0

def apply_rain(img):
    # garante uint8 pra Albumentations funcionar
    img_up = (img * 255).astype(np.uint8) if img.dtype!=np.uint8 else img
    img_up = cv2.resize(img_up, (256, 256), cv2.INTER_LINEAR)
    rain = A.RandomRain(
            slant_range=(-10, 10),       # ângulo aleatório entre -10° e +10°
            drop_length=10,
            drop_width=3,
            drop_color=(50, 50, 50),
            blur_value=5,
            brightness_coefficient=0.8
        )    
    img_rain = rain(image=img_up)['image']
    img_down = cv2.resize(img_rain, (64, 64), cv2.INTER_AREA)
    return img_down.astype(np.float32) / 255.0

def apply_contrast(img):
    contrast = A.AutoContrast(cutoff=0, method="cdf")
    return albumentations_tf(img, contrast)

def apply_defocus(img):
    defocus = A.Defocus(radius=[3,10], alias_blur=[0.1,0.5])
    return albumentations_tf(img, defocus)

def apply_brightness_contrast(img):
    rbc = A.RandomBrightnessContrast(brightness_limit=[-0.8,0], contrast_limit=[-0.3,0.2])
    return albumentations_tf(img, rbc)

def apply_rotate_upside_down(img):
    upside = A.Rotate(limit=[180,180], border_mode=cv2.BORDER_REPLICATE, fill=0)
    return albumentations_tf(img, upside)

def apply_gaussian_noise(img, mean=0.0, std=0.1):
    img_f = img.astype(np.float32)
    noise = np.random.normal(mean, std, img_f.shape)
    noisy = np.clip(img_f + noise, 0, 1)
    return noisy