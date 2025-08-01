import utils
import utils.path_manager
import utils.view
import utils.view.plot as uplt
from utils.preprocessing import *
from utils.image_metrics import calculate_all_metrics
import tensorflow as tf
import tensorflow.image

import numpy as np

import matplotlib.pyplot as plt

PathManager = utils.path_manager.PathManager(dataset_path='/opt/datasets')

# Imagem de teste
img, label= utils.view.plot.return_random_image_from_PKLot(dataset_path=PathManager.get_dataset_path(), label="Occupied")
img = tf.convert_to_tensor(img, dtype=tf.float32)
img = tf.image.resize(img, (64, 64))
img = img.numpy()

#Aplicar os efeitos:
img_noise = apply_gaussian_noise(img)
img_rain = apply_rain(img)
img_contrast = apply_contrast(img)
img_defocus = apply_defocus(img)
img_rbc = apply_brightness_contrast(img)
img_upside_down = apply_rotate_upside_down(img)

plt.figure(figsize=(18,6))
plt.subplot(1, 7, 1)
plt.imshow(img)
plt.title("Imagem Original")
metrics_str = '\n'.join([f'{k} : {v:.3f}' for k, v in calculate_all_metrics(img, img).items()])
plt.xlabel(metrics_str)

plt.subplot(1, 7, 2)
plt.imshow(img_noise)
plt.title("Noise")
metrics_str = '\n'.join([f'{k} : {v:.3f}' for k, v in calculate_all_metrics(img, img_noise).items()])
plt.xlabel(metrics_str)

plt.subplot(1, 7, 3)
plt.imshow(img_rain)
plt.title("Rain")
metrics_str = '\n'.join([f'{k} : {v:.3f}' for k, v in calculate_all_metrics(img, img_rain).items()])
plt.xlabel(metrics_str)

plt.subplot(1, 7, 4)
plt.imshow(img_contrast)
plt.title("Contrast")
metrics_str = '\n'.join([f'{k} : {v:.3f}' for k, v in calculate_all_metrics(img, img_contrast).items()])
plt.xlabel(metrics_str)

plt.subplot(1, 7, 5)
plt.imshow(img_defocus)
plt.title("Defocus")
metrics_str = '\n'.join([f'{k} : {v:.3f}' for k, v in calculate_all_metrics(img, img_defocus).items()])
plt.xlabel(metrics_str)

plt.subplot(1, 7, 6)
plt.imshow(img_rbc)
plt.title("RBC")
metrics_str = '\n'.join([f'{k} : {v:.3f}' for k, v in calculate_all_metrics(img, img_rbc).items()])
plt.xlabel(metrics_str)

plt.subplot(1, 7, 7)
plt.imshow(img_upside_down)
plt.title("Upside Down")
metrics_str = '\n'.join([f'{k} : {v:.3f}' for k, v in calculate_all_metrics(img, img_upside_down).items()])
plt.xlabel(metrics_str)
plt.savefig("Images/Metrics.png")