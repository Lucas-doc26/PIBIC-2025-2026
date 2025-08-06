import os
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from keras import ops, layers
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tensorflow.image as tf_img
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint


from utils.preprocessing import *
from utils.config import *
from utils.view.plot import plot_autoencoder_with_ssim
from utils.loss_functions import MyLoss

set_seeds()
config_gpu()

#No VAE, ele gera dois vetores latentes: 
# z_mean = média gaussiana latente
# z_log_var = log da variância de z_mean
# Só que a gente quer amostrar um vetor z dessa distribuição pra cada input, mas de forma que dê pra fazer backpropagation (senão o gradiente trava).
#Por isso o sampling existe: deixar z derivável, ou seja, tem que ser reparametrizado
class Sampling(layers.Layer):
    """Usa (z_mean, z_log_var) para criar o z, o vetor que será codificado."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator) # rúido
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon #reparametrização(determinística + diferenciável) -> backpropagation mesmo usando amostragem

latent_dim = 256
encoder_inputs = keras.Input(shape=(128, 128, 3))
input_label = keras.Input(shape=(1,))  # one-hot encode

x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)

x = layers.Flatten()(x)
x = layers.Concatenate()([x, input_label])
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model([encoder_inputs, input_label], [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
decoder_label_input = keras.Input(shape=(1,))  # one-hot

x = layers.Concatenate()([latent_inputs, decoder_label_input])
x = layers.Dense(32 * 32 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((32, 32, 64))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model([latent_inputs, decoder_label_input], decoder_outputs, name="decoder")
decoder.summary()

class CCVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


    def call(self, inputs):
        x, y = inputs
        z_mean, z_log_var, z = self.encoder([x, y])
        reconstruction = self.decoder([z, y])
        return reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x, y = data
            z_mean, z_log_var, z = self.encoder([x, y])

            reconstruction = self.decoder([z, y])

            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(x, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1)) 
            total_loss = reconstruction_loss + kl_loss #loss final de tudo

        grads = tape.gradient(total_loss, self.trainable_weights)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        z_mean, z_log_var, z = self.encoder([x, y])

        reconstruction = self.decoder([z, y])

        reconstruction_loss = ops.mean(
            ops.sum(
                keras.losses.binary_crossentropy(x, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1)) 
        total_loss = reconstruction_loss + kl_loss #loss final de tudo

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
    
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

train, _ = preprocessing_dataframe("/home/lucas/PIBIC-2025-2026/CSV/PUC/PUC_train.csv", autoencoder=False, data_algumentantation=False, input_shape=(128, 128))
valid, _ = preprocessing_dataframe("/home/lucas/PIBIC-2025-2026/CSV/PUC/PUC_valid.csv", autoencoder=False, data_algumentantation=False, input_shape=(128, 128))
test, _ = preprocessing_dataframe("/home/lucas/PIBIC-2025-2026/CSV/PUC/PUC_train.csv", autoencoder=False, data_algumentantation=False, input_shape=(128, 128))

myloss = MyLoss()

ccvae = CCVAE(encoder, decoder)
ccvae.compile(optimizer=keras.optimizers.Adam(), loss=myloss)

ccvae.fit(train,epochs=20, steps_per_epoch=len(train), validation_data=valid, validation_steps=len(valid))
plot_autoencoder_with_ssim(test, ccvae, 128, 128, "Images/CCVAE-MyLoss-20.png")
