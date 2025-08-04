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
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)

x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(32 * 32 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((32, 32, 64))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
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
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0] # se vier em x, y -> pego só o x

        # GradientesTape -> monitora todas as operações pra saber quem afeta quem.
        with tf.GradientTape() as tape:
            #z_mean = média gausiana latente
            #z_log_var - log da variância 
            #z = z_mean + eps * std
            z_mean, z_log_var, z = self.encoder(data)

            #passo z para reconstruir 
            reconstruction = self.decoder(z)


            #calcula o erro de reconstrução
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            #KL = força cada z a ficar perto de uma Gaussiana padrão N(0, I).
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1)) 
            total_loss = reconstruction_loss + kl_loss #loss final de tudo

        grads = tape.gradient(total_loss, self.trainable_weights)
        #total_loss: valor escalar da função de custo.
        #self.trainable_weights: todos os pesos treináveis do encoder + decoder.
        #grads: lista de tensores com o gradiente parcial pra cada peso.

        # att os pesos usando esses gradientes
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #aqui encima de fato aconteceu a backpropagation:
        # encoder -> decoder -> total_loss
        # gradientes -> aplica os gradientes        

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)

        reconstruction = self.decoder(z)

        reconstruction_loss = ops.mean(
            ops.sum(
                keras.losses.binary_crossentropy(data, reconstruction),
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

train, _ = preprocessing_dataframe("/home/lucas/PIBIC-2025-2026/CSV/PUC/PUC_train.csv", autoencoder=True, data_algumentantation=False, input_shape=(128, 128))
valid, _ = preprocessing_dataframe("/home/lucas/PIBIC-2025-2026/CSV/PUC/PUC_valid.csv", autoencoder=True, data_algumentantation=False, input_shape=(128, 128))
test, _ = preprocessing_dataframe("/home/lucas/PIBIC-2025-2026/CSV/PUC/PUC_train.csv", autoencoder=True, data_algumentantation=False, input_shape=(128, 128))

myloss = MyLoss()

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(), loss=myloss)
vae.fit(train, epochs=100, steps_per_epoch=len(train), validation_data=valid, validation_steps=len(valid))
plot_autoencoder_with_ssim(test, vae, 128, 128, "Images/VAE-MyLoss.png")
