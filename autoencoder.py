import tensorflow as tf
tf.config.optimizer.set_jit(False) 
from tensorflow.keras import layers, models

from utils.loss_functions import *
from utils.preprocessing import *
from utils.view.plot import plot_autoencoder, plot_autoencoder_with_ssim
from utils.config import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--loss", default='mse', help="Loss function")
args = parser.parse_args()

SEED = 42

tf.random.set_seed(SEED)

def fit_with_loss(model, loss, train_gen, valid_gen, test_gen, savefig=None):
    model.summary()
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    history = model.fit(train_gen, validation_data=valid_gen, epochs=20, steps_per_epoch=len(train_gen), validation_steps=len(valid_gen))
    plot_autoencoder_with_ssim(test_gen, model, 128, 128, savefig)
    #pd.DataFrame(history.history).plot()
    clear_session()

autoencoder = tf.keras.Sequential([
    # Entrada 128×128×3
    layers.Input(shape=(128, 128, 3)),

    # Encoder
    layers.Conv2D(64, 3, activation='relu', padding='same'),    # → 128×128×32
    layers.MaxPooling2D(2, padding='same'),                     # → 64×64×32
    layers.Conv2D(128, 3, activation='relu', padding='same'),    # → 64×64×64
    layers.MaxPooling2D(2, padding='same'),                     # → 32×32×64

    layers.Flatten(),                                           # → 32*32*64 = 65536
    layers.Dense(256, activation='relu', name='latent_vector'), # → vetor latente 256

    # Decoder
    layers.Dense(32 * 32 * 64, activation='relu'),              # → 65536
    layers.Reshape((32, 32, 64)),                               # → 32×32×64
    layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same'),  # → 64×64×64
    layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'),  # → 128×128×32

    # Saída RGB
    layers.Conv2D(3, 3, activation='sigmoid', padding='same')   # → 128×128×3, valores em [0,1]
])

# Imagens de treino e teste
train_gen, train_df = preprocessing_dataframe(path_csv='CSV/PUC/PUC_train.csv', autoencoder=True, data_algumentantation=False, input_shape=(128, 128))
valid_gen, valid_df = preprocessing_dataframe(path_csv='CSV/PUC/PUC_valid.csv', autoencoder=True, data_algumentantation=False, input_shape=(128, 128))
test_gen, test_df = preprocessing_dataframe(path_csv='CSV/PUC/PUC_test.csv', autoencoder=True, data_algumentantation=False, input_shape=(128, 128))

print(len(train_gen))

psnr = PSNR()
ssim = SSIM()
ncc = NCC()
myloss = MyLoss()

if args.loss == 'ncc':
    fit_with_loss(autoencoder, ncc, train_gen, valid_gen, test_gen, 'Images/autoencoder_ncc-20.png')
    clear_session()

elif args.loss == 'ssim':
    fit_with_loss(autoencoder, ssim, train_gen, valid_gen, test_gen, 'Images/autoencoder_ssim-20.png')
    clear_session()

elif args.loss == 'psnr':
    fit_with_loss(autoencoder, psnr, train_gen, valid_gen, test_gen, 'Images/autoencoder_psnr-20.png')
    clear_session()

elif args.loss == 'myloss':
    fit_with_loss(autoencoder, myloss, train_gen, valid_gen, test_gen, 'Images/autoencoder_myloss-20.png')
    clear_session()

else:
    fit_with_loss(autoencoder, 'mse', train_gen, valid_gen, test_gen, 'Images/autoencoder_mse-20.png')
    clear_session()
