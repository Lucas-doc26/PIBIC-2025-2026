import tensorflow as tf
tf.config.optimizer.set_jit(False) 
from tensorflow.keras import layers, models

from utils.loss_functions import *
from utils.preprocessing import *
from utils.view.plot import plot_history, plot_autoencoder_with_ssim
from utils.config import *
from utils.path_manager import * 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--loss", default='mse', help="Loss function")
parser.add_argument("--dataset_path", default='/opt/datasets', help="Caminho para download dos datasets")
args = parser.parse_args()

SEED = 42

tf.random.set_seed(SEED)

PathManager = PathManager(dataset_path=args.dataset_path)

def fit_with_loss(model, loss, train_gen, valid_gen, test_gen, base='PUC'):
    model.summary()
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    history = model.fit(train_gen, validation_data=valid_gen, epochs=20, steps_per_epoch=len(train_gen), validation_steps=len(valid_gen))

    #Pega os caminhos para salvar
    PathManager.create_folders_for_model(model.name)
    model_path, weights_path, history_path, plot_path = PathManager.return_paths_autoencoder(name=model.name, base=base)

    #Salva o modelo
    model.save(model_path)
    model.save_weights(weights_path)

    #Salva os plots
    plot_history(history, type='Autoencoder', save_fig=os.path.join(history_path, f'History-Autoencoder-{base}-{str(loss)}.png'))
    plot_autoencoder_with_ssim(test_gen, model, 128, 128, os.path.join(plot_path, f'Autoencoder-{base}-{str(loss)}.png'))

    clear_session()

autoencoder = tf.keras.Sequential([
    # Entrada 128×128×3
    layers.Input(shape=(128, 128, 3)),

    # Encoder
    layers.Conv2D(64, 3, activation='relu', padding='same'),   
    layers.MaxPooling2D(2, padding='same'),                    
    layers.Conv2D(128, 3, activation='relu', padding='same'),    
    layers.MaxPooling2D(2, padding='same'),                     

    layers.Flatten(),                                           
    layers.Dense(256, activation='relu', name='latent_vector'), 

    # Decoder
    layers.Dense(32 * 32 * 64, activation='relu'),              
    layers.Reshape((32, 32, 64)),                               
    layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same'),  
    layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'),  

    layers.Conv2D(3, 3, activation='sigmoid', padding='same')   
], name='Mark01')

# Imagens de treino e teste
train_gen, train_df = preprocessing_dataframe(path_csv='CSV/PUC/PUC_train.csv', autoencoder=True, data_algumentantation=False, input_shape=(128, 128), name='PUC')
valid_gen, valid_df = preprocessing_dataframe(path_csv='CSV/PUC/PUC_valid.csv', autoencoder=True, data_algumentantation=False, input_shape=(128, 128), name='PUC')
test_gen, test_df = preprocessing_dataframe(path_csv='CSV/PUC/PUC_test.csv', autoencoder=True, data_algumentantation=False, input_shape=(128, 128), name='PUC')

print(len(train_gen))
print(train_gen.name)

psnr = PSNR()
ssim = SSIM()
ncc = NCC()
myloss = MyLoss()

if args.loss == 'ncc':
    fit_with_loss(autoencoder, ncc, train_gen, valid_gen, test_gen)
    clear_session()

elif args.loss == 'ssim':
    fit_with_loss(autoencoder, ssim, train_gen, valid_gen, test_gen)
    clear_session()

elif args.loss == 'psnr':
    fit_with_loss(autoencoder, psnr, train_gen, valid_gen, test_gen)
    clear_session()

elif args.loss == 'myloss':
    fit_with_loss(autoencoder, myloss, train_gen, valid_gen, test_gen)
    clear_session()

else:
    fit_with_loss(autoencoder, 'mse', train_gen, valid_gen, test_gen)
    clear_session()
