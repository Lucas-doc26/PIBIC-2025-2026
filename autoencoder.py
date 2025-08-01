import tensorflow as tf
tf.config.optimizer.set_jit(False) 

from tensorflow.keras import layers, models
from utils.loss_functions import *
from utils.preprocessing import *
from utils.view.plot import plot_autoencoder, plot_autoencoder_with_ssim

import os
import random
import gc
import keras

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--loss", default='mse', help="Loss function")
args = parser.parse_args()

SEED = 42

def set_seeds():
    """
    Configura todas as seeds necessárias para reprodutibilidade
    """
    # TensorFlow
    tf.random.set_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    
    # NumPy
    np.random.seed(SEED)
    
    # Python random
    random.seed(SEED)
    
    # Variáveis de ambiente
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

    # Configurações do TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0' = tudo, '1' = warnings, '2' = info, '3' = apenas erros graves
    
    # Configurações do Pandas
    pd.set_option('mode.chained_assignment', None)

def config_gpu():
    """
    Configura a GPU para uso determinístico
    """
    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configurada para uso determinístico")
    else:
        print("Nenhuma GPU encontrada")

def config_tensorflow():
    """
    Configura o TensorFlow para uso determinístico e mixed precision
    """
    # Configurar logs do TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0' = tudo, '1' = warnings, '2' = info, '3' = apenas erros graves
    
    # Configurar mixed precision
    try:
        # Habilitar mixed precision independente da GPU
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision configurado com sucesso")
        
        # Configurar GPU se disponível
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs disponíveis: {len(gpus)}")
        else:
            print("Usando CPU com mixed precision")
            
    except Exception as e:
        print(f"Aviso: Erro ao configurar mixed precision: {e}")
        print("Continuando com precisão padrão...")

def clear_session():
    keras.backend.clear_session()  
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        print("GPU")
        tf.config.experimental.reset_memory_stats('GPU:0') #limpa memória da gpu

# Executa as configurações ao importar o módulo
set_seeds()
config_gpu() 
config_tensorflow()
clear_session()

tf.random.set_seed(42)

def fit_with_loss(model, loss, train_gen, valid_gen, test_gen, savefig=None):
    model.summary()
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    history = model.fit(train_gen, validation_data=valid_gen, epochs=100, steps_per_epoch=len(train_gen), validation_steps=len(valid_gen))
    x, y = next(test_gen)
    plot_autoencoder_with_ssim(x, model, 128, 128, savefig)
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
    fit_with_loss(autoencoder, ncc, train_gen, valid_gen, test_gen, 'Images/autoencoder_ncc.png')
    clear_session()

elif args.loss == 'ssim':
    fit_with_loss(autoencoder, ssim, train_gen, valid_gen, test_gen, 'Images/autoencoder_ssim.png')
    clear_session()

elif args.loss == 'psnr':
    fit_with_loss(autoencoder, psnr, train_gen, valid_gen, test_gen, 'Images/autoencoder_psnr.png')
    clear_session()

elif args.loss == 'myloss':
    fit_with_loss(autoencoder, myloss, train_gen, valid_gen, test_gen, 'Images/autoencoder_myloss.png')
    clear_session()

else:
    fit_with_loss(autoencoder, 'mse', train_gen, valid_gen, test_gen, 'Images/autoencoder_mse.png')
    clear_session()
