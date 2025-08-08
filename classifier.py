import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.utils import plot_model 
from sklearn.metrics import accuracy_score

import datetime
import os

import utils
from utils.view.plot import * 
from utils.preprocessing import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default='/opt/datasets', help="Caminho para download dos datasets")
args = parser.parse_args()

PathManager = utils.PathManager(dataset_path=args.dataset_path)

class Classifier:
    def __init__(self, name_model):
        self.name = name_model
        self.encoder = None
        self.type_encoder=None
        self.classifier = None
        self.PathManager = None

    def return_encoder(self, PathManager, type='Autoencoder'):
        self.type_encoder = type
        self.PathManager = PathManager
        model_path, weigths, _, _ = self.PathManager.return_paths_autoencoder(self.name, type=type)
        
        model = tf.keras.models.load_model(model_path)

        if isinstance(model, tf.keras.Sequential):
            latent_vector = 0
            layers = model.layers
            for i, layer in enumerate(layers):
                if layer.name == 'latent_vector':
                    latent_vector = i
                    break
        
            encoder_layers = layers[:latent_vector + 1]
            self.encoder = tf.keras.Sequential(encoder_layers, name='Encoder')

        return self.encoder

    def create_classifier(self, encoder, freeze=True):
        if encoder != None:
            if freeze:
                for layer in encoder.layers:
                    layer.trainable = False
                encoder.trainable = False

            #crio o classificador com o enconder
            self.classifier = tf.keras.models.Sequential([
                    encoder,  
                    tf.keras.layers.Dropout(0.2),  
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dense(128,activation='relu'),  
                    tf.keras.layers.Dense(2, activation='softmax')  
                ], name=f'Classifier-{self.name}')

            self.classifier.summary()
            return self.classifier

        else:
            raise ValueError("Encoder é None, não pode criar classificador.")

    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False)

    def train(self, train, valid, test, epochs, y_true=None, batch_size=32):
        #Callbacks:

        monitor_cb = MonitorCallback()
        logs_path = self.PathManager.get_logs_path(self.name)
        now = datetime.datetime.now().strftime("%d%m%y-%H%M")
        log_dir = os.path.join(
            logs_path,
            f'Classifier-{self.name}',
            f'fit-{now}'
        )
        
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = self.classifier.fit(
            train, 
            epochs=epochs, 
            callbacks=[monitor_cb, tensorboard_callback], 
            batch_size=batch_size,
            validation_data=valid,
            steps_per_epoch=len(train), 
            validation_steps=len(valid),
            verbose=1
        )

        _, weights_path, history_path, accs = PathManager.return_paths_classifier(name=self.name, type=self.type_encoder, base=train.name)

        self.classifier.save_weights(os.path.join(weights_path))
        plot_history(history, save_fig=os.path.join(history_path, f'Classifier-{train.name}.png'))

        self.classifier.evaluate(test)

        if y_true is not None:
            y_pred = self.classifier.predict(test)
            y_pred = tf.argmax(y_pred, axis=1)

            plot_confusion_matrix(
                y_true, y_pred, 
                labels=['Vazio', 'Ocupado'], 
                legend=[f'Treinado: {str(train.name)}', f'Teste: {str(train.name)}'], 
                save_path=os.path.join(
                    accs,
                    'Confusion-Matrix',
                    f'Classifier-{str(train.name)}-{str(test.name)}'
                )
            )
        

class MonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_acc = -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        if acc is not None:
            print(f"\nEpoch {epoch+1}: accuracy = {acc:.4f}")
            if acc > self.best_acc:
                self.best_acc = acc
                print(f"Novo melhor accuracy: {acc:.4f}")

"""Mark01 = Classifier(name_model='Mark01')
encoder = Mark01.return_encoder(PathManager)
Mark01.create_classifier(encoder)"""

train_gen, train_df = preprocessing_dataframe(path_csv='CSV/PUC/PUC_train.csv', autoencoder=False, data_algumentantation=False, input_shape=(128, 128), name='PUC')
valid_gen, valid_df = preprocessing_dataframe(path_csv='CSV/PUC/PUC_valid.csv', autoencoder=False, data_algumentantation=False, input_shape=(128, 128), name='PUC')
test_gen, test_df = preprocessing_dataframe(path_csv='CSV/PUC/PUC_test.csv', autoencoder=False, data_algumentantation=False, input_shape=(128, 128), name='PUC')

train_gen, train_df = preprocessing_dataframe(path_csv='CSV/UFPR04/UFPR04_train.csv', autoencoder=False, data_algumentantation=False, input_shape=(128, 128), name='UFPR04')
valid_gen, valid_df = preprocessing_dataframe(path_csv='CSV/UFPR04/UFPR04_valid.csv', autoencoder=False, data_algumentantation=False, input_shape=(128, 128), name='UFPR04')
test_gen, test_df = preprocessing_dataframe(path_csv='CSV/UFPR04/UFPR04_test.csv', autoencoder=False, data_algumentantation=False, input_shape=(128, 128), name='UFPR04')

train_gen, train_df = preprocessing_dataframe(path_csv='CSV/UFPR05/UFPR05_train.csv', autoencoder=False, data_algumentantation=False, input_shape=(128, 128), name='UFPR05')
valid_gen, valid_df = preprocessing_dataframe(path_csv='CSV/UFPR05/UFPR05_valid.csv', autoencoder=False, data_algumentantation=False, input_shape=(128, 128), name='UFPR05')
test_gen, test_df = preprocessing_dataframe(path_csv='CSV/UFPR05/UFPR05_test.csv', autoencoder=False, data_algumentantation=False, input_shape=(128, 128), name='UFPR05')



#Mark01.compile_model()
#Mark01.train(train_gen, valid_gen, test_gen, 20, test_df['class'], 32)