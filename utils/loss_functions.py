import tensorflow as tf
import utils.image_metrics as uimc

class SSIM(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    @tf.function(jit_compile=False)
    def call(self, y_true, y_pred):
        mse= uimc.calculate_mse(y_true, y_pred)
        ssim = uimc.calculate_ssim(y_true, y_pred)
        return self.alpha * mse + self.beta * (1.0 - ssim)

class PSNR(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    @tf.function(jit_compile=False)
    def call(self, y_true, y_pred):
        mse= uimc.calculate_mse(y_true, y_pred)
        psnr = uimc.calculate_psnr(y_true, y_pred, 50.0)
        return self.alpha * mse + self.beta * (1.0 - psnr)
    
class NCC(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    @tf.function(jit_compile=False)
    def call(self, y_true, y_pred):
        mse= uimc.calculate_mse(y_true, y_pred)
        ncc = uimc.calculate_ncc(y_true, y_pred)
        return self.alpha * mse + self.beta * (1.0 - ncc)
    
class MyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=0.7, teta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.teta = teta

    @tf.function(jit_compile=False)
    def call(self, y_true, y_pred):
        mse= uimc.calculate_mse(y_true, y_pred)
        ncc = uimc.calculate_ncc(y_true, y_pred)
        psnr = uimc.calculate_psnr(y_true, y_pred, 50.0)
        return self.alpha * mse + self.beta * (1.0 - ncc) + + self.beta * (1.0 - psnr)