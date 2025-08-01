import tensorflow as tf
import tensorflow.image as tf_img
import numpy as np
import sewar.full_ref

def calculate_mse(image1, image2):
    """
    Avalia: Erro quadrático médio entre as imagens.
    Varia: >= 0 (0 = perfeito).
    Quanto menor, melhor.
    """
    return tf.reduce_mean(tf.square(image1 - image2))

def calculate_ssim(image1, image2):
    """
    Avalia: Estrutura, contraste e luminância.
    Varia de 0 - 1:
    - 1: perfeito
    - ~0.95: excelente
    - ~0.9: muito bom
    - ~0.8: bom
    - < 0.7: distorções visíveis
    """
    image1 = tf.convert_to_tensor(image1, dtype=tf.float32)
    image2 = tf.convert_to_tensor(image2, dtype=tf.float32)
    return tf_img.ssim(image1, image2, max_val=1.0)

def calculate_msssim(image1, image2):
    """
    Avalia: Mesmo princípio do SSIM, mas em várias escalas de detalhe.
    Faixa: 0–1 (1 = igual).
    Serve pra: Ser mais sensível a artefatos em diferentes tamanhos de padrão (textura fina e grossa).
    """
    return sewar.full_ref.msssim(image1, image2, MAX=1.0)

def calculate_psnr(image1, image2, psnr_max=None):
    """
    Avalia: Diferença global (erro quadrático médio).
    Varia de 0 - 60 dB:
    - > 50 dB: muito raro, reconstrução quase perfeita
    - 30 - 50 dB: reconstrução muito boa
    - < 30 dB: qualidade visivelmente ruim
    """
    psnr = tf.image.psnr(image1, image2, max_val=1.0)
    if psnr_max is None:
        return psnr
    else:
        psnr = tf.clip_by_value(psnr, 0.0, psnr_max)  # limita ao máximo escolhido
        psnr_normalized = psnr / psnr_max
        return psnr_normalized

def calculate_ncc(image1, image2):
    """
    Avalia: Correlação entre os padrões (Normalized Cross-Correlation).
    Varia de -1 a +1:
    - 1.0: perfeito
    - ~0.9: muito bom
    - ~0.8: aceitável
    - < 0.8: distorções grandes
    - 0: sem correlação
    - -1.0: correlação negativa perfeita
    """
    image1 = tf.convert_to_tensor(image1, dtype=tf.float32)
    image2 = tf.convert_to_tensor(image2, dtype=tf.float32)
    image1_mean = tf.reduce_mean(image1)
    image2_mean = tf.reduce_mean(image2)

    num = tf.reduce_sum((image1 - image1_mean) * (image2 - image2_mean))

    # Denominador: raiz( soma((x-mean_x)^2) * soma((y-mean_y)^2) )
    denom = tf.sqrt(
        tf.reduce_sum(tf.square(image1 - image1_mean)) *
        tf.reduce_sum(tf.square(image2 - image2_mean))
    )

    # Retorna NCC
    return num / (denom + 1e-8)

def calculate_vif(image1, image2):
    """
    Avalia: Quantidade de informação visual preservada (fidelidade perceptual).
    Varia de 0 - 1:
    - 1.0: perfeito
    - ~0.9: excelente
    - ~0.8: muito bom
    - < 0.7: distorções perceptíveis
    """
    if hasattr(image1, "numpy"):
        image1 = image1.numpy()
    if hasattr(image2, "numpy"):
        image2 = image2.numpy()

    return sewar.full_ref.vifp(image1, image2)

def calculate_scc(image1, image2):
    """
    Avalia: Correlação espacial entre as imagens.
    Faixa: -1 a 1.
    Serve pra: Mostra alinhamento de padrões (parecido com NCC).
    """
    return sewar.full_ref.scc(image1, image2)

def calculate_all_metrics(image1, image2):
    """
    As duas imagens devem estar em formatos: 
    - normalizadas entre 0 e 1
    - Min 64x64
    - img.numpy
    """
    mse = calculate_mse(image1, image2)
    ssim = calculate_ssim(image1, image2)
    msssim = calculate_msssim(image1, image2)
    psnr = calculate_psnr(image1, image2)
    ncc = calculate_ncc(image1, image2)
    vif = calculate_vif(image1, image2)
    scc = calculate_scc(image1, image2)

    print(f"As seguintes métricas foram calculadas: \n MSE: {mse:.3f} \n SSIM: {ssim:.3f} \n MSSSIM: {msssim:.3f} \n PSNR: {psnr:.3f} \n NCC: {ncc:.3f} \n VIF: {vif:.3f} \n SCC: {scc:}")

    return {"MSE":mse, "SSIM":ssim, "MSSSIM":msssim, "PSNR":psnr, "NCC":ncc, "VIF":vif, "SCC":scc}