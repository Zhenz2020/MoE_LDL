import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree
from tensorflow.keras.losses import Loss
from label_enhancement import get_nearest_neighbor_parameters_manifold_learning
class KLDivergenceLoss(Loss):
    def __init__(self, name='kl_divergence_loss', **kwargs):
        super(KLDivergenceLoss, self).__init__(name=name, **kwargs)

    def kl_divergence(self, mu1, sigma1, mu2, sigma2):
        # log_square_sigma1 = tf.math.log(tf.square(sigma1))
        # log_square_sigma2 = tf.math.log(tf.square(sigma2))
        log_sigma1 = tf.math.log(sigma1 + 1e-5)
        log_sigma2 = tf.math.log(sigma2 + 1e-5)

        KL_loss = 0.5 * (log_sigma2 - log_sigma1 + tf.exp(log_sigma1 - log_sigma2) + tf.square(
            mu1 - mu2) / tf.exp(log_sigma2) - 1.0)
        return KL_loss

    def DLDL(self):
        pass

    # 定义自定义损失函数
    def call(self, y_true, y_pred):
        kl_loss = tf.reduce_mean(self.kl_divergence(mu1=y_true, sigma1=1.0, mu2=y_pred, sigma2=1.0))
        return kl_loss


class KLDivergenceLoss_dyna_sigma(Loss):
    def __init__(self, name='kl_divergence_loss', **kwargs):
        super(KLDivergenceLoss_dyna_sigma, self).__init__(name=name, **kwargs)

    def kl_divergence_dyna_sigma(self, mu1, sigma1, mu2, sigma2, lambda_):
        # log_square_sigma1 = tf.math.log(tf.square(sigma1))
        # log_square_sigma2 = tf.math.log(tf.square(sigma2))
        log_sigma1 = tf.math.log(sigma1 + 1e-5)
        log_sigma2 = tf.math.log(sigma2 + 1e-5)

        term1 = log_sigma2 - log_sigma1
        term2 = (tf.square(sigma1) + tf.square(mu1 - mu2)) / (2 * tf.square(sigma2)) - 0.5
        term3 = lambda_ * tf.square(mu1 - mu2)
        return term1 + term2 + term3

    # 定义自定义损失函数
    def call(self, y_true, y_pred):
        kl_loss = tf.reduce_mean(self.kl_divergence_dyna_sigma(mu1=y_true[:, 0], sigma1=y_true[:, 1], mu2=y_pred[:, 0],
                                                               sigma2=y_pred[:, 1],
                                                               lambda_=1.0))
        return kl_loss


class KLDivergenceLoss_dyna(Loss):
    # 直接调用tensorflow的KL散度
    def __init__(self, name='kl_divergence_dyna', **kwargs):
        super(KLDivergenceLoss_dyna, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        mean_true = y_true[:, 0]
        log_var_true = y_true[:, 1]
        log_var_true = tf.math.log(log_var_true + 1e-5)
        mean_pred = y_pred[:, 0]
        log_var_pred = y_pred[:, 1]
        log_var_pred = tf.math.log(log_var_pred + 1e-5)

        # 计算 KL 散度
        kl_loss = 0.5 * (log_var_pred - log_var_true + tf.exp(log_var_true - log_var_pred) + tf.square(
            mean_true - mean_pred) / tf.exp(log_var_pred) - 1.0)

        # 返回 KL 散度的均值作为损失
        return tf.reduce_mean(kl_loss)

class KLDivergenceLoss_GMM(Loss):
    #真值为一元高斯分布，预测值为一个混合高斯
    def __init__(self,GMM_mean, GMM_variance, GMM_weight):
        super(KLDivergenceLoss_GMM, self).__init__(name='GMM_KL')
        self.GMM_mean = GMM_mean
        self.GMM_variance = GMM_variance
        self.GMM_weight = GMM_weight
    def call(self, y_true, y_pred):

        # kd_tree = KDTree(self.X_train)
        # result = {'mean': np.array([]), 'weight': np.array([])}
        # result = get_nearest_neighbor_parameters_manifold_learning(result, self.X_train, y_pred, kd_tree, 4, 0.5)
        # var_determined_by_weight = 1 / result['weight'] * 4  # 方差与权重的相反数成正比
        # mean_true = y_true[:, 0]
        # log_var_true = y_true[:, 1]
        # log_var_true = tf.math.log(log_var_true + 1e-5)
        # mean_pred = result['mean']
        # log_var_pred = var_determined_by_weight
        # log_var_pred = tf.math.log(log_var_pred + 1e-5)

        mean_true = y_true
        mean_true = mean_true[:, np.newaxis]
        log_var_true = tf.constant(np.ones(y_true.shape[0]), dtype=tf.float32)
        log_var_true = log_var_true[:, np.newaxis]
        log_var_true = tf.math.log(log_var_true + 1e-5)

        mean_pred = self.GMM_mean
        mean_pred = tf.concat([y_pred, mean_pred[:, 1:]], axis=1)
        log_var_pred = self.GMM_variance
        log_var_pred = tf.math.log(log_var_pred + 1e-5)

        # 计算 KL 散度
        kl_loss = tf.multiply(self.GMM_weight,(0.5 * (log_var_pred - log_var_true + tf.exp(log_var_true - log_var_pred) + tf.square(
            mean_true - mean_pred) / tf.exp(log_var_pred) - 1.0)))

        # 返回 KL 散度的均值作为损失
        return tf.reduce_mean(kl_loss)