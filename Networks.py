import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import xgboost as xgb
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
# tf.compat.v1.enable_eager_execution()
from Loss_func import KLDivergenceLoss, KLDivergenceLoss_dyna, KLDivergenceLoss_GMM
from label_enhancement import get_nearest_neighbor_parameters_manifold_learning
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from generate_simulation_data import simulation_data_generation


class XGBoostTensorFlowWrapper(tf.keras.Model):
    def __init__(self, xgb_model):
        super(XGBoostTensorFlowWrapper, self).__init__()
        self.xgb_model = xgb_model

    def call(self, inputs):
        # 将TensorFlow张量转换为NumPy数组，因为XGBoost需要NumPy数组作为输入
        numpy_inputs = inputs.numpy()
        # 使用XGBoost模型进行预测
        predictions = self.xgb_model.predict(numpy_inputs)
        # 将预测结果转换回TensorFlow张量
        return tf.convert_to_tensor(predictions)

    def fit(self, X, y, validation_data=None, **kwargs):
        # 将TensorFlow数据集转换为NumPy数组

        # train_dataset = train_dataset.map(lambda x, y: (x.numpy(), y.numpy()))
        # 准备训练数据和标签
        # x_train, y_train = train_dataset[0], train_dataset[1]
        x_train, y_train = X, y
        # 训练XGBoost模型
        self.xgb_model.fit(x_train, y_train, **kwargs)
        # 可选地，如果提供了验证数据，可以使用XGBoost的评估方法进行评估
        if validation_data:
            x_val, y_val = validation_data
            x_val, y_val = x_val.numpy(), y_val.numpy()
            validation_score = self.xgb_model.eval(x_val, y_val, verbose=False)
            print(f'Validation score: {validation_score}')

    def predict(self, test_dataset):
        # 将TensorFlow数据集转换为NumPy数组
        # test_dataset = test_dataset.map(lambda x: x.numpy())
        # 使用XGBoost模型进行预测
        predictions = self.xgb_model.predict(test_dataset)
        # 将预测结果转换回TensorFlow张量
        return predictions


class LSTM_KLD(tf.keras.Model):
    def __init__(self, custom_loss, optimizer):
        super(LSTM_KLD, self).__init__()
        self.custom_loss = custom_loss
        self.optimizer = optimizer
        self.dense_units = 20
        self.num_units = 10

    def train_step(self, data, y):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        with tf.GradientTape() as tape:
            predictions = self(data, training=True)  # 前向传播
            loss = self.custom_loss(y, predictions)  # 计算损失

        gradients = tape.gradient(loss, self.trainable_variables)  # 计算梯度
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))  # 更新权重

        return loss

    def call(self, inputs, training=False):
        x = tf.keras.layers.Dense(self.dense_units, name="Hiddenlayer", input_shape=(10, 1))(inputs)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.LSTM(units=self.num_units)(x)
        results = tf.keras.layers.Dense(units=1, activation='softplus')(x)
        return results


class LSTM_GMM(tf.keras.Model):
    def __init__(self, optimizer):
        super(LSTM_GMM, self).__init__()
        # self.custom_loss = custom_loss
        self.optimizer = optimizer
        self.dense_units = 20
        self.num_units = 10

    def train_step(self, data, y, GMM_mean, GMM_variance, GMM_weight):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        KLD_GMM = KLDivergenceLoss_GMM(GMM_mean, GMM_variance, GMM_weight)
        with tf.GradientTape() as tape:
            predictions = self(data, training=True)  # 前向传播
            loss = KLD_GMM(y, predictions)  # 计算损失

        gradients = tape.gradient(loss, self.trainable_variables)  # 计算梯度
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))  # 更新权重

        return loss

    def call(self, inputs, training=False):
        x = tf.keras.layers.Dense(self.dense_units, name="Hiddenlayer", input_shape=(10, 1))(inputs)
        # x = tf.keras.layers.Dropout(0.7)(x)
        #x = tf.keras.layers.Dense(self.dense_units, name="Hiddenlayer")(x)
        x = tf.keras.layers.LSTM(units=self.num_units)(x)
        results = tf.keras.layers.Dense(units=1, activation='softplus')(x)
        return results


if __name__ == "__main__":
    dataset='simulation'
    if dataset=="huawei":
        data_ = pd.read_csv("data/supply_chain_data.csv")
        X = np.array(
            data_[['202107', '202108', '202109', '202110', '202111', '202112', '202201', '202202', '202203', '202204']])
        y = np.array(data_[['control']])
        true = np.array(data_[['202205']])
    elif dataset=='simulation':
        simulation = simulation_data_generation()
        X, true, y = simulation.generate_dataset1()
        X = np.array(X)
        true = np.array(true)
    print("--------------Data Generated--------------")
    kd_tree = KDTree(X)
    result = {'mean': np.array([]), 'weight': np.array([])}
    result = get_nearest_neighbor_parameters_manifold_learning(result, X, true.reshape(-1, ), kd_tree, 4, 0.5)
    var_determined_by_weight = 1 / result['weight'] * 1  # 方差与权重的相反数成正比
    GMM_mean = result['mean']
    GMM_weight = result['weight']
    X_train, X_test, y_train, y_test, true_train, true_test, GMM_mean_train, GMM_mean_test, GMM_variance_train, GMM_variance_test, \
        GMM_weight_train, GMM_weight_test = train_test_split(
        X, y,
        true, GMM_mean, var_determined_by_weight, GMM_weight,
        test_size=0.1,
        random_state=1)

    y_train = tf.constant(y_train, dtype=tf.float32)
    true_train = tf.constant(true_train, dtype=tf.float32)
    X_train = tf.constant(X_train, dtype=tf.float32)
    GMM_mean_train = tf.constant(GMM_mean_train, dtype=tf.float32)
    GMM_variance_train = tf.constant(GMM_variance_train, dtype=tf.float32)
    GMM_weight_train = tf.constant(GMM_weight_train, dtype=tf.float32)
    X_train = tf.expand_dims(X_train, axis=2)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, true_train, GMM_mean_train, GMM_variance_train, GMM_weight_train))

    # 设置 batch_size
    batch_size = 16

    # 批处理数据集
    dataset = dataset.batch(batch_size)
    # X_train = tf.expand_dims(X_train, axis=2)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # custom_loss = KLDivergenceLoss_GMM()
    # 实例化模型
    model = LSTM_GMM(
        # custom_loss=custom_loss,
        optimizer=optimizer
    )
    num_epochs = 50
    # 假设 train_dataset 是一个包含数据和标签的 tf.data.Dataset 对象
    # 训练模型
    loss_list=[]
    for epoch in range(num_epochs):
        for batch_n, (data, labels, GMM_mean, GMM_variance, GMM_weight) in enumerate(dataset):
            # data=tf.expand_dims(data, axis=2)
            loss = model.train_step(data, labels, GMM_mean, GMM_variance, GMM_weight)
        loss_list.append(loss.numpy())
            # if batch_n % 10 == 0:
        print(f"Epoch {epoch + 1} Loss {loss.numpy()}")
    print(loss_list)
