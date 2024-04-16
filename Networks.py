import xgboost as xgb
from sklearn.base import BaseEstimator
import numpy as np
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

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

        #train_dataset = train_dataset.map(lambda x, y: (x.numpy(), y.numpy()))
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