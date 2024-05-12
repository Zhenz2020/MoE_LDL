import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.losses import KLD
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import os
import xgboost as xgb
from cleanlab.regression.learn import CleanLearning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Multiply
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow_probability as tfp
from MoE import MixtureOfExperts, MixtureOfExperts_multi_experts
from Networks import XGBoostTensorFlowWrapper
from Loss_func import KLDivergenceLoss, KLDivergenceLoss_dyna, KLDivergenceLoss_GMM
from label_enhancement import get_nearest_neighbor_parameters_manifold_learning
from sklearn.neighbors import KDTree
# 测试混合高斯分布
tfd = tfp.distributions
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
my_init = tf.keras.initializers.glorot_uniform(seed=seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '0'


class simulation_data_generation():
    def true_relation(self, x):
        noise = np.random.normal(0, 20, size=(x.shape[0], 1))  # 添加噪声
        # y = np.sum(2 * x[:, :10] ^ 2, axis=1, keepdims=True) + 5 + noise
        y = np.sum(2 * x[:, :10], axis=1, keepdims=True) + 5 + noise
        return y

    def generate_dataset1(self):
        # 生成真实的y
        x = np.random.randint(0, 100, size=(10000, 10))
        true_y = self.true_relation(x)
        data = pd.DataFrame(np.hstack((x, true_y)))
        # 给定专家的误差率
        rate = [0.04, 0.08, 0.10]
        # 模拟专家的预测，引入一些误差
        for i in range(len(rate)):
            prediction = true_y + np.random.normal(np.mean(true_y * rate[i]), np.var(true_y * rate[i]),
                                                   size=true_y.shape)
            data[f'prediction{i + 1}'] = prediction
        return x, true_y, data.iloc[:, -len(rate):]



class LDL_models:
    def __init__(self, X, y, true, n_iter):
        self.X = X
        self.experts = y
        self.y = true  # expert
        self.true = true  # true value
        self.X_train = None
        self.y_train = None
        self.y_denoise_train = None
        self.y_denoise_test = None
        self.X_test = None
        self.n_iter = n_iter
        self.y_test = None
        self.y_train_logvar = None
        self.true_test = None
        self.true_train = None
        self.y_denoise = None
        self.num_units = 20
        self.batch_size = 8
        self.LEARNING_RATE = 1e-4
        self.patience = 10
        self.dense_units = 20
        self.y_gp = None
        self.sigma = None
        self.y_gp_train = None
        self.y_gp_test = None
        self.sigma_train = None
        self.sigma_test = None
        self.history_mse = None
        self.history_LDL = None
        self.history_denoise_LDL = None
        self.history_gp_LDL = None

    def dataset_splite(self):
        print("--------------Denoise Start--------------")
        self.label_noise_fix()
        print("--------------Denoise End--------------")
        print("--------------GP Start--------------")
        self.gp_pred_variance()
        print("--------------GP End--------------")
        X_train, self.X_test, y_train, self.y_test, y_denoise_train, self.y_denoise_test, y_gp_train, \
            self.y_gp_test, sigma_train, self.sigma_test, true_train, true_test, experts_train, experts_test = train_test_split(
            self.X, self.y,
            self.y_denoise,
            self.y_gp,
            self.sigma,
            self.true,
            self.experts,
            test_size=0.1,
            random_state=1)
        y_std = np.std(y_train)
        self.X_train_array = X_train.copy()
        self.y_train_array = y_train.copy()

        self.y_train = tf.constant(y_train, dtype=tf.float32)
        self.X_train = tf.constant(X_train, dtype=tf.float32)
        self.true_train = tf.constant(true_train, dtype=tf.float32)
        self.true_test = tf.constant(true_test, dtype=tf.float32)
        self.experts_train = tf.constant(experts_train, dtype=tf.float32)
        self.experts_test = tf.constant(experts_test, dtype=tf.float32)
        self.y_denoise_train = tf.constant(y_denoise_train, dtype=tf.float32)
        self.y_gp_train = tf.constant(y_gp_train, dtype=tf.float32)
        self.sigma_train = tf.constant(sigma_train, dtype=tf.float32)
        self.y_train_logvar = tf.abs(tf.random.normal((X_train.shape[0], 1), mean=y_std, stddev=0.2), name='abs')

    def gp_pred_variance(self):
        # 在数据上拟合高斯过程
        kernel = 1.0 * RBF(length_scale=10.0)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gp.fit(self.X, self.y_denoise)
        # 在数据上进行预测
        y_pred, sigma = gp.predict(self.X, return_std=True)
        self.y_gp = y_pred.reshape(-1, 1)
        self.sigma = sigma.reshape(-1, 1)

    def label_noise_fix(self):
        model_xgb = xgb.XGBRegressor(max_depth=10,  # 每一棵树最大深度，默认6；
                                     learning_rate=0.005,  # 学习率，每棵树的预测结果都要乘以这个学习率，默认0.3；
                                     n_estimators=200,  # 使用多少棵树来拟合，也可以理解为多少次迭代。默认100；
                                     objective='reg:squarederror',
                                     booster='gbtree',
                                     # 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。默认为gbtree
                                     gamma=0,  # 叶节点上进行进一步分裂所需的最小"损失减少"。默认0；
                                     min_child_weight=2,  # 可以理解为叶子节点最小样本数，默认1；
                                     subsample=0.8,  # 训练集抽样比例，每次拟合一棵树之前，都会进行该抽样步骤。默认1，取值范围(0, 1]
                                     colsample_bytree=1,  # 每次拟合一棵树之前，决定使用多少个特征，参数默认1，取值范围(0, 1]。
                                     reg_alpha=1,  # 默认为0，控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
                                     reg_lambda=1,  # 默认为1，控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                                     seed=1023)
        cl = CleanLearning(model_xgb)
        label_issues = cl.find_label_issues(self.X, self.y)
        for index, row in label_issues.iterrows():

            given_label = row['given_label']
            label_quality = row['label_quality']
            predicted_label = row['predicted_label']
            is_label_issue = row['is_label_issue']
            # if label_quality < 0.000001:
            if is_label_issue:
                label_issues.at[index, 'merge'] = predicted_label
            else:
                label_issues.at[index, 'merge'] = given_label

        self.y_denoise = label_issues['merge']
        return label_issues

    def LSTM_KLD(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense_units, name="Hiddenlayer", input_shape=(10, 1)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(units=self.num_units, activation='leaky_relu'),
            tf.keras.layers.Dense(units=1, activation='softplus', kernel_constraint=NonNeg()),
        ])
        optimizer = Adam(learning_rate=self.LEARNING_RATE, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=KLDivergenceLoss())
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        print("GPU Available:", tf.config.list_physical_devices('GPU'))
        with tf.device('/GPU:0'):
            self.history_LDL = model.fit(self.X_train, self.y_train, epochs=self.n_iter, batch_size=self.batch_size,
                                         validation_split=0.2, callbacks=[early_stopping])
        # 进行预测
        with tf.device('/GPU:0'):
            y_pred = model.predict(self.X_test)
            self.pred_LDL = model.predict(self.X_train)
            self.model_LDL = model
        # 提取预测的均值和方差

        return y_pred

    def LSTM_MSE(self):
        model = Sequential()
        model.add(Dense(self.dense_units, name="Hiddenlayer", input_shape=(10, 1)))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(LSTM(units=self.num_units, activation='leaky_relu'))
        model.add(Dense(units=1, activation='softplus', kernel_constraint=NonNeg()))
        optimizer = Adam(learning_rate=self.LEARNING_RATE, clipnorm=1.0)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        # 编译模型
        model.compile(optimizer=optimizer, loss='mse')  # 使用均方误差作为损失函数
        # early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        # 训练模型
        self.history_mse = model.fit(self.X_train, self.y_train, epochs=self.n_iter, batch_size=self.batch_size,
                                     validation_split=0.2, callbacks=[early_stopping])

        # 使用模型进行预测
        # 要预测的12个月的供应量数据

        predicted_value = model.predict(self.X_test)
        self.pred_mse = model.predict(self.X_train)
        self.model_mse = model
        return predicted_value

    def LSTM_denoise_LDL(self):
        # KL散度的均值为去噪过后的预测值，方差为定值
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense_units, name="Hiddenlayer", input_shape=(10, 1)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(units=self.num_units, activation='leaky_relu'),
            tf.keras.layers.Dense(units=1, activation='softplus', kernel_constraint=NonNeg()),
        ])
        optimizer = Adam(learning_rate=self.LEARNING_RATE, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=KLDivergenceLoss())
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        with tf.device('/GPU:0'):
            self.history_denoise_LDL = model.fit(self.X_train, self.y_denoise_train, epochs=self.n_iter,
                                                 batch_size=self.batch_size,
                                                 validation_split=0.2, callbacks=[early_stopping])
        # 进行预测
        with tf.device('/GPU:0'):
            y_pred = model.predict(self.X_test)
            self.pred_denoise_LDL = model.predict(self.X_train)
            self.model_denoise_LDL = model
        # 提取预测的均值和方差

        return y_pred

    def LSTM_GP_LDL(self):
        # KL散度的均值与方差都换成了高斯过程回归的预测值与方差
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense_units, name="Hiddenlayer", input_shape=(10, 1)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(units=self.num_units, activation='relu'),
            tf.keras.layers.Dense(units=2, activation='softplus', kernel_constraint=NonNeg()),  # 输出两个节点，分别表示均值和对数方差
            # tf.keras.layers.Dense(units=64, activation='tanh', input_shape=(10,)),
            # tf.keras.layers.Dense(units=32, activation='relu'),
            # tf.keras.layers.Dense(units=2, activation='softplus', kernel_constraint=NonNeg())
        ])
        optimizer = Adam(learning_rate=self.LEARNING_RATE, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=KLDivergenceLoss_dyna())
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        with tf.device('/GPU:0'):
            self.history_gp_LDL = model.fit(self.X_train, tf.concat([self.y_gp_train, self.sigma_train], axis=1),
                                            epochs=self.n_iter,
                                            batch_size=self.batch_size,
                                            validation_split=0.2, callbacks=[early_stopping])
        # 进行预测
        with tf.device('/GPU:0'):
            y_pred = model.predict(self.X_test)
            self.pred_gp_LDL = model.predict(self.X_train)[:, 0]
            self.model_gp_LDL = model
        # 提取预测的均值和方差

        return y_pred[:, 0]

    def LSTM_GMM_LDL(self):
        # KL散度的均值与方差都换成了高斯过程回归的预测值与方差
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense_units, name="Hiddenlayer", input_shape=(10, 1)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(units=self.num_units, activation='relu'),
            tf.keras.layers.Dense(units=2, activation='softplus', kernel_constraint=NonNeg()),  # 输出两个节点，分别表示均值和对数方差
        ])
        optimizer = Adam(learning_rate=self.LEARNING_RATE, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=KLDivergenceLoss_GMM())
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        with tf.device('/GPU:0'):
            self.history_GMM_LDL = model.fit(self.X_train, tf.concat([self.y_train, self.sigma_train], axis=1),
                                            epochs=self.n_iter,
                                            batch_size=self.batch_size,
                                            validation_split=0.2, callbacks=[early_stopping])
        # 进行预测
        with tf.device('/GPU:0'):
            y_pred = model.predict(self.X_test)
            self.pred_GMM_LDL = model.predict(self.X_train)[:, 0]
            self.model_GMM_LDL = model
        # 提取预测的均值和方差

        return y_pred[:, 0]

    def XGBoost_experts(self):
        model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')

        # 创建TensorFlow包装器实例
        xgb_tf_wrapper = XGBoostTensorFlowWrapper(model)

        # 假设我们有一个TensorFlow数据集
        # train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train_array, self.y_train_array))
        # test_dataset = tf.data.Dataset.from_tensor_slices(self.X_test)

        # 训练模型
        xgb_tf_wrapper.fit(self.X_train_array, self.y_train_array,validation_data=None)

        # 进行预测
        y_pred = xgb_tf_wrapper.predict(self.X_test)
        # model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', use_label_encoder=False,
        #                                   eval_metric='rmse')
        # 训练XGBoost模型
        # model.fit(self.X_train_array, self.y_train_array)
        # y_pred = model.predict(self.X_test)
        self.xgb_model = xgb_tf_wrapper
        return y_pred



    def build_moe_with_experts(self, epochs):

        # gating_network = self.build_gating_network(input_shape, num_experts)
        # xgb_expert = XGBoostExpert(num_outputs=1)
        #y_pred_GMM = self.LSTM_GMM_LDL()
        y_pred_xgb = self.XGBoost_experts()
        print("-------------XGBoost_experts End-------------")
        y_pred_MSE = self.LSTM_MSE()
        print("-------------LSTM End-------------")
        y_pred_LDL = self.LSTM_KLD()
        print("-------------LSTM KLD End-------------")
        y_pred_denoise_LDL = self.LSTM_denoise_LDL()
        print("-------------LSTM_denoise_LDL End-------------")
        y_pred_gp_LDL = self.LSTM_GP_LDL()
        print("-------------LSTM_GP_LDL End-------------")

        # X_new = np.hstack((X, Y))
        num_experts = 8
        expert_units = 32
        output_units = 1
        # experts = [self.pred_mse, self.pred_LDL, self.pred_denoise_LDL, self.pred_gp_LDL]
        experts_model = [self.xgb_model, self.model_mse, self.model_LDL, self.model_denoise_LDL, self.model_gp_LDL]
        model = MixtureOfExperts_multi_experts(experts_model, num_experts, output_units)
        optimizer = Adam(learning_rate=self.LEARNING_RATE, clipnorm=1.0)
        # Compile the model
        model.compile(optimizer=optimizer, loss=KLDivergenceLoss(),run_eagerly=True)
        # model.compile(optimizer=optimizer, loss='mse', run_eagerly=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        with tf.device('/GPU:0'):
            self.history_moe = model.fit([self.X_train,self.experts_train], self.true_train, epochs=epochs,
                                         batch_size=self.batch_size, validation_split=0.2, callbacks=[early_stopping])
            # self.history_moe = moe_model.fit(self.X_train, self.true_train, epochs=self.n_iter,
            #                                  batch_size=self.batch_size, validation_split=0.1)
            y_pred_moe = model.predict([self.X_test,self.experts_test])
        return y_pred_xgb, y_pred_moe, y_pred_MSE, y_pred_LDL, y_pred_denoise_LDL, y_pred_gp_LDL

    def metric_results(self, y_pred_xgb, y_pred_MSE, y_pred_LDL, y_pred_denoise_LDL, y_pred_GP_LDL, y_pred_moe, i):
        y_pred_MSE[y_pred_MSE < 0] = 1
        y_pred_LDL[y_pred_LDL < 0] = 1
        y_pred_denoise_LDL[y_pred_denoise_LDL < 0] = 1
        y_pred_GP_LDL[y_pred_GP_LDL < 0] = 1
        y_pred_xgb[y_pred_xgb < 0] = 1
        y_pred_moe[y_pred_moe < 0] = 1
        result_df = pd.DataFrame(
            {'True': np.array(self.true_test).ravel(),
             'Baseline': np.array(y_pred_MSE).ravel(),
             'LDL': np.array(y_pred_LDL).ravel(),
             'Denoise_LDL': np.array(y_pred_denoise_LDL).ravel(),
             'GP_LDL': np.array(y_pred_GP_LDL).ravel(),
             'XGB': np.array(y_pred_xgb).ravel(),
             'MoE': np.array(y_pred_moe).ravel()
             })
        result_experts = pd.DataFrame(np.array(self.experts_test))
        result_experts.to_csv('predictions_experts.csv',index=False)
        try:
            result_df.to_csv(f'prediction_results_{str(i)}.csv', index=False)
        except:
            pass
        mse_LDL = mean_squared_error(self.true_test, y_pred_LDL)
        mse_MSE = mean_squared_error(self.true_test, y_pred_MSE)
        mse_denoise_LDL = mean_squared_error(self.true_test, y_pred_denoise_LDL)
        mse_GP_LDL = mean_squared_error(self.true_test, y_pred_GP_LDL)
        mse_xgb = mean_squared_error(self.true_test, y_pred_xgb)


        print("测试集上Baseline的均方误差 (MSE):", mse_MSE)
        print("测试集上LDL的均方误差 (MSE):", mse_LDL)
        print("测试集上denoise_LDL的均方误差 (MSE):", mse_denoise_LDL)
        print("测试集上GP_LDL的均方误差 (MSE):", mse_GP_LDL)
        print("测试集上XGB的均方误差 (MSE):", mse_xgb)
        try:
            for i in range(self.experts_test.shape[1]):
                mse_experts = mean_squared_error(self.true_test, self.experts_test[:,i])
                print(f"测试集上Experts{i}的均方误差 (MSE):", mse_experts)
            mse_moe = mean_squared_error(self.true_test, y_pred_moe)
            print("测试集上MoE的均方误差 (MSE):", mse_moe)

        except:
            pass

    def plot_loss(self, i, if_moe=True):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 将四个子图分别绘制历史训练损失和验证损失曲线
        axes[0, 0].plot(self.history_mse.history['loss'], label='Model LSTM_MSE Training Loss')
        axes[0, 0].plot(self.history_mse.history['val_loss'], label='Model LSTM_MSE Validation Loss')
        axes[0, 0].set_title('LSTM_MSE')

        axes[0, 1].plot(self.history_LDL.history['loss'], label='Model LSTM_LDL Training Loss')
        axes[0, 1].plot(self.history_LDL.history['val_loss'], label='Model LSTM_LDL Validation Loss')
        axes[0, 1].set_title('LSTM_LDL')

        axes[1, 0].plot(self.history_denoise_LDL.history['loss'], label='Model LSTM_denoise_LDL Training Loss')
        axes[1, 0].plot(self.history_denoise_LDL.history['val_loss'], label='Model LSTM_denoise_LDL Validation Loss')
        axes[1, 0].set_title('LSTM_denoise_LDL')

        axes[1, 1].plot(self.history_gp_LDL.history['loss'], label='Model LSTM_gp_LDL Training Loss')
        axes[1, 1].plot(self.history_gp_LDL.history['val_loss'], label='Model LSTM_gp_LDL Validation Loss')
        axes[1, 1].set_title('LSTM_gp_LDL')

        # 设置图像标题和共享坐标轴
        fig.suptitle('Training and Validation Loss for Four Models')
        for ax in axes.flat:
            ax.set(xlabel='Epoch', ylabel='Loss')
            ax.legend()

        # 调整子图布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 显示图像
        plt.savefig(f"Figs/LDL_denoise_gp_mse{i}.jpg")
        plt.close()
        if if_moe:
            plt.figure()
            plt.plot(self.history_moe.history['loss'], label='Model MoE Training Loss')
            plt.plot(self.history_moe.history['val_loss'], label='Model MoE Validation Loss')
            plt.title('MoE')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f"Figs/LDL_MoE_{i}.jpg")
            plt.close()


if __name__ == "__main__":

    data_set = "MoE_true_y"

    if data_set == "huawei":
        data_ = pd.read_csv("data/supply_chain_data.csv")
        X = np.array(
            data_[['202107', '202108', '202109', '202110', '202111', '202112', '202201', '202202', '202203', '202204']])
        y = np.array(data_[['control']])
        true = np.array(data_[['202205']])
        print("--------------Data Generated--------------")
        X = np.array(X)
        experts = np.array(y)
        true = np.array(true)

        model = LDL_models(X, experts, true, n_iter=50)
        model.dataset_splite()
        y_pred_xgb,y_pred_moe, y_pred_MSE, y_pred_LDL, y_pred_denoise_LDL, y_pred_gp = model.build_moe_with_experts(epochs=50)
        print("--------------Data Generated--------------")
        model.metric_results(y_pred_xgb,y_pred_MSE, y_pred_LDL, y_pred_denoise_LDL, y_pred_gp, y_pred_moe, data_set)
        model.plot_loss(data_set, if_moe=True)
    elif data_set == 'simulation':
        simulation = simulation_data_generation()
        X, true, Y = simulation.generate_dataset1()
        X = np.array(X)
        true = np.array(true)
        for i in range(3):
            y = Y.iloc[:, i]
            y = np.array(y)
            model = LDL_models(X, y, true, n_iter=100)
            model.dataset_splite()
            print("下面是GP_LDL")
            y_pred_gp = model.LSTM_GP_LDL()
            print("下面是MSE")
            y_pred_MSE = model.LSTM_MSE()
            print("下面是KLD")
            y_pred_LDL = model.LSTM_KLD()
            print("下面是Denoise_LDL")
            y_pred_denoise_LDL = model.LSTM_denoise_LDL()
            temp = 1
            model.metric_results(y_pred_MSE, y_pred_LDL, y_pred_denoise_LDL, y_pred_gp, temp, i)
            model.plot_loss(i)
    elif data_set == 'MoE':
        for i in range(3):
            simulation = simulation_data_generation()
            X, true, Y = simulation.generate_dataset1()
            X = np.array(X)
            true = np.array(true)
            y = Y.iloc[:, i]
            model = LDL_models(X, y, true, n_iter=10)
            model.dataset_splite()
            y_pred_xgb, y_pred_moe, y_pred_MSE, y_pred_LDL, y_pred_denoise_LDL, y_pred_gp = model.build_moe(epochs=10)
            model.metric_results(y_pred_xgb, y_pred_MSE, y_pred_LDL, y_pred_denoise_LDL, y_pred_gp, y_pred_moe, str(i) + '_moe')
            model.plot_loss(i, if_moe=True)
    elif data_set == 'MoE_true_y':
        simulation = simulation_data_generation()
        X, true, Y = simulation.generate_dataset1()
        print("--------------Data Generated--------------")
        X = np.array(X)
        experts=np.array(Y)
        true = np.array(true)

        model = LDL_models(X,experts, true, n_iter=30)
        model.dataset_splite()
        y_pred_xgb, y_pred_moe, y_pred_MSE, y_pred_LDL, y_pred_denoise_LDL, y_pred_gp = model.build_moe_with_experts(epochs=30)
        print("--------------Data Generated--------------")
        model.metric_results(y_pred_xgb, y_pred_MSE, y_pred_LDL, y_pred_denoise_LDL, y_pred_gp, y_pred_moe, 'simulation')
        model.plot_loss(0, if_moe=True)
    # model.metric_results(y_pred_gp, y_pred_gp, y_pred_gp, y_pred_gp)
