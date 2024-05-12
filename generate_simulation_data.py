import numpy as np
import pandas as pd
class simulation_data_generation():
    def true_relation(self, x):
        noise = np.random.normal(0, 20, size=(x.shape[0], 1))  # 添加噪声
        # y = np.sum(2 * x[:, :10] ^ 2, axis=1, keepdims=True) + 5 + noise
        y = np.sum(2 * x[:, :10]**2, axis=1, keepdims=True) + 5 + noise
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

if __name__ == "__main__":

    data_set = "MoE_true_y"

    if data_set == "huawei":
        data_ = pd.read_csv("data/supply_chain_data.csv")
        X = np.array(
            data_[['202107', '202108', '202109', '202110', '202111', '202112', '202201', '202202', '202203', '202204']])
        y = np.array(data_[['control']])
        true = np.array(data_[['202205']])

    elif data_set == 'MoE_true_y':
        simulation = simulation_data_generation()
        X, true, Y = simulation.generate_dataset1()
        X = np.array(X)
        experts=np.array(Y)
        true = np.array(true)
        temp_save=pd.DataFrame(np.hstack((X, true,Y)),columns=['1','2','3','4','5','6','7','8','9','10','true','prediction1','prediction2','prediction3'])
        temp_save.to_csv('data/simulation_data.csv',index=False)
