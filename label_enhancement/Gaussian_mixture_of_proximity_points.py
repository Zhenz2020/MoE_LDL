import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt


def W_matrix_by_manifold_learning(indices_all, points):
    # 声明模型
    model = Model("manifold_learning")
    # 模型设置
    # 关闭输出
    model.setParam('OutputFlag', 0)
    # 创建变量
    W = [[[] for _ in range(indices_all.shape[1])] for _ in range(indices_all.shape[0])]
    for i in range(indices_all.shape[0]):
        for j in range(indices_all.shape[1]):
            W[i][j] = model.addVar(0, 1, vtype=GRB.CONTINUOUS, name=f'W_{i}_{j}')
    # 目标函数
    obj = LinExpr(0)
    for i in range(indices_all.shape[0]):
        cur_obj = 0
        for k in range(points.shape[1]):
            err = points[i, k]
            for j in range(indices_all.shape[1]):
                err = err - points[indices_all[i, j], k] * W[i][j]
            cur_obj += err ** 2
        obj += cur_obj
    model.setObjective(obj, GRB.MINIMIZE)
    # 约束条件
    for i in range(indices_all.shape[0]):
        model.addConstr(sum(W[i][j] for j in range(indices_all.shape[1])) == 1)
    model.optimize()
    # 获取结果
    W_matrix = np.zeros((indices_all.shape[0], indices_all.shape[1]))
    for i in range(indices_all.shape[0]):
        for j in range(indices_all.shape[1]):
            W_matrix[i, j] = W[i][j].x
    # 查看目标函数值
    print('Obj:', model.objVal)
    return W_matrix


def get_nearest_neighbor_parameters_simple(result, points, y, kd_tree): # 使用1/distance作为权重
    # 查询最近邻居
    distances_all, indices_all = kd_tree.query(points, k=4)
    distances_all = np.array(distances_all)
    mean_distances = np.mean(np.mean(distances_all[:, 1:], axis=1))
    # 将distance_all第一列全部改为mean_distances
    distances_all[:, 0] = mean_distances / 10
    for i in range(indices_all.shape[0]):
        indices = indices_all[i][:]
        distances = distances_all[i][:]
        result['mean'] = np.vstack((result['mean'], y[indices])) if result['mean'].size else y[indices]
        result['weight'] = np.vstack((result['weight'], 1 / distances / np.sum(1 / distances))) if result[
            'weight'].size else 1 / distances / np.sum(1 / distances)
    return result


def get_nearest_neighbor_parameters_manifold_learning(result, points, y, kd_tree, k,true_proportions=0.5):  # 使用流形学习的方法
    # 查询最近邻居
    distances_all, indices_all = kd_tree.query(points, k=k)
    indices_all_near = indices_all[:, 1:]
    W_matrix = W_matrix_by_manifold_learning(indices_all_near, points)
    # 在W_matrix插入第一列1
    W_matrix = np.insert(W_matrix, 0, true_proportions * (1 / (1 - true_proportions)), axis=1)
    # W_matrix每行归一化
    W_matrix = W_matrix / np.sum(W_matrix, axis=1).reshape(-1, 1)
    result['weight'] = W_matrix
    result['weight'] = W_matrix
    for i in range(indices_all.shape[0]):
        indices = indices_all[i][:]
        result['mean'] = np.vstack((result['mean'], y[indices])) if result['mean'].size else y[indices]
    return result


def gaussian(x, mu, sigma, weight=1):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi)) * weight


def mix_gaussian(x, mu, sigma, distribution):
    y = np.zeros_like(x)
    for i in range(len(mu)):
        y += gaussian(x, mu[i], sigma[i], distribution[i])
    return y


def draw_mix_gaussion(mu, sigma, distribution):
    x = np.linspace(900, 1100, 1000)
    y = mix_gaussian(x, mu, sigma, distribution)
    for i in range(len(mu)):
        cur_mu = mu[i]
        cur_sigma = sigma[i]
        plt.plot(x, gaussian(x, cur_mu, cur_sigma, distribution[i]),
                 label=f'Gaussian Function: mu={round(cur_mu, 2)}, sigma={round(cur_sigma, 2)}')
    plt.plot(x, y, label='Mixture Gaussian Function')
    plt.title('Mixture Gaussian Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # 示例数据
    points = np.array(pd.read_csv('simulation_data.csv').iloc[:, 0:10].values)
    y = np.array(pd.read_csv('simulation_data.csv')['true'].values)
    kd_tree = KDTree(points)
    result = {'mean': np.array([]), 'weight': np.array([])}
    result = get_nearest_neighbor_parameters_manifold_learning(result, points, y, kd_tree, 4,0.5)  # 给定真正的比例
    var_settled = 9
    var_settled = np.ones(result['mean'].shape[1]) * var_settled  # 固定一个方差
    var_determined_by_weight = 1 / result['weight'][0] * 4  # 方差与权重的相反数成正比
    draw_mix_gaussion(result['mean'][0], var_determined_by_weight, result['weight'][0])
    print(result['mean'][0], result['weight'][0])
