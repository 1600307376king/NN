import numpy as np
import pandas as pd


# NN 神经网络
# sigmoid函数 对于二分类问题，如果使用sigmoid，label应为0，1
def sigmoid(z):
    return 1 / (1 + np.exp(z))


# 对j(w)求wj的偏导函数
def j_weight_p_d(labels, x_v, w_v, w_n=None, bias=0):
    """
    :param labels:样本类别 如 1，0 shape m 行 1 列 m x 1
    :param x_v: m个样本数据，n个特征，m x n
    :param w_v: 样本权重矩阵， 如w1, w2
    :param w_n: 输入值为0，1 ...，是特征权重位置的下标 对应w1，w2，指对w1求偏导或w2求偏导
    :param bias: 偏置
    :return:
    """

    m, n = x_v.shape
    res = None
    if w_n:
        for i, v in enumerate(labels):
            xj = x_v[i, w_n]
            yi = labels[i]
            res += xj * (sigmoid(w_v * x_v[i].T + bias) - yi)
    else:
        res = np.zeros((1, n))
        for i, v in enumerate(labels):
            x = x_v[i]
            yi = labels[i]
            res = res + (sigmoid(w_v * x.T + bias) - yi) * x
    return res / m


# 梯度下降法
def gradient_descent(x, y, bias=0, alpha=0.01, epoch=500):
    """
    :param x: m个样本数据，n个特征，m x n
    :param y: 样本类别 如 1，0 shape m 行 1 列 m x 1
    :param bias: 偏置
    :param alpha: 学习率
    :param epoch: 迭代次数
    :return: 返回 权重向量，如w1，w2 ...
    """
    _, n = x.shape
    w = np.ones((1, n))
    for i in range(epoch):
        w = w + alpha * j_weight_p_d(y, x, w, bias)
    return w


# data = np.mat([[1, 3],
#                [1, 4],
#                [3, 1],
#                [4, 1]])
#
# y = np.mat([[1, 1, 0, 0]]).T
# print(gradient_descent(data, y))

