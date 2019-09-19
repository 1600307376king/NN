import numpy as np
import pandas as pd


# NN 神经网络
# sigmoid函数 对于二分类问题，如果使用sigmoid，label应为0，1
def sigmoid(z):
    return 1 / (1 + np.exp(z))


# 对j(w)求wj的偏导函数
def j_weight_p_d(labels, x_v, w_v, bias=0):
    """
    :param labels:样本类别 如 1，0 shape m 行 1 列 m x 1
    :param x_v: m个样本数据，n个特征，m x n
    :param w_v: 样本权重矩阵， 如w1, w2
    :param bias: 偏置
    :return:
    """

    m, n = x_v.shape
    res = np.zeros((1, n))
    for i, v in enumerate(labels):
        x = x_v[i]
        yi = labels[i]
        res = res + (sigmoid(w_v * x_v + bias) - yi) * x
    return res / m


# 梯度下降法
def gradient_descent(x, y, w_init, bias=0, alpha=0.01, epoch=500):
    """
    :param x: m个样本数据，n个特征，m x n
    :param y: 样本类别 如 1，0 shape m 行 1 列 m x 1
    :param bias: 偏置
    :param alpha: 学习率
    :param epoch: 迭代次数
    :return: 返回 权重向量，如w1，w2 ...
    """
    for i in range(epoch):
        w_init = w_init + alpha * j_weight_p_d(y, x, w_init, bias)
    return w_init


def h(x):
    if x < 0.5:
        return np.array([[0]])
    else:
        return np.array([[1]])


def dnn(data, label, bias=0):
    n = 2
    w1 = np.random.randn(n, 1) / np.sqrt(n)
    w2 = np.random.randn(n, 1) / np.sqrt(n)

    w21 = np.random.randn(n, 1) / np.sqrt(n)

    for i, v in enumerate(data):
        # forward propagation
        a1 = sigmoid(v * w1 + bias)
        a2 = sigmoid(v * w2 + bias)

        out1 = sigmoid(np.c_[a1, a2] * w21 + bias)

        # back propagation
        w21 = gradient_descent(out1, label[i], w21)

        w1 = gradient_descent(a1, label[i], w21)

        w2 = gradient_descent(a2, label[i], w21)

    return w1, w2, w21


def forward(data, w1, w2, w21, bias=0):
    for i, v in enumerate(data):
        a1 = sigmoid(v * w1 + bias)
        a2 = sigmoid(v * w2 + bias)

        out1 = sigmoid(np.c_[a1, a2] * w21 + bias)
        print(out1, h(out1))


# forward
d = np.mat([[1, 3],
            [1, 4],
            [3, 1],
            [4, 1]])

y = np.mat([[1],
            [1],
            [0],
            [0]])

w1, w2, w21 = dnn(d, y)

test_d = np.mat([[1, 4]])
forward(test_d, w1, w2, w21)
# dnn(d, y)
# w1 = np.random.randn(3, 1) / np.sqrt(3)
# w2 = np.random.randn(3, 1) / np.sqrt(3)
# w3 = np.random.randn(3, 1) / np.sqrt(3)
#
# w21 = np.random.randn(3, 1) / np.sqrt(3)
#
# b = 0
#
# a1 = sigmoid(d[0] * w1 + b)
# a2 = sigmoid(d[0] * w2 + b)
# a3 = sigmoid(d[0] * w3 + b)
#
# out1 = sigmoid(np.c_[a1, a2, a3] * w21 + b)
#
# # back
#
# w_u21 = gradient_descent(out1, y[0], w21)
#
# m = np.c_[a1, a2, a3].T
# my = np.c_[h(a1), h(a2), h(a3)].T
#
# w_u11 = gradient_descent(a1,
#                          [1],
#                          w_u21)
#
# w_u12 = gradient_descent(a2, [1], w_u21)
# w_u13 = gradient_descent(a3, [1], w_u21)
