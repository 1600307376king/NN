from Neural_Network.single_layer import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


w_init = np.mat([[1, 1]])

data = np.mat([[1, 3],
               [1, 4],
               [3, 1],
               [4, 1]])

y = np.mat([[1, 1, 0, 0]]).T


def loss_d(labels, x_v, w_v, w_n, bias=0):
    m = labels.shape[0]
    res = 0
    for i, v in enumerate(labels):
        xj = x_v[i, w_n]
        yi = labels[i]
        res += xj * (1 / (1 + np.exp(w_v * x_v[i].T + bias)) - yi)

    return res / m


'''
yi = 1
p(xi) = 1 / (1 + np.exp(w*xi + b))
xj = xi[i, j]

'''
alpha = 0.01
epoch = 500
w1 = np.array([[1]])
w2 = np.array([[1]])
for k in range(epoch):
    w1 = w1 + alpha * loss_d(y, data, np.c_[w1, w2], 0)
    w2 = w2 + alpha * loss_d(y, data, np.c_[w1, w2], 1)
    # e1 = loss_d(y, data, np.c_[w1, w2], 0)
    # e2 = loss_d(y, data, np.c_[w1, w2], 1)
    # w1 = w1 + alpha * e1
    # w2 = w2 + alpha * e2
    # if k > 400:
    #     print(e1 * 4 / 9, e2 * 4 / 9)

#
print(w1, w2)
# #
#
# x = np.linspace(0, 4, 100)
# ys = - 15.42891078 / -4.22462404 * x
# x2 = [1, 1, 3, 4]
# y2 = [3, 4, 1, 1]
#
# # plt.plot(x, ys)
# plt.scatter(x2, y2, color='red')
# plt.show()
#
# a = np.array([[1]])
# b = np.array([[2]])
# print(np.c_[a, b])
