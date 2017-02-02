import sys, os
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pylab as plt


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size


def cross_entropy_error_one_hot_false(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    y_arange = y[np.arange(batch_size), t]
    return -np.sum(np.log(y_arange)) / batch_size


def batch_sample():
    # x 0から9の値
    # t 画素データ
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    d = (f(x + h) - f(x - h)) / (2 * h)
    return d


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def numerical_diff_sample():
    x = np.arange(0.0, 20.0, 0.1)  # 0.0,0.1,0.2, ... , 20.0
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.savefig('graph_4_6.png')
    pass


if __name__ == '__main__':
    numerical_diff_sample()
    pass
