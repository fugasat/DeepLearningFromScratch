import sys, os
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


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


def function_2(x):
    return x[0]**2 + x[1]**2


def numerical_diff_sample():
    x = np.arange(0.0, 20.0, 0.1)  # 0.0,0.1,0.2, ... , 20.0
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.savefig('graph_4_6.png')

    a5 = numerical_diff(function_1, 5)
    b5 = function_1(5) - a5 * 5
    y5 = x * a5 + b5
    plt.plot(x, y5)

    a10 = numerical_diff(function_1, 10)
    b10 = function_1(10) - a10 * 10
    y10 = x * a10 + b10
    plt.plot(x, y10)

    plt.savefig('graph_4_7.png')


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def numerical_diff_sample2():
    x0 = np.arange(-3, 3, 0.25)
    x1 = np.arange(-3, 3, 0.25)
    X, Y = np.meshgrid(x0, x1)
    Z = function_2(np.array([X, Y]))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X, Y, Z)

    plt.savefig('graph_4_8.png')


if __name__ == '__main__':
    numerical_diff_sample2()
    pass
