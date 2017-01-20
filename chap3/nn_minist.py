import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
import time
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def get_test_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


if __name__ == '__main__':
    x, t = get_test_data()  # 10,000件のテストデータを取得
    network = init_network()

    px = []  # batch_size
    py = []  # elasped_time

    for batch_size in [1, 10, 50, 100, 1000, 5000, 10000]:
        print("batch size : {0}".format(batch_size))
        if (batch_size % 100) == 0:
            print("{0}%".format(int(batch_size / 100)))

        if batch_size == 0:
            continue

        accuracy_cnt = 0
        start = time.time()

        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = predict(network, x_batch)
            p = np.argmax(y_batch, axis=1)  # 最大値のインデックスを返す（axis=1 二つ目の次元を基準にする）
            a = (p == t[i:i + batch_size])  # 各データが正解かどうかチェックした結果をbool配列で取得
            cnt = np.sum(a)  # Trueの数をカウント
            accuracy_cnt += cnt

        elasped_time = time.time() - start
        print("elasped time : {0}[sec]".format(elasped_time))
        print("Accuracy : {0}".format(float(accuracy_cnt) / len(x)))
        px.append(batch_size)
        py.append(elasped_time)

    plt.plot(px, py)
    plt.savefig('graph_minist_elasped_time.png')
