import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pylab as plt

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    # 認識(推論)を行う
    def predict(self, x):
        W1 = self.params["W1"]
        W2 = self.params["W2"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]

        # 入力層 => 隠れ層
        a1 = np.dot(x, W1) + b1  # 内積でNNの計算を行う
        z1 = sigmoid(a1)  # 活性化関数(シグモイド関数)を使って出力値の計算を行う

        # 隠れ層 => 出力層
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)  # 今回は分類問題なのでソフトマックス関数を使う(出力値が0〜1の範囲に正規化される)

        return y

    # 損失関数を計算（交差エントロピー誤差）
    # x:入力データ、t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)  # 交差エントロピー誤差

    # 認識精度(正解率)を求める
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        # 正解率 = 正解数 / 入力データの総数
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads


def sample1():
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params["W1"].shape)
    print(net.params["b1"].shape)
    print(net.params["W2"].shape)
    print(net.params["b2"].shape)

    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)

    y = net.predict(x)
    print(y)

    grads = net.numerical_gradient(x, t)
    print(grads["W1"].shape)
    print(grads["b1"].shape)
    print(grads["W2"].shape)
    print(grads["b2"].shape)


def nn_batch():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # ハイパーパラメータ
    iters_num = 10000
    train_size = x_train.shape[0]  # 60000
    batch_size = 100
    learning_rate = 0.1

    # 1エポックあたりの繰り返し数
    iter_per_epoch = 10  #max(train_size / batch_size, 1)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        grad = network.numerical_gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]

        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        print("{0} : {1}".format(i, loss))

        x_iter = np.arange(0, len(train_loss_list), 1)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.ylim(0, 3)
        plt.xlim(0, len(train_loss_list) + 1)
        plt.plot(x_iter, train_loss_list)
        plt.savefig('graph_4_11.png')
        plt.clf()

        # 1エポック毎に認識精度を計算
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + " , " + str(test_acc))

            x_iter = np.arange(0, len(train_acc_list), 1)
            plt.ylim(0, 1)
            plt.xlim(0, len(train_acc_list) + 1)
            plt.plot(x_iter, train_acc_list)
            plt.plot(x_iter, test_acc_list)
            plt.savefig('graph_4_12.png')
            plt.clf()

if __name__ == '__main__':
    nn_batch()
