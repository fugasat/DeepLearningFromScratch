import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from common.optimizer import SGD, Momentum, AdaGrad, Adam
from collections import OrderedDict
from dataset.mnist import load_mnist
import matplotlib.pylab as plt
import time
import pickle


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, use_ReLU=True, weight_init_std_sqrt=2):
        # 重み初期化
        self.params = {}

        weight_init_std1 = weight_init_std
        if weight_init_std1 < 0:
            weight_init_std1 = np.sqrt(weight_init_std_sqrt / input_size)
        self.params['W1'] = weight_init_std1 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        weight_init_std2 = weight_init_std
        if weight_init_std2 < 0:
            weight_init_std2 = np.sqrt(weight_init_std_sqrt / hidden_size)
        self.params['W2'] = weight_init_std2 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤ生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        if use_ReLU:
            self.layers['Relu1'] = Relu()
        else:
            self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """
        認識(推論)を実施
        :param x: 入力データ
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        誤差(損失関数の値)を計算
        :param x: 入力データ
        :param t: 教師データ
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        """
        認識精度を計算
        :param x: 入力データ
        :param t: 教師データ
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        重みパラメータに対する勾配を数値微分により計算(前章と同じ)
        :param x: 入力データ
        :param t: 教師データ
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        """
        重みパラメータに対する勾配を誤差逆伝搬方により計算
        :param x: 入力データ
        :param t: 教師データ
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

def train_nn(name, optimizer, network):
    print("Optimizer={}".format(name))

    # データ読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    #network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    start = time.time()
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 誤差逆伝搬方によって勾配を求める
        grads = network.gradient(x_batch, t_batch)

        # 更新
        """
        for key in network.params.keys():
            network.params[key] -= learning_rate * grad[key]
        """
        optimizer.update(network.params, grads)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        #  print("loss : {0}".format(loss))

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("loss={0:.4f} : train_acc={1:.3f} : test_acc={2:.3f}".format(loss, train_acc, test_acc))

            """
            x_iter = np.arange(0, len(train_loss_list), 1)
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.ylim(0, 3)
            plt.xlim(0, len(train_loss_list) + 1)
            plt.plot(x_iter, train_loss_list)
            plt.savefig('graph_' + name + '_nn_loss.png')
            plt.clf()

            x_iter = np.arange(0, len(train_acc_list), 1)
            plt.ylim(0, 1)
            plt.xlim(0, len(train_acc_list) + 1)
            plt.plot(x_iter, train_acc_list)
            plt.plot(x_iter, test_acc_list)
            plt.savefig('graph_' + name + '_nn_acc.png')
            plt.clf()
            """

    elapsed_time = time.time() - start
    print("elapsed_time:{0}[sec]".format(str(elapsed_time)))

    result = NetResult(network, train_loss_list, train_acc_list, test_acc_list, elapsed_time)
    with open(name + ".pkl", mode='wb') as f:
        pickle.dump(result, f)


class NetResult:
    def __init__(self, network, train_loss_list, train_acc_list, test_acc_list, elapsed_time):
        self.network = network
        self.train_loss_list = train_loss_list
        self.train_acc_list = train_acc_list
        self.test_acc_list = test_acc_list
        self.elapsed_time = elapsed_time


if __name__ == '__main__':
    optimizers = {
        "SGD": SGD(lr=0.01),
        "Momentum": Momentum(lr=0.01, momentum=0.9),
        "AdaGrad": AdaGrad(lr=0.01),
        "Adam": Adam(lr=0.001, beta1=0.9, beta2=0.999),
    }
    for key in optimizers.keys():
        train_nn(key, optimizers[key], TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01))

    train_nn("SGD_W0", SGD(lr=0.01), TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=0))
    train_nn("SGD_W1", SGD(lr=0.01), TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=1))

    train_nn("SGD_Sig_0.01", SGD(lr=0.01),
             TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01,
              use_ReLU=False, weight_init_std_sqrt=1))
    train_nn("SGD_Sig_Xavier", SGD(lr=0.01),
             TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=-1,
                         use_ReLU=False, weight_init_std_sqrt=1))
    train_nn("SGD_Sig_He", SGD(lr=0.01),
             TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=-1,
                         use_ReLU=False))

    train_nn("SGD_0.01", SGD(lr=0.01), TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01))
    train_nn("SGD_He", SGD(lr=0.01), TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=-1))
    train_nn("SGD_Xavier", SGD(lr=0.01), TwoLayerNet(input_size=784, hidden_size=50, output_size=10,
                                                     weight_init_std=-1, weight_init_std_sqrt=1))
