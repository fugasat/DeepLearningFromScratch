import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

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


if __name__ == '__main__':
    sample1()