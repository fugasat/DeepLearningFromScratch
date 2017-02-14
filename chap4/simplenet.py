import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:

    def __init__(self):
        # [2 x 3]の行列
        # - 入力層の変数は2こ
        # - 出力層の変数は3こ
        self.W = np.random.randn(2, 3) # ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)  # ベクトルの内積（NNの計算を行う => "X" x "W")

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


def simplenet_sample():
    net = simpleNet()

    # 重みの初期値
    print(net.W)

    # NNの計算
    x = np.array([0.6, 0.9])  # 入力層
    p = net.predict(x)
    print(p)

    # 最大値のインデックス
    print(np.argmax(p))

    # 損失関数を計算（交差エントロピー誤差）
    t = np.array([0, 0, 1])  # 出力層の正解値
    print(net.loss(x, t))

    def f(W):
        return net.loss(x, t)

    # 損失関数の勾配を求める
    #  損失関数の値が減るような重みパラメータの調整値を計算
    dW = numerical_gradient(f, net.W)
    print(dW)


if __name__ == '__main__':
    simplenet_sample()
    pass
