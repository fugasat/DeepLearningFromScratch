import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from chap5.two_layer_net import TwoLayerNet

# データ読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 最初の3件をバッチデータとして使う
x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backdrop = network.gradient(x_batch, t_batch)

# 各重みの絶対誤差の平均を求める
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backdrop[key] - grad_numerical[key]))
    print("{0} : {1}".format(key, str(diff)))
