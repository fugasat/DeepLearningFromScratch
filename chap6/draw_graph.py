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
from chap6.two_layer_net import TwoLayerNet, NetResult


def get_result(name):
    with open(name + ".pkl", mode='rb') as f:
        result = pickle.load(f)
    return result

"""
    Optimizer毎の性能を比較
"""
optimizers = OrderedDict()
optimizers["SGD"] = get_result("SGD")
optimizers["AdaGrad"] = get_result("AdaGrad")
optimizers["Momentum"] = get_result("Momentum")
optimizers["Adam"] = get_result("Adam")

batch_size = 100
x_size = len(optimizers["SGD"].train_loss_list)
x_iter = np.arange(0, x_size, batch_size)

plt.xlabel("iteration")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.xlim(0, x_size + 1)
for key in optimizers.keys():
    train_loss_list = np.array(optimizers[key].train_loss_list)
    # Batchサイズ毎に平均値で値をならす
    train_loss_list_ave = []
    for i in x_iter:
        batch_train_loss_list = train_loss_list[i:i + batch_size]
        train_loss_list_ave.append(batch_train_loss_list.mean())
    plt.plot(x_iter, train_loss_list_ave, label=key)

plt.legend()
plt.savefig('graph_optimizer_loss.png')
plt.clf()
