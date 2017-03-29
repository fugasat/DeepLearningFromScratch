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


name = "SGD_He"
with open(name + ".pkl", mode='rb') as f:
    result = pickle.load(f)

x_iter = np.arange(0, len(result.train_loss_list), 1)

plt.xlabel("iteration")
plt.ylabel("loss")
plt.ylim(0, 3)
plt.xlim(0, len(result.train_loss_list) + 1)
plt.plot(x_iter, result.train_loss_list)
plt.savefig('graph_' + name + '_nn_loss.png')
plt.clf()
