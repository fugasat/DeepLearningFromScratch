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


def draw_graph(optimizers, graph_name, ylim=2.5):
    batch_size = 100
    x_size = len(list(optimizers.values())[0].train_loss_list)
    x_iter = np.arange(0, x_size, batch_size)

    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.ylim(0, ylim)
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
    plt.savefig(graph_name + "_loss.png")
    plt.clf()

    x_size = len(list(optimizers.values())[0].train_acc_list)
    x_iter = np.arange(0, x_size, 1)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.xlim(0, x_size + 1)
    for key in optimizers.keys():
        train_acc_list = np.array(optimizers[key].train_acc_list)
        test_acc_list = np.array(optimizers[key].test_acc_list)
        # plt.plot(x_iter, train_acc_list, label=key)
        plt.plot(x_iter, test_acc_list, label=key)
    plt.legend(loc="center right")
    plt.savefig(graph_name + "_acc.png")
    plt.clf()

    pass


"""
    Optimizer毎の性能を比較
"""
optimizers = OrderedDict()
optimizers["SGD"] = get_result("SGD")
optimizers["AdaGrad"] = get_result("AdaGrad")
optimizers["Momentum"] = get_result("Momentum")
optimizers["Adam"] = get_result("Adam")
draw_graph(optimizers, "graph_optimizer_loss")

"""
    重みWの初期値の違いを比較
"""
optimizers = OrderedDict()
optimizers["SGD(W=0)"] = get_result("SGD_W0")
optimizers["SGD(W=1)"] = get_result("SGD_W1")
optimizers["SGD(W=0.01)"] = get_result("SGD")
draw_graph(optimizers, "graph_initial_W_value", ylim=10)

"""
    活性化関数をSigmoidにしたときの初期値比較(Xavier)
"""
optimizers = OrderedDict()
optimizers["SGD(W=0.01)"] = get_result("SGD_Sig_0.01")
optimizers["SGD(W=Xavier)"] = get_result("SGD_Sig_Xavier")
optimizers["SGD(W=He)"] = get_result("SGD_Sig_He")
draw_graph(optimizers, "graph_initial_W_Sig")

"""
    活性化関数をReLUにしたときの初期値比較(He & Xavier)
"""
optimizers = OrderedDict()
optimizers["SGD(W=0.01)"] = get_result("SGD_0.01")
optimizers["SGD(W=He)"] = get_result("SGD_He")
optimizers["SGD(W=Xavier)"] = get_result("SGD_Xavier")
draw_graph(optimizers, "graph_initial_W_ReLU")
