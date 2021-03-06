import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from chap5.two_layer_net import TwoLayerNet
import matplotlib.pylab as plt
import time

# データ読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

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
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    #  print("loss : {0}".format(loss))

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("loss : {0}".format(loss))
        print("train_acc : {0} / test_acc : {1}".format(train_acc, test_acc))

        x_iter = np.arange(0, len(train_loss_list), 1)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.ylim(0, 3)
        plt.xlim(0, len(train_loss_list) + 1)
        plt.plot(x_iter, train_loss_list)
        plt.savefig('graph_nn_loss.png')
        plt.clf()

        x_iter = np.arange(0, len(train_acc_list), 1)
        plt.ylim(0, 1)
        plt.xlim(0, len(train_acc_list) + 1)
        plt.plot(x_iter, train_acc_list)
        plt.plot(x_iter, test_acc_list)
        plt.savefig('graph_nn_acc.png')
        plt.clf()

elapsed_time = time.time() - start
print ("elapsed_time:{0}[sec]".format(str(elapsed_time)))
