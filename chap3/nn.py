import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def func_sample():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.savefig('graph_3_6.png')
    plt.clf()

    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.savefig('graph_3_7.png')
    plt.clf()

    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)

    y = sigmoid(x)
    plt.plot(x, y)

    plt.ylim(-0.1, 1.1)
    plt.savefig('graph_3_8.png')
    plt.clf()

    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 5.1)
    plt.savefig('graph_3_9.png')
    plt.clf()


def three_layered_nn_sample():
    # 3 layered nn
    print("3 layered nn")
    X = np.array([1.0, 0.5])
    W1 = np.array(
        [[0.1, 0.3, 0.5],  # for x1 to Y
         [0.2, 0.4, 0.6]]  # for x2 to Y
    )
    B1 = np.array([0.1, 0.2, 0.3]) # to Y

    print(W1.shape)
    print(X.shape)
    print(B1.shape)

    A1 = np.dot(X, W1) + B1
    print(A1.shape)
    print(A1)

    Z1 = sigmoid(A1)
    print(Z1)

    W2 = np.array(
        [[0.1, 0.4], # for x1 to Y
         [0.2, 0.5], # for x2 to Y
         [0.3, 0.6]] # for x3 to Y
    )
    B2 = np.array([0.1, 0.2]) # to Y
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    print(Z2)

    W3 = np.array(
        [[0.1, 0.3], # for x1 to Y
         [0.2, 0.4]] # for x2 to Y
    )
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)
    print(Y)


def init_network():
    network = {}

    # W1 : 2 => 3
    network['W1'] = np.array(
        [[0.1, 0.3, 0.5],  # for x1 to Y
         [0.2, 0.4, 0.6]]  # for x2 to Y
    )
    # b1
    network['b1'] = np.array([0.1, 0.2, 0.3]) # to Y

    # W2 : 3 => 2
    network['W2'] = np.array(
        [[0.1, 0.4], # for x1 to Y
         [0.2, 0.5], # for x2 to Y
         [0.3, 0.6]] # for x3 to Y
    )
    # b2
    network['b2'] = np.array([0.1, 0.2]) # to Y

    # W3 : 2 => 2
    network['W3'] = np.array(
        [[0.1, 0.3], # for x1 to Y
         [0.2, 0.4]] # for x2 to Y
    )
    # b3
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


def three_layered_nn_sample2():
    print("3 layered nn (2)")

    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)


if __name__ == '__main__':
    three_layered_nn_sample2()

