import numpy as np


def percept(x, w, b):
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NOT(x):
    return 1 - x


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return percept(x, w, b)


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return percept(x, w, b)


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    return percept(x, w, b)


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


def INC(x1, x2, x3, x4):
    ss4 = 1
    ss3 = AND(x4, ss4)
    ss2 = AND(x3, ss3)
    ss1 = AND(x2, ss2)

    s4 = XOR(x4, ss4)
    s3 = XOR(x3, ss3)
    s2 = XOR(x2, ss2)
    s1 = XOR(x1, ss1)
    return [s1, s2, s3, s4]


def ADD_old(a1, a2, a3, a4,
            aa1, aa2, aa3, aa4):

    # layer 1
    b4 = XOR(a4, aa4)
    b3 = XOR(a3, aa3)
    b2 = XOR(a2, aa2)
    b1 = XOR(a1, aa1)

    bb4 = 0
    bb3 = AND(a4, aa4)
    bb2 = AND(a3, aa3)
    bb1 = AND(a2, aa2)

    # layer 2
    c4 = XOR(b4, bb4)
    c3 = XOR(b3, bb3)
    c2 = XOR(b2, bb2)
    c1 = XOR(b1, bb1)

    cc4 = 0
    cc3 = AND(b4, bb4)
    cc2 = AND(b3, bb3)
    cc1 = AND(b2, bb2)

    # layer 3
    d4 = XOR(c4, cc4)
    d3 = XOR(c3, cc3)
    d2 = XOR(c2, cc2)
    d1 = XOR(c1, cc1)

    dd4 = 0
    dd3 = AND(c4, cc4)
    dd2 = AND(c3, cc3)
    dd1 = AND(c2, cc2)

    # layer 4
    e4 = XOR(d4, dd4)
    e3 = XOR(d3, dd3)
    e2 = XOR(d2, dd2)
    e1 = XOR(d1, dd1)

    ee4 = 0
    ee3 = AND(d4, dd4)
    ee2 = AND(d3, dd3)
    ee1 = AND(d2, dd2)

    return [e1, e2, e3, e4]


def ADD(a1, a2, a3, a4,
        aa1, aa2, aa3, aa4): # layer 1 (入力層)

    # layer 2
    b4 = XOR(a4, aa4)
    b3 = XOR(a3, aa3)
    b2 = XOR(a2, aa2)
    b1 = XOR(a1, aa1)
    bb3 = AND(a4, aa4)
    bb2 = AND(a3, aa3)
    bb1 = AND(a2, aa2)

    # layer 3
    c4 = b4
    c3 = XOR(b3, bb3)
    c2 = XOR(b2, bb2)
    c1 = XOR(b1, bb1)
    cc2 = AND(b3, bb3)
    cc1 = AND(b2, bb2)

    # layer 4
    d4 = c4
    d3 = c3
    d2 = XOR(c2, cc2)
    d1 = XOR(c1, cc1)
    dd1 = AND(c2, cc2)

    # layer 5 (出力層)
    e4 = d4
    e3 = d3
    e2 = d2
    e1 = XOR(d1, dd1)
    return [e1, e2, e3, e4]
