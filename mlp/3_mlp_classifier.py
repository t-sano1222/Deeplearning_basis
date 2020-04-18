#!/usr/bin/python
# -*- coding: utf-8 -*-

"""多層パーセプトロンの実装
Todo:
    *パーセプトロンのクラス化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from activation_functions import sigmoid
from activation_functions import softmax

# generate data
X = np.arange(-1.0, 1.0, 0.1)
Y = np.arange(-1.0, 1.0, 0.1)


# weights for inputs and bias
w_im = np.array([[1.0, -0.7],
                 [0.3, 0.8]])
w_mo = np.array([[-1.0, 1.0],
                 [1.0, -1.0]])

b_im = np.array([0.3, -0.3])
b_mo = np.array([0.4, 0.1])

def middle_layer(x, w, b):
    """middle layer
    多層パーセプトロンの中間層

    Args:
        x: input
        w: wight
        b: bias

    Retuens:
        output of neuron
    """
    u = np.dot(x, w) + b
    return sigmoid(u)

def output_layer(x, w, b):
    """output layer
    多層パーセプトロンの出力層

    Args:
        x: input
        w: wight
        b: bias

    Retuens:
        output of mlp
    """
    u = np.dot(x, w) + b
    return softmax(u)


x_1 = []
x_2 = []
y_1 = []
y_2 = []

for i in range(20):
    for j in range(20):
        inp = np.array([X[i], Y[j]])
        mid = middle_layer(inp, w_im, b_im)
        out = output_layer(mid, w_mo, b_mo)
        if out[0] > out[1]:
            x_1.append(X[i])
            y_1.append(Y[j])
        else:
            x_2.append(X[i])
            y_2.append(Y[j])

plt.scatter(x_1, y_1, marker='+')
plt.scatter(x_2, y_2, marker='o')

plt.savefig('out_mlp_classifier.png')
