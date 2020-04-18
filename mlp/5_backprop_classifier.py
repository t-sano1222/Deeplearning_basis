#!/usr/bin/python
# -*- coding: utf-8 -*-

"""誤差逆伝搬の実装
Todo:
    *パーセプトロンのクラス化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuron_class import Middle_layer
from neuron_class import Output_layer
from loss_func import *
from activation_functions import *

# coordinate
X = np.arange(-1.0, 1.1,0.1)
Y = np.arange(-1.0, 1.1,0.1)

input_data = []
correct_data = []
for x in X:
    for y in Y:
        input_data.append([x,y])
        if y < np.sin(np.pi * x):
            correct_data.append([0, 1])
        else:
            correct_data.append([1, 0])

n_data = len(correct_data)

input_data = np.array(input_data)
correct_data = np.array(correct_data)

# num of neurons in each layer
n_in = 2
n_mid = 6
n_out = 2

eta = 0.1
epoch = 101
interval = 10

middle_layer = Middle_layer(n_in, n_mid)
output_layer = Output_layer(n_mid, n_out, activation=softmax)

loss = plt.figure()
ax_loss = loss.add_subplot(1, 1, 1)

sin_data = np.sin(np.pi * X)

for i in range(epoch):
    index_random = np.arange(n_data)
    np.random.shuffle(index_random)

    # for output result
    total_error = 0
    x_1 = []
    x_2 = []
    y_1 = []
    y_2 = []

    for idx in index_random:
        x = input_data[idx]
        t = correct_data[idx]

        # forward
        middle_layer.forward(x.reshape(1,2))
        output_layer.forward(middle_layer.y)

        # backward
        output_layer.backward(t.reshape(1, 2))
        middle_layer.backward(output_layer.grad_x)

        # update
        middle_layer.update(eta)
        output_layer.update(eta)

        if i % interval == 0:
            # tensor to vector
            y = output_layer.y.reshape(-1)

            total_error += cross_entropy(y ,t)
            
            if y[0] > y[1]:
                x_1.append(x[0])
                y_1.append(x[1])
            else:
                x_2.append(x[0])
                y_2.append(x[1])

    if i % interval == 0:
        # output graph
        reg = plt.figure()
        ax_reg = reg.add_subplot(1,1,1)
        ax_reg.plot(X, sin_data, linestyle="dashed")
        ax_reg.scatter(x_1, y_1, marker="+")
        ax_reg.scatter(x_2, y_2, marker="x")
        reg.show()
        reg.savefig('out_backprop_class_{}.png'.format(i))
        ax_loss.plot(i, total_error/n_data, marker="o", color = 'b', linestyle="dashed")
        loss.savefig('loss_class.png')
        print("Epoch:" + str(i) + "/" + str(epoch),
              "Error:" + str(total_error/n_data))
    