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
from loss_func import square_sum

input_data = np.arange(0, np.pi*2, 0.1)
correct_data = np.sin(input_data)
input_data = (input_data - np.pi)/np.pi
n_data = len(correct_data)

# num of neurons in each layer
n_in = 1
n_mid = 3
n_out = 1

eta = 0.1
epoch = 2001
interval = 200

middle_layer = Middle_layer(n_in, n_mid)
output_layer = Output_layer(n_mid, n_out)

loss = plt.figure()
ax_loss = loss.add_subplot(1, 1, 1)

for i in range(epoch):
    index_random = np.arange(n_data)
    np.random.shuffle(index_random)

    # for output result
    total_error = 0
    plot_x = []
    plot_y = []

    for idx in index_random:
        x = input_data[idx: idx + 1]
        t = correct_data[idx: idx + 1]

        # forward
        middle_layer.forward(x.reshape(1,1))
        output_layer.forward(middle_layer.y)

        # backward
        output_layer.backward(t.reshape(1, 1))
        middle_layer.backward(output_layer.grad_x)

        # update
        middle_layer.update(eta)
        output_layer.update(eta)

        if i % interval == 0:
            # tensor to vector
            y = output_layer.y.reshape(-1)

            total_error += square_sum(y ,t)
            
            plot_x.append(x)
            plot_y.append(y)

    if i % interval == 0:
        # output graph
        reg = plt.figure()
        ax_reg = reg.add_subplot(1,1,1)
        ax_reg.plot(input_data, correct_data, linestyle="dashed")
        ax_reg.scatter(plot_x, plot_y, marker="+")
        reg.show()
        reg.savefig('out_backprop_{}.png'.format(i))
        ax_loss.plot(i, total_error/n_data, marker="o", color = 'b', linestyle="dashed")
        loss.savefig('loss.png')
        print("Epoch:" + str(i) + "/" + str(epoch),
              "Error:" + str(total_error/n_data))
    