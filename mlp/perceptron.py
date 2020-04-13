#!/usr/bin/python
# -*- coding: utf-8 -*-

"""単純パーセプトロンの実装
Todo:
    *パーセプトロンのクラス化
    *活性化関数の分離
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# generate data
X = np.arange(-1.0, 1.0, 0.2)
Y = np.arange(-1.0, 1.0, 0.2)

Z = np.zeros((10,10))

# weights for inputs and bias
w_x = 2.5
w_y = 3.0

bias = 0.8

for i in range(10):
    for j in range(10):
        u = X[i]*w_x + Y[j]*w_y + bias
        y = 1/(1+np.exp(-u)) # sigmoid
        Z[j][i] = y

plt.imshow(Z, "gray", vmin = 0.0, vmax = 1.0)
plt.colorbar()

plt.savefig('out_perceptron.png')
