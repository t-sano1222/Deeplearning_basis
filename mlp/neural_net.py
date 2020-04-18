#!/usr/bin/python
# -*- coding: utf-8 -*-

"""パーセプトロンのクラス化
   neuron_classから活性化関数を可変にし，層ごとの分離をやめた．
Todo:
    *勾配計算法を可変にする
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from activation_functions import sigmoid
from activation_functions import softmax

wb_width = 0.01

class Fully_conected_layer():
    """fully connected layer

    layer for multi layered perceptron
    
    Attributes:
        属性の名前 (属性の型): 属性の説明
        属性の名前 (:obj:`属性の型`): 属性の説明.
    """

    def __init__(self, n_upper, n, activation=sigmoid):
        """initialize
        set weights and biases

        Args:
            n_upper: num of neurons in upper layer
            n: num of neurons in this layer
        """
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.random(n)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = activation(u)       
    
    def backward(self, grad_y):
        # differential for sigmoid
        delta = grad_y * (1 - self.y) * self.y

        # calculate grad vector
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        self.grad_x = np.dot(delta, self.w.T)

        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

class Model():
    """Model of neural net

    Fully connected layerを結合してモデル全体をまとめて計算できるようにした

    Attributes:
        属性の名前 (属性の型): 属性の説明
        属性の名前 (:obj:`属性の型`): 属性の説明.

    """
    def __init__(self, *args, eta=0.1):
        self.layers = args
        for l in self.layers:
            l.set_lr(eta=eta)
    
    def forward(self, x):
        # write your process
    
    def backward(self, t):
        # write your process