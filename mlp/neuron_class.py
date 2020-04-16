#!/usr/bin/python
# -*- coding: utf-8 -*-

"""パーセプトロンのクラス化
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

class Middle_layer():
    """middle layer

    middle layer for multi layered perceptron
    
    Attributes:
        属性の名前 (属性の型): 属性の説明
        属性の名前 (:obj:`属性の型`): 属性の説明.
    """

    def __init__(self, n_upper, n):
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
        self.y = sigmoid(u)       
    
    def backward(self, grad_y):
        # differential for sigmoid
        delta = grad_y * (1 - self.y) * self.y

        # calculate grad vector
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        self.grad_x = np.dot(delta, self.w.T)
    
    def update(self, eta):
        # SGD
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

class Output_layer():
    """output layer

    output layer for multi layered perceptron

    Attributes:
        属性の名前 (属性の型): 属性の説明
        属性の名前 (:obj:`属性の型`): 属性の説明.

    """

    def __init__(self, n_upper, n):
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
        self.y = u       
    
    def backward(self, t):
        delta = self.y - t

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

        self.grad_x = np.dot(delta, self.w.T)
    
    def update(self, eta):
        # SGD
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b    

       