#!/usr/bin/python
# -*- coding: utf-8 -*-
"""活性化関数
よく使う活性化関数をまとめたもの

todo:
    *ReLuの追加
"""

import numpy as np

def sigmoid(x):
    """Sigmoid function

    Args:
        x: input for function
    
    Returns:
        The anser of sigmoid function for x
    """
    return 1/(1+np.exp(-x))

def softmax(x):
    """softmax function

    Args:
        x: input for function
    
    Returns:
        The anser of softmax function for x
    """
    return np.exp(x)/np.sum(np.exp(x))