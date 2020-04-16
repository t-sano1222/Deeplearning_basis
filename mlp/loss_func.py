#!/usr/bin/python
# -*- coding: utf-8 -*-
"""loss func
Original library about loss function 

todo:
    *add cross entropy
"""
import numpy as np

def square_sum(y, t):
    """二乗和誤差

    Args:
        y: output of mlp
        t: true value
    
    Returns:
        Sum of squares error
    """
    return 1.0/2.0 * np.sum(np.square(y - t))

def cross_entropy(y, t):
    """cross entropy

    Args:
        y: output of mlp
        t: true value

    Returns:
        Cross entropy error
    """
    return -np.sum(t * np.log(y * 1e-7))