# Based on gptorch code by Steven Atkinson (steven@atkinson.mn)
# ------------------------------------------------
# Recyclable Gaussian Processes
# Pablo Moreno-Munoz (pmoreno@tsc.uc3m.es)
# Universidad Carlos III de Madrid
# January 2020
# ------------------------------------------------


import torch
import numpy as np


def true_function(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi) + \
        5*torch.cos(7*np.pi*x + 2.4*np.pi)
    return y

def smooth_function(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi)
    return y

def smooth_function_bias(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi) + \
        3.0*x - 7.5
    return y

def squared_distance(x1, x2=None):
    """
    Given points x1 [n1 x d1] and x2 [n2 x d2], return a [n1 x n2] matrix with
    the pairwise squared distances between the points.
    Entry (i, j) is sum_{j=1}^d (x_1[i, j] - x_2[i, j]) ^ 2
    """
    if x2 is None:
        return squared_distance(x1, x1)

    x1s = x1.pow(2).sum(1, keepdim=True)
    x2s = x2.pow(2).sum(1, keepdim=True)

    r2 = x1s + x2s.t() -2.0 * x1 @ x2.t()

    # Prevent negative squared distances using torch.clamp
    # NOTE: Clamping is for numerics.
    # This use of .detach() is to avoid breaking the gradient flow.
    return r2 - (torch.clamp(r2, max=0.0)).detach()