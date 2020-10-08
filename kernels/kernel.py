# Recyclable Gaussian Processes
# Pablo Moreno-Munoz (pmoreno@tsc.uc3m.es)
# Universidad Carlos III de Madrid
# January 2020


import torch
import numpy as np
from util import squared_distance

class Kernel(torch.nn.Module):
    """
    Base class for kernels
    """
    def __init__(self, input_dim=None):
        super(Kernel, self).__init__()

        # Input dimension -- x
        if input_dim is None:
            input_dim = 1
        else:
            input_dim = int(input_dim)

        self.input_dim = input_dim