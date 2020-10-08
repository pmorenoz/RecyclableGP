# Recyclable Gaussian Processes
# Pablo Moreno-Munoz (pmoreno@tsc.uc3m.es)
# Universidad Carlos III de Madrid
# January 2020

import torch
from util import squared_distance
from kernels.kernel import Kernel

class Stationary(Kernel):
    """
    Class for Stationary Kernel
    """

    def __init__(self, variance=None, length_scale=None, input_dim=None):
        super().__init__(input_dim)

        # Length-scale/smoothness of the kernel -- l
        if length_scale is None:
            length_scale = 0.1

        # Variance/amplitude of the kernel - /sigma
        if variance is None:
            variance = 2.0

        self.length_scale = torch.nn.Parameter(length_scale*torch.ones(1), requires_grad=True)
        self.variance = torch.nn.Parameter(variance*torch.ones(1), requires_grad=True)
        self.register_parameter('length_scale', self.length_scale)
        self.register_parameter('variance', self.variance)

    def squared_dist(self, X, X2):
        """
        Returns the SCALED squared distance between X and X2.
        """
        length_scale = self.length_scale.abs().clamp(min=0.0, max=10.0)

        if X2 is None:
            dist = squared_distance(X/length_scale)
        else:
            dist = squared_distance(X/length_scale, X2/length_scale)

        return dist

    def Kdiag(self, X):
        variance = torch.abs(self.variance)
        return variance.expand(X.size(0))