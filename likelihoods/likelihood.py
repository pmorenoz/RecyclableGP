# Recyclable Gaussian Processes
# Pablo Moreno-Munoz (pmoreno@tsc.uc3m.es)
# Universidad Carlos III de Madrid
# January 2020

import torch
import numpy as np

class Likelihood(torch.nn.Module):
    """
    Base class for likelihoods
    """
    def __init__(self):
        super(Likelihood, self).__init__()

    def gh_points(self, T=20):
        # Gaussian-Hermite Quadrature points
        gh_p, gh_w = np.polynomial.hermite.hermgauss(T)
        gh_p, gh_w = torch.from_numpy(gh_p), torch.from_numpy(gh_w)
        return gh_p, gh_w
