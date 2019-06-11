from typing import Any, Union

import numpy as np
from numpy.core.multiarray import ndarray

"""
DO NOT EDIT ANY PARTS OTHER THAN "EDIT HERE" !!! 

[Description]
__init__ - Initialize necessary variables for optimizer class
input   : gamma, epsilon
return  : X

update   - Update weight for one minibatch
input   : w - current weight, grad - gradient for w, lr - learning rate
return  : updated weight 
"""


class SGD:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        pass

        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        updated_weight = w - grad * lr

        # =============================================================
        return updated_weight


class Momentum:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        self.gamma = gamma
        self.v = None
        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        if self.v is None:
            self.v = np.zeros_like(w)
        self.v = self.gamma * self.v - lr * grad
        updated_weight = w + self.v
        # =============================================================
        return updated_weight


class RMSProp:
    # ========================= EDIT HERE =========================
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        self.epsilon = epsilon
        self.decay_rate = 0.99
        self.cache = None

        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        if self.cache is None:
            self.cache = np.zeros_like(w)
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * grad**2
        updated_weight = w - lr * grad / np.sqrt(self.cache) + self.epsilon

        # =============================================================
        return updated_weight
