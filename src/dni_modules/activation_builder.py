from abc import ABC, abstractmethod

import torch.nn as nn
import snntorch


class ActivationBuilder(ABC):
    @abstractmethod
    def build(self):
        pass


class ReLUActivationBuilder(ActivationBuilder):
    def build(self):
        return nn.ReLU()


class LeakyActivationBuilder(ActivationBuilder):
    def __init__(self, beta: float = 0.9, threshold: float = 1.0):
        self.beta = beta
        self.threshold = threshold

    def build(self):
        return snntorch.Leaky(beta=self.beta, threshold=self.threshold)
