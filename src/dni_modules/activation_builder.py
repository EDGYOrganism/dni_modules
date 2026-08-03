from abc import ABC, abstractmethod

import torch.nn as nn
import snntorch


class ActivationBuilder(ABC):
    """Abstract ActivationBuilder interface"""

    @abstractmethod
    def build(self):
        pass


class ReLUActivationBuilder(ActivationBuilder):
    """Builder for ReLU activations"""

    def build(self):
        """Builds ReLU activation.

        Returns
        -------
        torch.nn.modules.activation.ReLU
            torch ReLU activation
        """
        return nn.ReLU()


class LeakyActivationBuilder(ActivationBuilder):
    """Builder for Leaky Integrate & Fire activations

    Parameters
    ----------
    beta : float, optional
        Decay constant, by default 0.9
    threshold : float, optional
        Firing threshold, by default 1.0
    """

    def __init__(self, beta: float = 0.9, threshold: float = 1.0):
        self.beta = beta
        self.threshold = threshold

    def build(self):
        """Builds Leaky Integrate & Fire activation.

        Returns
        -------
        snntorch._neurons.leaky.Leaky
            snntorch Leaky activation
        """
        return snntorch.Leaky(beta=self.beta, threshold=self.threshold)
