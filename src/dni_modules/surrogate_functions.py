from abc import ABC, abstractmethod

import torch


class SurrogateGrad(ABC):
    """Abstract SurrogateGrad interface"""

    @abstractmethod
    def surrogate_grad(self, x: torch.Tensor):
        pass


class ATan(torch.autograd.Function):
    """
    ArcTan surrogate gradient
    **Forward pass:** Heaviside function
    **Backward pass:** Override Dirac Delta with the derivative of the ArcTan function
    """

    @staticmethod
    def surrogate_grad(x: torch.Tensor):
        """Computes ArcTan surrogate gradient.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            ArcTan surrogate gradient
        """
        return 1.0 / (1.0 + (torch.pi * x).pow(2))

    @staticmethod
    def forward(ctx, mem: torch.Tensor, hor_scaling: float = 1.0):
        """Forward pass

        Parameters
        ----------
        mem : torch.Tensor
            Membrane potential
        hor_scaling : float, optional
            Horizontal scaling used to change the effective range of surrogate function, by default 1.0

        Returns
        -------
        torch.Tensor
            Output spikes
        """
        spk = torch.zeros(0, requires_grad=True)
        spk = (mem > 0).float()  # Heaviside on the forward pass
        ctx.save_for_backward(
            mem
        )  # Store the membrane and hor_scaling for use in the backward pass
        ctx.hor_scaling = hor_scaling
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass"""
        (mem,) = ctx.saved_tensors  # Retrieve the membrane potential
        grad = (
            ATan.surrogate_grad(mem * ctx.hor_scaling) * grad_output
        )  # Apply horizontal scaling to change the effective range of surrogate function
        return grad, None


class ATanSurrogate(SurrogateGrad):
    """Wrapper for ATan class

    Parameters
    ----------
    hor_scaling : float, optional
        Horizontal scaling used to change the effective range of surrogate function, by default 1.0
    """

    def __init__(self, hor_scaling: float = 1.0):
        self.hor_scaling = hor_scaling

    def __call__(self, x: torch.Tensor):
        return ATan.apply(x, self.hor_scaling)

    def surrogate_grad(self, x: torch.Tensor):
        return ATan.surrogate_grad(x * self.hor_scaling)


class Triangle(torch.autograd.Function):
    """
    Triangle surrogate gradient
    **Forward pass:** Heaviside function
    **Backward pass:** Override Dirac Delta with triangle-shaped function
    """

    @staticmethod
    def surrogate_grad(x: torch.Tensor, gamma: float):
        """Computes Triangle surrogate gradient.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        gamma : float
            Gradient scaling parameter

        Returns
        -------
        torch.Tensor
            Triangle surrogate gradient
        """
        return gamma * torch.maximum(torch.tensor(0.0), 1 - torch.abs(x))

    @staticmethod
    def forward(ctx, mem: torch.Tensor, gamma: float = 0.3):
        """Forward pass

        Parameters
        ----------
        mem : torch.Tensor
            Membrane potential
        gamma : float, optional
             Gradient scaling parameter, by default 0.3

        Returns
        -------
        torch.Tensor
            Output spikes
        """
        ctx.save_for_backward(mem)
        ctx.gamma = gamma
        return (mem > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass"""
        (mem,) = ctx.saved_tensors
        grad = Triangle.surrogate_grad(mem, ctx.gamma) * grad_output
        return grad, None


class TriangleSurrogate(SurrogateGrad):
    """Wrapper for Triangle class

    Parameters
    ----------
    gamma : float, optional
        Gradient scaling parameter, by default 0.3

    """

    def __init__(self, gamma=0.3):
        self.gamma = gamma

    def __call__(self, x: torch.Tensor):
        return Triangle.apply(x, self.gamma)

    def surrogate_grad(self, x: torch.Tensor):
        return Triangle.surrogate_grad(x, self.gamma)
