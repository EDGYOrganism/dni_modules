from abc import ABC, abstractmethod

import torch.nn as nn

from .activation_builder import ActivationBuilder, ReLUActivationBuilder


class DNIBuilder(ABC):
    """Abstract DNIBuilder interface"""

    @abstractmethod
    def build(self):
        pass


class LinearDNIBuilder(DNIBuilder):
    """Builder for Linear Decoupled Neural Interfaces

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool, optional
        If set to False, the DNI will not learn an additive bias. By default True.
    activation_builder : ActivationBuilder, optional
        Activation builder, by default ReLUActivationBuilder()
    batch_norm : bool, optional
        If set to False, the DNI will not include a batch normalization layer. By default True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_builder: ActivationBuilder = ReLUActivationBuilder(),
        batch_norm: bool = True,
    ):

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_builder = activation_builder
        self.batch_norm = batch_norm

    def build(self):
        """Builds Linear Decoupled Neural Interface.

        Returns
        -------
        torch.nn.modules.container.Sequential
            built nn.Sequential() containing layers for a Linear DNI
        """
        layers = [nn.Linear(self.in_features, self.out_features, self.bias)]

        if self.batch_norm:
            layers.append(nn.BatchNorm1d(num_features=self.out_features))

        layers.append(self.activation_builder.build())
        return nn.Sequential(*layers)


class Conv2dDNIBuilder(DNIBuilder):
    """Builder for Conv2d Decoupled Neural Intefaces

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    out_channels : int
        Number of channels produced by the convolution
    kernel_size : int
        Size of convolving kernel
    stride : int, optional
        Stride of the convolution
    padding : int | str, optional
        Padding added to all sides of the input, by default "same"
    pooling : type[nn.Module], optional
        Type of pooling layer, by default nn.MaxPool2d
    pooling_kernel_size : int, optional
        Size of pooling kernel, by default 3
    bias : bool, optional
        If set to False, the DNI will not learn an additive bias. By default True.
    activation_builder : ActivationBuilder, optional
        Activation builder, by default ReLUActivationBuilder()
    batch_norm : bool, optional
        If set to False, the DNI will not include a batch normalization layer. By default True.

    Raises
    ------
    ValueError
        Convolution kernels with even size are not supported.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = "same",
        pooling: type[nn.Module] = nn.MaxPool2d,
        pooling_kernel_size: int = 3,
        bias: bool = True,
        activation_builder: ActivationBuilder = ReLUActivationBuilder(),
        batch_norm: bool = True,
    ):

        if kernel_size % 2 == 0:
            raise ValueError(
                f"Even convolution kernel sizes not supported: {kernel_size}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pooling = pooling
        self.pooling_kernel_size = pooling_kernel_size
        self.bias = bias
        self.activation_builder = activation_builder
        self.batch_norm = batch_norm

    def build(self):
        """Builds Conv2d Decoupled Neural Interface.

        Returns
        -------
        torch.nn.modules.container.Sequential
            built nn.Sequential() containing layers for a Conv2d DNI
        """
        layers = [
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                bias=self.bias,
            )
        ]

        if self.batch_norm:
            layers.append(nn.BatchNorm2d(num_features=self.out_channels))

        layers.append(self.activation_builder.build())
        layers.append(self.pooling(self.pooling_kernel_size))

        return nn.Sequential(*layers)
