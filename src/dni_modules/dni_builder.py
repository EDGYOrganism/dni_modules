from abc import ABC, abstractmethod

import torch.nn as nn

from .activation_builder import ActivationBuilder, ReLUActivationBuilder


class DNIBuilder(ABC):
    @abstractmethod
    def build(self):
        pass


class LinearDNIBuilder(DNIBuilder):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_builder: type(ActivationBuilder) = ReLUActivationBuilder(),
        batch_norm: bool = True,
    ):

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_builder = activation_builder
        self.batch_norm = batch_norm

    def build(self):
        layers = [nn.Linear(self.in_features, self.out_features, self.bias)]

        if self.batch_norm:
            layers.append(nn.BatchNorm1d(num_features=self.out_features))

        layers.append(self.activation_builder.build())
        return nn.Sequential(*layers)


class Conv2dDNIBuilder(DNIBuilder):
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
        activation_builder: type(ActivationBuilder) = ReLUActivationBuilder(),
        batch_norm: bool = True,
    ):

        if kernel_size % 2 == 0:
            raise ValueError(f"Odd kernel sizes not supported: {kernel_size}")

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
