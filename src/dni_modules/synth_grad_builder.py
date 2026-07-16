from abc import ABC, abstractmethod

import torch.nn as nn

from .activation_builder import ActivationBuilder, ReLUActivationBuilder


class SynthGradBuilder(ABC):
    @abstractmethod
    def build(self):
        pass


class LinearSynthGradBuilder(SynthGradBuilder):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layer_size: int,
        bias: bool,
        activation_builder: type(ActivationBuilder) = ReLUActivationBuilder(),
        num_hidden: int = 1,
    ):

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layer_size = hidden_layer_size
        self.bias = bias
        self.activation_builder = activation_builder
        self.num_hidden = num_hidden

    def build(self):
        layers = []
        if self.num_hidden == 0:
            layers.append(
                nn.Linear(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    bias=self.bias,
                )
            )
        else:
            for i in range(self.num_hidden):
                in_features = self.in_features if i == 0 else self.hidden_layer_size
                layers.append(
                    nn.Linear(
                        in_features=in_features,
                        out_features=self.hidden_layer_size,
                        bias=self.bias,
                    )
                )
                layers.append(nn.BatchNorm1d(num_features=self.hidden_layer_size))
                layers.append(self.activation_builder.build())
            layers.append(
                nn.Linear(
                    in_features=self.hidden_layer_size,
                    out_features=self.out_features,
                    bias=self.bias,
                )
            )

        net = nn.Sequential(*layers)
        # Initialize parameters of output layer with zeros
        nn.init.zeros_(net[-1].weight)
        if net[-1].bias is not None:
            nn.init.zeros_(net[-1].bias)

        return net


class Conv2dSynthGradBuilder(SynthGradBuilder):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_layer_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | str,
        bias: bool,
        activation_builder: type(ActivationBuilder) = ReLUActivationBuilder(),
        num_hidden: int = 1,
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_layer_channels = hidden_layer_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.activation_builder = activation_builder
        self.num_hidden = num_hidden

    def build(self):
        layers = []
        if self.num_hidden == 0:
            layers.append(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=self.bias,
                )
            )
        else:
            for i in range(self.num_hidden):
                in_channels = self.in_channels if i == 0 else self.hidden_layer_channels
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=self.hidden_layer_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        bias=self.bias,
                    )
                )
                layers.append(nn.BatchNorm2d(num_features=self.hidden_layer_channels))
                layers.append(self.activation_builder.build())
            layers.append(
                nn.Conv2d(
                    in_channels=self.hidden_layer_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=self.bias,
                )
            )

        net = nn.Sequential(*layers)
        # Initialize parameters of output layer with zeros
        nn.init.zeros_(net[-1].weight)
        if net[-1].bias is not None:
            nn.init.zeros_(net[-1].bias)

        return net
