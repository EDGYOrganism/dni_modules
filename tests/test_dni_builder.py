from itertools import product

import pytest

import torch
import torch.nn as nn
import snntorch

from dni_modules import (
    DNIBuilder,
    LinearDNIBuilder,
    Conv2dDNIBuilder,
    ReLUActivationBuilder,
    LeakyActivationBuilder,
)


LINEAR_DNI_BUILDER_TEST_CASES = list(
    product(
        [1, 10],  # in_features
        [10, 20],  # out_features
        [True, False],  # bias
        [
            ReLUActivationBuilder(),
            LeakyActivationBuilder(),
            LeakyActivationBuilder(beta=0.8, threshold=0.5),
        ],  # activation_builder
    )
)


CONV2D_DNI_BUILDER_TEST_CASES = list(
    product(
        [1, 3],  # in_channels
        [2, 3],  # out_channels
        [2, 3],  # kernel_size
        [1],  # stride
        [0, "same"],  # padding
        [nn.MaxPool2d, nn.AvgPool2d],  # pooling
        [3, 5],  # pooling_kernel_size
        [True, False],  # bias
        [
            ReLUActivationBuilder(),
            LeakyActivationBuilder(),
            LeakyActivationBuilder(beta=0.8, threshold=0.5),
        ],  # activation_builder
    )
)


def test_dni_builder_raises_error():
    """Test abstract class DNIBuilder raises TypeError if an attempt is made to create an instance"""
    with pytest.raises(
        TypeError,
    ):
        _ = DNIBuilder()


@pytest.mark.parametrize(
    "in_features, out_features, bias, activation_builder", LINEAR_DNI_BUILDER_TEST_CASES
)
def test_linear_dni_builder_init(in_features, out_features, bias, activation_builder):
    """Test the __init__ function of LinearDNIBuiler class"""
    builder = LinearDNIBuilder(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        activation_builder=activation_builder,
    )
    assert builder.in_features == in_features
    assert builder.out_features == out_features
    assert builder.bias == bias
    assert type(builder.activation_builder) is type(activation_builder)


@pytest.mark.parametrize(
    "in_features, out_features, bias, activation_builder", LINEAR_DNI_BUILDER_TEST_CASES
)
def test_linear_dni_builder_build(in_features, out_features, bias, activation_builder):
    """Test the build() function of LinearDNIBuiler class"""
    builder = LinearDNIBuilder(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        activation_builder=activation_builder,
    )
    dni = builder.build()
    assert len(dni) == 3
    assert type(dni[0]) is nn.Linear
    assert dni[0].weight.shape == (out_features, in_features)
    if bias is True:
        assert dni[0].bias is not None
        assert dni[0].bias.shape == (out_features,)
    else:
        assert dni[0].bias is None
    assert type(dni[1]) is nn.BatchNorm1d
    assert dni[1].num_features == out_features

    if type(activation_builder) is LeakyActivationBuilder:
        assert type(dni[2]) is snntorch.Leaky
        assert torch.isclose(dni[2].beta, torch.tensor(builder.activation_builder.beta))
        assert torch.isclose(
            dni[2].threshold, torch.tensor(builder.activation_builder.threshold)
        )


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, pooling, pooling_kernel_size, bias, activation_builder",
    CONV2D_DNI_BUILDER_TEST_CASES,
)
def test_conv2d_dni_builder_init(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    pooling,
    pooling_kernel_size,
    bias,
    activation_builder,
):
    """Test __init__ function of Conv2dDNIBuilder class"""
    builder = Conv2dDNIBuilder(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pooling=pooling,
        pooling_kernel_size=pooling_kernel_size,
        bias=bias,
        activation_builder=activation_builder,
    )

    assert builder.in_channels == in_channels
    assert builder.out_channels == out_channels
    assert builder.kernel_size == kernel_size
    assert builder.stride == stride
    assert builder.padding == padding
    assert builder.pooling == pooling
    assert builder.pooling_kernel_size == pooling_kernel_size
    assert builder.bias == bias
    assert type(builder.activation_builder) is type(activation_builder)


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, pooling, pooling_kernel_size, bias, activation_builder",
    CONV2D_DNI_BUILDER_TEST_CASES,
)
def test_conv2d_dni_builder_build(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    pooling,
    pooling_kernel_size,
    bias,
    activation_builder,
):
    """Test build() function of Conv2dDNIBuilder class"""
    builder = Conv2dDNIBuilder(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pooling=pooling,
        pooling_kernel_size=pooling_kernel_size,
        bias=bias,
        activation_builder=activation_builder,
    )

    dni = builder.build()
    assert len(dni) == 4
    assert type(dni[0]) is nn.Conv2d
    assert dni[0].weight.shape == (out_channels, in_channels, kernel_size, kernel_size)
    if bias is True:
        assert dni[0].bias is not None
        assert dni[0].bias.shape == (out_channels,)
    else:
        assert dni[0].bias is None
    assert type(dni[1]) is nn.BatchNorm2d
    assert dni[1].num_features == out_channels

    if type(activation_builder) is LeakyActivationBuilder:
        assert type(dni[2]) is snntorch.Leaky
        assert torch.isclose(dni[2].beta, torch.tensor(builder.activation_builder.beta))
        assert torch.isclose(
            dni[2].threshold, torch.tensor(builder.activation_builder.threshold)
        )

    assert type(dni[3]) is pooling
    assert dni[3].kernel_size == pooling_kernel_size
