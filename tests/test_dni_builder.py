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
        [True, False],  # batch_norm
    )
)


CONV2D_DNI_BUILDER_TEST_CASES = list(
    product(
        [1, 3],  # in_channels
        [2, 3],  # out_channels
        [2, 3, 4],  # kernel_size
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
        [True, False],  # batch_norm
    )
)


def test_dni_builder_raises_error():
    """Test abstract class DNIBuilder raises TypeError if an attempt is made to create an instance"""
    with pytest.raises(
        TypeError,
    ):
        _ = DNIBuilder()


@pytest.mark.parametrize(
    "in_features, out_features, bias, activation_builder, batch_norm",
    LINEAR_DNI_BUILDER_TEST_CASES,
)
def test_linear_dni_builder_init(
    in_features, out_features, bias, activation_builder, batch_norm
):
    """Test the __init__ function of LinearDNIBuiler class"""
    builder = LinearDNIBuilder(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        activation_builder=activation_builder,
        batch_norm=batch_norm,
    )
    assert builder.in_features == in_features
    assert builder.out_features == out_features
    assert builder.bias == bias
    assert type(builder.activation_builder) is type(activation_builder)
    assert builder.batch_norm == batch_norm


@pytest.mark.parametrize(
    "in_features, out_features, bias, activation_builder, batch_norm",
    LINEAR_DNI_BUILDER_TEST_CASES,
)
def test_linear_dni_builder_build(
    in_features, out_features, bias, activation_builder, batch_norm
):
    """Test the build() function of LinearDNIBuiler class"""
    builder = LinearDNIBuilder(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        activation_builder=activation_builder,
        batch_norm=batch_norm,
    )
    dni = builder.build()
    i = 0  # layer index
    assert len(dni) == 3 if batch_norm else 2
    assert type(dni[i]) is nn.Linear
    assert dni[i].weight.shape == (out_features, in_features)
    if bias is True:
        assert dni[i].bias is not None
        assert dni[i].bias.shape == (out_features,)
    else:
        assert dni[i].bias is None

    i += 1
    if batch_norm:
        assert type(dni[i]) is nn.BatchNorm1d
        assert dni[i].num_features == out_features
        i += 1

    if type(activation_builder) is LeakyActivationBuilder:
        assert type(dni[i]) is snntorch.Leaky
        assert torch.isclose(dni[i].beta, torch.tensor(builder.activation_builder.beta))
        assert torch.isclose(
            dni[i].threshold, torch.tensor(builder.activation_builder.threshold)
        )


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, pooling, pooling_kernel_size, bias, activation_builder, batch_norm",
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
    batch_norm,
):
    """Test __init__ function of Conv2dDNIBuilder class"""
    try:
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
            batch_norm=batch_norm,
        )
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == f"Odd kernel sizes not supported: {kernel_size}"
    else:
        assert builder.in_channels == in_channels
        assert builder.out_channels == out_channels
        assert builder.kernel_size == kernel_size
        assert builder.stride == stride
        assert builder.padding == padding
        assert builder.pooling == pooling
        assert builder.pooling_kernel_size == pooling_kernel_size
        assert builder.bias == bias
        assert type(builder.activation_builder) is type(activation_builder)
        assert batch_norm == batch_norm


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, pooling, pooling_kernel_size, bias, activation_builder, batch_norm",
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
    batch_norm,
):
    """Test build() function of Conv2dDNIBuilder class"""
    try:
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
            batch_norm=batch_norm,
        )
    except Exception as e:
        assert isinstance(e, ValueError)
        assert str(e) == f"Odd kernel sizes not supported: {kernel_size}"
    else:
        dni = builder.build()
        i = 0  # layer index
        assert len(dni) == 4 if batch_norm else 3
        assert type(dni[i]) is nn.Conv2d
        assert dni[i].weight.shape == (
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        )
        if bias is True:
            assert dni[i].bias is not None
            assert dni[i].bias.shape == (out_channels,)
        else:
            assert dni[i].bias is None

        i += 1
        if batch_norm:
            assert type(dni[i]) is nn.BatchNorm2d
            assert dni[i].num_features == out_channels
            i += 1

        if type(activation_builder) is LeakyActivationBuilder:
            assert type(dni[i]) is snntorch.Leaky
            assert torch.isclose(
                dni[i].beta, torch.tensor(builder.activation_builder.beta)
            )
            assert torch.isclose(
                dni[i].threshold, torch.tensor(builder.activation_builder.threshold)
            )

        i += 1
        assert type(dni[i]) is pooling
        assert dni[i].kernel_size == pooling_kernel_size
