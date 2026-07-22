from itertools import product

import pytest

import torch
import torch.nn as nn


from dni_modules import (
    DNI,
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
        [3, 5],  # kernel_size
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


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=LINEAR_DNI_BUILDER_TEST_CASES)
def linear_dni_builder_test_cases(request):
    return request.param


@pytest.fixture(params=CONV2D_DNI_BUILDER_TEST_CASES)
def conv2d_dni_builder_test_cases(request):
    return request.param


@pytest.fixture
def linear_dni_builder(linear_dni_builder_test_cases):
    in_features, out_features, bias, activation_builder, batch_norm = (
        linear_dni_builder_test_cases
    )
    return LinearDNIBuilder(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        activation_builder=activation_builder,
        batch_norm=batch_norm,
    )


@pytest.fixture
def conv2d_dni_builder(conv2d_dni_builder_test_cases):
    (
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
    ) = conv2d_dni_builder_test_cases
    return Conv2dDNIBuilder(
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


def test_linear_dni_init(linear_dni_builder):
    """Test __init__ function of DNI class with a LinearDNIBuilder instance"""
    dni = DNI(linear_dni_builder)

    # Check net
    assert len(dni.net) == 3 if linear_dni_builder.batch_norm else 2

    # Check elig_tr
    assert dni.elig_tr.shape == dni.net[0].weight.shape
    assert torch.count_nonzero(dni.elig_tr).item() == 0


def test_conv2d_dni_init(conv2d_dni_builder):
    """Test the __init__ function of DNI class with a Conv2dDNIBuilder instance"""
    dni = DNI(conv2d_dni_builder)

    # Check net
    assert len(dni.net) == 4 if conv2d_dni_builder.batch_norm else 3

    # Check elig_tr
    assert dni.elig_tr.shape == dni.net[0].weight.shape
    assert torch.count_nonzero(dni.elig_tr).item() == 0


def test_linear_dni_forward(linear_dni_builder, device):
    """Test the forward() function of a DNI instance built with LinearDNIBuilder"""
    # Batch size
    B = 4
    x = torch.randn((B, linear_dni_builder.in_features), device=device)

    dni = DNI(linear_dni_builder).to(device)
    out = dni(x)
    assert out.shape[0] == B
    assert out.shape[1] == linear_dni_builder.out_features


def test_conv2d_dni_forward(conv2d_dni_builder, device):
    """Test the forward() function of a DNI instance built with Conv2dDNIBuilder"""
    # Batch size
    B = 4
    height, width = (9, 9)
    x = torch.randn((B, conv2d_dni_builder.in_channels, height, width), device=device)

    dni = DNI(conv2d_dni_builder).to(device)
    out = dni(x)
    assert out.shape[0] == B
    assert out.shape[1] == conv2d_dni_builder.out_channels
    if conv2d_dni_builder.padding == "same":
        assert out.shape[2] == int(height / conv2d_dni_builder.pooling_kernel_size)
        assert out.shape[3] == int(width / conv2d_dni_builder.pooling_kernel_size)
