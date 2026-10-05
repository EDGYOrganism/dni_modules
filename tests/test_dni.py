from itertools import product

import pytest

import torch
import torch.nn as nn
import snntorch

from dni_modules import (
    DNI,
    LinearDNIBuilder,
    LazyLinearDNIBuilder,
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


LAZY_LINEAR_DNI_BUILDER_TEST_CASES = list(
    product(
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


@pytest.fixture(params=LAZY_LINEAR_DNI_BUILDER_TEST_CASES)
def lazy_linear_dni_builder_test_cases(request):
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
def lazy_linear_dni_builder(lazy_linear_dni_builder_test_cases):
    out_features, bias, activation_builder = lazy_linear_dni_builder_test_cases
    return LazyLinearDNIBuilder(
        out_features=out_features,
        bias=bias,
        activation_builder=activation_builder,
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

    # Check elig_eps
    assert dni.elig_eps.shape == dni.layer.weight.shape
    assert torch.count_nonzero(dni.elig_eps).item() == 0


def test_lazy_linear_dni_init(lazy_linear_dni_builder):
    """Test __init__ function of DNI class with a LazyLinearDNIBuilder instance"""
    dni = DNI(lazy_linear_dni_builder)

    assert type(dni.net[0]) is nn.Flatten
    assert type(dni.layer) is nn.LazyLinear

    # Check net
    assert len(dni.net) == 3


def test_conv2d_dni_init(conv2d_dni_builder):
    """Test the __init__ function of DNI class with a Conv2dDNIBuilder instance"""
    dni = DNI(conv2d_dni_builder)

    # Check net
    assert len(dni.net) == 4 if conv2d_dni_builder.batch_norm else 3

    # Check elig_eps
    assert dni.elig_eps.shape == dni.layer.weight.shape
    assert torch.count_nonzero(dni.elig_eps).item() == 0


@pytest.mark.parametrize("eval", [True, False])
def test_linear_dni_forward(linear_dni_builder, device, eval):
    """Test the forward() function of a DNI instance built with LinearDNIBuilder"""
    # Batch size
    B = 4
    x = torch.randn((B, linear_dni_builder.in_features), device=device)

    dni = DNI(linear_dni_builder).to(device)

    if eval:
        dni.eval()

    out = dni(x)
    assert out.shape[0] == B
    assert out.shape[1] == linear_dni_builder.out_features
    if type(dni.activation) is snntorch.Leaky and dni.training:
        assert torch.equal(
            dni.elig_eps,
            torch.mean(x, axis=0).unsqueeze(0).expand((dni.layer.out_features, -1)),
        )
        elig_eps_prev = dni.elig_eps
        x_new = torch.randn((B, linear_dni_builder.in_features), device=device)
        _ = dni(x_new)
        assert torch.equal(
            dni.elig_eps,
            dni.activation.beta * elig_eps_prev
            + torch.mean(x_new, axis=0)
            .unsqueeze(0)
            .expand((dni.layer.out_features, -1)),
        )
    else:
        assert torch.count_nonzero(dni.elig_eps).item() == 0


@pytest.mark.parametrize("eval", [True, False])
def test_lazy_linear_dni_forward(lazy_linear_dni_builder, device, eval):
    """Test the forward() function of a DNI instance built with LinearDNIBuilder"""
    # Batch size
    B = 16
    c, w, h = 3, 10, 10
    x = torch.randn((B, c, w, h), device=device)

    dni = DNI(lazy_linear_dni_builder).to(device)

    if eval:
        dni.eval()
        assert dni.training is False

    out = dni(x)
    assert out.shape[0] == B
    assert out.shape[1] == lazy_linear_dni_builder.out_features
    if type(dni.activation) is snntorch.Leaky and dni.training:
        assert torch.equal(
            dni.elig_eps,
            torch.mean(x.reshape(B, -1), axis=0)
            .unsqueeze(0)
            .expand((dni.layer.out_features, -1)),
        )
        elig_eps_prev = dni.elig_eps
        x_new = torch.randn((B, c, w, h), device=device)
        _ = dni(x_new)
        assert torch.equal(
            dni.elig_eps,
            dni.activation.beta * elig_eps_prev
            + torch.mean(x_new.reshape(B, -1), axis=0)
            .unsqueeze(0)
            .expand((dni.layer.out_features, -1)),
        )
    else:
        assert torch.equal(dni.elig_eps, torch.zeros(1, device=device))


@pytest.mark.parametrize("eval", [True, False])
@pytest.mark.parametrize("B", [1, 4])  # Batch size
def test_conv2d_dni_forward(conv2d_dni_builder, device, B, eval):
    """Test the forward() function of a DNI instance built with Conv2dDNIBuilder"""
    height, width = (9, 9)
    x = torch.randn((B, conv2d_dni_builder.in_channels, height, width), device=device)

    dni = DNI(conv2d_dni_builder).to(device)

    if eval:
        dni.eval()

    out = dni(x)
    assert out.shape[0] == B
    assert out.shape[1] == conv2d_dni_builder.out_channels
    if conv2d_dni_builder.padding == "same":
        assert out.shape[2] == int(height / conv2d_dni_builder.pooling_kernel_size)
        assert out.shape[3] == int(width / conv2d_dni_builder.pooling_kernel_size)
    if type(dni.activation) is snntorch.Leaky and dni.training:
        # If padding has been set to "same" and kernel_size is odd, manually compute padding because unfold() function cannot accept string "same" for padding parameter.
        if dni.layer.padding == "same" and dni.layer.kernel_size[0] % 2 == 1:
            padding = dni.layer.kernel_size[0] // 2
        else:
            padding = dni.layer.padding

        # Extract input patches of size k x k that the convolution sees
        x_unfold = torch.nn.functional.unfold(
            x,
            kernel_size=dni.layer.kernel_size,
            stride=dni.layer.stride,
            padding=padding,
        )  # Shape: (B, c_in*k*k, h_out*w_out)

        # Reshape to separate channels and kernel dimensions
        x_unfold = x_unfold.view(
            x.shape[0],
            dni.layer.in_channels,
            dni.layer.kernel_size[0],
            dni.layer.kernel_size[1],
            -1,
        )  # Shape: (B, c_in, k, k, h_out*w_out)

        # Sum over spatial output locations
        x_unfold = x_unfold.sum(dim=-1)  # Shape: (B, c_in, k, k)

        dv = torch.mean(x_unfold, axis=0)  # Shape: (1, c_in, k, k)

        assert torch.equal(dni.elig_eps, dv.expand(dni.layer.out_channels, -1, -1, -1))
    else:
        assert torch.count_nonzero(dni.elig_eps).item() == 0
