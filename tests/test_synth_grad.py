from itertools import product

import pytest
import torch

from dni_modules import (
    Conv2dSynthGradBuilder,
    LeakyActivationBuilder,
    LinearSynthGradBuilder,
    ReLUActivationBuilder,
    SynthGrad,
)

LINEAR_SYNTH_GRAD_BUILDER_TEST_CASES = list(
    product(
        [2, 3],  # in_features
        [2, 3],  # out_features
        [10, 20],  # hidden_layer_size
        [True, False],  # bias
        [
            ReLUActivationBuilder(),
            LeakyActivationBuilder(),
            LeakyActivationBuilder(beta=0.8, threshold=0.8),
        ],  # activation_builder
        [1, 2],  # num_hidden
        [True, False],  # output_zero_init
    )
)

CONV2D_SYNTH_GRAD_BUILDER_TEST_CASES = list(
    product(
        [1, 3],  # in_channels
        [2, 3],  # out_channels
        [4, 5],  # hidden_layer_channels
        [3, 5],  # kernel_size
        [1],  # stride
        [0, "same"],  # padding
        [True, False],  # bias
        [
            ReLUActivationBuilder(),
            LeakyActivationBuilder(),
            LeakyActivationBuilder(beta=0.8, threshold=0.5),
        ],  # activation_builder
        [0, 1, 2],  # num_hidden
        [True, False],  # output_zero_init
    )
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=LINEAR_SYNTH_GRAD_BUILDER_TEST_CASES)
def linear_synth_grad_builder_test_cases(request):
    return request.param


@pytest.fixture(params=CONV2D_SYNTH_GRAD_BUILDER_TEST_CASES)
def conv2d_synth_grad_builder_test_cases(request):
    return request.param


@pytest.fixture
def linear_synth_grad_builder(linear_synth_grad_builder_test_cases):
    (
        in_features,
        out_features,
        hidden_layer_size,
        bias,
        activation_builder,
        num_hidden,
        output_zero_init,
    ) = linear_synth_grad_builder_test_cases
    return LinearSynthGradBuilder(
        in_features=in_features,
        out_features=out_features,
        hidden_layer_size=hidden_layer_size,
        bias=bias,
        activation_builder=activation_builder,
        num_hidden=num_hidden,
        output_zero_init=output_zero_init,
    )


@pytest.fixture
def conv2d_synth_grad_builder(conv2d_synth_grad_builder_test_cases):
    (
        in_channels,
        out_channels,
        hidden_layer_channels,
        kernel_size,
        stride,
        padding,
        bias,
        activation_builder,
        num_hidden,
        output_zero_init,
    ) = conv2d_synth_grad_builder_test_cases

    return Conv2dSynthGradBuilder(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_layer_channels=hidden_layer_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        activation_builder=activation_builder,
        num_hidden=num_hidden,
        output_zero_init=output_zero_init,
    )


def test_linear_synth_grad_init(linear_synth_grad_builder):
    """Test __init__ function of SynthGrad class built with LinearSynthGradBuilder"""
    synth_grad = SynthGrad(linear_synth_grad_builder)

    # Check net
    assert len(synth_grad.net) == linear_synth_grad_builder.num_hidden * 3 + 1


def test_conv2d_synth_grad_init(conv2d_synth_grad_builder):
    """Test __init__ function of SynthGrad class built with Conv2dSynthGradBuilder"""
    synth_grad = SynthGrad(conv2d_synth_grad_builder)

    # Check net
    assert len(synth_grad.net) == conv2d_synth_grad_builder.num_hidden * 3 + 1


def test_linear_synth_grad_forward(linear_synth_grad_builder, device):
    """Test the forward() function of a SynthGrad class instance built with LinearSynthGradBuilder"""
    # Batch size
    B = 4
    x = torch.randn((B, linear_synth_grad_builder.in_features), device=device)

    synth_grad = SynthGrad(linear_synth_grad_builder).to(device)
    out = synth_grad(x)
    assert out.shape[0] == B
    assert out.shape[1] == linear_synth_grad_builder.out_features

    if linear_synth_grad_builder.output_zero_init:
        assert torch.count_nonzero(out).item() == 0


def test_conv2d_synth_grad_forward(conv2d_synth_grad_builder, device):
    """Test the forward() function of a SynthGrad class instance built with Conv2dSynthGradBuilder"""
    # Batch size
    B = 4
    height, width = (20, 20)
    x = torch.randn(
        (B, conv2d_synth_grad_builder.in_channels, height, width), device=device
    )

    synth_grad = SynthGrad(conv2d_synth_grad_builder).to(device)
    out = synth_grad(x)
    assert out.shape[0] == B
    assert out.shape[1] == conv2d_synth_grad_builder.out_channels
    if conv2d_synth_grad_builder.padding == "same":
        assert out.shape[2] == height
        assert out.shape[3] == width

    if conv2d_synth_grad_builder.output_zero_init:
        assert torch.count_nonzero(out).item() == 0
