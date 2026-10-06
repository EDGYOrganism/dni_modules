from itertools import product

import pytest
import snntorch
import torch
from torch import nn

from dni_modules import (
    Conv2dSynthGradBuilder,
    LeakyActivationBuilder,
    LinearSynthGradBuilder,
    ReLUActivationBuilder,
    SynthGradBuilder,
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
        [0, 1, 2],  # num_hidden
        [True, False],  # output_zero_init
    )
)

CONV2D_SYNTH_GRAD_BUILDER_TEST_CASES = list(
    product(
        [1, 3],  # in_channels
        [2, 3],  # out_channels
        [4, 5],  # hidden_layer_channels
        [2, 3],  # kernel_size
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


def test_synth_grad_builder_raises_error():
    """Test abstract class SynthGraBuiler raises TypeError if an attempt is made to create an instance"""
    with pytest.raises(TypeError):
        _ = SynthGradBuilder()


@pytest.mark.parametrize(
    "in_features, out_features, hidden_layer_size, bias, activation_builder, num_hidden, output_zero_init",
    LINEAR_SYNTH_GRAD_BUILDER_TEST_CASES,
)
def test_linear_synth_grad_builder_init(
    in_features,
    out_features,
    hidden_layer_size,
    bias,
    activation_builder,
    num_hidden,
    output_zero_init,
):
    """Test __init__ function of LinearSynthGradBuilder class"""
    builder = LinearSynthGradBuilder(
        in_features=in_features,
        out_features=out_features,
        hidden_layer_size=hidden_layer_size,
        bias=bias,
        activation_builder=activation_builder,
        num_hidden=num_hidden,
        output_zero_init=output_zero_init,
    )

    assert builder.in_features == in_features
    assert builder.out_features == out_features
    assert builder.hidden_layer_size == hidden_layer_size
    assert builder.bias == bias
    assert type(builder.activation_builder) is type(activation_builder)
    assert builder.num_hidden == num_hidden
    assert builder.output_zero_init == output_zero_init


@pytest.mark.parametrize(
    "in_features, out_features, hidden_layer_size, bias, activation_builder, num_hidden, output_zero_init",
    LINEAR_SYNTH_GRAD_BUILDER_TEST_CASES,
)
def test_linear_synth_grad_builder_build(
    in_features,
    out_features,
    hidden_layer_size,
    bias,
    activation_builder,
    num_hidden,
    output_zero_init,
):
    """Test build() function of LinearSynthGradBuilder class"""
    builder = LinearSynthGradBuilder(
        in_features=in_features,
        out_features=out_features,
        hidden_layer_size=hidden_layer_size,
        bias=bias,
        activation_builder=activation_builder,
        num_hidden=num_hidden,
        output_zero_init=output_zero_init,
    )

    synth_grad = builder.build()

    assert len(synth_grad) == 3 * num_hidden + 1

    if num_hidden == 0:
        assert synth_grad[0].weight.shape == (out_features, in_features)
    else:
        for i in range(num_hidden):
            assert (
                synth_grad[3 * i].weight.shape == (hidden_layer_size, in_features)
                if i == 0
                else (hidden_layer_size, hidden_layer_size)
            )
            assert type(synth_grad[3 * i + 1]) is nn.BatchNorm1d
            assert synth_grad[3 * i + 1].num_features == hidden_layer_size
            if type(activation_builder) is LeakyActivationBuilder:
                assert type(synth_grad[3 * i + 2]) is snntorch.Leaky
            else:
                assert type(synth_grad[3 * i + 2]) is nn.ReLU
        assert synth_grad[-1].weight.shape == (out_features, hidden_layer_size)

    if builder.output_zero_init:
        # Check output layer parameters are initialized to zero
        assert torch.count_nonzero(synth_grad[-1].weight).item() == 0
        if bias:
            assert torch.count_nonzero(synth_grad[-1].bias).item() == 0


@pytest.mark.parametrize(
    "in_channels, out_channels, hidden_layer_channels, kernel_size, stride, padding, bias, activation_builder, num_hidden, output_zero_init",
    CONV2D_SYNTH_GRAD_BUILDER_TEST_CASES,
)
def test_conv2d_synth_grad_builder_init(
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
):
    """Test __init__ function of Conv2dSynthGradBuilder class"""
    builder = Conv2dSynthGradBuilder(
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

    assert builder.in_channels == in_channels
    assert builder.out_channels == out_channels
    assert builder.hidden_layer_channels == hidden_layer_channels
    assert builder.kernel_size == kernel_size
    assert builder.stride == stride
    assert builder.padding == padding
    assert builder.bias == bias
    assert type(builder.activation_builder) is type(activation_builder)
    assert builder.num_hidden == num_hidden
    assert builder.output_zero_init == output_zero_init


@pytest.mark.parametrize(
    "in_channels, out_channels, hidden_layer_channels, kernel_size, stride, padding, bias, activation_builder, num_hidden, output_zero_init",
    CONV2D_SYNTH_GRAD_BUILDER_TEST_CASES,
)
def test_conv2d_synth_grad_builder_build(
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
):
    """Test build() function of Conv2dSynthGradBuilder class"""
    builder = Conv2dSynthGradBuilder(
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

    synth_grad = builder.build()

    assert len(synth_grad) == 3 * num_hidden + 1

    if num_hidden == 0:
        assert synth_grad[0].weight.shape == (
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        )
    else:
        for i in range(num_hidden):
            assert (
                synth_grad[3 * i].weight.shape
                == (hidden_layer_channels, in_channels, kernel_size, kernel_size)
                if i == 0
                else (
                    hidden_layer_channels,
                    hidden_layer_channels,
                    kernel_size,
                    kernel_size,
                )
            )
            assert type(synth_grad[3 * i + 1]) is nn.BatchNorm2d
            assert synth_grad[3 * i + 1].num_features == hidden_layer_channels
            if type(activation_builder) is LeakyActivationBuilder:
                assert type(synth_grad[3 * i + 2]) is snntorch.Leaky
            else:
                assert type(synth_grad[3 * i + 2]) is nn.ReLU
        assert synth_grad[-1].weight.shape == (
            out_channels,
            hidden_layer_channels,
            kernel_size,
            kernel_size,
        )

    if builder.output_zero_init:
        # Check output layer parameters are initialized to zero
        assert torch.count_nonzero(synth_grad[-1].weight).item() == 0
        if bias:
            assert torch.count_nonzero(synth_grad[-1].bias).item() == 0
