from itertools import product

import pytest

import torch
import torch.nn as nn

from dni_modules import (
    Conv2dDNIBuilder,
    Conv2dSynthGradBuilder,
    ReLUActivationBuilder,
    LeakyActivationBuilder,
    Conv2dDecoupledNet
)


CONV2D_DECOUPLED_NET_TEST_CASES = list(
    product(
        [1, 3],  # in_channels,
        [4, 10],  # hidden_dni_channels,
        [2, 10],  # out_features,
        [1, 2, 3],  # num_dni
    )
)


CONV2D_DNI_BUILDER_TEST_CASES = list(
    product(
        [1],  # in_channels
        [1],  # out_channels
        [3, 5],  # kernel_size
        [1],  # stride
        [0, "same"],  # padding
        [nn.AvgPool2d],  # pooling
        [3],  # pooling_kernel_size
        [True, False],  # bias
        [
            ReLUActivationBuilder(),
            LeakyActivationBuilder(beta=0.8, threshold=0.5),
        ],  # activation_builder
        [True, False],  # batch_norm
    )
)


CONV2D_SYNTH_GRAD_BUILDER_TEST_CASES = list(
    product(
        [1],  # in_channels
        [1],  # out_channels
        [4, 5],  # hidden_layer_channels
        [3, 5],  # kernel_size
        [1],  # stride
        [0, "same"],  # padding
        [True, False],  # bias
        [
            ReLUActivationBuilder(),
            LeakyActivationBuilder(beta=0.8, threshold=0.5),
        ],  # activation_builder
        [0, 2],  # num_hidden
    )
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=CONV2D_DNI_BUILDER_TEST_CASES)
def conv2d_dni_builder_test_cases(request):
    return request.param


@pytest.fixture(params=CONV2D_SYNTH_GRAD_BUILDER_TEST_CASES)
def conv2d_synth_grad_builder_test_cases(request):
    return request.param


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
    )


@pytest.mark.parametrize(
    "in_channels, hidden_dni_channels, out_features, num_dni",
    CONV2D_DECOUPLED_NET_TEST_CASES,
)
def test_conv2d_decoupled_net_init(
    in_channels,
    hidden_dni_channels,
    out_features,
    num_dni,
    conv2d_dni_builder,
    conv2d_synth_grad_builder,
):
    """Test __init__ of Conv2dDecoupledNet class"""
    try:
        net = Conv2dDecoupledNet(
            in_channels=in_channels,
            hidden_dni_channels=hidden_dni_channels,
            out_features=out_features,
            dni_builder=conv2d_dni_builder,
            synth_grad_builder=conv2d_synth_grad_builder,
            num_dni=num_dni,
        )
    except Exception as e:
        assert isinstance(e, ValueError)
        assert (
            str(e)
            == f"Number of DNI layers cannot be smaller than 2: num_dni = {num_dni}"
        )
    else:
        assert net.in_channels == in_channels
        assert net.hidden_dni_channels == hidden_dni_channels
        assert net.out_features == out_features
        assert id(net.dni_builder) == id(conv2d_dni_builder)
        assert id(net.synth_grad_builder) == id(conv2d_synth_grad_builder)
        assert net.num_dni == num_dni
        assert len(net.arch) == num_dni + 1

        # Check DNI layers
        for i in range(0, num_dni):
            assert net.arch[i]["dni"].layer.out_channels == hidden_dni_channels

            if i == 0:
                assert net.arch[i]["dni"].layer.in_channels == in_channels
                assert type(net.arch[i]["dni"].net[-1]) is nn.MaxPool2d
                assert net.arch[i]["synth_grad"] is None
            else:
                assert net.arch[i]["dni"].layer.in_channels == hidden_dni_channels
                assert type(net.arch[i]["dni"].net[-1]) is nn.AvgPool2d
                assert (
                    net.arch[i]["synth_grad"].net[0].in_channels
                    == net.arch[i - 1]["dni"].layer.out_channels
                )
                assert (
                    net.arch[i]["synth_grad"].net[-1].out_channels
                    == net.arch[i - 1]["dni"].layer.out_channels
                )

            # Check output layer
            assert net.arch[-1]["dni"][1].out_features == out_features

            assert (
                net.arch[-1]["synth_grad"].net[0].in_channels
                == net.arch[num_dni - 1]["dni"].layer.out_channels
            )
            assert (
                net.arch[-1]["synth_grad"].net[-1].out_channels
                == net.arch[num_dni - 1]["dni"].layer.out_channels
            )


@pytest.mark.parametrize(
    "in_channels, hidden_dni_channels, out_features, num_dni",
    CONV2D_DECOUPLED_NET_TEST_CASES,
)
@pytest.mark.parametrize("B", [1])  # batch size
def test_conv2d_decoupled_net_forward(
    in_channels,
    hidden_dni_channels,
    out_features,
    num_dni,
    conv2d_dni_builder,
    conv2d_synth_grad_builder,
    device,
    B,
):
    """Test forward() function of Conv2dDecoupledNet class"""
    try:
        net = Conv2dDecoupledNet(
            in_channels=in_channels,
            hidden_dni_channels=hidden_dni_channels,
            out_features=out_features,
            dni_builder=conv2d_dni_builder,
            synth_grad_builder=conv2d_synth_grad_builder,
            num_dni=num_dni,
        )
    except Exception as e:
        assert isinstance(e, ValueError)
        assert (
            str(e)
            == f"Number of DNI layers cannot be smaller than 2: num_dni = {num_dni}"
        )
    else:
        net.to(device)
        net.eval()
        x = torch.randn((B, in_channels, 80, 80), device=device)

        with torch.inference_mode():
            out = net(x)

        assert out.shape == (B, out_features)
        del x
        del out
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch._C._cuda_clearCublasWorkspaces()
