from itertools import product

import pytest
import torch
from torch import nn

from dni_modules import (
    LeakyActivationBuilder,
    LinearDecoupledNet,
    LinearDNIBuilder,
    LinearSynthGradBuilder,
    ReLUActivationBuilder,
)

LINEAR_DECOUPLED_NET_TEST_CASES = list(
    product(
        [10, 100],  # in_features,
        [10, 30],  # hidden_dni_features,
        [5, 10],  # out_features
        [1, 2],  # num_hidden_dni
    )
)

LINEAR_DNI_BUILDER_TEST_CASES = list(
    product(
        [1],  # in_features - overwritten by LinearDecoupledNet parameters
        [1],  # out_features
        [True, False],  # bias
        [
            ReLUActivationBuilder(),
            LeakyActivationBuilder(beta=0.8, threshold=0.5),
        ],  # activation_builder
        [True, False],  # batch_norm
    )
)

LINEAR_SYNTH_GRAD_BUILDER_TEST_CASES = list(
    product(
        [1],  # in_features - overwritten by LinearDecoupledNet parameters
        [1],  # out_features
        [10, 20],  # hidden_layer_size
        [True, False],  # bias
        [
            ReLUActivationBuilder(),
            # LeakyActivationBuilder(beta=0.8, threshold=0.8),
        ],  # activation_builder
        [0, 1, 2],  # num_hidden
    )
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=LINEAR_DNI_BUILDER_TEST_CASES)
def linear_dni_builder_test_cases(request):
    return request.param


@pytest.fixture(params=LINEAR_SYNTH_GRAD_BUILDER_TEST_CASES)
def linear_synth_grad_builder_test_cases(request):
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
def linear_synth_grad_builder(linear_synth_grad_builder_test_cases):
    (
        in_features,
        out_features,
        hidden_layer_size,
        bias,
        activation_builder,
        num_hidden,
    ) = linear_synth_grad_builder_test_cases
    return LinearSynthGradBuilder(
        in_features=in_features,
        out_features=out_features,
        hidden_layer_size=hidden_layer_size,
        bias=bias,
        activation_builder=activation_builder,
        num_hidden=num_hidden,
        output_zero_init=False,
    )


@pytest.mark.parametrize(
    "in_features, hidden_dni_features, out_features, num_hidden_dni",
    LINEAR_DECOUPLED_NET_TEST_CASES,
)
def test_linear_decoupled_net_init(
    in_features,
    hidden_dni_features,
    out_features,
    num_hidden_dni,
    linear_dni_builder,
    linear_synth_grad_builder,
):
    """Test __init__ of LinearDecoupledNet class"""
    try:
        net = LinearDecoupledNet(
            in_features=in_features,
            hidden_dni_features=hidden_dni_features,
            out_features=out_features,
            dni_builder=linear_dni_builder,
            synth_grad_builder=linear_synth_grad_builder,
            num_hidden_dni=num_hidden_dni,
        )
    except ValueError as e:
        assert (
            str(e)
            == f"Number of hidden DNI layers cannot be smaller than 2: num_hidden_dni = {num_hidden_dni}"
        )
    else:
        assert net.in_features == in_features
        assert net.hidden_dni_features == hidden_dni_features
        assert net.out_features == out_features
        assert id(net.dni_builder) == id(linear_dni_builder)
        assert id(net.synth_grad_builder) == id(linear_synth_grad_builder)
        assert net.num_hidden_dni == num_hidden_dni
        assert len(net.arch) == num_hidden_dni + 1

        # Check hidden DNI layers
        for i in range(num_hidden_dni):
            if i == 0:
                assert net.arch[i]["dni"].layer.in_features == in_features
                assert net.arch[i]["synth_grad"] is None
            else:
                assert net.arch[i]["dni"].layer.in_features == hidden_dni_features
                assert (
                    net.arch[i]["synth_grad"].net[0].in_features
                    == net.arch[i - 1]["dni"].layer.out_features
                )
                assert (
                    net.arch[i]["synth_grad"].net[-1].out_features
                    == net.arch[i - 1]["dni"].layer.out_features
                )

            assert net.arch[i]["dni"].layer.out_features == hidden_dni_features

        # Check output DNI layer
        assert net.arch[-1]["dni"].layer.in_features == hidden_dni_features
        assert net.arch[-1]["dni"].layer.out_features == out_features

        assert (
            net.arch[-1]["synth_grad"].net[0].in_features
            == net.arch[num_hidden_dni - 1]["dni"].layer.out_features
        )
        assert (
            net.arch[-1]["synth_grad"].net[-1].out_features
            == net.arch[num_hidden_dni - 1]["dni"].layer.out_features
        )


@pytest.mark.parametrize(
    "in_features, hidden_dni_features, out_features, num_hidden_dni",
    LINEAR_DECOUPLED_NET_TEST_CASES,
)
@pytest.mark.parametrize("B", [2, 4])  # batch size
@pytest.mark.parametrize("eval", [True, False])
@pytest.mark.parametrize("local_update", [True, False])
def test_linear_decoupled_net_forward(
    in_features,
    hidden_dni_features,
    out_features,
    num_hidden_dni,
    linear_dni_builder,
    linear_synth_grad_builder,
    device,
    B,
    eval,
    local_update,
):
    """Test forward() function of LinearDecoupledNet class"""
    try:
        net = LinearDecoupledNet(
            in_features=in_features,
            hidden_dni_features=hidden_dni_features,
            out_features=out_features,
            dni_builder=linear_dni_builder,
            synth_grad_builder=linear_synth_grad_builder,
            num_hidden_dni=num_hidden_dni,
        )
    except ValueError as e:
        assert isinstance(e, ValueError)
        assert (
            str(e)
            == f"Number of hidden DNI layers cannot be smaller than 2: num_hidden_dni = {num_hidden_dni}"
        )
    else:
        net.to(device)
        if eval:
            net.eval()
        T = 10
        x = 100 * torch.randint(
            low=0, high=2, size=(T, B, in_features), device=device
        ).to(torch.float)
        targets = torch.randint(low=0, high=out_features, size=(B,), device=device)
        lr = 0.01

        loss_fn = nn.MSELoss(reduction="sum")

        linear_clones = {}
        for name, module in net.arch.named_modules():
            if isinstance(module, nn.Linear):
                linear_clones[f"{name}.weight"] = module.weight.detach().clone()

        out = net(x, targets, lr, loss_fn, local_update)

        if local_update and not eval:
            for name, module in net.arch.named_modules():
                if isinstance(module, nn.Linear):
                    for param_type in ["weight"]:
                        param_tensor = getattr(module, param_type, None)
                        if param_tensor is not None:
                            full_name = f"{name}.{param_type}"
                            old_tensor = linear_clones[full_name]

                            if torch.count_nonzero(param_tensor.grad).item() > 0:
                                assert not torch.allclose(param_tensor, old_tensor)

        assert out.shape == (T, B, out_features)

        # Check that elig_eps for every DNI contains only zeros
        for layer_index in range(len(net.arch)):
            assert torch.count_nonzero(net.arch[layer_index]["dni"].elig_eps) == 0
