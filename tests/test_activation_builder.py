import math

import pytest
import torch
from torch import nn

from dni_modules import (
    ActivationBuilder,
    ATanSurrogate,
    LeakyActivationBuilder,
    ReLUActivationBuilder,
    TriangleSurrogate,
)


def test_activation_builder_raises_error():
    """Test abstract class ActivationBuilder raises TypeError if an attempt is made to create an instance"""
    with pytest.raises(
        TypeError,
    ):
        _ = ActivationBuilder()


def test_relu_activation_builder_build():
    """Test the build() function of ReLUActivationBuilder class"""
    builder = ReLUActivationBuilder()
    activation = builder.build()
    assert type(activation) is nn.ReLU


@pytest.mark.parametrize("beta", [0.5, 0.95])
@pytest.mark.parametrize("threshold", [1.0, 0.8])
@pytest.mark.parametrize("spike_grad", [ATanSurrogate, TriangleSurrogate])
def test_leaky_activation_builder_init(beta, threshold, spike_grad):
    """Test the __init__ function of LeakyActivationBuilder class"""
    builder = LeakyActivationBuilder(
        beta=beta, threshold=threshold, spike_grad=spike_grad()
    )
    assert math.isclose(builder.beta, beta)
    assert math.isclose(builder.threshold, threshold)
    assert type(builder.spike_grad) == spike_grad


def test_leaky_activation_builder_init_no_parameters():
    """Test the __init__ function of LeakyActivationBuilder class when no parameters are passed"""
    builder = LeakyActivationBuilder()
    assert math.isclose(builder.beta, 0.9)
    assert math.isclose(builder.threshold, 1.0)
    assert type(builder.spike_grad) == ATanSurrogate


@pytest.mark.parametrize("beta", [0.5, 0.95])
@pytest.mark.parametrize("threshold", [1.0, 0.8])
@pytest.mark.parametrize("spike_grad", [ATanSurrogate, TriangleSurrogate])
def test_leaky_activation_builder_build(beta, threshold, spike_grad):
    """Test the build() function of LeakyActivationBuilder class"""
    builder = LeakyActivationBuilder(
        beta=beta, threshold=threshold, spike_grad=spike_grad()
    )
    activation = builder.build()
    assert torch.isclose(activation.beta, torch.tensor(beta))
    assert torch.isclose(activation.threshold, torch.tensor(threshold))
    assert type(activation.spike_grad) == spike_grad
    assert hasattr(activation.spike_grad, "surrogate_grad") and callable(
        getattr(activation.spike_grad, "surrogate_grad", None)
    )
