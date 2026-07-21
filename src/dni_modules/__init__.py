"""Network modules for snnTorch experiments with Decoupled Neural Intefaces."""

from .dni_builder import DNIBuilder, LinearDNIBuilder, Conv2dDNIBuilder
from .activation_builder import (
    ActivationBuilder,
    ReLUActivationBuilder,
    LeakyActivationBuilder,
)
from .synth_grad_builder import (
    SynthGradBuilder,
    LinearSynthGradBuilder,
    Conv2dSynthGradBuilder,
)
from .dni import DNI

__all__ = [
    "DNIBuilder",
    "LinearDNIBuilder",
    "Conv2dDNIBuilder",
    "ActivationBuilder",
    "ReLUActivationBuilder",
    "LeakyActivationBuilder",
    "SynthGradBuilder",
    "LinearSynthGradBuilder",
    "Conv2dSynthGradBuilder",
    "DNI"
]
