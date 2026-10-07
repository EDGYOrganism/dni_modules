"""Network modules for experiments with Decoupled Neural Intefaces."""

from .dni_builder import DNIBuilder, LinearDNIBuilder, LazyLinearDNIBuilder, Conv2dDNIBuilder
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
from .synth_grad import SynthGrad

from .decoupled_net import LinearDecoupledNet, Conv2dDecoupledNet

from .surrogate_functions import SurrogateGrad, ATan, ATanSurrogate, Triangle, TriangleSurrogate

__all__ = [
    "DNIBuilder",
    "LinearDNIBuilder",
    "LazyLinearDNIBuilder",
    "Conv2dDNIBuilder",
    "ActivationBuilder",
    "ReLUActivationBuilder",
    "LeakyActivationBuilder",
    "SynthGradBuilder",
    "LinearSynthGradBuilder",
    "Conv2dSynthGradBuilder",
    "DNI",
    "SynthGrad",
    "LinearDecoupledNet",
    "Conv2dDecoupledNet",
    "SurrogateGrad",
    "ATan",
    "ATanSurrogate",
    "Triangle",
    "TriangleSurrogate"
]
