import torch.nn as nn

from .synth_grad_builder import SynthGradBuilder


class SynthGrad(nn.Module):
    def __init__(self, builder: type(SynthGradBuilder)):
        super().__init__()

        self.net = builder.build()

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
            # If returned output is a tuple, keep only the first element
            # Ensures compatibility between non-spiking and spiking activations
            if type(out) is tuple:
                out = out[0]
        return out
