import torch
import torch.nn as nn

from .dni_builder import DNIBuilder


class DNI(nn.Module):
    def __init__(self, builder: type(DNIBuilder)):
        super().__init__()

        self.net = builder.build()
        self.elig_tr = torch.zeros_like(self.net[0].weight)

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
            # If returned output is a tuple, keep only the first element
            # Ensures compatibility between non-spiking and spiking activations
            if type(out) is tuple:
                out = out[0]
        return out
