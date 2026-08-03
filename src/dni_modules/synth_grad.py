import torch.nn as nn

from .synth_grad_builder import SynthGradBuilder


class SynthGrad(nn.Module):
    """Synthetic Gradient class

    Parameters
    ----------
    builder : SynthGradBuilder
        SynthGrad instance used for constructing the SynthGrad module's internal net
    """

    def __init__(self, builder: SynthGradBuilder):
        super().__init__()

        self.net = builder.build()

    def forward(self, x):
        """Propagates input through SynthGrad module's internal net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape :math:`(B, H_{in})` for Linear SynthGrad and :math:`(B, C_{in}, H, W)` for Conv2d SynthGrad,
            where :math:`B` is the batch size, :math:`H_{in}` is the number of input features, :math:`C_{in}` is the number of input channels and
            :math:`H` and :math:`W` are the input height and width.
        Returns
        -------
        torch.Tensor
            Output tensor
        """
        out = x
        for layer in self.net:
            out = layer(out)
            # If returned output is a tuple, keep only the first element
            # Ensures compatibility between non-spiking and spiking activations
            if type(out) is tuple:
                out = out[0]
        return out
