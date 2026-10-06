import torch
from torch import nn

from .dni_builder import DNIBuilder, LazyLinearDNIBuilder


class DNI(nn.Module):
    """Decoupled Neural Interface class

    Parameters
    ----------
    builder : DNIBuilder
        DNIBuilder instance used for constructing the DNI's internal net
    """

    def __init__(self, builder: DNIBuilder):
        super().__init__()

        self.net = builder.build()

        if type(builder) is LazyLinearDNIBuilder:
            self.layer = self.net[1]
            self.activation = self.net[2]
            self.register_buffer("elig_eps", torch.zeros(1))
        else:
            self.layer = self.net[0]  # alias for Linear or Conv2d layer
            self.activation = (
                self.net[2] if builder.batch_norm else self.net[1]
            )  # alias for activation layer
            self.register_buffer("elig_eps", torch.zeros_like(self.layer.weight))

    def forward(self, x):
        """Propagates input through DNI's internal net and updates eligibility traces epsilon if spiking activations are used.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape :math:`(B, H_{in})` for Linear DNI and :math:`(B, C_{in}, H, W)` for Conv2d DNI,
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

                # Update eligibility traces epsilon if activation is spiking
                if self.training:
                    if type(self.layer) is nn.Linear:
                        # LazyLinear DNI
                        if x.ndim > 2:
                            dv = torch.mean(
                                x.reshape(x.shape[0], -1), axis=0
                            ).unsqueeze(0)  # Shape: (1, flattened vector dim)
                            # Initialize elig_eps if it hasn't been initialized yet
                            if torch.equal(
                                self.elig_eps,
                                torch.zeros(1, device=self.layer.weight.device),
                            ):
                                self.elig_eps = torch.zeros_like(self.layer.weight)
                        else:
                            dv = torch.mean(x, axis=0).unsqueeze(0)  # Shape: (1, D_in)

                    if type(self.layer) is nn.Conv2d:
                        # If padding has been set to "same" and kernel_size is odd, manually compute padding because unfold() function cannot accept string "same" for padding parameter.
                        if (
                            self.layer.padding == "same"
                            and self.layer.kernel_size[0] % 2 == 1
                        ):
                            padding = self.layer.kernel_size[0] // 2
                        else:
                            padding = self.layer.padding

                        # Extract input patches of size k x k that the convolution sees
                        x_unfold = torch.nn.functional.unfold(
                            x,
                            kernel_size=self.layer.kernel_size,
                            stride=self.layer.stride,
                            padding=padding,
                        )  # Shape: (B, c_in*k*k, h_out*w_out)

                        # Reshape to separate channels and kernel dimensions
                        x_unfold = x_unfold.view(
                            x.shape[0],
                            self.layer.in_channels,
                            self.layer.kernel_size[0],
                            self.layer.kernel_size[1],
                            -1,
                        )  # Shape: (B, c_in, k, k, h_out*w_out)

                        # Sum over spatial output locations
                        x_unfold = x_unfold.sum(dim=-1)  # Shape: (B, c_in, k, k)

                        dv = torch.mean(x_unfold, axis=0)  # Shape: (1, c_in, k, k)

                    self.elig_eps = self.activation.beta * self.elig_eps + dv

        return out

    def clear_elig_eps(self):
        """Sets eligibility traces epsilon to zero."""
        self.elig_eps = torch.zeros_like(self.elig_eps)
