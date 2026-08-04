import torch.nn as nn

from dni_modules import (
    DNI,
    SynthGrad,
    DNIBuilder,
    SynthGradBuilder,
    LinearDNIBuilder,
    Conv2dDNIBuilder,
    LinearSynthGradBuilder,
    Conv2dSynthGradBuilder,
)


class DecoupledNet(nn.Module):
    """Generic Decoupled Net class - not to be used directly


    Parameters
    ----------
    dni_builder : DNIBuilder
        DNIBuilder instance used for building DNI layers
    synth_grad_builder : SynthGradBuilder
        SynthGradBuilder instance used for building SynthGrad modules
    num_dni : int, optional
        Number of DNI layers, by default 2

    Raises
    ------
    ValueError
        Number of DNI layers cannot be smaller than 2.
    """

    def __init__(
        self,
        dni_builder: DNIBuilder,
        synth_grad_builder: SynthGradBuilder,
        num_dni: int = 2,
    ):
        super().__init__()

        if num_dni < 2:
            raise ValueError(
                f"Number of DNI layers cannot be smaller than 2: num_dni = {num_dni}"
            )
        self.dni_builder = dni_builder
        self.synth_grad_builder = synth_grad_builder
        self.num_dni = num_dni

        self.arch = nn.ModuleList()

    # Common forward() function for subclasses of DecoupledNet
    def forward(self, x):
        """Propagates input through all the DNI layers of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape :math:`(B, H_{in})` for Linear DNI and :math:`(B, C_{in}, H, W)` for Conv2d DNI,
            where :math:`B` is the batch size, :math:`H_{in}` is the number of input features, :math:`C_{in}` is the number of input channels and
            :math:`H` and :math:`W` are the input height and width.


        Returns
        -------
        torch.Tensor
            Output tensor with shape :math:`(B, H_{out})`, where :math:`B` is the batch size, :math:`H_{out}` is the number of output features.
            :math:`H_{out}` is set as a parameter in the constructor of the children classes of DecoupledNet.
        """
        out = x
        for layer in self.arch:
            out = layer["dni"](out)
            # If returned output is a tuple, keep only the first element
            # Ensures compatibility between non-spiking and spiking activations
            if type(out) is tuple:
                out = out[0]
        return out


class LinearDecoupledNet(DecoupledNet):
    """Creates a Decoupled Network consisting of linear DNIs and linear SynthGrad modules.

    Parameters
    ----------
    in_features : int
        Number of input features
    hidden_dni_features : int
        Number of features produced by hidden DNI layers
    out_features : int
        Number of output features
    dni_builder : LinearDNIBuilder
        LinearDNIBuilder instance used for building DNI layers
    synth_grad_builder : LinearSynthGradBuilder
        LinearSynthGrad instance used for building SynthGrad modules
    num_dni : int, optional
        Number of DNI layers, by default 2
    """

    def __init__(
        self,
        in_features: int,
        hidden_dni_features: int,
        out_features: int,
        dni_builder: LinearDNIBuilder,
        synth_grad_builder: LinearSynthGradBuilder,
        num_dni: int = 2,
    ):
        super().__init__(
            dni_builder=dni_builder,
            synth_grad_builder=synth_grad_builder,
            num_dni=num_dni,
        )

        self.in_features = in_features
        self.hidden_dni_features = hidden_dni_features
        self.out_features = out_features

        # Set out_features for dni_builder
        self.dni_builder.out_features = self.hidden_dni_features

        # Set in_features and out_features for synth_grad_builder
        self.synth_grad_builder.in_features = self.hidden_dni_features
        self.synth_grad_builder.out_features = self.hidden_dni_features

        for i in range(self.num_dni):
            self.dni_builder.in_features = (
                self.in_features if i == 0 else self.hidden_dni_features
            )
            synth_grad = SynthGrad(self.synth_grad_builder) if i > 0 else None
            self.arch.append(
                nn.ModuleDict({"dni": DNI(self.dni_builder), "synth_grad": synth_grad})
            )

        self.arch.append(
            nn.ModuleDict(
                {
                    "dni": nn.Sequential(
                        nn.Linear(
                            in_features=self.hidden_dni_features,
                            out_features=self.out_features,
                            bias=self.dni_builder.bias,
                        ),
                        self.dni_builder.activation_builder.build(),
                    ),
                    "synth_grad": SynthGrad(self.synth_grad_builder),
                }
            )
        )


class Conv2dDecoupledNet(DecoupledNet):
    """Creates a Decoupled Network consisting of Conv2d DNIs and Conv2d SynthGrad modules.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    hidden_dni_channels : int
        Number of channels in the tensors produced by hidden Conv2d DNIs
    out_features : int
        Number of output features
    dni_builder : Conv2dDNIBuilder
        Conv2dDNIBuilder instance used for building DNI layers
    synth_grad_builder : Conv2dSynthGradBuilder
        Conv2dSynthGradBuilder instance used for building SynthGrad layers
    num_dni : int, optional
        Number of DNI layers, by default 2

    """

    def __init__(
        self,
        in_channels: int,
        hidden_dni_channels: int,
        out_features: int,
        dni_builder: Conv2dDNIBuilder,
        synth_grad_builder: Conv2dSynthGradBuilder,
        num_dni: int = 2,
    ):
        super().__init__(
            dni_builder=dni_builder,
            synth_grad_builder=synth_grad_builder,
            num_dni=num_dni,
        )

        self.in_channels = in_channels
        self.hidden_dni_channels = hidden_dni_channels
        self.out_features = out_features

        # Set out_channels for dni_builder
        self.dni_builder.out_channels = self.hidden_dni_channels

        # Set in_channels and out_channels for synth_grad_builder
        self.synth_grad_builder.in_channels = self.hidden_dni_channels
        self.synth_grad_builder.out_channels = self.hidden_dni_channels

        for i in range(self.num_dni):
            if i == 0:
                self.dni_builder.in_channels = self.in_channels
                self.dni_builder.pooling = nn.MaxPool2d
            else:
                self.dni_builder.in_channels = self.hidden_dni_channels
                self.dni_builder.pooling = nn.AvgPool2d

            synth_grad = SynthGrad(self.synth_grad_builder) if i > 0 else None
            self.arch.append(
                nn.ModuleDict({"dni": DNI(self.dni_builder), "synth_grad": synth_grad})
            )

        self.arch.append(
            nn.ModuleDict(
                {
                    "dni": nn.Sequential(
                        nn.Flatten(),  # Flatten the output of the last hidden DNI to pass it to a linear layer
                        nn.LazyLinear(
                            out_features=self.out_features,
                            bias=self.dni_builder.bias,
                        ),
                        self.dni_builder.activation_builder.build(),
                    ),
                    "synth_grad": SynthGrad(self.synth_grad_builder),
                }
            )
        )
