import torch.nn as nn

from dni_modules import (
    DNI,
    SynthGrad,
    DNIBuilder,
    SynthGradBuilder,
    LinearDNIBuilder,
    LinearSynthGradBuilder,
)


class DecoupledNet(nn.Module):
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
        out = x
        for layer in self.arch:
            out = layer["dni"](out)
            # If returned output is a tuple, keep only the first element
            # Ensures compatibility between non-spiking and spiking activations
            if type(out) is tuple:
                out = out[0]
        return out


class LinearDecoupledNet(DecoupledNet):
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
