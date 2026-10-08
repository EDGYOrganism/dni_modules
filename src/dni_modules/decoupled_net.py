import snntorch
import torch
from torch import nn

from dni_modules import (
    DNI,
    Conv2dDNIBuilder,
    Conv2dSynthGradBuilder,
    DNIBuilder,
    LazyLinearDNIBuilder,
    LinearDNIBuilder,
    LinearSynthGradBuilder,
    SynthGrad,
    SynthGradBuilder,
)


class DecoupledNet(nn.Module):
    """Generic Decoupled Net class - not to be used directly


    Parameters
    ----------
    dni_builder : DNIBuilder
        DNIBuilder instance used for building DNI layers
    synth_grad_builder : SynthGradBuilder
        SynthGradBuilder instance used for building SynthGrad modules
    num_hidden_dni : int, optional
        Number of hidden DNI layers, by default 2

    Raises
    ------
    ValueError
        Number of hidden DNI layers cannot be smaller than 2.
    """

    def __init__(
        self,
        dni_builder: DNIBuilder,
        synth_grad_builder: SynthGradBuilder,
        num_hidden_dni: int = 2,
    ):
        super().__init__()

        if num_hidden_dni < 2:
            raise ValueError(
                f"Number of hidden DNI layers cannot be smaller than 2: num_hidden_dni = {num_hidden_dni}"
            )
        self.dni_builder = dni_builder
        self.synth_grad_builder = synth_grad_builder
        self.num_hidden_dni = num_hidden_dni

        self.arch = nn.ModuleList()


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
    num_hidden_dni : int, optional
        Number of hidden DNI layers, by default 2
    """

    def __init__(
        self,
        in_features: int,
        hidden_dni_features: int,
        out_features: int,
        dni_builder: LinearDNIBuilder,
        synth_grad_builder: LinearSynthGradBuilder,
        num_hidden_dni: int = 2,
    ):
        super().__init__(
            dni_builder=dni_builder,
            synth_grad_builder=synth_grad_builder,
            num_hidden_dni=num_hidden_dni,
        )

        self.in_features = in_features
        self.hidden_dni_features = hidden_dni_features
        self.out_features = out_features

        # Set out_features for dni_builder
        self.dni_builder.out_features = self.hidden_dni_features

        # Set in_features and out_features for synth_grad_builder
        self.synth_grad_builder.in_features = self.hidden_dni_features
        self.synth_grad_builder.out_features = self.hidden_dni_features

        for i in range(self.num_hidden_dni):
            self.dni_builder.in_features = (
                self.in_features if i == 0 else self.hidden_dni_features
            )
            synth_grad = SynthGrad(self.synth_grad_builder) if i > 0 else None
            self.arch.append(
                nn.ModuleDict({"dni": DNI(self.dni_builder), "synth_grad": synth_grad})
            )

        # Set out_features for dni_builder and deactivate batch_norm
        self.dni_builder.out_features = self.out_features
        self.dni_builder.batch_norm = False

        self.arch.append(
            nn.ModuleDict(
                {
                    "dni": DNI(self.dni_builder),
                    "synth_grad": SynthGrad(self.synth_grad_builder),
                }
            )
        )

    # forward() function
    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        lr: float,
        loss_fn,
    ):
        """Propagates input through all the DNI layers of the LinearDecoupledNet. If LinearDecoupledNet is in training mode, DNI and SynthGrad parameters are updated.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape :math:`(T, B, H_{in})`,
            where  :math:`T` is the number of time steps, :math:`B` is the batch size, :math:`H_{in}` is the number of input features.
        targets : torch.Tensor
            Label tensor with shape :math:`(B,)`:
        lr : float,
            Learning rate
        loss_fn :
            Output loss function

        Returns
        -------
        torch.Tensor
            Output tensor with shape :math:`(T, B, H_{out})`, where :math:`T` is the number of time steps, :math:`B` is the batch size, :math:`H_{out}` is the number of output features.
        """

        T = x.shape[0]
        B = x.shape[1]
        synth_grad_loss_fn = nn.MSELoss(reduction="sum")

        y = nn.functional.one_hot(targets, num_classes=self.out_features).to(
            torch.float
        )
        out = []

        # Reset membrane potentials for Leaky activations
        for layer in self.arch:
            if type(layer["dni"].activation) is snntorch.Leaky:
                layer["dni"].activation.reset_mem()
                assert layer["dni"].activation.mem.numel() == 0

        for t in range(T):
            z_l = self.arch[0]["dni"](x[t]).detach()

            if self.training:
                synth_delta_l = self.arch[1]["synth_grad"](
                    z_l
                )  # shape (B, hidden_dni_features)
                # Update DNI
                self.arch[0]["dni"].update_dni_parameters(x[t], synth_delta_l, lr)

            for layer_index in range(len(self.arch) - 1):
                # Compute z_{l+1}
                z_l_plus_1 = (
                    self.arch[layer_index + 1]["dni"](z_l).detach().requires_grad_(True)
                )

                if self.training:
                    if layer_index < len(self.arch) - 2:
                        synth_delta_l_plus_1 = self.arch[layer_index + 2]["synth_grad"](
                            z_l_plus_1
                        )
                    else:
                        loss = loss_fn(z_l_plus_1, y)
                        synth_delta_l_plus_1 = torch.autograd.grad(
                            loss, z_l_plus_1, retain_graph=False
                        )[0]

                    with torch.no_grad():
                        # Compute f'_{l+1}(z_l)
                        if (
                            type(self.arch[layer_index + 1]["dni"].activation)
                            is snntorch.Leaky
                        ):
                            temp_mem = self.arch[layer_index + 1]["dni"].activation.mem

                        jacobian_l_plus_1 = torch.autograd.functional.jacobian(
                            self.arch[layer_index + 1]["dni"], z_l
                        )
                        jacobian_l_plus_1_reshaped = jacobian_l_plus_1[
                            torch.arange(B), :, torch.arange(B), :
                        ]

                        if (
                            type(self.arch[layer_index + 1]["dni"].activation)
                            is snntorch.Leaky
                        ):
                            self.arch[layer_index + 1]["dni"].activation.mem = temp_mem

                        delta_l = torch.einsum(
                            "bij,bi->bj",
                            jacobian_l_plus_1_reshaped,
                            synth_delta_l_plus_1,
                        )  # (B, M, D) (B,  M) -> (B, D)

                    # Update DNI
                    self.arch[layer_index + 1]["dni"].update_dni_parameters(
                        z_l, synth_delta_l_plus_1, lr
                    )

                    loss_grad = synth_grad_loss_fn(synth_delta_l, delta_l)
                    loss_grad.backward()

                    # Update SynthGrad
                    with torch.no_grad():
                        for param in self.arch[layer_index + 1][
                            "synth_grad"
                        ].parameters():
                            if param.grad is not None:
                                param.sub_(lr * param.grad)
                                param.grad.zero_()

                z_l = z_l_plus_1
                if self.training:
                    synth_delta_l = synth_delta_l_plus_1

            out.append(z_l)

        # Clear eligibility traces if in training mode
        if self.training:
            for layer in self.arch:
                layer["dni"].clear_elig_eps()

        return torch.stack(out, dim=0)


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
    num_hidden_dni : int, optional
        Number of hidden DNI layers, by default 2

    """

    def __init__(
        self,
        in_channels: int,
        hidden_dni_channels: int,
        out_features: int,
        dni_builder: Conv2dDNIBuilder,
        synth_grad_builder: Conv2dSynthGradBuilder,
        num_hidden_dni: int = 2,
    ):
        super().__init__(
            dni_builder=dni_builder,
            synth_grad_builder=synth_grad_builder,
            num_hidden_dni=num_hidden_dni,
        )

        self.in_channels = in_channels
        self.hidden_dni_channels = hidden_dni_channels
        self.out_features = out_features

        # Set out_channels for dni_builder
        self.dni_builder.out_channels = self.hidden_dni_channels

        # Set in_channels and out_channels for synth_grad_builder
        self.synth_grad_builder.in_channels = self.hidden_dni_channels
        self.synth_grad_builder.out_channels = self.hidden_dni_channels

        for i in range(self.num_hidden_dni):
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

        # Create a LazyLinearDNIBUilder
        lazy_dni_builder = LazyLinearDNIBuilder(
            out_features=self.out_features,
            bias=self.dni_builder.bias,
            activation_builder=self.dni_builder.activation_builder,
        )

        self.arch.append(
            nn.ModuleDict(
                {
                    "dni": DNI(lazy_dni_builder),
                    "synth_grad": SynthGrad(self.synth_grad_builder),
                }
            )
        )

    # forward() function
    def forward(self, x: torch.Tensor, targets: torch.Tensor, lr: float, loss_fn):
        """Propagates input through all the DNI layers of the Conv2dDecoupledNet. If Conv2dDecoupledNet is in training mode, DNI and SynthGrad parameters are updated

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape :math:`(T, B, C_{in}, H, W)`,
            where :math:`T` is the number of time steps, :math:`B` is the batch size, :math:`H_{in}` is the number of input features, :math:`C_{in}` is the number of input channels and
            :math:`H` and :math:`W` are the input height and width.
        targets : torch.Tensor
            Label tensor with shape :math:`(B,)`:
        lr : float,
            Learning rate
        loss_fn :
            Output loss function

        Returns
        -------
        torch.Tensor
            Output tensor with shape :math:`(T, B, H_{out})`, where :math:`T` is the number of time steps, :math:`B` is the batch size, :math:`H_{out}` is the number of output features.
        """

        raise NotImplementedError(
            "forward() function for Conv2dDecoupledNet has not been implemented yet."
        )
