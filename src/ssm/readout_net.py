import torch
import torch.nn as nn


class ReadoutNet(nn.Module):
    """Readout map from feature space to target space.

    Implements r_xi as a shallow MLP for nonlinear readout.
    Used for both state readout (d_A -> r) and observation readout (d_B -> m).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = [32],
        activation: str = "ReLU"
    ):
        """
        Args:
            input_dim: Input dimension (d_A or d_B)
            output_dim: Output dimension (r or m)
            hidden_dims: List of hidden layer widths
            activation: Activation function name
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(nn, activation)())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (*, input_dim)

        Returns:
            torch.Tensor: Output (*, output_dim)
        """
        return self.net(x)
