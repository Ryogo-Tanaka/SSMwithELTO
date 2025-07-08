import torch
from torch import nn

class _mlpEncoder(nn.Module):
    """
    Simple feed-forward encoder: d-dimensional input → p-dimensional output.
    Configuration:
      - input_dim:  dimensionality of d
      - output_dim: dimensionality of y
      - hidden_sizes: list of ints for intermediate layers
      - activation: name of activation ("ReLU", "LeakyReLU", etc.)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list[int],
        activation: str = "ReLU",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        act_cls = getattr(nn, activation)
        # build hidden layers
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        # final projection
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d)
        return self.net(x)


class _mlpDecoder(nn.Module):
    """
    Mirror of MLPEncoder: p-dimensional input → d-dimensional reconstruction
    """
    def __init__(
        self,
        input_dim: int,    # should match encoder's output_dim
        output_dim: int,   # should match original d
        hidden_sizes: list[int],
        activation: str = "ReLU",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        act_cls = getattr(nn, activation)
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        # layers.append(nn.BatchNorm1d(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (batch, p)
        return self.net(y)
