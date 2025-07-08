import torch
from torch import nn

class _cnnEncoder(nn.Module):
    """
    1D-CNN encoder over a sequence dimension.
    Configuration:
      - in_channels: number of features d
      - out_channels: number of filters in the last conv
      - kernel_sizes: list of kernel sizes for successive Conv1d
      - hidden_channels: list of channel sizes for intermediate conv layers
      - final_dim: dimensionality p (project conv output → p)
    Expects input of shape (batch, in_channels, seq_len).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        kernel_sizes: list[int],
        final_dim: int,
    ):
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes), "channels vs kernels mismatch"
        layers: list[nn.Module] = []
        prev_ch = in_channels
        for hc, k in zip(hidden_channels, kernel_sizes):
            layers.append(nn.Conv1d(prev_ch, hc, kernel_size=k, padding=k//2))
            layers.append(nn.ReLU())
            prev_ch = hc
        # flatten and linear to final_dim
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse seq_len → 1
        self.proj = nn.Linear(prev_ch, final_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, seq_len)
        h = self.conv(x)                # (batch, hidden, seq_len)
        h = self.pool(h).squeeze(-1)    # (batch, hidden)
        return self.proj(h)             # (batch, final_dim)


class _cnnDecoder(nn.Module):
    """
    Reverse of CNNEncoder: map y features → sequence reconstructions.
    Configuration:
      - input_dim: dimensionality p
      - hidden_channels: list of channel sizes for deconv layers
      - kernel_sizes: list of kernel sizes
      - out_channels: original feature dimension d
      - seq_len: length of output sequence
    """
    def __init__(
        self,
        input_dim: int,
        hidden_channels: list[int],
        kernel_sizes: list[int],
        out_channels: int,
        seq_len: int,
    ):
        super().__init__()
        # project y → hidden_channels[0] * seq_len
        self.init_proj = nn.Linear(input_dim, hidden_channels[0] * seq_len)
        layers: list[nn.Module] = []
        prev_ch = hidden_channels[0]
        for hc, k in zip(hidden_channels[1:], kernel_sizes[1:]):
            layers.append(nn.ConvTranspose1d(prev_ch, hc, kernel_size=k, padding=k//2))
            layers.append(nn.ReLU())
            prev_ch = hc
        # final layer back to out_channels
        layers.append(nn.ConvTranspose1d(prev_ch, out_channels, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2))
        self.deconv = nn.Sequential(*layers)
        self.seq_len = seq_len
        self.hidden_channels = hidden_channels

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (batch, input_dim)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        b = y.size(0)
        h = self.init_proj(y)  # (batch, hidden0 * seq_len)
        h = h.view(b, self.hidden_channels[0], self.seq_len)
        return self.deconv(h)  # (batch, out_channels, seq_len)
