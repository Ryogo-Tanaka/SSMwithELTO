# src/models/architectures/time_invariant.py

"""
Time-invariant encoder/decoder architecture.

Config example:
```yaml
model:
  encoder:
    type: "time_invariant"
    input_dim: 7                     # observation dim n
    output_dim: 16                   # feature dim m
    architecture: "mlp"              # "mlp" | "resnet"
    hidden_dims: [64, 32]
    activation: "GELU"
    dropout: 0.1
    normalize_input: true
    normalize_output: true
    track_running_stats: true
    momentum: 0.1
    eps: 1e-5

  decoder:
    type: "time_invariant"
    input_dim: 16                    # feature dim m (matches encoder)
    output_dim: 7                    # observation dim n
    architecture: "mlp"
    hidden_dims: [32, 64]
    activation: "GELU"
    dropout: 0.1
```

Key parameters:
- input_dim/output_dim: Required for dimensional consistency
- normalize_input/output: Recommended for time-invariance and weak stationarity
- track_running_stats: Recommended for inference consistency
- architecture: "mlp" is standard; "resnet" for complex data
"""

import math
from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class time_invariantEncoder(nn.Module):
    """
    Time-invariant encoder u_eta: R^n -> R^m

    Requirements:
    - Time invariance: shared parameters eta across all time steps
    - Weak stationarity: E[u_eta(y_t)] = const
    - Dimensionality reduction: observation y_t in R^n -> features m_t in R^m

    Pipeline:
    1. Input normalization: y_t -> (y_t - mu_y) / sigma_y
    2. Time-invariant transform: m_t = u_eta(normalized_y_t)
    3. Output normalization: m_t -> (m_t - mu_m) / sigma_m
    4. Statistics management for inference consistency
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: str = "mlp",
        hidden_dims: Optional[List[int]] = None,
        activation: str = "GELU",
        dropout: float = 0.0,
        normalize_input: bool = True,
        normalize_output: bool = True,
        track_running_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        **kwargs
    ):
        """
        Args:
            input_dim: Input dimension n
            output_dim: Output dimension m
            architecture: Internal architecture ("mlp", "resnet")
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
            normalize_input: Whether to normalize input
            normalize_output: Whether to normalize output
            track_running_stats: Whether to track statistics
            momentum: Statistics update momentum
            eps: Numerical stability parameter
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.architecture = architecture
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.eps = eps

        if hidden_dims is None:
            if output_dim <= 4:
                hidden_dims = [64, 32]
            elif output_dim <= 16:
                hidden_dims = [128, 64]
            else:
                hidden_dims = [256, 128, 64]

        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.GELU()

        if self.normalize_input:
            self.input_norm = nn.BatchNorm1d(input_dim, momentum=momentum, eps=eps)

        if architecture == "mlp":
            self.core_net = self._build_mlp(input_dim, output_dim, hidden_dims, dropout)
        elif architecture == "resnet":
            self.core_net = self._build_resnet(input_dim, output_dim, hidden_dims, dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Supported: ['mlp', 'resnet']")

        if self.normalize_output:
            self.output_norm = nn.BatchNorm1d(output_dim, momentum=momentum, eps=eps)

        if track_running_stats:
            self.register_buffer('input_mean', torch.zeros(input_dim))
            self.register_buffer('input_var', torch.ones(input_dim))
            self.register_buffer('output_mean', torch.zeros(output_dim))
            self.register_buffer('output_var', torch.ones(output_dim))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self._initialize_weights()

    def _build_mlp(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
        """Standard MLP architecture."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_resnet(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
        """ResNet-style architecture with residual connections."""
        layers = []
        prev_dim = input_dim

        if len(hidden_dims) > 0:
            first_hidden = hidden_dims[0]
            layers.append(nn.Linear(input_dim, first_hidden))
            prev_dim = first_hidden

        for i, hidden_dim in enumerate(hidden_dims):
            if i > 0:
                layers.append(ResidualBlock(prev_dim, hidden_dim, self.activation, dropout))
                prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Time-invariant forward pass.

        Args:
            y: [B, T, n] or [B, n] or [T, n]

        Returns:
            m: [B, T, m] or [B, m] or [T, m]
        """
        original_shape = y.shape
        is_single_step = len(original_shape) == 1
        is_no_batch = len(original_shape) == 2 and y.size(0) != 1

        # Unify to [B, T, n]
        if is_single_step:
            y = y.unsqueeze(0).unsqueeze(0)
        elif is_no_batch:
            y = y.unsqueeze(0)
        elif len(original_shape) == 2:
            y = y.unsqueeze(1)

        B, T, n = y.shape

        if n != self.input_dim:
            raise ValueError(f"Input dim mismatch: expected {self.input_dim}, got {n}")

        # Process each time step independently (time invariance)
        y_flat = y.view(-1, n)

        if self.normalize_input:
            if self.training:
                y_flat = self.input_norm(y_flat)
            else:
                if self.track_running_stats:
                    input_mean = self.input_mean.to(y_flat.device)
                    input_var = self.input_var.to(y_flat.device)
                    y_flat = (y_flat - input_mean) / torch.sqrt(input_var + self.eps)
                else:
                    y_flat = self.input_norm(y_flat)

        if hasattr(self.core_net, 'to'):
            self.core_net = self.core_net.to(y_flat.device)
        m_flat = self.core_net(y_flat)

        if self.normalize_output:
            if self.training:
                m_flat = self.output_norm(m_flat)
            else:
                if self.track_running_stats:
                    output_mean = self.output_mean.to(m_flat.device)
                    output_var = self.output_var.to(m_flat.device)
                    m_flat = (m_flat - output_mean) / torch.sqrt(output_var + self.eps)
                else:
                    m_flat = self.output_norm(m_flat)

        m = m_flat.view(B, T, self.output_dim)

        if self.training and self.track_running_stats:
            self._update_statistics(y_flat, m_flat)

        # Restore original shape
        if is_single_step:
            return m.squeeze(0).squeeze(0)
        elif is_no_batch:
            return m.squeeze(0)
        elif len(original_shape) == 2:
            return m.squeeze(1)
        else:
            return m

    def _update_statistics(self, y_flat: torch.Tensor, m_flat: torch.Tensor):
        """Exponential moving average update of statistics."""
        with torch.no_grad():
            input_mean_batch = y_flat.mean(dim=0)
            input_var_batch = y_flat.var(dim=0, unbiased=False)
            output_mean_batch = m_flat.mean(dim=0)
            output_var_batch = m_flat.var(dim=0, unbiased=False)

            n = self.num_batches_tracked.item()
            momentum = self.momentum if n > 0 else 1.0

            input_mean_batch = input_mean_batch.to(self.input_mean.device)
            input_var_batch = input_var_batch.to(self.input_var.device)
            output_mean_batch = output_mean_batch.to(self.output_mean.device)
            output_var_batch = output_var_batch.to(self.output_var.device)

            self.input_mean.mul_(1 - momentum).add_(input_mean_batch, alpha=momentum)
            self.input_var.mul_(1 - momentum).add_(input_var_batch, alpha=momentum)
            self.output_mean.mul_(1 - momentum).add_(output_mean_batch, alpha=momentum)
            self.output_var.mul_(1 - momentum).add_(output_var_batch, alpha=momentum)

            self.num_batches_tracked += 1

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get normalization statistics."""
        if not self.track_running_stats:
            return {}

        return {
            'input_mean': self.input_mean.clone(),
            'input_var': self.input_var.clone(),
            'output_mean': self.output_mean.clone(),
            'output_var': self.output_var.clone(),
            'num_batches_tracked': self.num_batches_tracked.clone()
        }

    def load_statistics(self, stats: Dict[str, torch.Tensor]):
        """Load normalization statistics."""
        if not self.track_running_stats:
            return

        for key, value in stats.items():
            if hasattr(self, key):
                getattr(self, key).copy_(value)

    def verify_time_invariance(self, y1: torch.Tensor, y2: torch.Tensor, tol: float = 1e-6) -> bool:
        """Verify time invariance: same input at different times -> same output."""
        with torch.no_grad():
            self.eval()
            m1 = self.forward(y1)
            m2 = self.forward(y2)
            max_diff = torch.max(torch.abs(m1 - m2)).item()
            return max_diff < tol


class time_invariantDecoder(nn.Module):
    """
    Decoder g_alpha: R^m -> R^n

    Multivariate feature m_t in R^m contains sufficient instantaneous information,
    so explicit time-delay embedding is not needed.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: str = "mlp",
        hidden_dims: Optional[List[int]] = None,
        activation: str = "GELU",
        dropout: float = 0.0,
        **kwargs
    ):
        """
        Args:
            input_dim: Input feature dimension m
            output_dim: Output observation dimension n
            architecture: Architecture type ("mlp", "resnet")
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.architecture = architecture

        if hidden_dims is None:
            if input_dim <= 4:
                hidden_dims = [32, 64]
            elif input_dim <= 16:
                hidden_dims = [64, 128]
            else:
                hidden_dims = [64, 128, 256]

        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.GELU()

        if architecture == "mlp":
            self.net = self._build_mlp(input_dim, output_dim, hidden_dims, dropout)
        elif architecture == "resnet":
            self.net = self._build_resnet(input_dim, output_dim, hidden_dims, dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Supported: ['mlp', 'resnet']")

        self._initialize_weights()

    def _build_mlp(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
        """Standard MLP architecture."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_resnet(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
        """ResNet-style architecture."""
        layers = []
        prev_dim = input_dim

        if len(hidden_dims) > 0:
            first_hidden = hidden_dims[0]
            layers.append(nn.Linear(input_dim, first_hidden))
            prev_dim = first_hidden

        for i, hidden_dim in enumerate(hidden_dims):
            if i > 0:
                layers.append(ResidualBlock(prev_dim, hidden_dim, self.activation, dropout))
                prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct observations from multivariate features.

        Args:
            m: [B, T, m] or [B, m] or [T, m] features

        Returns:
            y: [B, T, n] or [B, n] or [T, n] reconstructed observations
        """
        original_shape = m.shape
        is_single_step = len(original_shape) == 1
        is_no_batch = len(original_shape) == 2 and m.size(0) != 1

        # Unify to [B, T, m]
        if is_single_step:
            m = m.unsqueeze(0).unsqueeze(0)
        elif is_no_batch:
            m = m.unsqueeze(0)
        elif len(original_shape) == 2:
            m = m.unsqueeze(1)

        B, T, m_dim = m.shape

        if m_dim != self.input_dim:
            raise ValueError(f"Input dim mismatch: expected {self.input_dim}, got {m_dim}")

        m_flat = m.view(-1, m_dim)
        if hasattr(self.net, 'to'):
            self.net = self.net.to(m_flat.device)
        y_flat = self.net(m_flat)

        y = y_flat.view(B, T, self.output_dim)

        # Restore original shape
        if is_single_step:
            return y.squeeze(0).squeeze(0)
        elif is_no_batch:
            return y.squeeze(0)
        elif len(original_shape) == 2:
            return y.squeeze(1)
        else:
            return y


class ResidualBlock(nn.Module):
    """Residual block."""

    def __init__(self, input_dim: int, hidden_dim: int, activation: nn.Module, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, input_dim)  # Back to input dim for residual
        )

        if input_dim != hidden_dim:
            self.shortcut = nn.Linear(input_dim, input_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
