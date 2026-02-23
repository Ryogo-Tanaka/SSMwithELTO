# src/models/architectures/cnn_image.py
"""
CNN image encoder/decoder architecture - Factory Pattern compliant.
File: cnn_image.py -> Classes: cnn_imageEncoder, cnn_imageDecoder
(Originally adapted from RKN codebase)

- cnn_imageEncoder: image (H,W,C) -> latent representation (100-dim)
- cnn_imageDecoder: latent representation (100-dim) -> image (H,W,C)
- cnn_image_targetDecoder: latent representation -> state mean (d_state-dim)

Shape handling (consistent with time_invariant.py):
- Single image: (H, W, C)
- Time series images: (T, H, W, C) <- primary shape
- Batched time series: (B, T, H, W, C)
"""

import math
from typing import Optional, List, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn_imageEncoder(nn.Module):
    """
    Image encoder: (H,W,C) -> latent (100-dim).
    Factory Pattern: type="cnn_image" for auto-registration.

    Architecture (Table 6):
    (H x W x C) -> Conv(5x5, 32), stride=1, same -> ReLU
                 -> MaxPool(2x2)
                 -> Conv(3x3, 64), stride=1, same -> ReLU
                 -> MaxPool(2x2)
                 -> Flatten -> FC(hidden=200) -> ReLU
                 -> FC(100)
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        feature_dim: int = 100,
        hidden: int = 200,
        conv_channels: Tuple[int, int] = (32, 64),
        conv_strides: Tuple[int, int] = (1, 1),  # Table 6: (2, 2)
        activation: str = "relu",
        normalize_input: bool = False,
        normalize_output: bool = False,
        track_running_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        **kwargs
    ):
        """
        Args:
            input_resolution: Input image resolution (H, W, C)
            feature_dim: Latent feature dimension (100 recommended)
            hidden: FC hidden dimension
            conv_channels: CNN channel sizes (conv1_ch, conv2_ch)
            conv_strides: CNN strides (stride1, stride2) -- Table 6: (2, 2)
            activation: Activation function name
            normalize_input: Input normalization (usually False for images)
            normalize_output: Output normalization
            track_running_stats: Track running statistics
            momentum: BatchNorm momentum
            eps: Numerical stability parameter
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.feature_dim = feature_dim
        self.hidden = hidden
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.eps = eps

        H, W, C = input_resolution
        s1, s2 = conv_strides

        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()

        # CNN layers (Table 6 compliant)
        self.conv1 = nn.Conv2d(C, conv_channels[0], kernel_size=5, stride=s1, padding=2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=s2, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # Compute post-convolution spatial size
        conv_h = ((H + 2*2 - 5) // s1 + 1) // 2  # Conv1 + Pool1
        conv_h = ((conv_h + 2*1 - 3) // s2 + 1) // 2  # Conv2 + Pool2
        conv_w = ((W + 2*2 - 5) // s1 + 1) // 2
        conv_w = ((conv_w + 2*1 - 3) // s2 + 1) // 2
        conv_output_size = conv_h * conv_w * conv_channels[1]

        self.fc1 = nn.Linear(conv_output_size, hidden)
        self.fc2 = nn.Linear(hidden, feature_dim)

        if self.normalize_input:
            self.input_norm = nn.BatchNorm2d(C, momentum=momentum, eps=eps)

        if self.normalize_output:
            self.output_norm = nn.BatchNorm1d(feature_dim, momentum=momentum, eps=eps)

        if track_running_stats:
            self.register_buffer('input_mean', torch.zeros(C, H, W))
            self.register_buffer('input_var', torch.ones(C, H, W))
            self.register_buffer('output_mean', torch.zeros(feature_dim))
            self.register_buffer('output_var', torch.ones(feature_dim))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self._initialize_weights()

    def _initialize_weights(self):
        """Weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """
        Time-invariant forward pass.

        Args:
            o: Image data
               - (H, W, C): single image
               - (T, H, W, C): time series <- primary shape
               - (B, T, H, W, C): batched time series

        Returns:
            z: Latent representation
               - (feature_dim,): single
               - (T, feature_dim): time series <- primary shape
               - (B, T, feature_dim): batched time series
        """
        original_shape = o.shape
        is_single_step = len(original_shape) == 3  # (H, W, C)
        is_no_batch = len(original_shape) == 4  # (T, H, W, C)

        # Unify to (B, T, H, W, C)
        if is_single_step:
            o = o.unsqueeze(0).unsqueeze(0)
        elif is_no_batch:
            o = o.unsqueeze(0)
        elif len(original_shape) == 4:
            o = o.unsqueeze(1)

        B, T, H, W, C = o.shape

        expected_H, expected_W, expected_C = self.input_resolution
        if H != expected_H or W != expected_W or C != expected_C:
            raise ValueError(f"Input shape mismatch: expected {self.input_resolution}, got ({H}, {W}, {C})")

        # Process each time step independently (time invariance)
        o_flat = o.view(B * T, H, W, C)
        o_flat = o_flat.permute(0, 3, 1, 2)  # -> (B*T, C, H, W)

        if self.normalize_input:
            if self.training:
                o_flat = self.input_norm(o_flat)
            else:
                if self.track_running_stats:
                    input_mean = self.input_mean.to(o_flat.device)
                    input_var = self.input_var.to(o_flat.device)
                    o_flat = (o_flat - input_mean.unsqueeze(0)) / torch.sqrt(input_var.unsqueeze(0) + self.eps)
                else:
                    o_flat = self.input_norm(o_flat)

        # CNN forward
        x = self.activation(self.conv1(o_flat))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)

        # Flatten + FC
        x = x.reshape(x.size(0), -1)
        x = self.activation(self.fc1(x))
        z_flat = self.fc2(x)

        if self.normalize_output:
            if self.training:
                z_flat = self.output_norm(z_flat)
            else:
                if self.track_running_stats:
                    output_mean = self.output_mean.to(z_flat.device)
                    output_var = self.output_var.to(z_flat.device)
                    z_flat = (z_flat - output_mean) / torch.sqrt(output_var + self.eps)
                else:
                    z_flat = self.output_norm(z_flat)

        z = z_flat.view(B, T, self.feature_dim)

        if self.training and self.track_running_stats:
            self._update_statistics(o_flat, z_flat)

        # Restore original shape
        if is_single_step:
            return z.squeeze(0).squeeze(0)
        elif is_no_batch:
            return z.squeeze(0)
        elif len(original_shape) == 4:
            return z.squeeze(1)
        else:
            return z

    def _update_statistics(self, o_flat: torch.Tensor, z_flat: torch.Tensor):
        """Exponential moving average update of statistics."""
        with torch.no_grad():
            input_mean_batch = o_flat.mean(dim=0)
            input_var_batch = o_flat.var(dim=0, unbiased=False)
            output_mean_batch = z_flat.mean(dim=0)
            output_var_batch = z_flat.var(dim=0, unbiased=False)

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

    def verify_time_invariance(self, o1: torch.Tensor, o2: torch.Tensor, tol: float = 1e-6) -> bool:
        """Verify time invariance: same input at different times -> same output."""
        with torch.no_grad():
            self.eval()
            z1 = self.forward(o1)
            z2 = self.forward(o2)
            max_diff = torch.max(torch.abs(z1 - z2)).item()
            return max_diff < tol


class cnn_imageDecoder(nn.Module):
    """
    Image decoder: latent (100-dim) -> (H,W,C).
    Factory Pattern: type="cnn_image" for auto-registration.

    Architecture:
    z in R^100 -> FC(hidden=200) -> ReLU
               -> FC(S) -> ReLU           # S = H'*W'*C'
               -> Reshape (H', W', C')
               -> Upsample(x2) -> Conv(3x3, 64), same -> ReLU
               -> Upsample(x2) -> Conv(5x5, 32), same -> ReLU
               -> Conv(1x1, C), same -> Sigmoid
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        feature_dim: int = 100,
        grid: Optional[Tuple[int, int, int]] = None,
        hidden: int = 200,
        upsample_mode: str = "nearest",  # "nearest"=legacy, "conv_transpose"=Table 6
        conv_channels: Tuple[int, ...] = (64, 32, 1),
        activation: str = "relu",
        output_activation: str = "sigmoid",
        **kwargs
    ):
        """
        Args:
            input_resolution: Output image resolution (H, W, C)
            feature_dim: Input latent feature dimension
            grid: Intermediate grid size (H', W', C') - auto if None
            hidden: FC hidden dimension (unused in conv_transpose mode)
            upsample_mode: "nearest"/"bilinear"=Upsample+Conv, "conv_transpose"=TransposedConv (Table 6)
            conv_channels: Channel sizes -- conv_transpose: (16,12,1), legacy: (64,32,1)
            activation: Activation function name
            output_activation: Output activation function name
        """
        super().__init__()

        self.input_resolution = input_resolution
        self.feature_dim = feature_dim
        self.upsample_mode = upsample_mode

        H, W, C = input_resolution

        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()

        if output_activation.lower() == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation.lower() == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation.lower() == "linear":
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Sigmoid()

        if grid is None:
            if upsample_mode == "conv_transpose":
                grid = (3, 3, 16)  # Table 6: 3x3x16=144
            else:
                grid = (H // 4, W // 4, conv_channels[0])
        self.grid = grid

        grid_h, grid_w, grid_c = grid
        grid_size = grid_h * grid_w * grid_c

        if upsample_mode == "conv_transpose":
            # Table 6: FC + ConvTranspose2d
            self.fc1 = nn.Linear(feature_dim, grid_size)
            self.deconv1 = nn.ConvTranspose2d(
                grid_c, conv_channels[0], kernel_size=5, stride=4,
                padding=1, output_padding=1
            )
            self.deconv2 = nn.ConvTranspose2d(
                conv_channels[0], conv_channels[1], kernel_size=3, stride=4,
                padding=0, output_padding=1
            )
            self.final_conv = nn.Conv2d(conv_channels[1], C, kernel_size=1)
        else:
            # Legacy: FC + Upsample+Conv
            self.fc1 = nn.Linear(feature_dim, hidden)
            self.fc2 = nn.Linear(hidden, grid_size)

            self.upsample1 = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.conv1 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)

            self.upsample2 = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.conv2 = nn.Conv2d(conv_channels[1], conv_channels[2] if len(conv_channels) > 2 else C, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        """Weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Latent -> image: time-invariant decoding.

        Args:
            z: Latent representation
               - (feature_dim,): single
               - (T, feature_dim): time series <- primary shape
               - (B, T, feature_dim): batched time series

        Returns:
            o: Reconstructed image
               - (H, W, C): single
               - (T, H, W, C): time series <- primary shape
               - (B, T, H, W, C): batched time series
        """
        original_shape = z.shape
        is_single_step = len(original_shape) == 1
        is_no_batch = len(original_shape) == 2

        # Unify to (B, T, feature_dim)
        if is_single_step:
            z = z.unsqueeze(0).unsqueeze(0)
        elif is_no_batch:
            z = z.unsqueeze(0)
        elif len(original_shape) == 2:
            z = z.unsqueeze(1)

        B, T, feature_dim = z.shape

        if feature_dim != self.feature_dim:
            raise ValueError(f"Feature dim mismatch: expected {self.feature_dim}, got {feature_dim}")

        z_flat = z.view(B * T, feature_dim)

        grid_h, grid_w, grid_c = self.grid

        if self.upsample_mode == "conv_transpose":
            x = self.activation(self.fc1(z_flat))
            x = x.view(-1, grid_c, grid_h, grid_w)
            x = self.activation(self.deconv1(x))
            x = self.activation(self.deconv2(x))
            x = self.output_activation(self.final_conv(x))
        else:
            x = self.activation(self.fc1(z_flat))
            x = self.activation(self.fc2(x))
            x = x.view(-1, grid_c, grid_h, grid_w)
            x = self.upsample1(x)
            x = self.activation(self.conv1(x))
            x = self.upsample2(x)
            x = self.output_activation(self.conv2(x))

        # (B*T, C, H, W) -> (B*T, H, W, C)
        x = x.permute(0, 2, 3, 1)

        H, W, C = self.input_resolution
        o = x.view(B, T, H, W, C)

        # Restore original shape
        if is_single_step:
            return o.squeeze(0).squeeze(0)
        elif is_no_batch:
            return o.squeeze(0)
        elif len(original_shape) == 2:
            return o.squeeze(1)
        else:
            return o


class cnn_image_targetDecoder(nn.Module):
    """
    Target prediction decoder: latent (d-dim) -> state mean (d_state-dim).
    Naming: <type>_targetDecoder (factory pattern compliant).

    Per-timestep structure:
    z_t in R^{feature_dim} -> FC(hidden) -> ReLU -> FC(d_state) -> output_activation
    """

    def __init__(
        self,
        feature_dim: int = 100,
        state_dim: int = 8,
        hidden: int = 50,
        activation: str = "relu",
        output_activation: str = "linear",
        **kwargs
    ):
        """
        Args:
            feature_dim: Input latent feature dimension
            state_dim: Output state dimension (e.g., 8 for quadcopter)
            hidden: FC hidden dimension
            activation: Activation function name
            output_activation: Output activation ("linear", "tanh", "sigmoid")
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.hidden = hidden

        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()

        if output_activation.lower() == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation.lower() == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation.lower() == "linear":
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Identity()

        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, state_dim)

        self._initialize_weights()

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Latent -> state mean: time-invariant state estimation.

        Args:
            z: Latent representation
               - (feature_dim,): single
               - (T, feature_dim): time series <- primary shape
               - (B, T, feature_dim): batched time series

        Returns:
            mu_s: State mean
                - (state_dim,): single
                - (T, state_dim): time series <- primary shape
                - (B, T, state_dim): batched time series
        """
        original_shape = z.shape
        is_single_step = len(original_shape) == 1
        is_no_batch = len(original_shape) == 2

        # Unify to (B, T, feature_dim)
        if is_single_step:
            z = z.unsqueeze(0).unsqueeze(0)
        elif is_no_batch:
            z = z.unsqueeze(0)
        elif len(original_shape) == 2:
            z = z.unsqueeze(1)

        B, T, feature_dim = z.shape

        if feature_dim != self.feature_dim:
            raise ValueError(f"Feature dim mismatch: expected {self.feature_dim}, got {feature_dim}")

        z_flat = z.view(B * T, feature_dim)

        x = self.activation(self.fc1(z_flat))
        x = self.fc2(x)
        mu_s_flat = self.output_activation(x)

        mu_s = mu_s_flat.view(B, T, self.state_dim)

        # Restore original shape
        if is_single_step:
            return mu_s.squeeze(0).squeeze(0)
        elif is_no_batch:
            return mu_s.squeeze(0)
        elif len(original_shape) == 2:
            return mu_s.squeeze(1)
        else:
            return mu_s


def make_state_decoder(cfg: Dict[str, Any]) -> cnn_image_targetDecoder:
    """
    Factory function for state decoder.

    Args:
        cfg: Config dict
            - feature_dim: Latent feature dimension (default 100)
            - state_dim: State dimension (required)
            - hidden: Hidden layer dimension (default 50)

    Returns:
        cnn_image_targetDecoder instance
    """
    feature_dim = cfg.get('feature_dim', 100)
    state_dim = cfg.get('state_dim')
    hidden = cfg.get('hidden', 50)

    if state_dim is None:
        raise ValueError("state_dim is required")

    return cnn_image_targetDecoder(
        feature_dim=feature_dim,
        state_dim=state_dim,
        hidden=hidden,
        **{k: v for k, v in cfg.items() if k not in ['feature_dim', 'state_dim', 'hidden']}
    )


# Backward-compatible aliases for legacy checkpoint loading
rknEncoder = cnn_imageEncoder
rknDecoder = cnn_imageDecoder
rkn_targetDecoder = cnn_image_targetDecoder
