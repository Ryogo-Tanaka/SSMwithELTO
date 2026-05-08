"""RKN image model wrapping ALRhub/rkn_share `RKNCell` for 48x48 image prediction.

Pipeline:
    y[t] -- encoder --> (w_t, sigma_t)             [latent observation + uncertainty]
    (w_t, sigma_t), prior_t -- RKNCell --> post_t, prior_{t+1}
    post_t (or prior_{t+1}) -- decoder --> reconstructed / predicted image

Training (teacher forcing, MSE on next-step image):
    For each window y[0..T-1], filter through the encoder + RKN cell, and decode
    next_prior_means[0..T-2] (the predicted prior for time 1..T-1) -- compare
    against y[1..T-1] with image-space MSE.

This deviates from the original RKN paper which uses a Bernoulli image likelihood
with a separate variance head. The image-MSE objective matches the DSE-aligned
evaluation protocol (`evaluate_kalman_image_mse.py`) and matches the LSTM baseline
already in this codebase, so DSE / LSTM / RKN comparisons remain consistent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make `rkn_cell.RKNCell` and `util.ConfigDict` importable via external/rkn.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RKN_PATH = str(_REPO_ROOT / "external" / "rkn")
if _RKN_PATH not in sys.path:
    sys.path.insert(0, _RKN_PATH)

from rkn_cell.RKNCell import RKNCell, var_activation, var_activation_inverse  # noqa: E402

from src.models.architectures.cnn_image import cnn_imageDecoder  # noqa: E402


class RknImageEncoder(nn.Module):
    """CNN encoder producing (mean, var) for the RKN cell.

    Backbone matches `cnn_imageEncoder` (used by the LSTM baseline / DSE):
        conv(5x5, 32) -> ReLU -> MaxPool(2)
        conv(3x3, 64) -> ReLU -> MaxPool(2)
        flatten -> FC(hidden) -> ReLU -> [optional L2-normalize]
    Then two separate Linear heads produce ``mean`` and ``log_var``; the
    positive variance is obtained via ``var_activation`` (= log(exp(x) + 1))
    matching the official RKN encoder.
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        lod: int = 100,
        hidden: int = 200,
        conv_channels: Tuple[int, int] = (32, 64),
        normalize_pre: bool = True,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.lod = int(lod)
        self.hidden = int(hidden)
        self.normalize_pre = bool(normalize_pre)

        H, W, C = input_resolution
        self.conv1 = nn.Conv2d(C, conv_channels[0], kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # 48 -> 48 -> 24 -> 24 -> 12 (with default settings)
        h_out = ((H + 2 * 2 - 5) // 1 + 1) // 2
        h_out = ((h_out + 2 * 1 - 3) // 1 + 1) // 2
        w_out = ((W + 2 * 2 - 5) // 1 + 1) // 2
        w_out = ((w_out + 2 * 1 - 3) // 1 + 1) // 2
        self.flat_dim = h_out * w_out * conv_channels[1]

        self.fc1 = nn.Linear(self.flat_dim, self.hidden)
        self.mean_head = nn.Linear(self.hidden, self.lod)
        self.log_var_head = nn.Linear(self.hidden, self.lod)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """y: (B, T, H, W, C) -> (mean, var) each shape (B, T, lod)."""
        y = y.contiguous()
        B, T = y.shape[0], y.shape[1]
        H, W, C = y.shape[2], y.shape[3], y.shape[4]

        if (H, W, C) != tuple(self.input_resolution):
            raise ValueError(
                f"Input shape mismatch: expected {self.input_resolution}, got ({H}, {W}, {C})"
            )

        x = y.reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B*T, C, H, W)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.normalize_pre:
            x = F.normalize(x, p=2, dim=-1, eps=1e-8)

        mean = self.mean_head(x)
        log_var = self.log_var_head(x)
        var = var_activation(log_var)

        mean = mean.view(B, T, self.lod)
        var = var.view(B, T, self.lod)
        return mean, var


class RknImageModel(nn.Module):
    """RKN for image-to-image prediction on quad-link 48x48 sequences."""

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        lod: int = 100,
        num_basis: int = 15,
        bandwidth: int = 3,
        trans_net_hidden_units: Tuple[int, ...] = (128, 128),
        trans_net_hidden_activation: str = "Tanh",
        trans_covar: float = 0.1,
        encoder_hidden: int = 200,
        decoder_hidden: int = 200,
        decoder_upsample_mode: str = "nearest",
        initial_state_variance: float = 10.0,
        normalize_pre: bool = True,
    ) -> None:
        super().__init__()

        self.input_resolution = tuple(input_resolution)
        self.lod = int(lod)
        self.lsd = 2 * self.lod

        self.encoder = RknImageEncoder(
            input_resolution=input_resolution,
            lod=self.lod,
            hidden=encoder_hidden,
            normalize_pre=normalize_pre,
        )

        # RKN cell config
        cell_conf = RKNCell.get_default_config()
        cell_conf.num_basis = int(num_basis)
        cell_conf.bandwidth = int(bandwidth)
        cell_conf.never_invalid = True
        cell_conf.trans_net_hidden_units = list(trans_net_hidden_units)
        cell_conf.trans_net_hidden_activation = trans_net_hidden_activation
        cell_conf.trans_covar = float(trans_covar)
        cell_conf.finalize_modifying()
        self.cell = RKNCell(self.lod, cell_conf)

        self.decoder = cnn_imageDecoder(
            input_resolution=input_resolution,
            feature_dim=self.lsd,
            hidden=decoder_hidden,
            upsample_mode=decoder_upsample_mode,
            output_activation="sigmoid",
        )

        # Initial belief (mean fixed at 0; covariance learnable like the official repo)
        self.register_buffer("initial_mean", torch.zeros(1, self.lsd))
        log_ic_init = float(var_activation_inverse(initial_state_variance))
        self.log_icu = nn.Parameter(log_ic_init * torch.ones(1, self.lod))
        self.log_icl = nn.Parameter(log_ic_init * torch.ones(1, self.lod))
        self.register_buffer("initial_cs", torch.zeros(1, self.lod))

    def _get_initial_state(self, batch_size: int, device: torch.device):
        m = self.initial_mean.to(device).expand(batch_size, -1)
        cu = var_activation(self.log_icu).to(device).expand(batch_size, -1)
        cl = var_activation(self.log_icl).to(device).expand(batch_size, -1)
        cs = self.initial_cs.to(device).expand(batch_size, -1)
        return m, [cu, cl, cs]

    def filter(self, y: torch.Tensor, return_priors: bool = False):
        """Filter through input sequence ``y`` of shape (B, T, H, W, C).

        Returns:
            post_means: (B, T, lsd)
            (optional) next_prior_means: (B, T, lsd) where ``next_prior_means[:, t]``
                is the predicted prior for time ``t + 1`` (i.e., RKNCell._predict
                applied to ``post_means[:, t]``).
        """
        B = y.shape[0]
        device = y.device
        w, w_var = self.encoder(y)  # (B, T, lod)
        T = w.shape[1]

        prior_mean, prior_cov = self._get_initial_state(B, device)
        post_means = []
        next_prior_means = [] if return_priors else None
        for t in range(T):
            post_mean, post_cov, next_prior_mean, next_prior_cov = self.cell(
                prior_mean, prior_cov, w[:, t], w_var[:, t]
            )
            post_means.append(post_mean)
            if return_priors:
                next_prior_means.append(next_prior_mean)
            prior_mean = next_prior_mean
            prior_cov = next_prior_cov

        post_means = torch.stack(post_means, dim=1)
        if return_priors:
            next_prior_means = torch.stack(next_prior_means, dim=1)
            return post_means, next_prior_means
        return post_means

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing forward.

        Returns y_pred (B, T-1, H, W, C) where y_pred[:, t] predicts y[:, t+1].
        Loss = MSE(y_pred, y[:, 1:]).
        """
        _, next_prior_means = self.filter(y, return_priors=True)
        # next_prior_means[:, :-1] are predicted priors for t = 1..T-1.
        pred_priors = next_prior_means[:, :-1].contiguous()
        y_pred = self.decoder(pred_priors)
        return y_pred

    @torch.no_grad()
    def rollout(self, y_ctx: torch.Tensor, horizon: int) -> torch.Tensor:
        """Free rollout from context y_ctx for ``horizon`` future frames.

        Args:
            y_ctx: (B, C, H, W, Cimg) context frames.
            horizon: number of frames to predict after the context.

        Returns:
            (B, horizon, H, W, Cimg). Index h-1 corresponds to time C+h-1.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        B = y_ctx.shape[0]
        device = y_ctx.device
        w, w_var = self.encoder(y_ctx)  # (B, C, lod)
        T = w.shape[1]

        prior_mean, prior_cov = self._get_initial_state(B, device)
        # Filter through context
        for t in range(T):
            _, _, next_prior_mean, next_prior_cov = self.cell(
                prior_mean, prior_cov, w[:, t], w_var[:, t]
            )
            prior_mean = next_prior_mean
            prior_cov = next_prior_cov

        # ``prior_mean`` is now the predicted prior for time C (h=1).
        priors = [prior_mean]
        for _ in range(horizon - 1):
            prior_mean, prior_cov = self.cell._predict(prior_mean, prior_cov)
            priors.append(prior_mean)

        priors = torch.stack(priors, dim=1).contiguous()  # (B, horizon, lsd)
        y_pred = self.decoder(priors)
        return y_pred.clamp(0.0, 1.0)
