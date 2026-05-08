"""ELTO-KF model: encoder + decoder + spectral realization + operator-based KF.

Architecture (paper §6 quad-link description):
    encoder f_theta: 2x Conv-ReLU-MaxPool + 1x FC -> feature dim m
    decoder mean:   2x ConvTranspose-ReLU + sigmoid
    decoder var:    2x ConvTranspose-ReLU + softplus

Spectral realization (paper §4.2):
    Past block u_p(n) = [phi(y_{n-h+1}), ..., phi(y_{n-1})] in R^{m*(h-1)}
    Future block u_f(n) = [phi(y_n), ..., phi(y_{n+h-2})] in R^{m*(h-1)}
    C_pp = (1/N) U_p U_p^T, C_ff = (1/N) U_f U_f^T, C_fp = (1/N) U_f U_p^T
    Cholesky: C_ff = L L^T, C_pp = M M^T (with jitter)
    SVD: L^{-1} C_fp M^{-T} = U S V^T
    State: x_n = B u_p(n) where B = S^{1/2} V^T M^{-1} (truncated to rank r)

Closed-form operators (paper §4.2 eq. 1):
    T_e = Psi_2 Psi_1^T (Psi_1 Psi_1^T + eps_te I)^{-1}
    O_e = Phi   Psi^T (Psi Psi^T + eps_oe I)^{-1}
    where Psi[:, n] = x_n (state realisation, r-dim) and Phi[:, n] = phi(y_{idx[n]}).

Sequential state estimation (paper §5):
    Predict:    mu^- = T_e mu^+,           C^- = T_e C^+ T_e^T + C_V
    Innovation: G    = C^- O_e^T (O_e C^- O_e^T + C_W)^{-1}
    Update:     mu^+ = mu^- + G (phi - O_e mu^-)
                C^+  = C^- - G O_e C^-

Notes:
- Feature-space approximation: kernel k_theta is taken to be the linear kernel on
  encoder features f_theta(y), so spectral realization works directly on features.
- C_V, C_W are estimated from training-data residuals once the operators are fit.
- gamma_Q / gamma_R add small jitter to C_V / C_W; clean uses 1e-6, corrupted 1e-3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class EltoEncoder(nn.Module):
    """Conv-Conv-FC encoder producing a single m-dim feature per frame."""

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        feature_dim: int = 50,
        hidden: int = 200,
        conv_channels: Tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)

        H, W, C = input_resolution
        self.conv1 = nn.Conv2d(C, conv_channels[0], kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        h_out = ((H + 4 - 5) // 1 + 1) // 2
        h_out = ((h_out + 2 - 3) // 1 + 1) // 2
        w_out = ((W + 4 - 5) // 1 + 1) // 2
        w_out = ((w_out + 2 - 3) // 1 + 1) // 2
        self.flat_dim = h_out * w_out * conv_channels[1]

        self.fc1 = nn.Linear(self.flat_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.feature_dim)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, T, H, W, C) -> (B, T, feature_dim).

        Single-frame inputs (T omitted) are also supported.
        """
        y = y.contiguous()
        if y.dim() == 4:  # (T, H, W, C) -> (1, T, H, W, C)
            y = y.unsqueeze(0)
        B, T = y.shape[0], y.shape[1]
        H, W, C = y.shape[2], y.shape[3], y.shape[4]

        if (H, W, C) != tuple(self.input_resolution):
            raise ValueError(
                f"Input shape mismatch: expected {self.input_resolution}, got ({H}, {W}, {C})"
            )
        x = y.reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(B, T, self.feature_dim)


class EltoImageHead(nn.Module):
    """ConvTranspose decoder (mean or var head) producing per-pixel output.

    Two heads share the same architecture but differ in the final activation:
        mean head: sigmoid (pixel range [0, 1])
        var head:  softplus + small floor (positive values)
    """

    def __init__(
        self,
        feature_dim: int = 50,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        grid: Tuple[int, int, int] = (3, 3, 16),
        deconv_channels: Tuple[int, int] = (32, 16),
        output_activation: str = "sigmoid",
        var_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.input_resolution = input_resolution
        self.grid = grid
        self.var_floor = float(var_floor)
        self.output_activation_kind = output_activation

        H, W, C = input_resolution
        gh, gw, gc = grid
        self.fc = nn.Linear(self.feature_dim, gh * gw * gc)
        self.deconv1 = nn.ConvTranspose2d(
            gc, deconv_channels[0], kernel_size=5, stride=4,
            padding=1, output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            deconv_channels[0], deconv_channels[1], kernel_size=3, stride=4,
            padding=0, output_padding=1,
        )
        self.final = nn.Conv2d(deconv_channels[1], C, kernel_size=1)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., feature_dim) -> image (..., H, W, C)."""
        z = z.contiguous()
        leading = z.shape[:-1]
        flat = z.reshape(-1, self.feature_dim)
        gh, gw, gc = self.grid
        x = F.relu(self.fc(flat))
        x = x.view(-1, gc, gh, gw)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.final(x)
        if self.output_activation_kind == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation_kind == "softplus":
            x = F.softplus(x) + self.var_floor
        H, W, C = self.input_resolution
        x = x.permute(0, 2, 3, 1).contiguous().reshape(*leading, H, W, C)
        return x


# ---------------------------------------------------------------------------
# Spectral realization (one-shot, on training feature trajectory)
# ---------------------------------------------------------------------------


@dataclass
class RealizationArtifacts:
    """Outputs of ``compute_realization``."""

    B_state: torch.Tensor       # (r, m*(h-1)) state extraction matrix
    T_e: torch.Tensor           # (r, r) transfer operator
    O_e: torch.Tensor           # (m, r) observable operator
    C_V: torch.Tensor           # (r, r) state-noise covariance
    C_W: torch.Tensor           # (m, m) observation-noise covariance
    init_mean: torch.Tensor     # (r,) prior mean for warmup initialization
    init_cov: torch.Tensor      # (r, r) prior covariance for warmup
    past_horizon: int           # h
    state_dim: int              # r
    feature_dim: int            # m


def _regularize_cholesky(M: torch.Tensor, jitter: float = 1e-6,
                         max_jitter: float = 1.0) -> torch.Tensor:
    """Cholesky with jitter ladder; falls back to eigh-based square root."""
    n = M.shape[0]
    eye = torch.eye(n, device=M.device, dtype=M.dtype)
    cur = float(jitter)
    last_err: Optional[Exception] = None
    while cur <= max_jitter:
        try:
            return torch.linalg.cholesky(M + cur * eye)
        except RuntimeError as e:
            last_err = e
            cur *= 10
    # eigh fallback
    eigvals, eigvecs = torch.linalg.eigh(M + max_jitter * eye)
    eigvals = torch.clamp(eigvals, min=jitter)
    return eigvecs @ torch.diag(torch.sqrt(eigvals))


def _ridge_solve(A: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute (A + eps*I)^{-1} for a symmetric PSD matrix A."""
    n = A.shape[0]
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    try:
        return torch.linalg.solve(A + eps * eye, eye)
    except RuntimeError:
        eigvals, eigvecs = torch.linalg.eigh(A + eps * eye)
        eigvals = torch.clamp(eigvals, min=eps)
        return eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T


def compute_realization(
    features: torch.Tensor,
    past_horizon: int,
    state_dim: int,
    eps_te: float = 1e-3,
    eps_oe: float = 1e-3,
    cov_jitter: float = 1e-6,
    gamma_Q: float = 1e-6,
    gamma_R: float = 1e-6,
) -> RealizationArtifacts:
    """Run the spectral realization pipeline on a single feature trajectory.

    Args:
        features: (T, m) feature trajectory (single trajectory).
        past_horizon: h (number of frames in past/future block + 1).
        state_dim: r (target state dim, truncated from SVD).
        eps_te / eps_oe: ridge regularization for closed-form T_e / O_e.
        cov_jitter: Cholesky jitter for C_pp / C_ff.
        gamma_Q / gamma_R: extra diagonal jitter added to C_V / C_W for KF stability.

    Returns:
        RealizationArtifacts with B_state, T_e, O_e, C_V, C_W and init priors.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be (T, m), got shape {tuple(features.shape)}")
    T, m = features.shape
    h = int(past_horizon)
    r = int(state_dim)
    if h < 2:
        raise ValueError(f"past_horizon must be >= 2, got {h}")
    if r < 1:
        raise ValueError(f"state_dim must be >= 1, got {r}")

    # Window the features into past/future blocks of length (h-1).
    block_len = h - 1
    block_dim = m * block_len
    # We use t indexing such that block past-window-n is [phi(y_{n-h+1}), ..., phi(y_{n-1})]
    # and the corresponding future is [phi(y_n), ..., phi(y_{n+h-2})].
    # Sample indices: n = h-1 .. T-h. Number of samples N = T - 2h + 2.
    N = T - 2 * h + 2
    if N <= max(block_dim, r):
        raise ValueError(
            f"Insufficient training samples N={N} for block_dim={block_dim}, state_dim={r}"
        )

    device = features.device
    dtype = features.dtype

    U_p = torch.empty(block_dim, N, device=device, dtype=dtype)
    U_f = torch.empty(block_dim, N, device=device, dtype=dtype)
    Phi_obs = torch.empty(m, N, device=device, dtype=dtype)
    for i, n in enumerate(range(h - 1, T - h + 1)):
        past = features[n - block_len : n].reshape(-1)            # [phi_{n-h+1}, ..., phi_{n-1}]
        future = features[n : n + block_len].reshape(-1)          # [phi_n, ..., phi_{n+h-2}]
        U_p[:, i] = past
        U_f[:, i] = future
        Phi_obs[:, i] = features[n]

    inv_N = 1.0 / float(N)
    C_pp = inv_N * U_p @ U_p.T
    C_ff = inv_N * U_f @ U_f.T
    C_fp = inv_N * U_f @ U_p.T

    # Cholesky factors (with jitter ladder fallback).
    L = _regularize_cholesky(C_ff, jitter=cov_jitter)
    M = _regularize_cholesky(C_pp, jitter=cov_jitter)

    # Whitened cross-covariance.
    Linv = torch.linalg.solve_triangular(L, torch.eye(block_dim, device=device, dtype=dtype), upper=False)
    Minv = torch.linalg.solve_triangular(M, torch.eye(block_dim, device=device, dtype=dtype), upper=False)
    T_white = Linv @ C_fp @ Minv.T

    # SVD with rank truncation.
    Uw, Sw, Vh = torch.linalg.svd(T_white, full_matrices=False)
    r_eff = min(r, Sw.shape[0])
    Uw_r = Uw[:, :r_eff]
    Sw_r = Sw[:r_eff]
    Vh_r = Vh[:r_eff, :]
    # B = S^{1/2} V^T M^{-1}, where V is right singular vectors -> Vh has rows = V^T already.
    B_state = torch.diag(torch.sqrt(Sw_r)) @ Vh_r @ Minv  # (r, block_dim)

    # Realised state trajectory.
    Psi = B_state @ U_p  # (r, N)
    Psi_1 = Psi[:, :-1]
    Psi_2 = Psi[:, 1:]
    Phi_aligned = Phi_obs            # phi(y_n) for n = h-1 .. T-h

    # Closed-form operators.
    G1 = Psi_1 @ Psi_1.T
    G_xx = Psi @ Psi.T
    T_e = Psi_2 @ Psi_1.T @ _ridge_solve(G1, eps_te)
    O_e = Phi_aligned @ Psi.T @ _ridge_solve(G_xx, eps_oe)

    # Residual covariances for KF noise terms.
    state_resid = Psi_2 - T_e @ Psi_1
    obs_resid = Phi_aligned - O_e @ Psi
    N1 = state_resid.shape[1]
    N2 = obs_resid.shape[1]
    C_V = state_resid @ state_resid.T / max(1, N1 - 1)
    C_W = obs_resid @ obs_resid.T / max(1, N2 - 1)

    # Symmetrise + add noise jitter.
    C_V = 0.5 * (C_V + C_V.T) + gamma_Q * torch.eye(r_eff, device=device, dtype=dtype)
    C_W = 0.5 * (C_W + C_W.T) + gamma_R * torch.eye(m, device=device, dtype=dtype)

    init_mean = Psi.mean(dim=1)       # (r,)
    init_centred = Psi - init_mean[:, None]
    init_cov = init_centred @ init_centred.T / max(1, N - 1)
    init_cov = 0.5 * (init_cov + init_cov.T) + gamma_Q * torch.eye(r_eff, device=device, dtype=dtype)

    return RealizationArtifacts(
        B_state=B_state.detach(),
        T_e=T_e.detach(),
        O_e=O_e.detach(),
        C_V=C_V.detach(),
        C_W=C_W.detach(),
        init_mean=init_mean.detach(),
        init_cov=init_cov.detach(),
        past_horizon=h,
        state_dim=int(r_eff),
        feature_dim=int(m),
    )


# ---------------------------------------------------------------------------
# Kalman primitives (paper §5)
# ---------------------------------------------------------------------------


class EltoKalmanFilter:
    """Sequential state estimation with the embedded latent transfer operator.

    Operates on torch tensors (CPU or CUDA). State / covariance are maintained
    externally; each method takes (mu, C) and returns an updated pair so that
    callers can build differentiable or autoregressive flows easily.
    """

    def __init__(
        self,
        T_e: torch.Tensor,
        O_e: torch.Tensor,
        C_V: torch.Tensor,
        C_W: torch.Tensor,
    ) -> None:
        self.T_e = T_e
        self.O_e = O_e
        self.C_V = C_V
        self.C_W = C_W

    def predict(self, mu: torch.Tensor, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_new = self.T_e @ mu
        C_new = self.T_e @ C @ self.T_e.T + self.C_V
        C_new = 0.5 * (C_new + C_new.T)
        return mu_new, C_new

    def update(self, mu: torch.Tensor, C: torch.Tensor, phi: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        innov_cov = self.O_e @ C @ self.O_e.T + self.C_W
        innov_cov = 0.5 * (innov_cov + innov_cov.T)
        # Solve K = C O_e^T innov_cov^{-1} via solve.
        try:
            K = torch.linalg.solve(innov_cov, self.O_e @ C).T
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(innov_cov)
            eigvals = torch.clamp(eigvals, min=1e-6)
            inv_innov = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T
            K = (self.O_e @ C).T @ inv_innov.T
        innov = phi - self.O_e @ mu
        mu_new = mu + K @ innov
        C_new = C - K @ self.O_e @ C
        C_new = 0.5 * (C_new + C_new.T)
        return mu_new, C_new


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class EltoKfModel(nn.Module):
    """ELTO-KF model: encoder + Gaussian decoder + (post-training) operator-based KF.

    Training mode:
        forward(y) -> (mean_image, var_image) via per-frame autoencoding through
        the encoder + Gaussian decoder. Used with Gaussian NLL loss.

    After training:
        ``fit_operators(train_features)`` -- run the spectral realization pipeline
        and store T_e / O_e / C_V / C_W as buffers. The decoder is then used as
        the observable readout: image = decoder_mean(O_e @ mu_state).
    """

    def __init__(
        self,
        input_resolution: Tuple[int, int, int] = (48, 48, 1),
        feature_dim: int = 50,
        encoder_hidden: int = 200,
        past_horizon: int = 5,
        state_dim: int = 50,
        eps_te: float = 1e-3,
        eps_oe: float = 1e-3,
        cov_jitter: float = 1e-6,
        gamma_Q: float = 1e-6,
        gamma_R: float = 1e-6,
        var_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.input_resolution = tuple(input_resolution)
        self.feature_dim = int(feature_dim)
        self.past_horizon = int(past_horizon)
        self.state_dim_target = int(state_dim)
        self.eps_te = float(eps_te)
        self.eps_oe = float(eps_oe)
        self.cov_jitter = float(cov_jitter)
        self.gamma_Q = float(gamma_Q)
        self.gamma_R = float(gamma_R)
        self.var_floor = float(var_floor)

        self.encoder = EltoEncoder(
            input_resolution=input_resolution,
            feature_dim=self.feature_dim,
            hidden=encoder_hidden,
        )
        self.decoder_mean = EltoImageHead(
            feature_dim=self.feature_dim,
            input_resolution=input_resolution,
            output_activation="sigmoid",
        )
        self.decoder_var = EltoImageHead(
            feature_dim=self.feature_dim,
            input_resolution=input_resolution,
            output_activation="softplus",
            var_floor=self.var_floor,
        )

        # Operator buffers are populated by ``fit_operators`` and persisted in
        # ``state_dict`` so checkpoints can be reloaded for evaluation.
        block_dim = self.feature_dim * (self.past_horizon - 1)
        self.register_buffer("B_state", torch.zeros(self.state_dim_target, block_dim))
        self.register_buffer("T_e", torch.zeros(self.state_dim_target, self.state_dim_target))
        self.register_buffer("O_e", torch.zeros(self.feature_dim, self.state_dim_target))
        self.register_buffer("C_V", torch.eye(self.state_dim_target))
        self.register_buffer("C_W", torch.eye(self.feature_dim))
        self.register_buffer("init_mean", torch.zeros(self.state_dim_target))
        self.register_buffer("init_cov", torch.eye(self.state_dim_target))
        self.register_buffer("operators_fitted", torch.tensor(0, dtype=torch.long))
        self.register_buffer("state_dim_actual", torch.tensor(self.state_dim_target, dtype=torch.long))

    # -- training-mode forward -------------------------------------------------

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-frame autoencoder forward: y -> (mean_image, var_image)."""
        phi = self.encoder(y)                # (B, T, m)
        mean = self.decoder_mean(phi)        # (B, T, H, W, C)
        var = self.decoder_var(phi)          # (B, T, H, W, C)
        return mean, var

    # -- realization (post-training) ------------------------------------------

    @torch.no_grad()
    def fit_operators(
        self,
        train_features: torch.Tensor,
        past_horizon: Optional[int] = None,
        state_dim: Optional[int] = None,
    ) -> RealizationArtifacts:
        """Compute spectral realization on a single training feature trajectory.

        Args:
            train_features: (T, m) torch tensor of encoder features on training data.
            past_horizon / state_dim: override if non-None.

        Updates the model's buffers in-place and returns the artifacts dataclass.
        """
        h = int(past_horizon) if past_horizon is not None else self.past_horizon
        r = int(state_dim) if state_dim is not None else self.state_dim_target
        artifacts = compute_realization(
            features=train_features,
            past_horizon=h,
            state_dim=r,
            eps_te=self.eps_te,
            eps_oe=self.eps_oe,
            cov_jitter=self.cov_jitter,
            gamma_Q=self.gamma_Q,
            gamma_R=self.gamma_R,
        )
        # Buffers were sized to ``state_dim_target``; if SVD truncation produced a
        # smaller r_eff, pad with zeros to keep tensors fixed-size.
        r_eff = artifacts.state_dim
        block_dim = artifacts.feature_dim * (artifacts.past_horizon - 1)

        device = self.B_state.device
        B_full = torch.zeros(self.state_dim_target, block_dim, device=device, dtype=artifacts.B_state.dtype)
        B_full[:r_eff] = artifacts.B_state.to(device)
        T_full = torch.zeros(self.state_dim_target, self.state_dim_target, device=device, dtype=artifacts.T_e.dtype)
        T_full[:r_eff, :r_eff] = artifacts.T_e.to(device)
        O_full = torch.zeros(self.feature_dim, self.state_dim_target, device=device, dtype=artifacts.O_e.dtype)
        O_full[:, :r_eff] = artifacts.O_e.to(device)
        CV_full = torch.eye(self.state_dim_target, device=device, dtype=artifacts.C_V.dtype) * self.gamma_Q
        CV_full[:r_eff, :r_eff] = artifacts.C_V.to(device)
        CW_full = artifacts.C_W.to(device)
        init_mean_full = torch.zeros(self.state_dim_target, device=device, dtype=artifacts.init_mean.dtype)
        init_mean_full[:r_eff] = artifacts.init_mean.to(device)
        init_cov_full = torch.eye(self.state_dim_target, device=device, dtype=artifacts.init_cov.dtype) * self.gamma_Q
        init_cov_full[:r_eff, :r_eff] = artifacts.init_cov.to(device)

        self.B_state = B_full
        self.T_e = T_full
        self.O_e = O_full
        self.C_V = CV_full
        self.C_W = CW_full
        self.init_mean = init_mean_full
        self.init_cov = init_cov_full
        self.operators_fitted = torch.tensor(1, dtype=torch.long, device=device)
        self.state_dim_actual = torch.tensor(r_eff, dtype=torch.long, device=device)
        return artifacts

    # -- inference helpers -----------------------------------------------------

    def _make_kf(self) -> EltoKalmanFilter:
        if int(self.operators_fitted.item()) != 1:
            raise RuntimeError("Operators are not fitted. Call ``fit_operators`` first.")
        r = int(self.state_dim_actual.item())
        return EltoKalmanFilter(
            T_e=self.T_e[:r, :r],
            O_e=self.O_e[:, :r],
            C_V=self.C_V[:r, :r],
            C_W=self.C_W,
        )

    def _initial_state(self, past_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise (mu, C) at time h-1 from the first ``h-1`` features.

        Args:
            past_features: (h-1, m) tensor.
        Returns:
            mu: (r,), C: (r, r)
        """
        h = self.past_horizon
        if past_features.shape[0] != h - 1:
            raise ValueError(f"past_features must have shape (h-1, m) = ({h-1}, m)")
        r = int(self.state_dim_actual.item())
        u_p = past_features.reshape(-1)
        mu = self.B_state[:r] @ u_p
        # Use empirical state covariance from training as initial uncertainty.
        C = self.init_cov[:r, :r]
        return mu, C

    @torch.no_grad()
    def filter_window(self, y_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter through a single sequence ``y_seq`` of shape (L, H, W, C).

        Returns posterior (mu^+, C^+) at the last time step.
        """
        h = self.past_horizon
        if y_seq.shape[0] < h:
            raise ValueError(f"sequence length must be >= past_horizon ({h}), got {y_seq.shape[0]}")
        phi = self.encoder(y_seq.unsqueeze(0)).squeeze(0)  # (L, m)
        kf = self._make_kf()
        mu, C = self._initial_state(phi[: h - 1])
        # First update at t = h-1
        mu, C = kf.update(mu, C, phi[h - 1])
        for t in range(h, phi.shape[0]):
            mu, C = kf.predict(mu, C)
            mu, C = kf.update(mu, C, phi[t])
        return mu, C

    @torch.no_grad()
    def predict_image(self, mu: torch.Tensor) -> torch.Tensor:
        """Decode predicted feature O_e mu to image (H, W, C)."""
        r = int(self.state_dim_actual.item())
        feat = self.O_e[:, :r] @ mu
        img = self.decoder_mean(feat.unsqueeze(0)).squeeze(0)
        return img.clamp(0.0, 1.0)

    @torch.no_grad()
    def rollout(self, y_ctx: torch.Tensor, horizon: int) -> torch.Tensor:
        """Free rollout for ``horizon`` steps after context y_ctx (C, H, W, C_img).

        Returns (horizon, H, W, C_img). Index h-1 corresponds to time C+h-1.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        kf = self._make_kf()
        h = self.past_horizon
        if y_ctx.shape[0] < h:
            raise ValueError(
                f"context length C={y_ctx.shape[0]} must be >= past_horizon ({h}) for warmup"
            )
        phi = self.encoder(y_ctx.unsqueeze(0)).squeeze(0)  # (C, m)
        mu, C = self._initial_state(phi[: h - 1])
        mu, C = kf.update(mu, C, phi[h - 1])
        for t in range(h, phi.shape[0]):
            mu, C = kf.predict(mu, C)
            mu, C = kf.update(mu, C, phi[t])

        # Free rollout: prior at C, C+1, ...
        preds = []
        for _ in range(horizon):
            mu, C = kf.predict(mu, C)
            preds.append(self.predict_image(mu))
        return torch.stack(preds, dim=0)

    @torch.no_grad()
    def predict_dse(self, y_window: torch.Tensor) -> torch.Tensor:
        """DSE-aligned 1-step prediction: y[s..s+L-1] -> y[s+L+1].

        Args:
            y_window: (L, H, W, C). Filters through the window then performs two
                prediction steps (skip 1) to produce a single image.

        Returns:
            (H, W, C) predicted image.
        """
        kf = self._make_kf()
        mu, _C = self.filter_window(y_window)
        mu, _C = kf.predict(mu, _C)
        mu, _C = kf.predict(mu, _C)
        return self.predict_image(mu)
