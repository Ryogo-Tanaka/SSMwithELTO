"""DSE checkpoint loader helpers (PROJ-REBUTTAL Step 4).

Encapsulates the build_* functions that scripts/evaluate_kalman_image_mse.py uses
to reconstruct an encoder/decoder/df_state/df_obs/realization from a saved
checkpoint, plus Q/R noise covariance estimation. Imported by
scripts/rebuttal/evaluate_multistep_rollout_dse.py to avoid touching the
existing evaluate_kalman_image_mse.py.

The semantics are kept identical to evaluate_kalman_image_mse.py; this module
exists so the original 1-step evaluation script remains unmodified per the
project's invariance principle.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.architectures.cnn_image import cnn_imageEncoder, cnn_imageDecoder
from src.ssm.df_state_layer import DFStateLayer
from src.ssm.df_observation_layer import DFObservationLayer
from src.ssm.realization import StochasticRealizationWithEncoder
from src.inference.utils import (
    estimate_noise_covariances,
    compute_residuals_from_operators,
)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(model_path: str) -> Tuple[Dict, Dict]:
    """Load a DSE checkpoint and return (state_dict, config)."""
    # Backward compatibility shim for legacy module name 'rkn'
    import src.models.architectures.cnn_image as _cnn_image_module
    sys.modules.setdefault("src.models.architectures.rkn", _cnn_image_module)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    return ckpt, config


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

def build_encoder(ckpt: Dict, config: Dict, device: torch.device) -> nn.Module:
    enc_cfg = config.get("model", {}).get("encoder", {})
    encoder = cnn_imageEncoder(
        input_resolution=tuple(enc_cfg.get("input_resolution", [48, 48, 1])),
        feature_dim=enc_cfg.get("feature_dim", 100),
        hidden=enc_cfg.get("hidden", 200),
        conv_channels=enc_cfg.get("conv_channels", [12, 12]),
        conv_strides=enc_cfg.get("conv_strides", [2, 2]),
        activation=enc_cfg.get("activation", "relu"),
        normalize_input=enc_cfg.get("normalize_input", False),
        normalize_output=enc_cfg.get("normalize_output", False),
        track_running_stats=enc_cfg.get("track_running_stats", True),
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def build_decoder(ckpt: Dict, config: Dict, device: torch.device) -> nn.Module:
    dec_cfg = config.get("model", {}).get("decoder", {})
    decoder = cnn_imageDecoder(
        input_resolution=tuple(dec_cfg.get("input_resolution", [48, 48, 1])),
        feature_dim=dec_cfg.get("feature_dim", 100),
        grid=tuple(dec_cfg.get("grid", [3, 3, 16])),
        upsample_mode=dec_cfg.get("upsample_mode", "conv_transpose"),
        conv_channels=dec_cfg.get("conv_channels", [16, 12, 1]),
        output_activation=dec_cfg.get("output_activation", "sigmoid"),
        activation=dec_cfg.get("activation", "relu"),
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.to(device).eval()
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder


# ---------------------------------------------------------------------------
# DF-A (state) and DF-B (observation) layers
# ---------------------------------------------------------------------------

def build_df_state(ckpt: Dict, config: Dict, device: torch.device) -> DFStateLayer:
    ds_cfg = config.get("ssm", {}).get("df_state", {})
    ds_ckpt = ckpt["df_state"]

    phi_state = ds_ckpt["phi_theta"]
    weight_keys = sorted([k for k in phi_state if "weight" in k])
    state_dim = phi_state[weight_keys[0]].shape[1]
    feature_dim = phi_state[weight_keys[-1]].shape[0]

    fn_cfg = ds_cfg.get("feature_net", {})
    readout_cfg = ds_cfg.get("readout", None)

    df_state = DFStateLayer(
        state_dim=state_dim,
        feature_dim=feature_dim,
        lambda_A=float(ds_cfg.get("lambda_A", 1e-2)),
        lambda_B=float(ds_cfg.get("lambda_B", 1e-3)),
        feature_net_config=fn_cfg,
        cross_fitting_config=ds_cfg.get("cross_fitting", None),
        readout_config=readout_cfg,
    )
    df_state.phi_theta.load_state_dict(ds_ckpt["phi_theta"])
    df_state.V_A = ds_ckpt["V_A"].to(device)

    readout_type = ds_ckpt.get("readout_type", "linear")
    if readout_type == "nonlinear" and "readout_net" in ds_ckpt:
        df_state.readout_net.load_state_dict(ds_ckpt["readout_net"])
        df_state.readout_type = "nonlinear"
    else:
        df_state.U_A = ds_ckpt["U_A"].to(device)
    df_state._is_fitted = True

    df_state = df_state.to(device).eval()
    for p in df_state.parameters():
        p.requires_grad = False
    return df_state


def build_df_obs(
    ckpt: Dict,
    config: Dict,
    df_state: DFStateLayer,
    device: torch.device,
) -> DFObservationLayer:
    do_cfg = config.get("ssm", {}).get("df_observation", {})
    do_ckpt = ckpt["df_obs"]

    obs_net_cfg = do_cfg.get("obs_net", {})
    readout_cfg = do_cfg.get("readout", None)

    df_obs = DFObservationLayer(
        df_state_layer=df_state,
        obs_feature_dim=int(do_cfg.get("obs_feature_dim", 50)),
        multivariate_feature_dim=int(do_cfg.get("multivariate_feature_dim", 100)),
        lambda_B=float(do_cfg.get("lambda_B", 1e-3)),
        lambda_dB=float(do_cfg.get("lambda_dB", 1e-3)),
        obs_net_config=obs_net_cfg,
        cross_fitting_config=do_cfg.get("cross_fitting", None),
        readout_config=readout_cfg,
    )
    df_obs.psi_omega.load_state_dict(do_ckpt["psi_omega"])
    df_obs.V_B = do_ckpt["V_B"].to(device)

    readout_type = do_ckpt.get("readout_type", "linear")
    if readout_type == "nonlinear" and "readout_net" in do_ckpt:
        df_obs.readout_net.load_state_dict(do_ckpt["readout_net"])
        df_obs.readout_type = "nonlinear"
    else:
        df_obs.U_B = do_ckpt["U_B"].to(device)
    df_obs._is_fitted = True

    df_obs = df_obs.to(device).eval()
    for p in df_obs.parameters():
        p.requires_grad = False
    return df_obs


# ---------------------------------------------------------------------------
# Realization (only needed for Q/R estimation, not for rollout init)
# ---------------------------------------------------------------------------

def build_realization(
    ckpt: Dict,
    config: Dict,
    encoder: nn.Module,
    device: torch.device,
) -> StochasticRealizationWithEncoder:
    real_cfg = config.get("ssm", {}).get("realization", {})
    fm_cfg = real_cfg.get("feature_mapping", {})

    realization = StochasticRealizationWithEncoder(
        encoder=encoder,
        encoder_output_dim=int(real_cfg.get("encoder_output_dim", 100)),
        past_horizon=int(real_cfg.get("past_horizon", 30)),
        rank=int(real_cfg.get("rank", 20)),
        ridge_param=float(real_cfg.get("ridge_param", 1e-3)),
        jitter=float(real_cfg.get("jitter", 1e-6)),
        device=str(device),
        feature_mapping_type=fm_cfg.get("type", "mlp"),
        feature_mapping_hidden_dims=fm_cfg.get("hidden_dims", [32]),
        feature_mapping_activation=fm_cfg.get("activation", "relu"),
    )

    real_saved = ckpt.get("realization_config", {})
    if isinstance(real_saved, dict) and "_modules" in real_saved:
        ct_saved = real_saved["_modules"].get("component_transforms")
        if ct_saved is not None and realization.component_transforms is not None:
            try:
                realization.component_transforms = ct_saved.to(device)
            except Exception:
                try:
                    realization.component_transforms.load_state_dict(
                        ct_saved.state_dict()
                    )
                except Exception:
                    pass

    realization = realization.to(device)
    for p in realization.parameters():
        p.requires_grad = False
    return realization


# ---------------------------------------------------------------------------
# Test data loading
# ---------------------------------------------------------------------------

def load_test_data(data_path: str, device: torch.device) -> torch.Tensor:
    """Load test_obs from an .npz file and normalize to float32 in [0, 1].

    Handles both 5D clean (1, T, H, W, C) and 4D noisy (T, H, W, C) shapes.
    """
    data = np.load(data_path)
    test_obs = data["test_obs"]
    if test_obs.ndim == 5 and test_obs.shape[0] == 1:
        test_obs = test_obs[0]
    if test_obs.dtype == np.uint8:
        test_obs = test_obs.astype(np.float32) / 255.0
    else:
        test_obs = test_obs.astype(np.float32)
    return torch.from_numpy(test_obs).to(device)


# ---------------------------------------------------------------------------
# Q / R noise covariance estimation
# ---------------------------------------------------------------------------

def estimate_noise_QR(
    encoder: nn.Module,
    df_state: DFStateLayer,
    df_obs: DFObservationLayer,
    realization: StochasticRealizationWithEncoder,
    test_obs: torch.Tensor,
    device: torch.device,
    gamma_q: float = 1e-6,
    gamma_r: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate Q, R from residuals on test data features (matches evaluate_kalman_image_mse.py)."""
    h = realization.h
    with torch.no_grad():
        if not realization.is_fitted:
            realization.fit(test_obs, encoder)
        X_states = realization.estimate_states(test_obs)
        N = X_states.shape[0]
        phi_seq = df_state.phi_theta(X_states)
        M_features = encoder(test_obs)
        M_aligned = M_features[h:h + N]
        psi_seq = df_obs.psi_omega(M_aligned)

        V_A = df_state.V_A
        V_B = df_obs.V_B
        res_state, res_obs = compute_residuals_from_operators(
            phi_seq, psi_seq, V_A, V_B
        )
        reg = {"gamma_Q": gamma_q, "gamma_R": gamma_r}
        Q, R = estimate_noise_covariances(res_state, res_obs, reg)
    return Q, R


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

class DSEModel:
    """Bundle of DSE components reconstructed from a checkpoint."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        df_state: DFStateLayer,
        df_obs: DFObservationLayer,
        realization: StochasticRealizationWithEncoder,
        config: Dict[str, Any],
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.df_state = df_state
        self.df_obs = df_obs
        self.realization = realization
        self.config = config


def load_dse(model_path: str, device: torch.device) -> DSEModel:
    """One-shot loader: returns a bundled DSEModel."""
    ckpt, config = load_checkpoint(model_path)
    encoder = build_encoder(ckpt, config, device)
    decoder = build_decoder(ckpt, config, device)
    df_state = build_df_state(ckpt, config, device)
    df_obs = build_df_obs(ckpt, config, df_state, device)
    realization = build_realization(ckpt, config, encoder, device)
    return DSEModel(encoder, decoder, df_state, df_obs, realization, config)
