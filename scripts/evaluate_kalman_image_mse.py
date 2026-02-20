#!/usr/bin/env python3
"""
Kalman Filtering Image Prediction MSE Evaluation

Evaluates a trained model's Image Prediction MSE using three methods:
  Step 1: Direct prediction baseline (no Kalman filtering)
  Step 2: Kalman filtering + df_obs round-trip prediction
  Step 3: Kalman filtering + feature-space direct prediction (no round-trip)

Usage:
  PYTHONUNBUFFERED=1 python scripts/evaluate_kalman_image_mse.py \
    --model results/phase_c_clean_1500_v3/seed1/models/final_model.pth \
    --data data/rkn_quad/quad1_n.npz \
    --output results/kalman_eval \
    --device cuda
"""

import sys
import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architectures.cnn_image import cnn_imageEncoder, cnn_imageDecoder
from src.ssm.df_state_layer import DFStateLayer, StateFeatureNet
from src.ssm.df_observation_layer import DFObservationLayer, ObservationFeatureNet
from src.ssm.realization import StochasticRealizationWithEncoder
from src.inference.kalman_filter import OperatorBasedKalmanFilter
from src.inference.utils import (
    estimate_noise_covariances,
    compute_residuals_from_operators,
)


# ========== Utility Functions ==========

def compute_image_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Per-pixel MSE averaged over all dimensions (T, H, W, C)."""
    return torch.mean((y_true - y_pred) ** 2).item()


def compute_image_mse_per_timestep(y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
    """MSE per timestep, averaged over spatial dims."""
    spatial_dims = tuple(range(1, y_true.dim()))
    return torch.mean((y_true - y_pred) ** 2, dim=spatial_dims).cpu().numpy()


# ========== Model Reconstruction ==========

def load_checkpoint(model_path: str) -> Tuple[Dict, Dict]:
    """Load checkpoint and config."""
    print(f"Loading checkpoint: {model_path}")
    # Register legacy module alias for checkpoints saved before Phase E rename
    import src.models.architectures.cnn_image as _cnn_image_module
    sys.modules['src.models.architectures.rkn'] = _cnn_image_module
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    print(f"  Top-level keys: {list(ckpt.keys())}")
    return ckpt, config


def build_encoder(ckpt: Dict, config: Dict, device: torch.device) -> nn.Module:
    enc_cfg = config.get('model', {}).get('encoder', {})
    encoder = cnn_imageEncoder(
        input_resolution=tuple(enc_cfg.get('input_resolution', [48, 48, 1])),
        feature_dim=enc_cfg.get('feature_dim', 100),
        hidden=enc_cfg.get('hidden', 200),
        conv_channels=enc_cfg.get('conv_channels', [12, 12]),
        conv_strides=enc_cfg.get('conv_strides', [2, 2]),
        activation=enc_cfg.get('activation', 'relu'),
        normalize_input=enc_cfg.get('normalize_input', False),
        normalize_output=enc_cfg.get('normalize_output', False),
        track_running_stats=enc_cfg.get('track_running_stats', True),
    )
    encoder.load_state_dict(ckpt['encoder'])
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters())} params")
    return encoder


def build_decoder(ckpt: Dict, config: Dict, device: torch.device) -> nn.Module:
    dec_cfg = config.get('model', {}).get('decoder', {})
    decoder = cnn_imageDecoder(
        input_resolution=tuple(dec_cfg.get('input_resolution', [48, 48, 1])),
        feature_dim=dec_cfg.get('feature_dim', 100),
        grid=tuple(dec_cfg.get('grid', [3, 3, 16])),
        upsample_mode=dec_cfg.get('upsample_mode', 'conv_transpose'),
        conv_channels=dec_cfg.get('conv_channels', [16, 12, 1]),
        output_activation=dec_cfg.get('output_activation', 'sigmoid'),
        activation=dec_cfg.get('activation', 'relu'),
    )
    decoder.load_state_dict(ckpt['decoder'])
    decoder = decoder.to(device).eval()
    for p in decoder.parameters():
        p.requires_grad = False
    print(f"  Decoder: {sum(p.numel() for p in decoder.parameters())} params")
    return decoder


def build_df_state(ckpt: Dict, config: Dict, device: torch.device) -> DFStateLayer:
    ds_cfg = config.get('ssm', {}).get('df_state', {})
    ds_ckpt = ckpt['df_state']

    # Infer dimensions from checkpoint weights
    phi_state = ds_ckpt['phi_theta']
    weight_keys = sorted([k for k in phi_state if 'weight' in k])
    state_dim = phi_state[weight_keys[0]].shape[1]    # first layer input = r
    feature_dim = phi_state[weight_keys[-1]].shape[0]  # last layer output = dA

    fn_cfg = ds_cfg.get('feature_net', {})

    df_state = DFStateLayer(
        state_dim=state_dim,
        feature_dim=feature_dim,
        lambda_A=float(ds_cfg.get('lambda_A', 1e-2)),
        lambda_B=float(ds_cfg.get('lambda_B', 1e-3)),
        feature_net_config=fn_cfg,
        cross_fitting_config=ds_cfg.get('cross_fitting', None),
    )
    df_state.phi_theta.load_state_dict(ds_ckpt['phi_theta'])
    df_state.V_A = ds_ckpt['V_A'].to(device)
    df_state.U_A = ds_ckpt['U_A'].to(device)
    df_state._is_fitted = True

    df_state = df_state.to(device).eval()
    for p in df_state.parameters():
        p.requires_grad = False
    print(f"  DFState: r={state_dim}, dA={feature_dim}, "
          f"V_A={tuple(df_state.V_A.shape)}, U_A={tuple(df_state.U_A.shape)}")
    return df_state


def build_df_obs(ckpt: Dict, config: Dict, df_state: DFStateLayer,
                 device: torch.device) -> DFObservationLayer:
    do_cfg = config.get('ssm', {}).get('df_observation', {})
    do_ckpt = ckpt['df_obs']

    obs_net_cfg = do_cfg.get('obs_net', {})

    df_obs = DFObservationLayer(
        df_state_layer=df_state,
        obs_feature_dim=int(do_cfg.get('obs_feature_dim', 50)),
        multivariate_feature_dim=int(do_cfg.get('multivariate_feature_dim', 100)),
        lambda_B=float(do_cfg.get('lambda_B', 1e-3)),
        lambda_dB=float(do_cfg.get('lambda_dB', 1e-3)),
        obs_net_config=obs_net_cfg,
        cross_fitting_config=do_cfg.get('cross_fitting', None),
    )
    df_obs.psi_omega.load_state_dict(do_ckpt['psi_omega'])
    df_obs.V_B = do_ckpt['V_B'].to(device)
    df_obs.U_B = do_ckpt['U_B'].to(device)
    df_obs._is_fitted = True

    df_obs = df_obs.to(device).eval()
    for p in df_obs.parameters():
        p.requires_grad = False
    print(f"  DFObs: V_B={tuple(df_obs.V_B.shape)}, U_B={tuple(df_obs.U_B.shape)}")
    return df_obs


def build_realization(ckpt: Dict, config: Dict, encoder: nn.Module,
                      device: torch.device) -> StochasticRealizationWithEncoder:
    real_cfg = config.get('ssm', {}).get('realization', {})
    fm_cfg = real_cfg.get('feature_mapping', {})

    realization = StochasticRealizationWithEncoder(
        encoder=encoder,
        encoder_output_dim=int(real_cfg.get('encoder_output_dim', 100)),
        past_horizon=int(real_cfg.get('past_horizon', 30)),
        rank=int(real_cfg.get('rank', 20)),
        ridge_param=float(real_cfg.get('ridge_param', 1e-3)),
        jitter=float(real_cfg.get('jitter', 1e-6)),
        m=int(real_cfg.get('m', 100)),
        device=str(device),
        feature_mapping_type=fm_cfg.get('type', 'mlp'),
        feature_mapping_hidden_dims=fm_cfg.get('hidden_dims', [32]),
        feature_mapping_activation=fm_cfg.get('activation', 'relu'),
    )

    # Load component_transforms from checkpoint (learned during training)
    real_saved = ckpt.get('realization_config', {})
    ct_loaded = False
    if isinstance(real_saved, dict) and '_modules' in real_saved:
        ct_saved = real_saved['_modules'].get('component_transforms')
        if ct_saved is not None and realization.component_transforms is not None:
            try:
                # Direct assignment (pickled nn.ModuleList)
                realization.component_transforms = ct_saved.to(device)
                ct_loaded = True
                print("  Realization: component_transforms loaded (direct)")
            except Exception as e:
                try:
                    # Fallback: load via state_dict
                    realization.component_transforms.load_state_dict(
                        ct_saved.state_dict())
                    ct_loaded = True
                    print("  Realization: component_transforms loaded (state_dict)")
                except Exception as e2:
                    print(f"  Warning: component_transforms load failed: {e2}")

    if not ct_loaded:
        print("  Warning: component_transforms not loaded from checkpoint!")
        print("  Using randomly initialized transforms (results may differ)")

    realization = realization.to(device)
    for p in realization.parameters():
        p.requires_grad = False
    print(f"  Realization: h={realization.h}, r={realization.rank}, "
          f"m={realization.m}, mapping={realization.feature_mapping_type}")
    return realization


# ========== Data Loading ==========

def load_test_data(data_path: str, device: torch.device) -> torch.Tensor:
    """Load and normalize test observations from npz file."""
    print(f"\nLoading data: {data_path}")
    data = np.load(data_path)
    print(f"  Keys: {list(data.keys())}")

    test_obs = data['test_obs']  # (1, T, 48, 48, 1)
    print(f"  test_obs: shape={test_obs.shape}, dtype={test_obs.dtype}")

    # Handle both 5D (1,T,H,W,C) clean data and 4D (T,H,W,C) noisy data
    if test_obs.ndim == 5 and test_obs.shape[0] == 1:
        test_obs = test_obs[0]

    # Normalize: uint8 [0,255] -> float32 [0,1]
    if test_obs.dtype == np.uint8:
        test_obs = test_obs.astype(np.float32) / 255.0
        print(f"  Normalized: uint8 -> float32 [0, 1]")
    else:
        test_obs = test_obs.astype(np.float32)
        print(f"  Range: [{test_obs.min():.4f}, {test_obs.max():.4f}]")

    tensor = torch.from_numpy(test_obs).to(device)
    print(f"  Tensor: {tuple(tensor.shape)}, device={tensor.device}")
    return tensor


# ========== Step 1: Direct Prediction Baseline ==========

def evaluate_direct_prediction(
    encoder: nn.Module,
    decoder: nn.Module,
    df_state: DFStateLayer,
    df_obs: DFObservationLayer,
    realization: StochasticRealizationWithEncoder,
    test_obs: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Step 1: encode -> realization -> DF-A predict -> DF-B predict -> decode -> MSE."""
    print("\n" + "=" * 60)
    print("STEP 1: Direct Prediction Baseline (no Kalman)")
    print("=" * 60)

    T = test_obs.shape[0]
    h = realization.h  # past_horizon = 30

    t0 = time.time()
    with torch.no_grad():
        # Fit realization on test data
        print(f"  Fitting realization (T={T}, h={h})...")
        realization.fit(test_obs, encoder)

        # Estimate states: x_t for t in [h, h+1, ..., h+N-1]
        X_states = realization.estimate_states(test_obs)  # (N, r)
        N, r = X_states.shape
        print(f"  States: N={N}, r={r}")

        # One-step-ahead state prediction: x_hat_{t+1|t} for t=0,...,N-2
        X_hat = df_state.predict_sequence(X_states)  # (N-1, r)
        T_pred = X_hat.shape[0]

        # Feature prediction (batch): m_hat = U_B^T V_B phi_theta(x_hat)
        M_hat = df_obs.predict_one_step(X_hat)  # (N-1, m)
        print(f"  Predictions: {T_pred} steps, features {tuple(M_hat.shape)}")

        # Decode to images
        Y_hat = decoder(M_hat)  # (N-1, 48, 48, 1)

        # Ground truth alignment:
        # X_states[t] = state at time (h + t)
        # X_hat[t] = prediction for time (h + t + 1) from state at time (h + t)
        # Ground truth for prediction[t] = test_obs[h + t + 1]
        gt_start = h + 1
        gt_end = gt_start + T_pred
        Y_gt = test_obs[gt_start:gt_end]

        if Y_hat.shape != Y_gt.shape:
            raise RuntimeError(
                f"Shape mismatch: Y_hat={tuple(Y_hat.shape)} vs Y_gt={tuple(Y_gt.shape)}")

        mse = compute_image_mse(Y_gt, Y_hat)
        mse_per_t = compute_image_mse_per_timestep(Y_gt, Y_hat)

    elapsed = time.time() - t0

    print(f"\n  === Step 1 Results ===")
    print(f"  Image MSE:         {mse:.6f}")
    print(f"  Image RMSE:        {np.sqrt(mse):.6f}")
    print(f"  MSE range:         [{mse_per_t.min():.6f}, {mse_per_t.max():.6f}]")
    print(f"  MSE median:        {np.median(mse_per_t):.6f}")
    print(f"  T_pred={T_pred}, gt=[{gt_start}:{gt_end}]")
    print(f"  Time: {elapsed:.1f}s")

    return {
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'mse_per_timestep': mse_per_t.tolist(),
        'T_pred': int(T_pred),
        'gt_range': [int(gt_start), int(gt_end)],
        'elapsed_seconds': round(elapsed, 1),
    }


# ========== Q/R Estimation ==========

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
    """Estimate Q, R from residuals on test data features."""
    print("\n  --- Estimating Q, R ---")

    h = realization.h

    with torch.no_grad():
        # Ensure realization is fitted
        if not realization.is_fitted:
            realization.fit(test_obs, encoder)

        # States from realization
        X_states = realization.estimate_states(test_obs)  # (N, r)
        N = X_states.shape[0]

        # phi features: phi_theta(x_t) for each state
        phi_seq = df_state.phi_theta(X_states)  # (N, dA)

        # psi features: psi_omega(m_t) for aligned observations
        M_features = encoder(test_obs)  # (T, m)
        M_aligned = M_features[h:h + N]  # (N, m)
        psi_seq = df_obs.psi_omega(M_aligned)  # (N, dB)

        print(f"  phi: {tuple(phi_seq.shape)}, psi: {tuple(psi_seq.shape)}")

        # Compute residuals: eps_t = phi_{t+1} - V_A phi_t, rho_t = psi_t - V_B phi_t
        V_A = df_state.V_A
        V_B = df_obs.V_B
        res_state, res_obs = compute_residuals_from_operators(
            phi_seq, psi_seq, V_A, V_B)
        print(f"  residuals_state: {tuple(res_state.shape)}, "
              f"residuals_obs: {tuple(res_obs.shape)}")

        # Estimate Q, R
        reg = {"gamma_Q": gamma_q, "gamma_R": gamma_r}
        Q, R = estimate_noise_covariances(res_state, res_obs, reg)

        print(f"  Q: {tuple(Q.shape)}, trace={Q.trace().item():.6f}, "
              f"cond={torch.linalg.cond(Q).item():.1f}")
        print(f"  R: {tuple(R.shape)}, trace={R.trace().item():.6f}, "
              f"cond={torch.linalg.cond(R).item():.1f}")

    return Q, R


# ========== Step 2: Kalman + Round-trip ==========

def evaluate_kalman_roundtrip(
    encoder: nn.Module,
    decoder: nn.Module,
    df_state: DFStateLayer,
    df_obs: DFObservationLayer,
    test_obs: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    device: torch.device,
    warmup: int = 50,
) -> Dict[str, Any]:
    """Step 2: Kalman filter -> df_state.predict -> df_obs.predict -> decode -> MSE."""
    print("\n" + "=" * 60)
    print("STEP 2: Kalman Filtering + Round-trip Prediction")
    print("=" * 60)

    T = test_obs.shape[0]
    V_A = df_state.V_A
    V_B = df_obs.V_B
    U_A = df_state.U_A
    U_B = df_obs.U_B

    t0 = time.time()
    with torch.no_grad():
        # Build Kalman filter
        kf = OperatorBasedKalmanFilter(
            V_A=V_A, V_B=V_B, U_A=U_A, U_B=U_B,
            Q=Q, R=R,
            encoder=encoder,
            df_obs_layer=df_obs,
            device=str(device),
        )

        # Filter all observations
        print(f"  Running Kalman filter (T={T})...")
        X_means, X_covs = kf.filter_sequence(test_obs)
        print(f"  Filtered states: {tuple(X_means.shape)}")

        # One-step-ahead prediction from filtered states:
        # x_hat_{t+1|t} = U_A^T V_A phi_theta(x_t|t) for t=0,...,T-2
        X_hat_next = df_state.predict_sequence(X_means)  # (T-1, r)

        # Feature prediction (batch)
        M_hat = df_obs.predict_one_step(X_hat_next)  # (T-1, m)

        # Decode to images
        Y_hat = decoder(M_hat)  # (T-1, 48, 48, 1)

        # Ground truth: y_{t+1} for t=0,...,T-2
        Y_gt = test_obs[1:T]  # (T-1, 48, 48, 1)

        if Y_hat.shape != Y_gt.shape:
            raise RuntimeError(
                f"Shape mismatch: Y_hat={tuple(Y_hat.shape)} vs Y_gt={tuple(Y_gt.shape)}")

        # MSE (full and post-warmup)
        mse_full = compute_image_mse(Y_gt, Y_hat)
        mse_warmup = compute_image_mse(Y_gt[warmup:], Y_hat[warmup:])
        mse_per_t = compute_image_mse_per_timestep(Y_gt, Y_hat)

    elapsed = time.time() - t0

    print(f"\n  === Step 2 Results ===")
    print(f"  Image MSE (full):        {mse_full:.6f}")
    print(f"  Image MSE (post-warmup): {mse_warmup:.6f}")
    print(f"  Image RMSE (post-warmup):{np.sqrt(mse_warmup):.6f}")
    print(f"  MSE range:               [{mse_per_t.min():.6f}, {mse_per_t.max():.6f}]")
    print(f"  MSE median:              {np.median(mse_per_t):.6f}")
    print(f"  T_pred={T - 1}, warmup={warmup}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        'mse_full': float(mse_full),
        'mse_post_warmup': float(mse_warmup),
        'rmse_post_warmup': float(np.sqrt(mse_warmup)),
        'mse_per_timestep': mse_per_t.tolist(),
        'warmup': int(warmup),
        'T_pred': int(T - 1),
        'elapsed_seconds': round(elapsed, 1),
    }


# ========== Step 3: Kalman + Feature-space Direct ==========

def evaluate_kalman_feature_space(
    encoder: nn.Module,
    decoder: nn.Module,
    df_obs: DFObservationLayer,
    test_obs: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    V_A: torch.Tensor,
    V_B: torch.Tensor,
    U_A: torch.Tensor,
    U_B: torch.Tensor,
    device: torch.device,
    warmup: int = 50,
) -> Dict[str, Any]:
    """Step 3: Manual Kalman loop capturing mu_minus -> V_B @ mu -> U_B^T -> decode -> MSE.

    This avoids the lossy round-trip through U_A^T and phi_theta by directly using
    the predicted feature-space state mu_minus_t for image prediction.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Kalman Filtering + Feature-space Direct Prediction")
    print("=" * 60)

    T = test_obs.shape[0]
    dA = V_A.shape[0]

    t0 = time.time()
    with torch.no_grad():
        # Build Kalman filter
        kf = OperatorBasedKalmanFilter(
            V_A=V_A, V_B=V_B, U_A=U_A, U_B=U_B,
            Q=Q, R=R,
            encoder=encoder,
            df_obs_layer=df_obs,
            device=str(device),
        )

        # Initialize state
        n_init = min(10, T)
        kf.initialize_state(test_obs[:n_init])

        # Manual Kalman loop: capture mu_minus (one-step-ahead prediction in feature space)
        print(f"  Running manual Kalman loop (T={T})...")
        mu_predicted = torch.zeros(T, dA, device=device)

        for t in range(T):
            mu_prev = kf.mu
            Sigma_prev = kf.Sigma

            # Predict: mu_minus_t = V_A @ mu_plus_{t-1}
            # mu_minus_t is the one-step-ahead prediction for time t
            mu_minus, Sigma_minus = kf.predict_step(mu_prev, Sigma_prev)
            mu_predicted[t] = mu_minus.detach()

            # Update with observation at time t
            mu_plus, Sigma_plus, _ = kf.update_step(
                mu_minus, Sigma_minus, test_obs[t])
            kf.mu = mu_plus
            kf.Sigma = Sigma_plus

            if (t + 1) % 500 == 0:
                print(f"    t={t + 1}/{T}")

        print(f"  Predicted feature states: {tuple(mu_predicted.shape)}")

        # Direct feature-space image prediction (no phi_theta round-trip):
        # m_hat_t = U_B^T @ (V_B @ mu_minus_t)
        h_pred = (V_B @ mu_predicted.T).T  # (T, dB)
        M_hat = h_pred @ U_B               # (T, m) = (T, dB) @ (dB, m)

        print(f"  Feature predictions: {tuple(M_hat.shape)}")

        # Decode to images
        Y_hat = decoder(M_hat)  # (T, 48, 48, 1)

        # Ground truth: y_t for all t (mu_minus_t predicts state at time t)
        Y_gt = test_obs  # (T, 48, 48, 1)

        # MSE (full and post-warmup)
        mse_full = compute_image_mse(Y_gt, Y_hat)
        mse_warmup = compute_image_mse(Y_gt[warmup:], Y_hat[warmup:])
        mse_per_t = compute_image_mse_per_timestep(Y_gt, Y_hat)

    elapsed = time.time() - t0

    print(f"\n  === Step 3 Results ===")
    print(f"  Image MSE (full):        {mse_full:.6f}")
    print(f"  Image MSE (post-warmup): {mse_warmup:.6f}")
    print(f"  Image RMSE (post-warmup):{np.sqrt(mse_warmup):.6f}")
    print(f"  MSE range:               [{mse_per_t.min():.6f}, {mse_per_t.max():.6f}]")
    print(f"  MSE median:              {np.median(mse_per_t):.6f}")
    print(f"  T_pred={T}, warmup={warmup}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        'mse_full': float(mse_full),
        'mse_post_warmup': float(mse_warmup),
        'rmse_post_warmup': float(np.sqrt(mse_warmup)),
        'mse_per_timestep': mse_per_t.tolist(),
        'warmup': int(warmup),
        'T_pred': int(T),
        'elapsed_seconds': round(elapsed, 1),
    }


# ========== Main ==========

def parse_args():
    parser = argparse.ArgumentParser(
        description="Kalman Filtering Image Prediction MSE Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data file (.npz)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--gamma-q", type=float, default=1e-6,
                        help="Q regularization (gamma_Q)")
    parser.add_argument("--gamma-r", type=float, default=1e-6,
                        help="R regularization (gamma_R)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Kalman warm-up steps to skip for MSE")
    parser.add_argument("--skip-step1", action="store_true",
                        help="Skip Step 1 (direct prediction)")
    parser.add_argument("--skip-step2", action="store_true",
                        help="Skip Step 2 (Kalman round-trip)")
    parser.add_argument("--skip-step3", action="store_true",
                        help="Skip Step 3 (Kalman feature-space)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and build models
    ckpt, config = load_checkpoint(args.model)

    print("\nReconstructing models...")
    encoder = build_encoder(ckpt, config, device)
    decoder = build_decoder(ckpt, config, device)
    df_state = build_df_state(ckpt, config, device)
    df_obs = build_df_obs(ckpt, config, df_state, device)
    realization = build_realization(ckpt, config, encoder, device)

    # Load test data
    test_obs = load_test_data(args.data, device)

    # Quick sanity: test encoder + decoder roundtrip
    print("\n  Sanity check: encoder -> decoder roundtrip...")
    with torch.no_grad():
        sample_m = encoder(test_obs[:5])  # (5, 100)
        sample_y = decoder(sample_m)       # (5, 48, 48, 1)
        recon_mse = compute_image_mse(test_obs[:5], sample_y)
        print(f"  Encoder-decoder recon MSE (5 samples): {recon_mse:.6f}")

    results = {
        'config': {
            'model_path': str(args.model),
            'data_path': str(args.data),
            'device': str(device),
            'gamma_q': args.gamma_q,
            'gamma_r': args.gamma_r,
            'warmup': args.warmup,
        }
    }

    # ===== Step 1 =====
    if not args.skip_step1:
        results['step1_direct'] = evaluate_direct_prediction(
            encoder, decoder, df_state, df_obs, realization, test_obs, device)

    # ===== Estimate Q, R (needed for Steps 2 and 3) =====
    need_kalman = not args.skip_step2 or not args.skip_step3
    if need_kalman:
        Q, R = estimate_noise_QR(
            encoder, df_state, df_obs, realization, test_obs, device,
            args.gamma_q, args.gamma_r)

    # ===== Step 2 =====
    if not args.skip_step2:
        results['step2_kalman_roundtrip'] = evaluate_kalman_roundtrip(
            encoder, decoder, df_state, df_obs, test_obs, Q, R, device,
            args.warmup)

    # ===== Step 3 =====
    if not args.skip_step3:
        results['step3_kalman_feature'] = evaluate_kalman_feature_space(
            encoder, decoder, df_obs, test_obs, Q, R,
            df_state.V_A, df_obs.V_B, df_state.U_A, df_obs.U_B,
            device, args.warmup)

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if 'step1_direct' in results:
        s1 = results['step1_direct']
        print(f"  Step 1 (Direct):     MSE = {s1['mse']:.6f}, RMSE = {s1['rmse']:.6f}")
    if 'step2_kalman_roundtrip' in results:
        s2 = results['step2_kalman_roundtrip']
        print(f"  Step 2 (Kalman+RT):  MSE = {s2['mse_post_warmup']:.6f}, "
              f"RMSE = {s2['rmse_post_warmup']:.6f}")
    if 'step3_kalman_feature' in results:
        s3 = results['step3_kalman_feature']
        print(f"  Step 3 (Kalman+FS):  MSE = {s3['mse_post_warmup']:.6f}, "
              f"RMSE = {s3['rmse_post_warmup']:.6f}")
    print(f"  Paper Table 1 target: MSE = 0.2006 (clean-1.5k)")
    print("=" * 60)

    # Save results (summary without per-timestep data)
    results_summary = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_summary[k] = {
                kk: vv for kk, vv in v.items() if kk != 'mse_per_timestep'}
        else:
            results_summary[k] = v

    summary_file = output_dir / 'kalman_eval_results.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  Summary: {summary_file}")

    # Save detailed results (with per-timestep MSE)
    detailed_file = output_dir / 'kalman_eval_detailed.json'
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Detailed: {detailed_file}")


if __name__ == "__main__":
    main()
