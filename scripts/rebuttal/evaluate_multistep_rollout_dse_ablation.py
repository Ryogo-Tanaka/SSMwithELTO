#!/usr/bin/env python3
"""DSE multi-step free rollout evaluation with ablation variant support.

Variant of scripts/rebuttal/evaluate_multistep_rollout_dse.py that accepts a
--variant argument and uses src.rebuttal.dse_loader_ablation.load_dse_variant
so the realization can be swapped to PCA for DSE_no_cca. Other variants
(full / joint_training / no_closed_form) use the standard CCA realization,
matching the original script's behaviour.

The evaluation protocol is identical to the Step 4 rollout script: free
rollout from y[0:C] context for horizons 1, 5, 10, 20, 50.

Usage:
  PYTHONUNBUFFERED=1 python scripts/rebuttal/evaluate_multistep_rollout_dse_ablation.py \
    --variant no_cca \
    --model results/rebuttal/dse_ablation/no_cca/seed1/models/final_model.pth \
    --data data/rkn_quad/quad1_n.npz \
    --regime clean --seed_label 42 --ctx_len 5 \
    --horizons 1,5,10,20,50 --gamma-q 1e-6 --gamma-r 1e-6 \
    --output results/rebuttal/dse_ablation/no_cca/seed1/rollout/ctx5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rebuttal.dse_loader_ablation import (  # noqa: E402
    estimate_noise_QR,
    load_dse_variant,
    load_test_data,
)
from src.inference.kalman_filter import OperatorBasedKalmanFilter  # noqa: E402


METHOD_NAME = "DSE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True,
                   choices=("full", "joint_training", "no_cca", "no_closed_form"),
                   help="DSE ablation variant identifier (controls realization).")
    p.add_argument("--model", type=str, required=True,
                   help="Path to DSE checkpoint (.pth)")
    p.add_argument("--data", type=str, required=True,
                   help="Path to data file (.npz)")
    p.add_argument("--regime", type=str, choices=("clean", "corrupted"),
                   required=True)
    p.add_argument("--seed_label", type=int, required=True,
                   help="Seed label (used for raw CSV 'seed' column)")
    p.add_argument("--ctx_len", type=int, default=5,
                   help="Context length C (default 5)")
    p.add_argument("--horizons", type=str, default="1,5,10,20,50",
                   help="Comma-separated horizon list")
    p.add_argument("--gamma-q", type=float, default=1e-6)
    p.add_argument("--gamma-r", type=float, default=1e-6)
    p.add_argument("--output", type=str, required=True,
                   help="Output directory")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def parse_horizons(s: str) -> List[int]:
    horizons = [int(h.strip()) for h in s.split(",") if h.strip()]
    if not horizons:
        raise ValueError("--horizons cannot be empty")
    return sorted(set(horizons))


@torch.no_grad()
def predict_image_from_mu(
    mu: torch.Tensor,
    V_B: torch.Tensor,
    df_obs,
    decoder: nn.Module,
) -> torch.Tensor:
    """mu (dA,) -> y_hat (H, W, C) via V_B, readout_B (or U_B), decoder."""
    h_pred = V_B @ mu  # (dB,)
    readout_type = getattr(df_obs, "readout_type", "linear")
    if readout_type == "nonlinear":
        m_hat = df_obs.readout_net(h_pred)
    else:
        m_hat = h_pred @ df_obs.U_B
    y_hat = decoder(m_hat.unsqueeze(0)).squeeze(0)
    return y_hat.clamp(0.0, 1.0)


def main() -> int:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    H_max = max(horizons)
    C = int(args.ctx_len)
    variant = args.variant

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[ablation-rollout] CUDA not available, using CPU", flush=True)
    else:
        device = torch.device(args.device)
    print(f"[ablation-rollout] variant={variant}, device={device}", flush=True)

    print(f"[ablation-rollout] loading model: {args.model}", flush=True)
    dse = load_dse_variant(args.model, device, variant=variant)

    print(f"[ablation-rollout] loading data : {args.data}", flush=True)
    test_obs = load_test_data(args.data, device)
    T_full, H_img, W_img, C_ch = test_obs.shape
    print(f"[ablation-rollout] test_obs: T={T_full}, image=({H_img},{W_img},{C_ch})",
          flush=True)
    if T_full < C + H_max:
        raise RuntimeError(
            f"test_obs length {T_full} < C+H_max ({C + H_max}); cannot rollout"
        )

    Q, R = estimate_noise_QR(
        dse.encoder, dse.df_state, dse.df_obs, dse.realization,
        test_obs, device, args.gamma_q, args.gamma_r,
    )
    print(
        f"[ablation-rollout] Q.trace={Q.trace().item():.4f}, "
        f"R.trace={R.trace().item():.4f}",
        flush=True,
    )

    V_A = dse.df_state.V_A
    V_B = dse.df_obs.V_B
    U_A = getattr(dse.df_state, "U_A", None)
    U_B = getattr(dse.df_obs, "U_B", None)
    readout_A = getattr(dse.df_state, "readout_net", None) \
        if getattr(dse.df_state, "readout_type", "linear") == "nonlinear" else None
    readout_B = getattr(dse.df_obs, "readout_net", None) \
        if getattr(dse.df_obs, "readout_type", "linear") == "nonlinear" else None

    kf = OperatorBasedKalmanFilter(
        V_A=V_A, V_B=V_B, U_A=U_A, U_B=U_B, Q=Q, R=R,
        encoder=dse.encoder, df_obs_layer=dse.df_obs, device=str(device),
        readout_A=readout_A, readout_B=readout_B,
    )

    s = 0
    context = test_obs[s:s + C]
    print(f"[ablation-rollout] init from y[0:{C}] using method='from_observations'",
          flush=True)
    kf.initialize_state(context, method="from_observations")

    t0 = time.time()
    with torch.no_grad():
        for t in range(C):
            mu_minus, Sigma_minus = kf.predict_step(kf.mu, kf.Sigma)
            mu_plus, Sigma_plus, _ = kf.update_step(
                mu_minus, Sigma_minus, test_obs[s + t]
            )
            kf.mu = mu_plus
            kf.Sigma = Sigma_plus

        mu = kf.mu.clone()
        records: List[Dict] = []
        for h in range(1, H_max + 1):
            mu = V_A @ mu
            if h in horizons:
                y_hat = predict_image_from_mu(mu, V_B, dse.df_obs, dse.decoder)
                y_gt = test_obs[s + C + h - 1]
                if y_hat.shape != y_gt.shape:
                    raise RuntimeError(
                        f"shape mismatch h={h}: y_hat={tuple(y_hat.shape)} "
                        f"vs y_gt={tuple(y_gt.shape)}"
                    )
                mse = ((y_hat - y_gt) ** 2).mean().item()
                records.append({
                    "method": METHOD_NAME,
                    "variant": variant,
                    "regime": args.regime,
                    "seed": int(args.seed_label),
                    "sequence_id": s,
                    "context_length": C,
                    "horizon": h,
                    "mse": float(mse),
                })
                print(
                    f"[ablation-rollout] h={h:>2d}: MSE={mse:.6f}, "
                    f"RMSE={mse ** 0.5:.6f}",
                    flush=True,
                )
    elapsed = time.time() - t0

    csv_path = output_dir / "eval_result.csv"
    json_path = output_dir / "eval_result.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "variant", "regime", "seed", "sequence_id",
                        "context_length", "horizon", "mse"],
        )
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "method": METHOD_NAME,
        "variant": variant,
        "regime": args.regime,
        "seed_label": int(args.seed_label),
        "data_path": args.data,
        "model_path": args.model,
        "ctx_len": C,
        "horizons": horizons,
        "gamma_q": args.gamma_q,
        "gamma_r": args.gamma_r,
        "Q_trace": float(Q.trace().item()),
        "R_trace": float(R.trace().item()),
        "device": str(device),
        "elapsed_seconds": round(elapsed, 2),
        "records": records,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[ablation-rollout] wrote {csv_path}\n[ablation-rollout] wrote {json_path}\n"
        f"[ablation-rollout] elapsed={elapsed:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
