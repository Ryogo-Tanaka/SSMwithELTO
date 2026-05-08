#!/usr/bin/env python3
"""NCDSSM-LL multi-step free rollout evaluation (PROJ-REBUTTAL Step 4, v3).

Protocol (matches scripts/rebuttal/evaluate_multistep_rollout_dse.py):
  - Sampling: s=0 fixed, 1 rollout per (regime, seed, C). No sliding windows.
  - Context length C: 5 (main) or 10 (extended).
  - Horizons H: {1, 5, 10, 20, 50}.
  - Indexing: standard "next-step" (target frame for h is y[C+h-1]).
  - past_times = [0, dt, ..., (C-1)*dt]
  - future_times = [C*dt, (C+1)*dt, ..., (C+H_max-1)*dt]   (length H_max=50)
  - num_samples=1, no_state_sampling=True (deterministic mean of emission).

Note on OOD: NCDSSM-LL is trained with ctx_len=30; eval here uses C=5/10
which is OOD. The aux_inference_net (bidirectional GRU) still produces a
latent for any sequence length, but quality is uncertain; this is a
documented limitation.

Usage:
  PYTHONUNBUFFERED=1 python scripts/rebuttal/evaluate_multistep_rollout_ncdssm.py \
    --ckpt results/rebuttal/ncdssm_quadlink/clean/seed1/ckpts/model_10000.pt \
    --regime clean --data_seed 1 --seed_label 1 --ctx_len 5 \
    --horizons 1,5,10,20,50 \
    --output results/rebuttal/multistep_rollout/ncdssm/clean/seed1/ctx5/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
NCDSSM_ROOT = REPO_ROOT / "external" / "ncdssm"
sys.path.insert(0, str(NCDSSM_ROOT / "src"))
sys.path.insert(0, str(NCDSSM_ROOT))


METHOD_NAME = "NCDSSM-LL"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to NCDSSM checkpoint (.pt)")
    p.add_argument("--regime", type=str, choices=("clean", "corrupted"),
                   required=True)
    p.add_argument("--data_seed", type=int, required=True,
                   help="quad{N} data file index (1..5)")
    p.add_argument("--seed_label", type=int, required=True,
                   help="Seed label for raw CSV 'seed' column (typically same as data_seed)")
    p.add_argument("--ctx_len", type=int, default=5,
                   help="Context length C (default 5)")
    p.add_argument("--horizons", type=str, default="1,5,10,20,50")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def parse_horizons(s: str) -> List[int]:
    horizons = [int(h.strip()) for h in s.split(",") if h.strip()]
    if not horizons:
        raise ValueError("--horizons cannot be empty")
    return sorted(set(horizons))


def load_test_sequence(data_root: Path, regime: str, data_seed: int):
    """Load test_obs as (T, y_dim) flat tensor + (T, H, W, C) shape info."""
    suffix = "n" if regime == "clean" else "y"
    path = data_root / f"quad{int(data_seed)}_{suffix}.npz"
    npz = np.load(path)
    obs = np.asarray(npz["test_obs"])
    if obs.ndim == 5 and obs.shape[0] == 1:
        obs = obs[0]
    assert obs.ndim == 4, f"Expected (T, H, W, C); got {obs.shape}"
    T, H, W, C = obs.shape
    obs = obs.astype(np.float32)
    if obs.max() > 1.5:
        obs = obs / 255.0
    flat = obs.reshape(T, H * W * C)
    return torch.from_numpy(flat), (T, H, W, C)


@torch.no_grad()
def forecast_multistep_mean(
    model,
    past_target: torch.Tensor,
    past_times: torch.Tensor,
    future_times: torch.Tensor,
    num_samples: int = 1,
    no_state_sampling: bool = True,
) -> torch.Tensor:
    """Same as evaluate_ncdssm_quadlink.forecast_one_step_mean but returns
    multi-step (length H_max) emission means.

    Returns: (B, len(future_times), y_dim)
    """
    from ncdssm.torch_utils import merge_leading_dims

    B, T, _ = past_target.shape
    mask = torch.ones(B, T, device=past_target.device)
    aux_samples, _, _ = model.aux_inference_net(
        past_target, mask, num_samples=1, deterministic=False
    )
    aux_samples = merge_leading_dims(aux_samples, ndims=2)
    mask3 = mask.unsqueeze(-1)
    base_predictions = model.base_ssm.forecast(
        aux_samples,
        (mask3.sum(-1) > 0).float(),
        past_times,
        future_times,
        num_samples=num_samples,
        no_state_sampling=no_state_sampling,
        use_smooth=False,
    )
    aux_forecast = base_predictions["forecast"]            # (S, B, H_max, aux_dim)
    forecast_emit_dist = model.y_emission_net(
        merge_leading_dims(aux_forecast, ndims=2)
    )
    forecast_mean = forecast_emit_dist.mean
    forecast_mean = forecast_mean.view(
        num_samples, B, aux_forecast.shape[-2], -1
    )
    return forecast_mean.mean(dim=0)  # (B, H_max, y_dim)


def main() -> int:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    H_max = max(horizons)
    C = int(args.ctx_len)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[ncdssm-rollout] device={device}", flush=True)

    # ---------------- Load checkpoint and model ----------------
    print(f"[ncdssm-rollout] loading checkpoint: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    dt = float(cfg.get("dt", 0.1))
    print(f"[ncdssm-rollout] step={ckpt.get('step', '?')}, dt={dt}, "
          f"img_size={cfg['img_size']}, y_dim={cfg['y_dim']}", flush=True)

    from experiments.setups import get_model
    model = get_model(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    # ---------------- Load test data ----------------
    data_root = Path(cfg.get(
        "quadlink_data_root",
        "/workspace/nas/SSMwithELTO/data/rkn_quad",
    ))
    test_seq, (T_full, H_img, W_img, C_ch) = load_test_sequence(
        data_root, args.regime, args.data_seed
    )
    test_seq = test_seq.to(device)
    y_dim = test_seq.shape[1]
    n_pixels = H_img * W_img * C_ch
    if T_full < C + H_max:
        raise RuntimeError(
            f"test_obs length {T_full} < C+H_max ({C + H_max})"
        )
    print(f"[ncdssm-rollout] test_seq: T={T_full}, y_dim={y_dim}", flush=True)

    # ---------------- s=0 fixed: forecast over horizons ----------------
    s = 0
    past = test_seq[s:s + C].unsqueeze(0)  # (1, C, y_dim)
    past_times = torch.tensor(
        np.arange(C) * dt, dtype=torch.float32, device=device
    )
    future_times = torch.tensor(
        np.arange(C, C + H_max) * dt, dtype=torch.float32, device=device
    )
    print(
        f"[ncdssm-rollout] s={s}, C={C}, H_max={H_max}; "
        f"past_times[0..-1]=[{past_times[0]:.2f}, ..., {past_times[-1]:.2f}], "
        f"future_times[0..-1]=[{future_times[0]:.2f}, ..., {future_times[-1]:.2f}]",
        flush=True,
    )

    t0 = time.time()
    forecast_mean = forecast_multistep_mean(
        model, past, past_times, future_times,
        num_samples=1, no_state_sampling=True,
    )  # (1, H_max, y_dim)
    forecast_mean = forecast_mean.clamp(0.0, 1.0)

    records: List[Dict] = []
    for h in horizons:
        pred = forecast_mean[0, h - 1]              # (y_dim,)
        target = test_seq[s + C + h - 1]            # (y_dim,)
        sq_err = (pred - target).pow(2).sum().item()
        mse = sq_err / float(n_pixels)
        records.append({
            "method": METHOD_NAME,
            "regime": args.regime,
            "seed": int(args.seed_label),
            "sequence_id": s,
            "context_length": C,
            "horizon": h,
            "mse": float(mse),
        })
        print(
            f"[ncdssm-rollout] h={h:>2d}: MSE={mse:.6f}, RMSE={mse ** 0.5:.6f}",
            flush=True,
        )
    elapsed = time.time() - t0

    # ---------------- Save ----------------
    csv_path = output_dir / "eval_result.csv"
    json_path = output_dir / "eval_result.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "regime", "seed", "sequence_id",
                        "context_length", "horizon", "mse"],
        )
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "method": METHOD_NAME,
        "regime": args.regime,
        "data_seed": int(args.data_seed),
        "seed_label": int(args.seed_label),
        "ckpt": args.ckpt,
        "ckpt_step": int(ckpt.get("step", -1)),
        "ctx_len": C,
        "horizons": horizons,
        "dt": dt,
        "device": str(device),
        "elapsed_seconds": round(elapsed, 2),
        "records": records,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[ncdssm-rollout] wrote {csv_path}\n[ncdssm-rollout] wrote {json_path}\n"
        f"[ncdssm-rollout] elapsed={elapsed:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
