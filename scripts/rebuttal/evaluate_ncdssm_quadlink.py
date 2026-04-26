#!/usr/bin/env python3
"""Evaluate NCDSSM-LL on the quad-link image dataset (PROJ-REBUTTAL).

Mirrors the DSE 1-step prediction MSE protocol used in
`scripts/evaluate_kalman_image_mse.py` (Step 1) for an apples-to-apples
comparison.

DSE-aligned protocol (default, --protocol=dse):
  DSE's "state at time t" is the prior at time t: it is computed from
  past frames y_{t-L..t-1} only, and y_t is NOT included
  (src/ssm/realization.py L231 builds past_indices = [t-1, ..., t-L]).
  V_A then advances the state to time t+1 and the decoder predicts
  y_{t+1} from this advanced state. So DSE predicts y_{t+1} given the
  30 frames y_{t-L..t-1} — i.e., one frame is "skipped" between the
  end of the input window and the prediction target. The first
  predictable frame is y_{h+1} = y_31; the last is y_{h + T_pred} =
  y_{1470} (T_pred = T - 2L = 1440 because the realization needs both
  past and future blocks at fit time, even though estimate_states
  itself is causal — see src/ssm/realization.py L218 N=T-2L+1, then
  predict_sequence drops one to get T_pred = N - 1 = 1440).

  We therefore:
    - Use input window past_target = y_{s..s+L-1} (L = ctx_len = 30).
    - Set future_times = [(L+1) * dt] (2*dt after the last past time).
    - Compare prediction with target = y_{s+L+1}.
    - Sweep s = 0..(T - 2L - 1) so that predicted frames cover
      y_{L+1}..y_{T-L} = y_31..y_1470 (1440 frames total).

Optional standard 1-step protocol (--protocol=onestep):
  Uses NCDSSM's natural 1-step convention: predict y_{s+L} from
  y_{s..s+L-1} (no skipped frame). This is the protocol most papers
  call "1-step prediction"; it is included for reference but is NOT
  what DSE reports.

Both protocols use the *mean* of the emission Gaussian (not a sample),
clamp predictions to [0, 1], and compute per-pixel MSE averaged over
all predicted timesteps.

Usage:
  PYTHONUNBUFFERED=1 python scripts/rebuttal/evaluate_ncdssm_quadlink.py \
    --ckpt results/rebuttal/ncdssm_quadlink/clean/seed1/ckpts/model_60000.pt \
    --regime clean --data_seed 1 \
    --output results/rebuttal/ncdssm_quadlink/clean/seed1/eval

Outputs:
  - {output}/eval_result.json            scalars + metadata
  - {output}/mse_per_timestep.npy        (T-ctx_len,) float32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
NCDSSM_ROOT = REPO_ROOT / "external" / "ncdssm"
sys.path.insert(0, str(NCDSSM_ROOT / "src"))
sys.path.insert(0, str(NCDSSM_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to the NCDSSM checkpoint (.pt)")
    p.add_argument("--regime", type=str, choices=("clean", "corrupted"),
                   required=True)
    p.add_argument("--data_seed", type=int, required=True,
                   help="quad{N} index (1..5).")
    p.add_argument("--output", type=str, required=True,
                   help="Output directory for the evaluation result.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Number of windows per forecast batch.")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Forecast num_samples; we use the mean across "
                        "samples to reduce noise. 1 means we use a single "
                        "no-state-sampling forward.")
    p.add_argument("--ctx_len", type=int, default=None,
                   help="Override context length (defaults to value in "
                        "checkpoint config).")
    p.add_argument("--protocol", type=str, default="dse",
                   choices=("dse", "onestep"),
                   help="dse: DSE-aligned protocol (predict y_{s+L+1} from "
                        "y_{s..s+L-1}, future_times=[(L+1)*dt], 1440 frames). "
                        "onestep: standard 1-step (predict y_{s+L} from "
                        "y_{s..s+L-1}, future_times=[L*dt], 1470 frames).")
    p.add_argument("--no_state_sampling", action="store_true", default=True,
                   help="Use mean of latent state distributions in forecast "
                        "(deterministic). Default True.")
    return p.parse_args()


def load_test_sequence(data_root: Path, regime: str, data_seed: int) -> torch.Tensor:
    suffix = "n" if regime == "clean" else "y"
    path = data_root / f"quad{int(data_seed)}_{suffix}.npz"
    print(f"[eval] loading {path}")
    npz = np.load(path)
    obs = npz["test_obs"]
    obs = np.asarray(obs)
    if obs.ndim == 5 and obs.shape[0] == 1:
        obs = obs[0]
    assert obs.ndim == 4, f"Expected (T, H, W, C); got {obs.shape}"
    T, H, W, C = obs.shape
    assert C == 1
    obs = obs.astype(np.float32)
    if obs.max() > 1.5:
        obs = obs / 255.0
    print(f"[eval] test_obs: T={T}, H={H}, W={W}")
    flat = obs.reshape(T, H * W * C)
    return torch.from_numpy(flat), (T, H, W, C)


@torch.no_grad()
def forecast_one_step_mean(
    model,
    past_target: torch.Tensor,
    past_times: torch.Tensor,
    future_times: torch.Tensor,
    num_samples: int,
    no_state_sampling: bool,
) -> torch.Tensor:
    """Re-implement model.forecast() but return the emission *mean* for
    the forecast block instead of a sample.

    Returns: (B, len(future_times), y_dim) — mean prediction.
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
    aux_forecast = base_predictions["forecast"]   # (S, B, H, aux_dim)
    forecast_emit_dist = model.y_emission_net(
        merge_leading_dims(aux_forecast, ndims=2)
    )
    # Use mean: works for both Bernoulli (probability map) and Gaussian
    # (pixel-mean). Has shape (S*B, H, y_dim).
    forecast_mean = forecast_emit_dist.mean
    forecast_mean = forecast_mean.view(num_samples, B, aux_forecast.shape[-2], -1)
    return forecast_mean.mean(dim=0)  # average over samples -> (B, H, y_dim)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[eval] device={device}")

    print(f"[eval] loading checkpoint {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"[eval] checkpoint step: {ckpt.get('step', '?')}, "
          f"img_size={cfg['img_size']}, y_dim={cfg['y_dim']}")

    ctx_len = int(args.ctx_len if args.ctx_len is not None else cfg["ctx_len"])
    dt = float(cfg.get("dt", 0.1))
    print(f"[eval] ctx_len={ctx_len}, dt={dt}")

    from experiments.setups import get_model
    model = get_model(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    data_root = Path(cfg.get(
        "quadlink_data_root",
        "/workspace/nas/SSMwithELTO/data/rkn_quad",
    ))
    test_seq, (T_full, H, W, C) = load_test_sequence(
        data_root, args.regime, args.data_seed
    )
    test_seq = test_seq.to(device)  # (T, y_dim)
    y_dim = test_seq.shape[1]

    # Protocol-dependent window indexing.
    # past_target windows are always y_{s..s+ctx_len-1} (length ctx_len).
    # protocol "dse":     target = y_{s + ctx_len + 1}, future_times = [(L+1)*dt]
    #                     (skip 1 frame; DSE prior-at-t convention; T_pred = T-2L = 1440 for T=1500, L=30).
    # protocol "onestep": target = y_{s + ctx_len},     future_times = [L*dt]
    #                     (standard 1-step; T_pred = T-L = 1470 for T=1500, L=30).
    if args.protocol == "dse":
        target_offset = ctx_len + 1  # frame to predict relative to s
        future_steps_ahead = ctx_len + 1  # in units of dt past time origin
        s_max = T_full - ctx_len - ctx_len - 1 + 1  # inclusive count: T-2L
        n_predict = s_max
        first_pred_idx = ctx_len + 1            # y_{L+1} = y_31
        last_pred_idx = first_pred_idx + n_predict - 1  # y_{T-L} = y_1470
    else:  # onestep
        target_offset = ctx_len
        future_steps_ahead = ctx_len
        s_max = T_full - ctx_len  # inclusive count: T-L
        n_predict = s_max
        first_pred_idx = ctx_len
        last_pred_idx = first_pred_idx + n_predict - 1

    past_times = torch.tensor(
        np.arange(ctx_len) * dt, dtype=torch.float32, device=device
    )
    future_times = torch.tensor(
        [future_steps_ahead * dt], dtype=torch.float32, device=device
    )

    print(
        f"[eval] protocol={args.protocol}: predicting {n_predict} timesteps "
        f"(y_{first_pred_idx}..y_{last_pred_idx}); "
        f"future_times[0]={future_steps_ahead}*dt; batch_size={args.batch_size}"
    )
    se_per_t = np.zeros(n_predict, dtype=np.float64)  # mean squared error per t
    n_pixels = H * W * C

    t0 = time.time()
    bs = int(args.batch_size)
    for batch_start in range(0, n_predict, bs):
        batch_end = min(batch_start + bs, n_predict)
        b = batch_end - batch_start
        # Window starts: past_target = y_{s..s+ctx_len-1}, target = y_{s+target_offset}.
        starts = np.arange(batch_start, batch_end)
        past = torch.stack(
            [test_seq[s : s + ctx_len] for s in starts], dim=0
        )  # (b, ctx_len, y_dim)
        target = test_seq[target_offset + starts]  # (b, y_dim)

        forecast_mean = forecast_one_step_mean(
            model,
            past_target=past,
            past_times=past_times,
            future_times=future_times,
            num_samples=int(args.num_samples),
            no_state_sampling=bool(args.no_state_sampling),
        )  # (b, 1, y_dim)
        pred = forecast_mean.squeeze(1)  # (b, y_dim)
        pred = pred.clamp(0.0, 1.0)  # mirror DSE pixel range

        sq_err = (pred - target).pow(2).sum(dim=-1)  # (b,) sum over pixels
        se_per_t[batch_start:batch_end] = sq_err.detach().cpu().numpy()

        if (batch_end // bs) % 5 == 0 or batch_end == n_predict:
            print(
                f"[eval] {batch_end:>5d}/{n_predict} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    mse_per_t = se_per_t / float(n_pixels)
    image_mse = float(mse_per_t.mean())
    image_rmse = float(np.sqrt(image_mse))

    print(
        f"\n[eval] === Results (ctx_len={ctx_len}, regime={args.regime}, "
        f"data_seed={args.data_seed}) ===\n"
        f"[eval]  Image MSE  : {image_mse:.6f}\n"
        f"[eval]  Image RMSE : {image_rmse:.6f}\n"
        f"[eval]  T_pred     : {n_predict}\n"
        f"[eval]  total time : {time.time() - t0:.1f}s",
        flush=True,
    )

    np.save(output_dir / "mse_per_timestep.npy", mse_per_t.astype(np.float32))
    summary: Dict = {
        "method": "NCDSSM-LL",
        "regime": args.regime,
        "data_seed": int(args.data_seed),
        "protocol": str(args.protocol),
        "ctx_len": int(ctx_len),
        "horizon": 1,
        "future_steps_ahead": int(future_steps_ahead),
        "first_pred_idx": int(first_pred_idx),
        "last_pred_idx": int(last_pred_idx),
        "image_mse": image_mse,
        "image_rmse": image_rmse,
        "T_pred": int(n_predict),
        "ckpt": str(args.ckpt),
        "ckpt_step": int(ckpt.get("step", -1)),
        "device": str(device),
        "num_samples": int(args.num_samples),
        "no_state_sampling": bool(args.no_state_sampling),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(output_dir / "eval_result.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[eval] wrote {output_dir / 'eval_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
