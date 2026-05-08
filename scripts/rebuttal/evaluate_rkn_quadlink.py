#!/usr/bin/env python3
"""Evaluate the RKN baseline on quad-link 1-step prediction (DSE-aligned).

Mirrors `evaluate_lstm_quadlink.py`. The RKN model produces predictions for
time t+1 from posterior at time t (via RKNCell._predict). For DSE-aligned
protocol the target is y[s + ctx_len + 1] (skip 1 frame), so we run a
free 2-step rollout from the context window and use the second prediction.

Usage:
    python scripts/rebuttal/evaluate_rkn_quadlink.py \
        --ckpt results/rebuttal/rkn/clean/seed1/ckpts/model_final.pt \
        --regime clean --data_seed 1 \
        --output results/rebuttal/rkn/clean/seed1/eval_final
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.rkn import RknImageModel
from src.baselines.rkn.data import load_test_sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--regime", type=str, choices=("clean", "corrupted"), required=True)
    p.add_argument("--data_seed", type=int, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--ctx_len", type=int, default=30)
    p.add_argument("--protocol", type=str, choices=("dse", "onestep"),
                   default="dse",
                   help="dse=skip 1 frame; onestep=standard next-step.")
    p.add_argument("--data_root", type=str,
                   default=str(REPO_ROOT / "data" / "rkn_quad"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> RknImageModel:
    cfg = ckpt.get("config", {})
    model = RknImageModel(
        input_resolution=tuple(cfg.get("input_resolution", (48, 48, 1))),
        lod=int(cfg.get("lod", 100)),
        num_basis=int(cfg.get("num_basis", 15)),
        bandwidth=int(cfg.get("bandwidth", 3)),
        trans_net_hidden_units=tuple(cfg.get("trans_net_hidden_units", (128, 128))),
        trans_net_hidden_activation=str(cfg.get("trans_net_hidden_activation", "Tanh")),
        trans_covar=float(cfg.get("trans_covar", 0.1)),
        encoder_hidden=int(cfg.get("encoder_hidden", 200)),
        decoder_hidden=int(cfg.get("decoder_hidden", 200)),
        decoder_upsample_mode=str(cfg.get("decoder_upsample_mode", "nearest")),
        initial_state_variance=float(cfg.get("initial_state_variance", 10.0)),
        normalize_pre=bool(cfg.get("normalize_pre", True)),
    ).to(device)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> int:
    args = parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, device)

    test_seq = load_test_sequence(args.data_root, args.regime, args.data_seed).to(device)
    T = test_seq.shape[0]
    L = int(args.ctx_len)
    if args.protocol == "dse":
        rollout_h = 2
        target_offset = L + 1
        first_s = 0
        last_s = T - L - 2
    elif args.protocol == "onestep":
        rollout_h = 1
        target_offset = L
        first_s = 0
        last_s = T - L - 1
    else:
        raise ValueError(f"unknown protocol={args.protocol}")
    if last_s < first_s:
        raise ValueError(f"insufficient test length T={T} for L={L}, protocol={args.protocol}")

    N = last_s - first_s + 1
    print(f"[eval] regime={args.regime}, data_seed={args.data_seed}, "
          f"protocol={args.protocol}, ctx_len={L}, T={T}, "
          f"first_s={first_s}, last_s={last_s}, N={N}", flush=True)

    H, W, C = test_seq.shape[1], test_seq.shape[2], test_seq.shape[3]
    contexts = torch.stack(
        [test_seq[s:s+L] for s in range(first_s, last_s + 1)], dim=0
    )  # (N, L, H, W, C)
    targets = torch.stack(
        [test_seq[s + target_offset] for s in range(first_s, last_s + 1)], dim=0
    )  # (N, H, W, C)

    bsz = int(args.batch_size)
    mse_per_target = np.zeros(N, dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, bsz):
            ctx_b = contexts[i:i+bsz]
            tgt_b = targets[i:i+bsz]
            preds = model.rollout(ctx_b, horizon=rollout_h)  # (b, h, H, W, C)
            pred_target = preds[:, -1]
            err = (pred_target - tgt_b) ** 2
            err = err.reshape(err.shape[0], -1).mean(dim=1)
            mse_per_target[i:i+err.shape[0]] = err.cpu().numpy()
    elapsed = time.time() - t0

    image_mse = float(mse_per_target.mean())
    image_rmse = float(np.sqrt(image_mse))

    result = {
        "method": "rkn",
        "regime": args.regime,
        "data_seed": int(args.data_seed),
        "protocol": args.protocol,
        "ctx_len": L,
        "horizon": rollout_h,
        "future_steps_ahead": rollout_h,
        "image_mse": image_mse,
        "image_rmse": image_rmse,
        "T_pred": int(N),
        "first_pred_idx": int(first_s + target_offset),
        "last_pred_idx": int(last_s + target_offset),
        "ckpt": str(args.ckpt),
        "ckpt_step": int(ckpt.get("step", -1)),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "num_samples": 1,
        "no_state_sampling": True,
    }
    with open(output / "eval_result.json", "w") as fp:
        json.dump(result, fp, indent=2)
    np.save(output / "mse_per_timestep.npy", mse_per_target)

    print(f"[eval] image_mse={image_mse:.6f} (rmse={image_rmse:.6f}) "
          f"N={N}  elapsed={elapsed:.1f}s", flush=True)
    print(f"[eval] result -> {output / 'eval_result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
