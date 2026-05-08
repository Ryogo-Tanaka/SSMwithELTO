#!/usr/bin/env python3
"""Multi-step rollout evaluation for the LSTM baseline.

Mirrors `evaluate_multistep_rollout_ncdssm.py`. Rollout protocol (v3):
    s = 0 fixed, single rollout per (regime, seed, C).
    Context y[0..C-1] -> free rollout of H_max frames.
    For each h in horizons, target = y[C + h - 1], pred = rollout output [h-1].

Usage:
    python scripts/rebuttal/evaluate_multistep_rollout_lstm.py \
        --ckpt results/rebuttal/lstm/clean/seed1/ckpts/model_final.pt \
        --regime clean --data_seed 1 --seed_label 1 \
        --ctx_len 5 \
        --horizons 1,5,10,20,50 \
        --output results/rebuttal/lstm/clean/seed1/rollout/ctx5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.lstm import CnnLstmCnn
from src.baselines.lstm.data import load_test_sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--regime", type=str, choices=("clean", "corrupted"), required=True)
    p.add_argument("--data_seed", type=int, required=True)
    p.add_argument("--seed_label", type=int, default=None,
                   help="Value used in the 'seed' column of CSV output. "
                        "Defaults to data_seed.")
    p.add_argument("--ctx_len", type=int, default=5)
    p.add_argument("--horizons", type=str, default="1,5,10,20,50")
    p.add_argument("--data_root", type=str,
                   default=str(REPO_ROOT / "data" / "rkn_quad"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> CnnLstmCnn:
    cfg = ckpt.get("config", {})
    model = CnnLstmCnn(
        input_resolution=tuple(cfg.get("input_resolution", (48, 48, 1))),
        feature_dim=int(cfg.get("feature_dim", 100)),
        lstm_hidden=int(cfg.get("lstm_hidden", 100)),
        lstm_layers=int(cfg.get("lstm_layers", 1)),
        encoder_hidden=int(cfg.get("encoder_hidden", 200)),
        decoder_hidden=int(cfg.get("decoder_hidden", 200)),
    ).to(device)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    seed_label = args.data_seed if args.seed_label is None else int(args.seed_label)

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    if not horizons:
        raise ValueError("--horizons cannot be empty")
    H_max = max(horizons)
    C = int(args.ctx_len)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, device)

    test_seq = load_test_sequence(args.data_root, args.regime, args.data_seed).to(device)
    T = test_seq.shape[0]
    if T < C + H_max:
        raise ValueError(f"insufficient test length T={T} for C={C}, H_max={H_max}")

    ctx = test_seq[:C].unsqueeze(0)  # (1, C, H, W, Cimg)

    t0 = time.time()
    with torch.no_grad():
        preds = model.rollout(ctx, horizon=H_max)  # (1, H_max, H, W, Cimg)
    elapsed = time.time() - t0

    records = []
    for h in horizons:
        idx = h - 1
        pred = preds[0, idx]
        target = test_seq[C + h - 1]
        mse = float(((pred - target) ** 2).mean().item())
        rec = {
            "method": "lstm",
            "regime": args.regime,
            "seed": int(seed_label),
            "sequence_id": 0,
            "context_length": C,
            "horizon": int(h),
            "mse": mse,
        }
        records.append(rec)
        print(f"[rollout] C={C} h={h:>2d} mse={mse:.6f}", flush=True)

    csv_path = output / "eval_result.csv"
    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    json_payload = {
        "method": "lstm",
        "regime": args.regime,
        "data_seed": int(args.data_seed),
        "seed_label": seed_label,
        "ckpt": str(args.ckpt),
        "ckpt_step": int(ckpt.get("step", -1)),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "ctx_len": C,
        "horizons": horizons,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "records": records,
    }
    with open(output / "eval_result.json", "w") as fp:
        json.dump(json_payload, fp, indent=2)

    print(f"[rollout] csv -> {csv_path}  elapsed={elapsed:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
