#!/usr/bin/env python3
"""Train NCDSSM-LL on the quad-link image dataset (PROJ-REBUTTAL).

This is a thin training driver around the cloned `external/ncdssm/`
repo. It mostly mirrors the structure of `external/ncdssm/train_pymunk.py`
but adds:
  - explicit RNG seeding (numpy / torch / python random),
  - CLI control of regime / data_seed / output dir / num_steps,
  - sys.path injection so the script can be invoked from
    `/workspace/nas/SSMwithELTO/`,
  - clean per-run logging into the output dir.

Usage:
  PYTHONUNBUFFERED=1 python scripts/rebuttal/train_ncdssm_quadlink.py \
      --regime clean --data_seed 1 \
      --output results/rebuttal/ncdssm_quadlink/clean/seed1 \
      --num_steps 60000

A short smoke run (~300 steps) finishes in a few minutes; the full
60_000-step run takes ~3-6h on a single GPU.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
NCDSSM_ROOT = REPO_ROOT / "external" / "ncdssm"
sys.path.insert(0, str(NCDSSM_ROOT / "src"))
sys.path.insert(0, str(NCDSSM_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(NCDSSM_ROOT / "configs" / "quadlink_ncdssmll.yaml"))
    p.add_argument("--regime", type=str, choices=("clean", "corrupted"),
                   default=None,
                   help="Override regime in config. clean => quad{N}_n.npz, corrupted => quad{N}_y.npz.")
    p.add_argument("--data_seed", type=int, default=None,
                   help="Override data_seed in config (which quad{N} npz to use).")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for python/numpy/torch (separate from data_seed).")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory; overrides exp_root_dir.")
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--log_steps", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--ssm_params_warmup_steps", type=int, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--skip_eval_during_training", action="store_true",
                   help="Skip the val Wasserstein/forecast plots during training "
                        "(faster; we run our own MSE eval afterwards).")
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str, args: argparse.Namespace) -> dict:
    with open(path, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

    overrides = {
        "regime": args.regime,
        "data_seed": args.data_seed,
        "num_steps": args.num_steps,
        "log_steps": args.log_steps,
        "save_steps": args.save_steps,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "ssm_params_warmup_steps": args.ssm_params_warmup_steps,
        "max_grad_norm": args.max_grad_norm,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    cfg["num_workers"] = args.num_workers
    cfg["skip_eval_during_training"] = bool(args.skip_eval_during_training)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    if args.output is not None:
        exp_root_dir = args.output
    else:
        exp_root_dir = cfg["exp_root_dir"].format(
            model=cfg["model"].lower(),
            dataset=cfg["dataset"].lower(),
            timestamp=timestamp,
            regime=str(cfg.get("regime", "default")),
            data_seed=str(cfg.get("data_seed", 0)),
        )
    cfg["exp_root_dir"] = exp_root_dir
    cfg["log_dir"] = os.path.join(exp_root_dir, "logs")
    cfg["ckpt_dir"] = os.path.join(exp_root_dir, "ckpts")
    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    return cfg


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")

    args = parse_args()
    rng_seed = args.seed if args.seed is not None else (args.data_seed or 0) * 42
    set_global_seed(rng_seed)
    print(f"[train] RNG seed: {rng_seed}")

    cfg = load_config(args.config, args)
    print(f"[train] regime={cfg.get('regime')}, data_seed={cfg.get('data_seed')}, "
          f"num_steps={cfg['num_steps']}, exp_root_dir={cfg['exp_root_dir']}")

    # Resolve dataset / model from the cloned NCDSSM repo.
    from experiments.setups import get_dataset, get_model
    from experiments.utils import LinearScheduler
    from ncdssm.torch_utils import grad_norm

    train_ds, val_ds, _ = get_dataset(cfg)
    print(
        f"[train] datasets: train={len(train_ds)} windows, val={len(val_ds)} windows"
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["train_batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        collate_fn=train_ds.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["test_batch_size"],
        num_workers=cfg["num_workers"],
        collate_fn=train_ds.collate_fn,
    )

    device = torch.device(cfg["device"])
    model = get_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params}")

    kf_param_names = {
        name for name, _ in model.named_parameters() if "base_ssm" in name
    }
    kf_params = [p for n, p in model.named_parameters() if n in kf_param_names]
    non_kf_params = [
        p for n, p in model.named_parameters() if n not in kf_param_names
    ]
    optim = torch.optim.Adam(
        params=[{"params": kf_params}, {"params": non_kf_params}],
        lr=cfg["learning_rate"],
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=cfg["lr_decay_rate"]
    )
    reg_scheduler = LinearScheduler(
        iters=cfg.get("reg_anneal_iters", 0),
        maxval=cfg.get("reg_coeff_maxval", 1.0),
    )

    # Persist config snapshot.
    with open(os.path.join(cfg["log_dir"], "config.yaml"), "w") as fp:
        yaml.dump(cfg, fp, default_flow_style=False, sort_keys=False)
    metrics_path = os.path.join(cfg["log_dir"], "train_metrics.jsonl")
    metrics_fp = open(metrics_path, "w", buffering=1)

    train_gen = iter(train_loader)
    num_steps = int(cfg["num_steps"])
    log_steps = int(cfg["log_steps"])
    save_steps = int(cfg["save_steps"])
    warmup_steps = int(cfg.get("ssm_params_warmup_steps", 0))
    max_grad_norm = float(cfg.get("max_grad_norm", float("inf")))

    t0 = datetime.now()
    print(f"[train] start: {t0.isoformat(timespec='seconds')}")
    for step in range(1, num_steps + 1):
        try:
            batch = next(train_gen)
        except StopIteration:
            train_gen = iter(train_loader)
            batch = next(train_gen)

        past_target = batch["past_target"].to(device)
        past_times = batch["past_times"].to(device)
        past_mask = batch["past_mask"].to(device)
        optim.zero_grad()
        out = model(
            past_target,
            past_mask,
            past_times,
            num_samples=cfg.get("num_samples", 1),
        )
        cond_ll = out["likelihood"]
        reg = out["regularizer"]
        loss = -(cond_ll + reg_scheduler.val * reg).mean(0)
        loss.backward()

        if step <= warmup_steps:
            saved_lr0 = optim.param_groups[0]["lr"]
            optim.param_groups[0]["lr"] = 0.0
        gn = grad_norm(model.parameters())
        if max_grad_norm != float("inf"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optim.step()
        if step <= warmup_steps:
            optim.param_groups[0]["lr"] = saved_lr0

        # Compact stdout: every step the upstream script prints; we only
        # log every 50 steps to keep output digestible.
        if step <= 5 or step % 50 == 0 or step == num_steps:
            print(
                f"step={step:>6d} loss={loss.item():.4f} "
                f"cond_ll={cond_ll.mean().item():.4f} "
                f"reg={reg.mean().item():.4f} grad_norm={gn.item():.2f}",
                flush=True,
            )
        metrics_fp.write(json.dumps({
            "step": step,
            "loss": float(loss.item()),
            "cond_ll": float(cond_ll.mean().item()),
            "reg": float(reg.mean().item()),
            "grad_norm": float(gn.item()),
            "reg_coeff": float(reg_scheduler.val),
        }) + "\n")

        if step % cfg["lr_decay_steps"] == 0:
            lr_scheduler.step()
        if step % cfg.get("reg_anneal_every", 1) == 0:
            reg_scheduler.step()
        if step % save_steps == 0 or step == num_steps:
            ckpt_path = os.path.join(cfg["ckpt_dir"], f"model_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"[train] saved checkpoint: {ckpt_path}", flush=True)

        # Optional val pass: lightweight loss-only val on first batch.
        if (
            (step % log_steps == 0 or step == num_steps)
            and not cfg["skip_eval_during_training"]
        ):
            try:
                vbatch = next(iter(val_loader))
                vpt = vbatch["past_target"].to(device)
                vpts = vbatch["past_times"].to(device)
                vpm = vbatch["past_mask"].to(device)
                with torch.no_grad():
                    vout = model(vpt, vpm, vpts, num_samples=1)
                    vloss = -(vout["likelihood"] + vout["regularizer"]).mean(0)
                print(f"[val] step={step} val_loss={vloss.item():.4f}", flush=True)
                metrics_fp.write(json.dumps({
                    "step": step,
                    "val_loss": float(vloss.item()),
                    "val_cond_ll": float(vout["likelihood"].mean().item()),
                    "val_reg": float(vout["regularizer"].mean().item()),
                }) + "\n")
            except Exception as e:
                print(f"[val] skipped due to error: {e}", flush=True)

    metrics_fp.close()
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[train] done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
