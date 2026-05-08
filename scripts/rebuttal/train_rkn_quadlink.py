#!/usr/bin/env python3
"""Train the RKN baseline on quad-link images (PROJ-REBUTTAL Step 8).

Mirrors `train_lstm_quadlink.py`. Outputs:
    {output}/
        logs/{config.yaml, train_metrics.jsonl}
        ckpts/{model_epoch_{N}.pt, model_final.pt}

Usage:
    PYTHONUNBUFFERED=1 python scripts/rebuttal/train_rkn_quadlink.py \
        --regime clean --data_seed 1 \
        --output results/rebuttal/rkn/clean/seed1 \
        --num_epochs 30
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.rkn import RknImageModel
from src.baselines.rkn.data import QuadlinkRknDataset, resolve_quadlink_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(REPO_ROOT / "configs" / "rebuttal" / "rkn_quadlink.yaml"))
    p.add_argument("--regime", type=str, choices=("clean", "corrupted"), default=None)
    p.add_argument("--data_seed", type=int, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (default = data_seed * 42).")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--save_every_epochs", type=int, default=None)
    p.add_argument("--val_every_epochs", type=int, default=None)
    p.add_argument("--log_every_steps", type=int, default=None)
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str, args: argparse.Namespace) -> dict:
    with open(path, "r") as fp:
        cfg = yaml.safe_load(fp)
    overrides = {
        "regime": args.regime,
        "data_seed": args.data_seed,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "num_workers": args.num_workers,
        "max_grad_norm": args.max_grad_norm,
        "save_every_epochs": args.save_every_epochs,
        "val_every_epochs": args.val_every_epochs,
        "log_every_steps": args.log_every_steps,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def compute_loss(model: RknImageModel, batch_y: torch.Tensor) -> torch.Tensor:
    """Teacher-forcing next-step image MSE (predict y[t+1] from y[0..t])."""
    y_pred = model(batch_y)  # (B, T-1, H, W, C)
    target = batch_y[:, 1:]
    return ((y_pred - target) ** 2).mean()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args)

    rng_seed = args.seed if args.seed is not None else int(cfg.get("data_seed", 0)) * 42
    set_global_seed(rng_seed)

    output = Path(args.output)
    log_dir = output / "logs"
    ckpt_dir = output / "ckpts"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] regime={cfg['regime']}, data_seed={cfg['data_seed']}, "
          f"num_epochs={cfg['num_epochs']}, output={output}", flush=True)
    print(f"[train] RNG seed: {rng_seed}", flush=True)

    npz_path = resolve_quadlink_path(
        cfg["quadlink_data_root"], cfg["regime"], int(cfg["data_seed"])
    )
    if not npz_path.is_file():
        raise FileNotFoundError(f"quad-link npz not found: {npz_path}")
    print(f"[train] data: {npz_path}", flush=True)

    train_ds = QuadlinkRknDataset(
        file_path=str(npz_path),
        split="train",
        window_len=int(cfg["window_len"]),
        stride=int(cfg["stride"]),
        val_size=int(cfg["val_size"]),
    )
    val_ds = QuadlinkRknDataset(
        file_path=str(npz_path),
        split="val",
        window_len=int(cfg["window_len"]),
        stride=int(cfg["stride"]),
        val_size=int(cfg["val_size"]),
    )
    print(f"[train] datasets: train={len(train_ds)} windows, val={len(val_ds)} windows",
          flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        drop_last=False,
    )

    device = torch.device(cfg["device"])
    model = RknImageModel(
        input_resolution=tuple(cfg["input_resolution"]),
        lod=int(cfg["lod"]),
        num_basis=int(cfg["num_basis"]),
        bandwidth=int(cfg["bandwidth"]),
        trans_net_hidden_units=tuple(cfg["trans_net_hidden_units"]),
        trans_net_hidden_activation=str(cfg["trans_net_hidden_activation"]),
        trans_covar=float(cfg["trans_covar"]),
        encoder_hidden=int(cfg["encoder_hidden"]),
        decoder_hidden=int(cfg["decoder_hidden"]),
        decoder_upsample_mode=str(cfg.get("decoder_upsample_mode", "nearest")),
        initial_state_variance=float(cfg["initial_state_variance"]),
        normalize_pre=bool(cfg.get("normalize_pre", True)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params}", flush=True)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    cfg_snapshot = copy.deepcopy(cfg)
    cfg_snapshot["rng_seed"] = rng_seed
    cfg_snapshot["data_path"] = str(npz_path)
    cfg_snapshot["num_params"] = n_params
    with open(log_dir / "config.yaml", "w") as fp:
        yaml.safe_dump(cfg_snapshot, fp, sort_keys=False)
    metrics_fp = open(log_dir / "train_metrics.jsonl", "w", buffering=1)

    num_epochs = int(cfg["num_epochs"])
    save_every = int(cfg.get("save_every_epochs", 5))
    val_every = int(cfg.get("val_every_epochs", 1))
    log_every = int(cfg.get("log_every_steps", 50))
    max_grad_norm = float(cfg.get("max_grad_norm", float("inf")))

    t0 = datetime.now()
    print(f"[train] start: {t0.isoformat(timespec='seconds')}", flush=True)
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_n = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optim.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            if max_grad_norm != float("inf"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optim.step()

            global_step += 1
            epoch_loss_sum += float(loss.item()) * batch.size(0)
            epoch_n += batch.size(0)

            if global_step <= 5 or global_step % log_every == 0:
                print(f"step={global_step:>6d} epoch={epoch:>3d} "
                      f"train_loss={loss.item():.6f}", flush=True)
                metrics_fp.write(json.dumps({
                    "type": "step",
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                }) + "\n")

        epoch_loss = epoch_loss_sum / max(1, epoch_n)
        log_entry = {
            "type": "epoch",
            "epoch": epoch,
            "step": global_step,
            "train_loss_epoch": epoch_loss,
        }
        if val_every > 0 and epoch % val_every == 0:
            model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device, non_blocking=True)
                    vloss = compute_loss(model, vbatch)
                    val_loss_sum += float(vloss.item()) * vbatch.size(0)
                    val_n += vbatch.size(0)
            val_loss = val_loss_sum / max(1, val_n)
            log_entry["val_loss"] = val_loss
            print(f"[epoch] {epoch:>3d}/{num_epochs} "
                  f"train_loss={epoch_loss:.6f} val_loss={val_loss:.6f}",
                  flush=True)
        else:
            print(f"[epoch] {epoch:>3d}/{num_epochs} train_loss={epoch_loss:.6f}",
                  flush=True)
        metrics_fp.write(json.dumps(log_entry) + "\n")

        if save_every > 0 and (epoch % save_every == 0 or epoch == num_epochs):
            ckpt_path = ckpt_dir / f"model_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "config": cfg_snapshot,
            }, ckpt_path)
            print(f"[train] saved checkpoint: {ckpt_path}", flush=True)

    final_path = ckpt_dir / "model_final.pt"
    torch.save({
        "epoch": num_epochs,
        "step": global_step,
        "model": model.state_dict(),
        "config": cfg_snapshot,
    }, final_path)
    print(f"[train] saved final: {final_path}", flush=True)

    metrics_fp.close()
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[train] done in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
