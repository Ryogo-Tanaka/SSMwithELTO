#!/usr/bin/env python3
"""Smoke test for the PROJ-REBUTTAL NCDSSM adaptation.

Checks:
  1. ImageEncoder/ImageDecoder work for img_size=48 (and still 32).
  2. QuadlinkDataset loads quad1_n.npz, returns shape-consistent dict.
  3. setups.get_model("NCDSSMLL", quadlink) builds a usable model.
  4. A single forward+backward pass on a real batch does not error and
     produces a finite loss.

Run with:
  PYTHONUNBUFFERED=1 python scripts/rebuttal/smoke_test_ncdssm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Make the cloned NCDSSM package importable.
NCDSSM_ROOT = Path(__file__).resolve().parents[2] / "external" / "ncdssm"
sys.path.insert(0, str(NCDSSM_ROOT / "src"))
sys.path.insert(0, str(NCDSSM_ROOT))


def section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    section("1. ImageEncoder / ImageDecoder shape sanity")
    from ncdssm.modules import ImageEncoder, ImageDecoder

    for img_size in (32, 48):
        enc = ImageEncoder(img_size=img_size, channels=1, out_dim=64)
        dec = ImageDecoder(in_dim=4, img_size=img_size, channels=1)
        x = torch.randn(2, 5, img_size * img_size)  # (B, T, H*W)
        h = enc(x)
        z = torch.randn(2, 5, 4)
        y = dec(z)
        print(
            f"  img_size={img_size}: enc out={tuple(h.shape)}, "
            f"dec out={tuple(y.shape)}"
        )
        assert h.shape == (2, 5, 64), h.shape
        assert y.shape == (2, 5, 1, img_size, img_size), y.shape

    section("2. QuadlinkDataset loading")
    from ncdssm.datasets import QuadlinkDataset, resolve_quadlink_path

    npz_path = resolve_quadlink_path(
        data_root=str(Path("/workspace/nas/SSMwithELTO/data/rkn_quad")),
        regime="clean",
        data_seed=1,
    )
    print(f"  npz path: {npz_path}")
    ds_train = QuadlinkDataset(
        file_path=str(npz_path),
        regime="clean",
        split="train",
        train=True,
        ctx_len=30,
        pred_len=1,
        stride=1,
        val_size=300,
        seed=1,
    )
    ds_val = QuadlinkDataset(
        file_path=str(npz_path),
        regime="clean",
        split="val",
        train=False,
        ctx_len=30,
        pred_len=1,
        stride=1,
        val_size=300,
        seed=2,
    )
    ds_test = QuadlinkDataset(
        file_path=str(npz_path),
        regime="clean",
        split="test",
        train=False,
        ctx_len=30,
        pred_len=1,
        stride=1,
        val_size=300,
        seed=3,
    )
    print(
        f"  train windows={len(ds_train)} (T={ds_train._T}), "
        f"val windows={len(ds_val)} (T={ds_val._T}), "
        f"test windows={len(ds_test)} (T={ds_test._T})"
    )
    print(f"  y_dim={ds_train.y_dim}, img_size={ds_train.img_size}")
    sample = ds_train[0]
    print(
        "  sample[0]: past_target",
        tuple(sample["past_target"].shape),
        "future_target",
        sample["future_target"],
        "past_times",
        tuple(sample["past_times"].shape),
        "past_mask",
        tuple(sample["past_mask"].shape),
    )
    assert sample["past_target"].shape == (30, 48 * 48)
    assert sample["future_target"] is None
    sample_val = ds_val[0]
    print(
        "  sample_val[0]: past_target",
        tuple(sample_val["past_target"].shape),
        "future_target",
        tuple(sample_val["future_target"].shape),
        "future_times",
        tuple(sample_val["future_times"].shape),
    )
    assert sample_val["future_target"].shape == (1, 48 * 48)

    # Pixel range
    pt = sample["past_target"]
    print(
        f"  past_target stats: min={pt.min():.4f}, max={pt.max():.4f}, "
        f"mean={pt.mean():.4f}"
    )
    assert pt.min() >= 0.0 and pt.max() <= 1.0 + 1e-5

    # Collate
    loader = torch.utils.data.DataLoader(
        ds_train, batch_size=4, collate_fn=ds_train.collate_fn, num_workers=0
    )
    batch = next(iter(loader))
    print(
        "  batch: past_target",
        tuple(batch["past_target"].shape),
        "past_times",
        tuple(batch["past_times"].shape),
        "past_mask",
        tuple(batch["past_mask"].shape),
    )

    section("3. setups.get_model — NCDSSM-LL on quadlink")
    from experiments.setups import get_model

    config = dict(
        dataset="quadlink",
        model="NCDSSMLL",
        img_size=48,
        y_dim=48 * 48,
        z_dim=10,
        u_dim=0,
        aux_dim=4,
        K=10,
        inference_img_enc_dim=64,
        inference_tied_cov=False,
        inference_trainable_cov=True,
        alpha_mlp_units=64,
        alpha_hidden_layers=1,
        alpha_nonlinearity="softplus",
        fixed_H=False,
        integration_step_size=0.05,
        integration_method="rk4",
        emission_init_sigma=0.1,
        emission_scale_function="exp",
        emission_min_scale=0.05,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    model = get_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params}")

    section("4. Forward + backward on a real batch")
    past_target = batch["past_target"].to(device)
    past_times = batch["past_times"].to(device)
    past_mask = batch["past_mask"].to(device)
    print(
        "  inputs:",
        "past_target",
        tuple(past_target.shape),
        "past_times",
        tuple(past_times.shape),
        "past_mask",
        tuple(past_mask.shape),
    )
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    optim.zero_grad()
    out = model(past_target, past_mask, past_times, num_samples=1)
    cond_ll = out["likelihood"]
    reg = out["regularizer"]
    loss = -(cond_ll + 1.0 * reg).mean(0)
    print(
        f"  forward ok: loss={loss.item():.4f}, "
        f"cond_ll mean={cond_ll.mean().item():.4f}, "
        f"reg mean={reg.mean().item():.4f}"
    )
    assert torch.isfinite(loss).all(), "loss is not finite"
    loss.backward()
    grad_norms = [
        p.grad.detach().norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    print(
        f"  backward ok: grad-norm sum={sum(grad_norms):.4f}, "
        f"max={max(grad_norms):.4f}"
    )
    optim.step()

    section("5. forecast() on val batch")
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=4, collate_fn=ds_val.collate_fn, num_workers=0
    )
    vbatch = next(iter(val_loader))
    past_target = vbatch["past_target"].to(device)
    past_times = vbatch["past_times"].to(device).view(-1)
    future_times = vbatch["future_times"].to(device).view(-1)
    past_mask = vbatch["past_mask"].to(device)
    fc = model.forecast(
        past_target,
        past_mask,
        past_times,
        future_times,
        num_samples=4,
    )
    rec = fc["reconstruction"]
    fc_y = fc["forecast"]
    print(f"  reconstruction shape: {tuple(rec.shape)}")
    print(f"  forecast       shape: {tuple(fc_y.shape)}")
    assert rec.shape[-1] == 48 * 48
    assert fc_y.shape[-1] == 48 * 48

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
