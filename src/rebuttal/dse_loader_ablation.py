"""Variant-aware DSE loader for the ablation experiments (Step 6).

Wraps `src.rebuttal.dse_loader` so the realization can be swapped to
`StochasticRealizationPCA` for the DSE_no_cca variant. Other variants reuse
the existing CCA realization unchanged.

Usage:
    from src.rebuttal.dse_loader_ablation import load_dse_variant

    dse = load_dse_variant(model_path, device, variant="no_cca")

variant values:
    "full"             — Full DSE (CCA realization)
    "joint_training"   — DSE_joint_training (CCA realization)
    "no_cca"           — DSE_no_cca (PCA realization)
    "no_closed_form"   — DSE_no_closed_form (CCA realization)

For "no_cca", `realization` is a `StochasticRealizationPCA` instance.
estimate_noise_QR / estimate_states therefore use PCA-based state
construction (matching how the model was trained).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.rebuttal.dse_loader import (  # noqa: F401
    DSEModel,
    build_decoder,
    build_df_obs,
    build_df_state,
    build_encoder,
    build_realization as build_realization_cca,
    estimate_noise_QR,
    load_checkpoint,
    load_test_data,
)
from src.rebuttal.realization_pca import StochasticRealizationPCA


def build_realization_pca(
    ckpt: Dict,
    config: Dict,
    encoder: nn.Module,
    device: torch.device,
) -> StochasticRealizationPCA:
    """Variant of dse_loader.build_realization that constructs a PCA realization."""
    real_cfg = config.get("ssm", {}).get("realization", {})
    fm_cfg = real_cfg.get("feature_mapping", {})

    realization = StochasticRealizationPCA(
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

    # Restore component_transforms weights if present in checkpoint
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


def load_dse_variant(
    model_path: str,
    device: torch.device,
    variant: str = "full",
) -> DSEModel:
    """Load a DSE checkpoint, optionally swapping realization to PCA.

    Args:
        model_path: path to final_model.pth
        device: target device
        variant: one of {"full", "joint_training", "no_cca", "no_closed_form"}

    Returns:
        DSEModel bundle.
    """
    if variant not in {"full", "joint_training", "no_cca", "no_closed_form"}:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from "
            "full / joint_training / no_cca / no_closed_form."
        )

    ckpt, config = load_checkpoint(model_path)
    encoder = build_encoder(ckpt, config, device)
    decoder = build_decoder(ckpt, config, device)
    df_state = build_df_state(ckpt, config, device)
    df_obs = build_df_obs(ckpt, config, df_state, device)

    if variant == "no_cca":
        realization = build_realization_pca(ckpt, config, encoder, device)
    else:
        realization = build_realization_cca(ckpt, config, encoder, device)

    return DSEModel(encoder, decoder, df_state, df_obs, realization, config)
