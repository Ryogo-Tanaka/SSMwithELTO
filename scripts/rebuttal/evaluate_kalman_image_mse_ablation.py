#!/usr/bin/env python3
"""1-step prediction MSE evaluation wrapper for DSE ablation variants.

For DSE_no_cca, monkey-patches scripts.evaluate_kalman_image_mse.build_realization
to construct a StochasticRealizationPCA instance instead of the default CCA.
For other variants, calls the original evaluation script unchanged.

Usage (inside CLAUDE):
  python scripts/rebuttal/evaluate_kalman_image_mse_ablation.py \
    --variant no_cca \
    --model results/rebuttal/dse_ablation/no_cca/seed1/models/final_model.pth \
    --data data/rkn_quad/quad1_n.npz \
    --output results/rebuttal/dse_ablation/no_cca/seed1/eval \
    --device cuda \
    --skip-step2 --skip-step3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    # Pull out --variant before evaluate_kalman_image_mse parses its own args
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True,
                        choices=("full", "joint_training", "no_cca", "no_closed_form"))
    args, remaining = parser.parse_known_args()
    variant = args.variant

    # Restore argv so evaluate_kalman_image_mse can re-parse with its own argparse
    sys.argv = [sys.argv[0]] + remaining

    import scripts.evaluate_kalman_image_mse as eval_module

    if variant == "no_cca":
        from src.rebuttal.realization_pca import StochasticRealizationPCA
        import torch.nn as nn
        import torch

        original_build_realization = eval_module.build_realization

        def patched_build_realization(ckpt, config, encoder: nn.Module,
                                      device: torch.device):
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
            print(
                f"[ablation-eval] (variant=no_cca) realization swapped to "
                f"StochasticRealizationPCA"
            )
            return realization

        eval_module.build_realization = patched_build_realization

    # Run the (possibly patched) evaluator
    if hasattr(eval_module, "main"):
        return eval_module.main()
    # Fallback: run the module's __main__ block via runpy
    import runpy
    runpy.run_module("scripts.evaluate_kalman_image_mse", run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
