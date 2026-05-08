#!/usr/bin/env python3
"""Aggregate DSE ablation results into raw CSV, summary CSV, and markdown table.

Usage:
  python scripts/rebuttal/build_dse_ablation_table.py \
    --paper-full-dse-mean 0.2006 \
    --paper-full-dse-std 0.0228 \
    --paper-num-seeds 5 \
    --joint-training-eval results/rebuttal/dse_ablation/joint_training/seed1/eval/kalman_eval_results.json \
    --output-raw results/rebuttal/dse_ablation_raw.csv \
    --output-summary results/rebuttal/dse_ablation_summary.csv \
    --output-markdown results/rebuttal/markdown_tables/dse_ablation.md

Future Step 6 will add --no-cca-eval and --no-closed-form-eval (each takes a list of
eval JSON paths). For Step 5 only Full DSE (paper value) and DSE_joint_training are
populated; the remaining variants render as "TBD" in the markdown.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


VARIANT_ROW_LABELS = {
    "Full DSE": "Full DSE",
    "DSE_no_cca": "w/o CCA-based state construction",
    "DSE_no_closed_form": "w/o closed-form operator estimation",
    "DSE_joint_training": "joint training instead of staged training",
}

VARIANT_NOTES = {
    "Full DSE": "proposed model (paper value)",
    "DSE_no_cca": "replace CCA state with PCA on past block features",
    "DSE_no_closed_form": "replace closed-form V_A/U_A/V_B/U_B with gradient-learned linear (cross-fitting also dropped)",
    "DSE_joint_training": "no staged schedule (`update_strategy: joint_all`, `phase1_warmup_epochs: 0`)",
}


def load_step1_mse(eval_json_path: str) -> float:
    with open(eval_json_path, "r") as f:
        data = json.load(f)
    if "step1_direct" not in data:
        raise KeyError(
            f"{eval_json_path} has no 'step1_direct' entry — "
            f"was --skip-step2 --skip-step3 used during evaluation?"
        )
    return float(data["step1_direct"]["mse"])


def aggregate_seed_evals(
    eval_paths: List[str],
    metric_name: str,
    sample_id_format: str = "seed{idx}",
) -> List[Dict]:
    rows = []
    for idx, path in enumerate(eval_paths, start=1):
        mse = load_step1_mse(path)
        rows.append({
            "sequence_id": sample_id_format.format(idx=idx),
            "mse": mse,
            "source_path": path,
        })
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper-full-dse-mean", type=float, required=True,
                   help="Full DSE mean MSE from the paper (clean regime).")
    p.add_argument("--paper-full-dse-std", type=float, required=True,
                   help="Full DSE std MSE from the paper.")
    p.add_argument("--paper-num-seeds", type=int, required=True,
                   help="Number of seeds reported in the paper for Full DSE.")
    p.add_argument("--regime", type=str, default="clean",
                   help="Regime label (clean | corrupted).")
    p.add_argument("--joint-training-eval", type=str, default=None,
                   help="Path to DSE_joint_training eval_kalman_image_mse JSON (Step 1 only).")
    p.add_argument("--joint-training-seed", type=int, default=42,
                   help="Random seed used for the DSE_joint_training run (default 42).")
    p.add_argument("--no-cca-eval", type=str, nargs="*", default=None,
                   help="(Step 6+) eval JSON path(s) for DSE_no_cca seeds.")
    p.add_argument("--no-closed-form-eval", type=str, nargs="*", default=None,
                   help="(Step 6+) eval JSON path(s) for DSE_no_closed_form seeds.")
    p.add_argument("--output-raw", type=str, required=True)
    p.add_argument("--output-summary", type=str, required=True)
    p.add_argument("--output-markdown", type=str, required=True)
    return p.parse_args()


def fmt_mean_std(mean: Optional[float], std: Optional[float], n: int) -> str:
    if mean is None:
        return "TBD"
    if n == 1 or std is None or math.isnan(std):
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} ± {std:.4f}"


def write_raw_csv(path: str, rows: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "regime", "seed", "sequence_id", "metric_name", "horizon", "mse"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_summary_csv(path: str, rows: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "regime", "metric_name", "horizon",
                  "mean_mse", "std_mse", "num_sequences", "num_seeds"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_markdown(path: str, regime: str, summary_by_variant: Dict[str, Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# DSE Core Component Ablation ({regime} regime)",
        "",
        "| Variant | Prediction MSE ↓ | Notes |",
        "|---|---:|---|",
    ]
    order = ["Full DSE", "DSE_no_cca", "DSE_no_closed_form", "DSE_joint_training"]
    for variant in order:
        info = summary_by_variant.get(variant)
        if info is None:
            cell = "TBD"
        else:
            cell = fmt_mean_std(info["mean"], info["std"], info["num_seeds"])
        label = VARIANT_ROW_LABELS[variant]
        notes = VARIANT_NOTES[variant]
        lines.append(f"| {label} | {cell} | {notes} |")
    lines.append("")
    lines.append(
        "Metric: 1-step prediction MSE on test_obs of `quad1_n.npz`, computed via "
        "`evaluate_kalman_image_mse.py` Step 1 (Direct prediction) for trained "
        "checkpoints; Full DSE row is the paper value (Table 1)."
    )
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    raw_rows: List[Dict] = []
    summary_by_variant: Dict[str, Dict] = {}

    # ---- Full DSE (paper value, treated as one aggregated row in raw CSV) ----
    raw_rows.append({
        "variant": "Full DSE",
        "regime": args.regime,
        "seed": "paper_aggregated",
        "sequence_id": f"paper_n{args.paper_num_seeds}",
        "metric_name": "image_mse_paper",
        "horizon": 1,
        "mse": args.paper_full_dse_mean,
    })
    summary_by_variant["Full DSE"] = {
        "mean": args.paper_full_dse_mean,
        "std": args.paper_full_dse_std,
        "num_sequences": args.paper_num_seeds,
        "num_seeds": args.paper_num_seeds,
        "metric_name": "image_mse_paper",
    }

    # ---- DSE_joint_training (n=1 measured) ----
    if args.joint_training_eval:
        mse = load_step1_mse(args.joint_training_eval)
        raw_rows.append({
            "variant": "DSE_joint_training",
            "regime": args.regime,
            "seed": args.joint_training_seed,
            "sequence_id": f"seed{args.joint_training_seed}",
            "metric_name": "image_mse_step1_direct",
            "horizon": 1,
            "mse": mse,
        })
        summary_by_variant["DSE_joint_training"] = {
            "mean": mse,
            "std": float("nan"),
            "num_sequences": 1,
            "num_seeds": 1,
            "metric_name": "image_mse_step1_direct",
        }

    # ---- DSE_no_cca (Step 6+) ----
    if args.no_cca_eval:
        seed_rows = aggregate_seed_evals(
            args.no_cca_eval, "image_mse_step1_direct")
        for r in seed_rows:
            raw_rows.append({
                "variant": "DSE_no_cca",
                "regime": args.regime,
                "seed": r["sequence_id"],
                "sequence_id": r["sequence_id"],
                "metric_name": "image_mse_step1_direct",
                "horizon": 1,
                "mse": r["mse"],
            })
        mses = [r["mse"] for r in seed_rows]
        n = len(mses)
        mean = sum(mses) / n
        std = (sum((m - mean) ** 2 for m in mses) / max(n - 1, 1)) ** 0.5 if n > 1 else float("nan")
        summary_by_variant["DSE_no_cca"] = {
            "mean": mean, "std": std, "num_sequences": n, "num_seeds": n,
            "metric_name": "image_mse_step1_direct",
        }

    # ---- DSE_no_closed_form (Step 6+) ----
    if args.no_closed_form_eval:
        seed_rows = aggregate_seed_evals(
            args.no_closed_form_eval, "image_mse_step1_direct")
        for r in seed_rows:
            raw_rows.append({
                "variant": "DSE_no_closed_form",
                "regime": args.regime,
                "seed": r["sequence_id"],
                "sequence_id": r["sequence_id"],
                "metric_name": "image_mse_step1_direct",
                "horizon": 1,
                "mse": r["mse"],
            })
        mses = [r["mse"] for r in seed_rows]
        n = len(mses)
        mean = sum(mses) / n
        std = (sum((m - mean) ** 2 for m in mses) / max(n - 1, 1)) ** 0.5 if n > 1 else float("nan")
        summary_by_variant["DSE_no_closed_form"] = {
            "mean": mean, "std": std, "num_sequences": n, "num_seeds": n,
            "metric_name": "image_mse_step1_direct",
        }

    write_raw_csv(args.output_raw, raw_rows)

    summary_rows = []
    for variant, info in summary_by_variant.items():
        summary_rows.append({
            "variant": variant,
            "regime": args.regime,
            "metric_name": info["metric_name"],
            "horizon": 1,
            "mean_mse": info["mean"],
            "std_mse": info["std"],
            "num_sequences": info["num_sequences"],
            "num_seeds": info["num_seeds"],
        })
    write_summary_csv(args.output_summary, summary_rows)

    write_markdown(args.output_markdown, args.regime, summary_by_variant)

    print(f"Wrote raw CSV       : {args.output_raw} ({len(raw_rows)} rows)")
    print(f"Wrote summary CSV   : {args.output_summary} ({len(summary_rows)} rows)")
    print(f"Wrote markdown table: {args.output_markdown}")
    print()
    for variant in ["Full DSE", "DSE_no_cca", "DSE_no_closed_form", "DSE_joint_training"]:
        info = summary_by_variant.get(variant)
        if info is None:
            print(f"  {variant:<22} TBD")
        else:
            cell = fmt_mean_std(info["mean"], info["std"], info["num_seeds"])
            print(f"  {variant:<22} {cell}")


if __name__ == "__main__":
    main()
