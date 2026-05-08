#!/usr/bin/env python3
"""Aggregate DSE ablation rollout JSONs into a markdown table.

Reads per-variant rollout eval_result.json files (produced by
evaluate_multistep_rollout_dse_ablation.py or evaluate_multistep_rollout_dse.py
for Full DSE) and emits:
  - results/rebuttal/dse_ablation_rollout_raw.csv
  - results/rebuttal/dse_ablation_rollout_summary.csv
  - results/rebuttal/markdown_tables/dse_ablation_rollout.md

Usage:
  python scripts/rebuttal/build_dse_ablation_rollout_table.py \
    --full-dse-eval results/rebuttal/multistep_rollout/dse/clean/seed{1..5}/ctx5/eval_result.json \
    --joint-training-eval results/rebuttal/dse_ablation/joint_training/seed1/rollout/ctx5/eval_result.json \
    --no-cca-eval results/rebuttal/dse_ablation/no_cca/seed1/rollout/ctx5/eval_result.json \
    --no-closed-form-eval results/rebuttal/dse_ablation/no_closed_form/seed1/rollout/ctx5/eval_result.json \
    --output-raw results/rebuttal/dse_ablation_rollout_raw.csv \
    --output-summary results/rebuttal/dse_ablation_rollout_summary.csv \
    --output-markdown results/rebuttal/markdown_tables/dse_ablation_rollout.md

`--full-dse-eval` accepts multiple JSON paths (e.g. 5 seeds), each producing
mean/std across seeds. The other variants accept multiple paths too but
typically have n=1.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


VARIANT_LABELS = {
    "Full DSE": "Full DSE",
    "DSE_no_cca": "w/o CCA-based state construction",
    "DSE_no_closed_form": "w/o closed-form operator estimation",
    "DSE_joint_training": "joint training instead of staged training",
}
VARIANT_ORDER = ["Full DSE", "DSE_no_cca", "DSE_no_closed_form", "DSE_joint_training"]


def load_records(json_path: Path, default_variant: str) -> List[Dict]:
    """Read records from a per-variant rollout JSON."""
    with open(json_path) as f:
        data = json.load(f)
    records = []
    variant_in_data = data.get("variant", None)
    for r in data.get("records", []):
        records.append({
            "variant": r.get("variant", variant_in_data) or default_variant,
            "regime": r.get("regime"),
            "seed": int(r.get("seed", 0)),
            "sequence_id": int(r.get("sequence_id", 0)),
            "context_length": int(r.get("context_length", 0)),
            "horizon": int(r.get("horizon", 0)),
            "mse": float(r.get("mse", float("nan"))),
        })
    return records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-dse-eval", nargs="*", default=[],
                   help="One or more eval_result.json paths from Full DSE rollout (multiple seeds OK).")
    p.add_argument("--joint-training-eval", nargs="*", default=[])
    p.add_argument("--no-cca-eval", nargs="*", default=[])
    p.add_argument("--no-closed-form-eval", nargs="*", default=[])
    p.add_argument("--output-raw", required=True)
    p.add_argument("--output-summary", required=True)
    p.add_argument("--output-markdown", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Map variant name -> list of JSON paths
    variant_paths = {
        "Full DSE": list(args.full_dse_eval),
        "DSE_joint_training": list(args.joint_training_eval),
        "DSE_no_cca": list(args.no_cca_eval),
        "DSE_no_closed_form": list(args.no_closed_form_eval),
    }

    raw_rows: List[Dict] = []
    for variant, paths in variant_paths.items():
        for path_str in paths:
            recs = load_records(Path(path_str), default_variant=variant)
            for r in recs:
                r["variant"] = variant  # force canonical name
                raw_rows.append(r)

    # Write raw CSV
    Path(args.output_raw).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_raw, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "regime", "seed", "sequence_id",
                        "context_length", "horizon", "mse"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    # Aggregate per (variant, regime, ctx_len, horizon)
    grouped: Dict[Tuple[str, str, int, int], List[float]] = defaultdict(list)
    for r in raw_rows:
        key = (r["variant"], r["regime"], r["context_length"], r["horizon"])
        grouped[key].append(r["mse"])

    summary_rows = []
    for key, vals in sorted(grouped.items()):
        variant, regime, C, h = key
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5 if n > 1 else float("nan")
        summary_rows.append({
            "variant": variant,
            "regime": regime,
            "context_length": C,
            "horizon": h,
            "mean_mse": mean,
            "std_mse": std,
            "num_seeds": n,
        })

    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_summary, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "regime", "context_length", "horizon",
                        "mean_mse", "std_mse", "num_seeds"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                **row,
                "mean_mse": f"{row['mean_mse']:.6f}",
                "std_mse": f"{row['std_mse']:.6f}" if not math.isnan(row['std_mse']) else "",
            })

    # Markdown table: rows = variants, columns = horizons (per (regime, C))
    horizons = sorted({row["horizon"] for row in summary_rows})
    regimes = sorted({row["regime"] for row in summary_rows})
    ctx_lens = sorted({row["context_length"] for row in summary_rows})

    Path(args.output_markdown).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_markdown, "w") as f:
        f.write("# DSE ablation x multi-step rollout — image-space MSE\n\n")
        f.write(
            "Protocol: starting from y[0:C] context, free rollout for H_max steps. "
            "Pixel range [0,1]. Standard \"next-step\" convention (h=1 predicts y_C). "
            "Each cell shows mean ± std across seeds (n shown in 'n' column); "
            "n=1 cells show the single-seed value.\n\n"
        )
        for C in ctx_lens:
            for regime in regimes:
                f.write(f"## C={C}, regime={regime}\n\n")
                f.write("| Variant | n | " + " | ".join(f"H={h}" for h in horizons) + " |\n")
                f.write("|---|---:|" + "|".join("---:" for _ in horizons) + "|\n")
                for variant in VARIANT_ORDER:
                    cells: List[str] = []
                    n_for_variant = 0
                    for h in horizons:
                        match = next(
                            (r for r in summary_rows
                             if r["variant"] == variant
                             and r["regime"] == regime
                             and r["context_length"] == C
                             and r["horizon"] == h),
                            None,
                        )
                        if match is None:
                            cells.append("-")
                        else:
                            n_for_variant = match["num_seeds"]
                            if match["num_seeds"] > 1 and not math.isnan(match["std_mse"]):
                                cells.append(
                                    f"{match['mean_mse']:.4f} ± {match['std_mse']:.4f}"
                                )
                            else:
                                cells.append(f"{match['mean_mse']:.4f}")
                    label = VARIANT_LABELS.get(variant, variant)
                    f.write(f"| {label} | {n_for_variant} | " + " | ".join(cells) + " |\n")
                f.write("\n")

    print(f"Wrote raw CSV       : {args.output_raw} ({len(raw_rows)} rows)")
    print(f"Wrote summary CSV   : {args.output_summary} ({len(summary_rows)} rows)")
    print(f"Wrote markdown table: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
