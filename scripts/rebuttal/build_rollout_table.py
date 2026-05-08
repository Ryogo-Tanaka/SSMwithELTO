#!/usr/bin/env python3
"""Build summary CSV + markdown table + figure from raw rollout CSV.

PROJ-REBUTTAL Step 4, v3. Reads results/rebuttal/multistep_rollout_raw.csv
(produced by run_multistep_rollout.py) and emits:
  - results/rebuttal/multistep_rollout_summary.csv
  - results/rebuttal/markdown_tables/multistep_rollout.md
  - results/rebuttal/figures/multistep_rollout_mse.png  (optional)

Usage:
  python scripts/rebuttal/build_rollout_table.py \
    --raw results/rebuttal/multistep_rollout_raw.csv \
    --summary results/rebuttal/multistep_rollout_summary.csv \
    --markdown results/rebuttal/markdown_tables/multistep_rollout.md \
    --figure results/rebuttal/figures/multistep_rollout_mse.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=str, required=True)
    p.add_argument("--summary", type=str, required=True)
    p.add_argument("--markdown", type=str, required=True)
    p.add_argument("--figure", type=str, default="",
                   help="If set, also produce a PNG figure")
    return p.parse_args()


def load_raw(path: Path) -> List[Dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({
                "method": r["method"],
                "regime": r["regime"],
                "seed": int(r["seed"]),
                "sequence_id": int(r["sequence_id"]),
                "context_length": int(r["context_length"]),
                "horizon": int(r["horizon"]),
                "mse": float(r["mse"]),
            })
    return rows


def aggregate(
    rows: List[Dict],
) -> Dict[Tuple[str, str, int, int], Tuple[float, float, int]]:
    """Group by (method, regime, ctx_len, horizon) and compute mean/std/n_seeds."""
    grouped: Dict[Tuple[str, str, int, int], List[float]] = defaultdict(list)
    for r in rows:
        key = (r["method"], r["regime"], r["context_length"], r["horizon"])
        grouped[key].append(r["mse"])
    summary: Dict[Tuple[str, str, int, int], Tuple[float, float, int]] = {}
    for key, vals in grouped.items():
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        summary[key] = (mean, std, len(arr))
    return summary


def write_summary_csv(
    summary: Dict[Tuple[str, str, int, int], Tuple[float, float, int]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "regime", "context_length", "horizon",
                         "mean_mse", "std_mse", "num_seeds"])
        for key in sorted(summary.keys()):
            method, regime, C, h = key
            mean, std, n = summary[key]
            writer.writerow([method, regime, C, h,
                             f"{mean:.6f}", f"{std:.6f}", n])


def fmt_cell(mean: float, std: float, n: int) -> str:
    if n > 1:
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.4f}"


def write_markdown(
    summary: Dict[Tuple[str, str, int, int], Tuple[float, float, int]],
    path: Path,
    horizons: List[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({key[0] for key in summary})
    regimes = ["clean", "corrupted"]
    ctx_lens = sorted({key[2] for key in summary})

    sections = []
    sections.append("# Multi-step free rollout — image-space MSE\n")
    sections.append(
        "Protocol: starting from frame s=0 of each test trajectory, give the model "
        "y[0:C] as observation context, then perform free rollout for "
        f"H_max={max(horizons)} steps (no further observation update). "
        "Standard \"next-step\" convention (h=1 predicts y_C). Pixel range [0,1]. "
        "C=5 matches the data convention: corrupted regime has clean ground-truth "
        "at frames t=0..4 and noise added from t=5 onward, so the C=5 context is "
        "fully clean even under corruption.\n"
    )
    for C in ctx_lens:
        for regime in regimes:
            label = f"## C={C}, regime={regime}\n"
            header = "| Method | n | " + " | ".join(f"H={h}" for h in horizons) + " |"
            sep = "|---|---:|" + "|".join("---:" for _ in horizons) + "|"
            lines = [label, "", header, sep]
            for m in methods:
                cells = []
                n_for_method = 0
                for h in horizons:
                    key = (m, regime, C, h)
                    if key in summary:
                        mean, std, n = summary[key]
                        n_for_method = n
                        cells.append(fmt_cell(mean, std, n))
                    else:
                        cells.append("-")
                lines.append(f"| {m} | {n_for_method} | " + " | ".join(cells) + " |")
            sections.append("\n".join(lines) + "\n")

    with open(path, "w") as f:
        f.write("\n".join(sections))


def write_figure(
    summary: Dict[Tuple[str, str, int, int], Tuple[float, float, int]],
    path: Path,
    horizons: List[int],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[build] matplotlib unavailable, skipping figure")
        return

    methods = sorted({key[0] for key in summary})
    regimes = ["clean", "corrupted"]
    ctx_lens = sorted({key[2] for key in summary})

    nrows = len(ctx_lens)
    ncols = len(regimes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    for i, C in enumerate(ctx_lens):
        for j, regime in enumerate(regimes):
            ax = axes[i][j]
            for m in methods:
                xs, ys, errs = [], [], []
                for h in horizons:
                    key = (m, regime, C, h)
                    if key not in summary:
                        continue
                    mean, std, n = summary[key]
                    xs.append(h)
                    ys.append(mean)
                    errs.append(std if n > 1 else 0.0)
                if xs:
                    ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3,
                                label=m)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("horizon h")
            ax.set_ylabel("image MSE")
            ax.set_title(f"C={C}, {regime}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"[build] wrote figure: {path}")


def main() -> int:
    args = parse_args()
    raw_path = Path(args.raw)
    summary_path = Path(args.summary)
    markdown_path = Path(args.markdown)

    rows = load_raw(raw_path)
    print(f"[build] loaded {len(rows)} raw rows from {raw_path}")
    if not rows:
        print("[build] no rows to process")
        return 1

    horizons = sorted({r["horizon"] for r in rows})
    summary = aggregate(rows)
    write_summary_csv(summary, summary_path)
    print(f"[build] wrote summary: {summary_path}")
    write_markdown(summary, markdown_path, horizons)
    print(f"[build] wrote markdown: {markdown_path}")

    if args.figure:
        write_figure(summary, Path(args.figure), horizons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
