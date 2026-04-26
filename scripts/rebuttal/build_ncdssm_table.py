#!/usr/bin/env python3
"""Build the NCDSSM comparison markdown table for PROJ-REBUTTAL.

Reads `eval_result.json` files produced by
`scripts/rebuttal/evaluate_ncdssm_quadlink.py` (one per seed * regime),
aggregates them, and writes:
  - results/rebuttal/ncdssm_quadlink_raw.csv     per-seed image MSE
  - results/rebuttal/ncdssm_quadlink_summary.csv mean / std / n_seeds
  - results/rebuttal/markdown_tables/ncdssm_clean_corrupted.md

Per the agreed plan (PROJ_REBUTTAL Step 1), the DSE / ELTO-KF / RKN /
LSTM rows in the markdown table are taken from the paper (Table 1) —
NOT recomputed.

Usage:
  python scripts/rebuttal/build_ncdssm_table.py \
    --eval_root results/rebuttal/ncdssm_quadlink \
    --output_root results/rebuttal
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


# Paper Table 1 values (locked per user decision).
PAPER_TABLE_1 = [
    ("DSE", 0.2006, 0.0228, 0.2593, 0.0274),
    ("ELTO-KF", 0.2175, 0.0130, 0.2942, 0.0215),
    ("RKN", 0.2198, 0.0132, 0.2944, 0.0236),
    ("LSTM", 0.2527, 0.0152, 0.3167, 0.0223),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_root", type=str,
                   default="results/rebuttal/ncdssm_quadlink",
                   help="Root containing {regime}/seed{N}/eval/eval_result.json")
    p.add_argument("--output_root", type=str,
                   default="results/rebuttal",
                   help="Root for raw/summary CSVs and markdown_tables/.")
    return p.parse_args()


def collect_eval_results(eval_root: Path) -> list[dict]:
    """Collect production eval results only.

    Glob: eval_root/{regime}/seed*/eval*/eval_result.json
    Skips ad-hoc directories like `_smoke*` so smoke results don't pollute
    the production table.
    """
    rows = []
    candidates: list[Path] = []
    for regime_dir in sorted(eval_root.glob("*")):
        if not regime_dir.is_dir() or regime_dir.name.startswith("_"):
            continue
        for seed_dir in sorted(regime_dir.glob("seed*")):
            for eval_dir in sorted(seed_dir.glob("eval*")):
                jp = eval_dir / "eval_result.json"
                if jp.exists():
                    candidates.append(jp)
    for json_path in candidates:
        try:
            with open(json_path) as fp:
                row = json.load(fp)
            row["_path"] = str(json_path)
            rows.append(row)
        except Exception as e:
            print(f"[warn] failed to read {json_path}: {e}")
    rows.sort(key=lambda r: (r.get("regime", ""), r.get("data_seed", 0)))
    return rows


def write_raw_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "method", "regime", "seed", "sequence_id",
            "context_length", "horizon", "mse",
        ])
        for r in rows:
            w.writerow([
                r.get("method", "NCDSSM-LL"),
                r.get("regime", ""),
                r.get("data_seed", 0),
                f"quad{r.get('data_seed', 0)}",  # sequence id
                r.get("ctx_len", ""),
                r.get("horizon", 1),
                f"{r.get('image_mse', float('nan')):.6f}",
            ])
    print(f"[build] wrote {output}")


def write_summary_csv(rows: list[dict], output: Path) -> dict[str, dict]:
    by_regime: dict[str, list[float]] = {}
    ctx_by_regime: dict[str, int] = {}
    for r in rows:
        regime = r.get("regime", "")
        by_regime.setdefault(regime, []).append(float(r["image_mse"]))
        ctx_by_regime[regime] = int(r.get("ctx_len", -1))

    summary = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "method", "regime", "context_length", "horizon",
            "mean_mse", "std_mse", "num_sequences", "num_seeds",
        ])
        for regime, mses in by_regime.items():
            n = len(mses)
            m = mean(mses)
            s = stdev(mses) if n > 1 else 0.0
            summary[regime] = dict(mean=m, std=s, n=n)
            w.writerow([
                "NCDSSM-LL", regime, ctx_by_regime[regime], 1,
                f"{m:.6f}", f"{s:.6f}", 1, n,
            ])
    print(f"[build] wrote {output} (regimes: {list(summary.keys())})")
    return summary


def build_markdown(summary: dict[str, dict], output: Path) -> None:
    lines = []
    lines.append(
        "# NCDSSM comparison: clean / corrupted prediction MSE (PROJ-REBUTTAL)"
    )
    lines.append("")
    lines.append(
        "**Protocol**: 1-step prediction MSE on test_obs (single 1500-frame "
        "trajectory, warmup ctx_len=30 skipped). Pixel range [0,1], "
        "image-space MSE averaged over predicted timesteps."
    )
    lines.append("")
    lines.append(
        "**DSE / ELTO-KF / RKN / LSTM rows are taken from the paper "
        "(Table 1).** NCDSSM rows are produced by "
        "`scripts/rebuttal/evaluate_ncdssm_quadlink.py` under the same "
        "evaluation protocol."
    )
    lines.append("")
    lines.append(
        "| Method | Clean prediction MSE ↓ | Corrupted prediction MSE ↓ |"
    )
    lines.append("|---|---:|---:|")
    for name, c_mu, c_std, x_mu, x_std in PAPER_TABLE_1:
        lines.append(
            f"| {name} | {c_mu:.4f} ± {c_std:.4f} | "
            f"{x_mu:.4f} ± {x_std:.4f} |"
        )

    def fmt(reg: str) -> str:
        if reg not in summary:
            return "(pending)"
        m = summary[reg]
        if m["n"] > 1:
            return f"{m['mean']:.4f} ± {m['std']:.4f}"
        return f"{m['mean']:.4f} (n={m['n']})"

    lines.append(
        f"| NCDSSM-LL | {fmt('clean')} | {fmt('corrupted')} |"
    )
    lines.append("")
    lines.append(
        "Notes: NCDSSM-LL uses amortized auxiliary inference and "
        "filtering-based dynamic-state inference. We adapt its image "
        "encoder/emission to 48×48 grayscale and replace the Bernoulli "
        "emission with a Gaussian emission for continuous pixel values."
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"[build] wrote {output}")


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root)
    output_root = Path(args.output_root)
    if not eval_root.exists():
        print(f"[build] eval_root does not exist: {eval_root}")
        # Still produce a stub markdown table with paper rows + NCDSSM pending.
        rows = []
    else:
        rows = collect_eval_results(eval_root)
    print(f"[build] found {len(rows)} eval results")

    raw_csv = output_root / "ncdssm_quadlink_raw.csv"
    summary_csv = output_root / "ncdssm_quadlink_summary.csv"
    md_path = output_root / "markdown_tables" / "ncdssm_clean_corrupted.md"

    write_raw_csv(rows, raw_csv)
    summary = write_summary_csv(rows, summary_csv)
    build_markdown(summary, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
