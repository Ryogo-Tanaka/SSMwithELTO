#!/usr/bin/env python3
"""
Mode Decomposition Results Evaluator

Aggregates and summarizes results from VDP and SL mode decomposition experiments.
Reads individual trial result JSONs and computes statistics.

Usage:
    # VDP results
    python scripts/evaluate_mode_decomposition.py \
        --results-dir results/vdp \
        --experiment vdp

    # SL results
    python scripts/evaluate_mode_decomposition.py \
        --results-dir results/sl \
        --experiment sl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Paper reference values (Table 2 / Table 3)
PAPER_VDP = {
    0.05: (0.016, 0.025),
    0.10: (0.014, 0.026),
    0.15: (0.023, 0.028),
    0.20: (0.020, 0.035),
}

PAPER_SL = {
    0.01: (0.035, 0.025),
    0.05: (0.043, 0.036),
    0.09: (0.042, 0.033),
}


def collect_results(results_dir, condition_prefix, n_trials=50):
    """Collect results for all conditions from a results directory.

    Args:
        results_dir: Root results directory
        condition_prefix: 'sig_o' for VDP, 'var_p' for SL
        n_trials: Max expected trials

    Returns:
        dict: condition_value -> list of total_eigenvalue_error values
    """
    results_dir = Path(results_dir)
    all_results = {}

    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        if not condition_dir.name.startswith(condition_prefix):
            continue

        # Parse condition value from directory name
        try:
            val_str = condition_dir.name.split('_')[-1]
            condition_val = float(val_str)
        except ValueError:
            continue

        errors = []
        for trial_dir in sorted(condition_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            result_file = trial_dir / 'vdp_result.json'
            if not result_file.exists():
                result_file = trial_dir / 'sl_result.json'
            if not result_file.exists():
                continue

            with open(result_file) as f:
                data = json.load(f)
            errors.append(data['total_eigenvalue_error'])

        if errors:
            all_results[condition_val] = errors

    return all_results


def print_summary_table(all_results, paper_ref, experiment_name):
    """Print a formatted summary table comparing results to paper."""
    print(f"\n{'='*70}")
    print(f"  {experiment_name} Mode Decomposition Results Summary")
    print(f"{'='*70}")

    if experiment_name.upper() == 'VDP':
        cond_label = 'sig_o^2'
    else:
        cond_label = 'var_p'

    print(f"\n{'Condition':<12} {'Our mean':>10} {'Our std':>10} "
          f"{'Paper mean':>12} {'Paper std':>12} {'N':>5}")
    print('-' * 65)

    all_means = []
    all_stds = []

    for cond_val in sorted(all_results.keys()):
        errors = np.array(all_results[cond_val])
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        n = len(errors)
        all_means.append(mean_err)
        all_stds.append(std_err)

        paper_mean = paper_ref.get(cond_val, (None, None))[0] if paper_ref else None
        paper_std = paper_ref.get(cond_val, (None, None))[1] if paper_ref else None

        paper_mean_str = f"{paper_mean:.3f}" if paper_mean is not None else "—"
        paper_std_str = f"{paper_std:.3f}" if paper_std is not None else "—"

        print(f"{cond_label}={cond_val:.2f}  {mean_err:10.4f} {std_err:10.4f} "
              f"{paper_mean_str:>12} {paper_std_str:>12} {n:5d}")

    if all_means:
        avg_mean = np.mean(all_means)
        avg_std = np.mean(all_stds)
        print('-' * 65)
        print(f"{'Average':<12} {avg_mean:10.4f} {avg_std:10.4f}")

    print(f"{'='*70}\n")


def save_summary(all_results, output_path, experiment_name):
    """Save summary to JSON."""
    summary = {
        'experiment': experiment_name,
        'conditions': {}
    }

    for cond_val in sorted(all_results.keys()):
        errors = np.array(all_results[cond_val])
        summary['conditions'][f"{cond_val:.2f}"] = {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'n_trials': len(errors),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'all_errors': errors.tolist(),
        }

    # Overall average
    all_errors = []
    for errors in all_results.values():
        all_errors.extend(errors)
    if all_errors:
        summary['overall'] = {
            'mean': float(np.mean(all_errors)),
            'std': float(np.std(all_errors)),
            'n_total': len(all_errors),
        }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate mode decomposition results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Results directory')
    parser.add_argument('--experiment', type=str, choices=['vdp', 'sl'], required=True,
                        help='Experiment type')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: results_dir/summary/evaluation.json)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    if args.experiment == 'vdp':
        condition_prefix = 'sig_o'
        paper_ref = PAPER_VDP
    else:
        condition_prefix = 'var_p'
        paper_ref = PAPER_SL

    all_results = collect_results(results_dir, condition_prefix)

    if not all_results:
        print("No results found!")
        sys.exit(1)

    print_summary_table(all_results, paper_ref, args.experiment.upper())

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / 'summary' / 'evaluation.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_summary(all_results, output_path, args.experiment)


if __name__ == '__main__':
    main()
