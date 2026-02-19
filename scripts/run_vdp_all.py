#!/usr/bin/env python3
"""
VDP Oscillator Full Experiment Orchestrator

Runs run_vdp_experiment.py for all conditions (sig_o^2) and trials.
Supports resumption by skipping already-completed trials.

Usage:
    # Smoke test: 5 trials per condition, 4 conditions
    python scripts/run_vdp_all.py \
        --sig-o-list 0.05 0.10 0.15 0.20 \
        --n-trials 5 \
        --device cuda:0

    # Full experiment: all 20 conditions, 50 trials each
    python scripts/run_vdp_all.py --n-trials 50 --device cuda:0

    # Generate data first, then run
    python scripts/run_vdp_all.py --generate-data --n-trials 50 --device cuda:0
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_trial_complete(output_dir):
    """Check if a trial has already been completed (result JSON exists)."""
    result_file = output_dir / 'vdp_result.json'
    return result_file.exists()


def generate_data(sig_o_sq_list, n_trials, data_dir, base_seed):
    """Generate VDP data for all conditions."""
    sig_o_args = ' '.join(str(s) for s in sig_o_sq_list)
    cmd = (
        f"python {PROJECT_ROOT}/scripts/generate_vdp_data.py "
        f"--output-dir {data_dir} "
        f"--sig-o-list {sig_o_args} "
        f"--n-trials {n_trials} "
        f"--base-seed {base_seed}"
    )
    print(f"Generating data: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Data generation failed:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout)


def run_single_trial(data_path, config_path, output_dir, device, seed):
    """Run a single trial using run_vdp_experiment.py."""
    cmd = (
        f"PYTHONUNBUFFERED=1 python {PROJECT_ROOT}/scripts/run_vdp_experiment.py "
        f"--data {data_path} "
        f"--config {config_path} "
        f"--output {output_dir} "
        f"--device {device} "
        f"--seed {seed}"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1200)
    return result


def main():
    parser = argparse.ArgumentParser(description='Run full VDP experiment')
    parser.add_argument('--sig-o-list', type=float, nargs='+', default=None,
                        help='List of sig_o^2 values (default: 0.01 to 0.20)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials per condition')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device')
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'configs/vdp_config.yaml'),
                        help='Path to config YAML')
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data/vdp'),
                        help='Data directory')
    parser.add_argument('--results-dir', type=str, default=str(PROJECT_ROOT / 'results/vdp'),
                        help='Results directory')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate data before running experiments')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print what would be done')
    args = parser.parse_args()

    # Default: 20 noise conditions
    if args.sig_o_list is None:
        sig_o_sq_list = [round(0.01 * i, 2) for i in range(1, 21)]
    else:
        sig_o_sq_list = args.sig_o_list

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    # Generate data if requested
    if args.generate_data:
        generate_data(sig_o_sq_list, args.n_trials, data_dir, args.base_seed)

    # Count total/completed/remaining
    total = len(sig_o_sq_list) * args.n_trials
    completed = 0
    remaining = []

    for sig_o_sq in sig_o_sq_list:
        for trial_idx in range(1, args.n_trials + 1):
            data_path = data_dir / f"sig_o_{sig_o_sq:.2f}" / f"trial_{trial_idx:03d}.npz"
            output_dir = results_dir / f"sig_o_{sig_o_sq:.2f}" / f"trial_{trial_idx:03d}"
            seed = args.base_seed + int(sig_o_sq * 10000) + trial_idx

            if check_trial_complete(output_dir):
                completed += 1
            else:
                remaining.append((sig_o_sq, trial_idx, data_path, output_dir, seed))

    print(f"=== VDP Full Experiment ===")
    print(f"Conditions: {len(sig_o_sq_list)}, Trials: {args.n_trials}")
    print(f"Total: {total}, Completed: {completed}, Remaining: {len(remaining)}")

    if args.dry_run:
        for sig_o_sq, trial_idx, data_path, output_dir, seed in remaining[:10]:
            print(f"  Would run: sig_o^2={sig_o_sq:.2f}, trial={trial_idx:03d}, seed={seed}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
        return

    if not remaining:
        print("All trials completed!")
        return

    # Run remaining trials
    start_time = time.time()
    successes = 0
    failures = 0

    for i, (sig_o_sq, trial_idx, data_path, output_dir, seed) in enumerate(remaining):
        if not data_path.exists():
            print(f"[{i+1}/{len(remaining)}] SKIP sig_o^2={sig_o_sq:.2f} trial={trial_idx:03d} "
                  f"- data file not found: {data_path}")
            failures += 1
            continue

        print(f"\n[{i+1}/{len(remaining)}] sig_o^2={sig_o_sq:.2f}, trial={trial_idx:03d}, seed={seed}")
        trial_start = time.time()

        try:
            result = run_single_trial(data_path, args.config, output_dir, args.device, seed)
            trial_time = time.time() - trial_start

            if result.returncode == 0:
                # Extract total error from output
                for line in result.stdout.split('\n'):
                    if 'total_eigenvalue_error' in line:
                        print(f"  {line.strip()}")
                        break
                print(f"  Completed in {trial_time:.1f}s")
                successes += 1
            else:
                print(f"  FAILED (exit code {result.returncode})")
                print(f"  stderr: {result.stderr[-500:]}")
                failures += 1
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>1200s)")
            failures += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failures += 1

    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Successes: {successes}, Failures: {failures}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Aggregate results
    if successes > 0:
        aggregate_results(results_dir, sig_o_sq_list, args.n_trials)


def aggregate_results(results_dir, sig_o_sq_list, n_trials):
    """Aggregate results across all conditions and trials."""
    import numpy as np

    results_dir = Path(results_dir)
    summary_dir = results_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    aggregate = {}
    for sig_o_sq in sig_o_sq_list:
        errors = []
        for trial_idx in range(1, n_trials + 1):
            result_file = results_dir / f"sig_o_{sig_o_sq:.2f}" / f"trial_{trial_idx:03d}" / 'vdp_result.json'
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                errors.append(data['total_eigenvalue_error'])

        if errors:
            errors = np.array(errors)
            aggregate[f"sig_o_{sig_o_sq:.2f}"] = {
                'sig_o_sq': sig_o_sq,
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'n_trials': len(errors),
                'min': float(np.min(errors)),
                'max': float(np.max(errors)),
            }
            print(f"sig_o^2={sig_o_sq:.2f}: {np.mean(errors):.4f} +/- {np.std(errors):.4f} "
                  f"({len(errors)}/{n_trials} trials)")

    # Save aggregate
    agg_path = summary_dir / 'aggregate_results.json'
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate saved: {agg_path}")


if __name__ == '__main__':
    main()
