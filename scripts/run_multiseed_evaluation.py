#!/usr/bin/env python3
"""
Multi-Seed Evaluation Pipeline

Orchestrates training and Kalman filtering evaluation across multiple data seeds
for all conditions (clean/noisy x 1.5k/15k).

Calls existing scripts via subprocess:
  - scripts/run_quadlink_experiment.py (training)
  - scripts/evaluate_kalman_image_mse.py (Kalman evaluation)

Usage:
    # Clean-1.5k (default condition)
    PYTHONUNBUFFERED=1 python scripts/run_multiseed_evaluation.py \
        --condition clean-1.5k \
        --base-output results/clean_1500 \
        --config configs/quad_image_reconstruction_config.yaml \
        --device cuda --seeds 1,2,3,4,5

    # Noisy-1.5k
    PYTHONUNBUFFERED=1 python scripts/run_multiseed_evaluation.py \
        --condition noisy-1.5k \
        --base-output results/noisy_1500 \
        --config configs/quad_image_reconstruction_config.yaml \
        --device cuda --seeds 1,2,3,4,5

    # Clean-15k (single data file, all seeds share quad_long_n.npz)
    PYTHONUNBUFFERED=1 python scripts/run_multiseed_evaluation.py \
        --condition clean-15k \
        --base-output results/clean_15000 \
        --config configs/quad_image_reconstruction_config.yaml \
        --device cuda --seeds 1,2,3,4,5

    # Aggregate only (skip training and evaluation)
    PYTHONUNBUFFERED=1 python scripts/run_multiseed_evaluation.py \
        --condition noisy-1.5k \
        --base-output results/noisy_1500 \
        --aggregate-only

    # Retry specific seeds
    PYTHONUNBUFFERED=1 python scripts/run_multiseed_evaluation.py \
        --condition noisy-1.5k \
        --base-output results/noisy_1500 \
        --config configs/quad_image_reconstruction_config.yaml \
        --device cuda --seeds 3,4
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Condition-specific configuration (Paper Table 1)
CONDITION_CONFIG = {
    "clean-1.5k": {
        "data_suffix": "_n.npz",
        "data_pattern": "per_seed",       # quad{seed}_n.npz
        "paper_mean": 0.2006,
        "paper_std": 0.0228,
    },
    "noisy-1.5k": {
        "data_suffix": "_y.npz",
        "data_pattern": "per_seed",       # quad{seed}_y.npz
        "paper_mean": 0.2593,
        "paper_std": 0.0274,
    },
    "clean-15k": {
        "data_suffix": "_n.npz",
        "data_pattern": "single_file",    # quad_long_n.npz
        "paper_mean": 0.2615,
        "paper_std": 0.0211,
    },
    "noisy-15k": {
        "data_suffix": "_y.npz",
        "data_pattern": "single_file",    # quad_long_y.npz
        "paper_mean": 0.2683,
        "paper_std": 0.0254,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="5-Seed Evaluation Pipeline"
    )
    parser.add_argument(
        "--base-output", type=str,
        default=str(PROJECT_ROOT / "results" / "clean_1500"),
        help="Base output directory for all 5 seeds"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "quad_image_reconstruction_config.yaml"),
        help="Training config file (must be identical for all seeds)"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "rkn_quad"),
        help="Directory containing data files"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="CUDA device (cuda, cuda:0, cuda:1)"
    )
    parser.add_argument(
        "--seeds", type=str, default="1,2,3,4,5",
        help="Comma-separated seed indices to process"
    )
    parser.add_argument(
        "--condition", type=str, default="clean-1.5k",
        choices=list(CONDITION_CONFIG.keys()),
        help="Experiment condition (determines data files and paper targets)"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training, only run evaluation on existing models"
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true",
        help="Skip Kalman evaluation, only run training"
    )
    parser.add_argument(
        "--reuse-seed1", type=str, default=None,
        help="Path to existing seed1 training results (e.g. results/paper_reconstruction_v4)"
    )
    parser.add_argument(
        "--reuse-seed1-eval", type=str, default=None,
        help="Path to existing seed1 Kalman eval results (e.g. results/kalman_eval_v4)"
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Only aggregate existing results, skip training and evaluation"
    )
    parser.add_argument(
        "--gamma-q", type=float, default=1e-6,
        help="Kalman Q regularization (passed to evaluate_kalman_image_mse.py)"
    )
    parser.add_argument(
        "--gamma-r", type=float, default=1e-6,
        help="Kalman R regularization (passed to evaluate_kalman_image_mse.py)"
    )
    return parser.parse_args()


def get_data_file(data_dir, seed_idx, condition_cfg):
    """Return path to data file based on condition.

    per_seed: quad{seed_idx}_n/y.npz (1.5k conditions)
    single_file: quad_long_n/y.npz (15k conditions, all seeds share same file)
    """
    suffix = condition_cfg["data_suffix"]
    if condition_cfg["data_pattern"] == "per_seed":
        return str(Path(data_dir) / f"quad{seed_idx}{suffix}")
    else:
        return str(Path(data_dir) / f"quad_long{suffix}")


def copy_seed1_training(src_dir, dst_dir):
    """Copy existing v4 training results into seed1 directory."""
    src = Path(src_dir)
    dst = Path(dst_dir)

    # Copy models/final_model.pth
    models_dst = dst / "models"
    models_dst.mkdir(parents=True, exist_ok=True)
    src_model = src / "models" / "final_model.pth"
    if src_model.exists():
        shutil.copy2(src_model, models_dst / "final_model.pth")
        print(f"  Copied model: {src_model} -> {models_dst / 'final_model.pth'}")
    else:
        raise FileNotFoundError(f"Source model not found: {src_model}")

    # Copy logs/ directory if exists
    src_logs = src / "logs"
    if src_logs.exists():
        dst_logs = dst / "logs"
        if dst_logs.exists():
            shutil.rmtree(dst_logs)
        shutil.copytree(src_logs, dst_logs)
        print(f"  Copied logs: {src_logs} -> {dst_logs}")


def copy_seed1_eval(src_dir, dst_dir):
    """Copy existing v4 Kalman evaluation results into seed1/kalman_eval/."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for fname in ["kalman_eval_results.json", "kalman_eval_detailed.json"]:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, dst / fname)
            print(f"  Copied eval: {src_file} -> {dst / fname}")


def run_training(config, data_file, output_dir, device, seed=None, timeout=14400):
    """Run training via subprocess."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_quadlink_experiment.py"),
        "--config", config,
        "--data", data_file,
        "--output", str(output_dir),
        "--device", device,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    print(f"  Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed for {data_file} (exit code {result.returncode})"
        )

    # Verify model was saved
    model_path = Path(output_dir) / "models" / "final_model.pth"
    if not model_path.exists():
        raise RuntimeError(f"Model not saved after training: {model_path}")

    print(f"  Model saved: {model_path}")


def run_kalman_eval(model_path, data_file, output_dir, device, gamma_q, gamma_r,
                    timeout=600):
    """Run Kalman evaluation via subprocess."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_kalman_image_mse.py"),
        "--model", str(model_path),
        "--data", data_file,
        "--output", str(output_dir),
        "--device", device,
        "--gamma-q", str(gamma_q),
        "--gamma-r", str(gamma_r),
    ]
    print(f"  Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Kalman eval failed for {model_path} (exit code {result.returncode})"
        )

    # Verify results
    results_file = Path(output_dir) / "kalman_eval_results.json"
    if not results_file.exists():
        raise RuntimeError(f"Eval results not saved: {results_file}")

    print(f"  Results saved: {results_file}")


def collect_results(base_output, seeds):
    """Collect evaluation results from all seeds."""
    results = {}
    for seed_idx in seeds:
        seed_dir = Path(base_output) / f"seed{seed_idx}"
        eval_file = seed_dir / "kalman_eval" / "kalman_eval_results.json"
        if eval_file.exists():
            with open(eval_file) as f:
                results[seed_idx] = json.load(f)
            print(f"  Seed {seed_idx}: loaded from {eval_file}")
        else:
            print(f"  Seed {seed_idx}: MISSING ({eval_file})")
    return results


def aggregate_results(results_per_seed, summary_dir, condition_name, paper_mean,
                      paper_std):
    """Compute mean +/- std across seeds and format for Table 1."""
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    mse_kalman = []
    mse_direct = []
    per_seed_summary = {}

    for seed_idx in sorted(results_per_seed.keys()):
        r = results_per_seed[seed_idx]

        # Primary: Kalman + Round-trip MSE (post-warmup)
        kalman_mse = None
        if "step2_kalman_roundtrip" in r:
            kalman_mse = r["step2_kalman_roundtrip"].get("mse_post_warmup")

        # Baseline: Direct prediction MSE
        direct_mse = None
        if "step1_direct" in r:
            direct_mse = r["step1_direct"].get("mse")

        if kalman_mse is not None:
            mse_kalman.append(kalman_mse)
        if direct_mse is not None:
            mse_direct.append(direct_mse)

        per_seed_summary[f"seed{seed_idx}"] = {
            "kalman_mse": kalman_mse,
            "direct_mse": direct_mse,
        }

    mse_kalman_arr = np.array(mse_kalman)
    mse_direct_arr = np.array(mse_direct)

    aggregate = {
        "condition": condition_name,
        "n_seeds": len(mse_kalman),
        "kalman_roundtrip_mse": {
            "mean": float(np.mean(mse_kalman_arr)) if len(mse_kalman_arr) > 0 else None,
            "std": float(np.std(mse_kalman_arr)) if len(mse_kalman_arr) > 0 else None,
            "values": mse_kalman_arr.tolist(),
        },
        "direct_prediction_mse": {
            "mean": float(np.mean(mse_direct_arr)) if len(mse_direct_arr) > 0 else None,
            "std": float(np.std(mse_direct_arr)) if len(mse_direct_arr) > 0 else None,
            "values": mse_direct_arr.tolist(),
        },
        "paper_target": {
            "mean": paper_mean,
            "std": paper_std,
        },
        "per_seed": per_seed_summary,
    }

    # Save JSON
    json_path = summary_dir / "aggregate_results.json"
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate results saved: {json_path}")

    # Save human-readable summary
    txt_path = summary_dir / "table1_comparison.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write(f"{condition_name} 5-Seed Evaluation Results\n")
        f.write("DSE 5-Seed Evaluation Results\n")
        f.write("=" * 65 + "\n\n")

        f.write(f"{'Seed':<8} {'Kalman+RT MSE':<18} {'Direct MSE':<18}\n")
        f.write("-" * 44 + "\n")
        for seed_idx in sorted(results_per_seed.keys()):
            ps = per_seed_summary[f"seed{seed_idx}"]
            k_str = f"{ps['kalman_mse']:.6f}" if ps['kalman_mse'] is not None else "N/A"
            d_str = f"{ps['direct_mse']:.6f}" if ps['direct_mse'] is not None else "N/A"
            f.write(f"seed{seed_idx:<4} {k_str:<18} {d_str:<18}\n")
        f.write("-" * 44 + "\n")

        if len(mse_kalman_arr) > 0:
            f.write(f"{'Mean':<8} {np.mean(mse_kalman_arr):<18.6f} {np.mean(mse_direct_arr):<18.6f}\n")
            f.write(f"{'Std':<8} {np.std(mse_kalman_arr):<18.6f} {np.std(mse_direct_arr):<18.6f}\n")

        f.write(f"\n{'Paper Table 1 (' + condition_name + '):':<30} {paper_mean:.4f} +/- {paper_std:.4f}\n")
        if len(mse_kalman_arr) > 0:
            f.write(
                f"{'Our result (Kalman+RT):':<30} "
                f"{np.mean(mse_kalman_arr):.4f} +/- {np.std(mse_kalman_arr):.4f}\n"
            )

        f.write("\n" + "=" * 65 + "\n")
        if len(mse_kalman_arr) > 0:
            our_mean = np.mean(mse_kalman_arr)
            if our_mean <= paper_mean:
                f.write("RESULT: Our mean MSE is BELOW paper target. Reproduction successful.\n")
            elif our_mean <= paper_mean + paper_std:
                f.write("RESULT: Our mean MSE is within 1 std of paper target. Acceptable.\n")
            else:
                f.write("RESULT: Our mean MSE exceeds paper target + 1 std. Investigation needed.\n")

    print(f"Comparison table saved: {txt_path}")

    # Print summary to console
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    if len(mse_kalman_arr) > 0:
        print(f"  Seeds evaluated: {len(mse_kalman_arr)}")
        print(f"  Kalman+RT MSE:   {np.mean(mse_kalman_arr):.4f} +/- {np.std(mse_kalman_arr):.4f}")
        print(f"  Direct MSE:      {np.mean(mse_direct_arr):.4f} +/- {np.std(mse_direct_arr):.4f}")
        print(f"  Paper target:    {paper_mean:.4f} +/- {paper_std:.4f}")
    else:
        print("  No results available.")
    print("=" * 65)

    return aggregate


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    base_output = Path(args.base_output)
    base_output.mkdir(parents=True, exist_ok=True)

    # Load condition configuration
    condition_cfg = CONDITION_CONFIG[args.condition]
    condition_name = args.condition

    # Dynamic timeouts: 15k conditions need ~10x longer
    is_15k = "15k" in condition_name
    train_timeout = 144000 if is_15k else 14400   # 40h or 4h
    eval_timeout = 6000 if is_15k else 600         # 100min or 10min

    print("=" * 65)
    print(f"{condition_name} 5-Seed Evaluation Pipeline")
    print("=" * 65)
    print(f"  Condition: {condition_name}")
    print(f"  Seeds: {seeds}")
    print(f"  Output: {base_output}")
    print(f"  Config: {args.config}")
    print(f"  Device: {args.device}")
    print(f"  Paper target: {condition_cfg['paper_mean']:.4f} +/- {condition_cfg['paper_std']:.4f}")
    print(f"  Training timeout: {train_timeout}s")
    print(f"  Aggregate only: {args.aggregate_only}")
    print(f"  Skip training: {args.skip_training}")
    print(f"  Skip evaluation: {args.skip_evaluation}")
    print()

    # Verify data files exist
    for seed_idx in seeds:
        data_file = get_data_file(args.data_dir, seed_idx, condition_cfg)
        if not Path(data_file).exists():
            print(f"ERROR: Data file not found: {data_file}")
            sys.exit(1)
    print("All data files verified.\n")

    if not args.aggregate_only:
        total_start = time.time()
        failed_seeds = []

        for seed_idx in seeds:
            seed_dir = base_output / f"seed{seed_idx}"
            data_file = get_data_file(args.data_dir, seed_idx, condition_cfg)
            model_path = seed_dir / "models" / "final_model.pth"
            eval_dir = seed_dir / "kalman_eval"

            print(f"\n{'='*65}")
            print(f"[SEED {seed_idx}/{len(seeds)}] Processing {Path(data_file).name}")
            print(f"{'='*65}")
            seed_start = time.time()

            try:
                # --- Training ---
                if not args.skip_training:
                    if seed_idx == 1 and args.reuse_seed1:
                        print(f"\n[SEED 1] Copying existing v4 results...")
                        copy_seed1_training(args.reuse_seed1, seed_dir)
                    elif model_path.exists():
                        print(f"\n[SEED {seed_idx}] Model already exists, skipping training: {model_path}")
                    else:
                        print(f"\n[SEED {seed_idx}] Starting training...")
                        train_seed = seed_idx * 42
                        run_training(args.config, data_file, str(seed_dir),
                                     args.device, seed=train_seed,
                                     timeout=train_timeout)

                # --- Kalman Evaluation ---
                if not args.skip_evaluation:
                    if seed_idx == 1 and args.reuse_seed1_eval:
                        print(f"\n[SEED 1] Copying existing Kalman eval results...")
                        copy_seed1_eval(args.reuse_seed1_eval, eval_dir)
                    else:
                        if not model_path.exists():
                            raise FileNotFoundError(f"Model not found: {model_path}")
                        print(f"\n[SEED {seed_idx}] Starting Kalman evaluation...")
                        run_kalman_eval(
                            model_path, data_file, str(eval_dir),
                            args.device, args.gamma_q, args.gamma_r,
                            timeout=eval_timeout
                        )

                elapsed = time.time() - seed_start
                print(f"\n[SEED {seed_idx}] Completed in {elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - seed_start
                print(f"\n[SEED {seed_idx}] FAILED after {elapsed:.1f}s: {e}")
                failed_seeds.append(seed_idx)
                continue

        total_elapsed = time.time() - total_start
        print(f"\n{'='*65}")
        print(f"All seeds processed in {total_elapsed:.1f}s")
        if failed_seeds:
            print(f"FAILED seeds: {failed_seeds}")
            print(f"Retry with: --seeds {','.join(str(s) for s in failed_seeds)}")
        print(f"{'='*65}\n")

    # --- Aggregation ---
    print("\nCollecting results...")
    all_seeds = [int(s) for s in args.seeds.split(",")]
    results = collect_results(base_output, all_seeds)

    if results:
        summary_dir = base_output / "summary"
        aggregate = aggregate_results(
            results, summary_dir, condition_name,
            condition_cfg["paper_mean"], condition_cfg["paper_std"]
        )
    else:
        print("No results found to aggregate.")


if __name__ == "__main__":
    main()
