#!/usr/bin/env python3
"""Orchestrator for multi-step free rollout (PROJ-REBUTTAL Step 4, v3).

Runs evaluate_multistep_rollout_dse.py and evaluate_multistep_rollout_ncdssm.py
for every (method, regime, seed, C) combination, then aggregates the per-eval
CSVs into a single raw CSV at results/rebuttal/multistep_rollout_raw.csv.

Cohorts (defaults):
  DSE clean:   seeds 1..5 (results/phase_c_clean_1500_v3/seed{N}/models/final_model.pth,
               data/rkn_quad/quad{N}_n.npz)
  DSE corrupted: seeds 1..3 (results/phase_c_noisy_1500_v3/seed{N}/models/final_model.pth,
               data/rkn_quad/quad{N}_y.npz)
  NCDSSM clean:    seed 1 (results/rebuttal/ncdssm_quadlink/clean/seed1/ckpts/model_10000.pt,
                   data/rkn_quad/quad1_n.npz)
  NCDSSM corrupted: seed 1 (results/rebuttal/ncdssm_quadlink/corrupted/seed1/ckpts/model_10000.pt,
                    data/rkn_quad/quad1_y.npz)
  Context lengths: 5, 10

Usage:
  PYTHONUNBUFFERED=1 python scripts/rebuttal/run_multistep_rollout.py \
    --ctx_lens 5,10 \
    --output_root results/rebuttal/multistep_rollout/ \
    --raw_csv results/rebuttal/multistep_rollout_raw.csv
"""
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "rkn_quad"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ctx_lens", type=str, default="5,10",
                   help="Comma-separated context lengths (default 5,10)")
    p.add_argument("--horizons", type=str, default="1,5,10,20,50")
    p.add_argument("--dse_clean_seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--dse_corrupted_seeds", type=str, default="1,2,3")
    p.add_argument("--ncdssm_clean_seeds", type=str, default="1")
    p.add_argument("--ncdssm_corrupted_seeds", type=str, default="1")
    p.add_argument("--dse_clean_root", type=str,
                   default="results/phase_c_clean_1500_v3")
    p.add_argument("--dse_noisy_root", type=str,
                   default="results/phase_c_noisy_1500_v3")
    p.add_argument("--ncdssm_root", type=str,
                   default="results/rebuttal/ncdssm_quadlink")
    p.add_argument("--ncdssm_ckpt_step", type=int, default=10000)
    p.add_argument("--output_root", type=str,
                   default="results/rebuttal/multistep_rollout/")
    p.add_argument("--raw_csv", type=str,
                   default="results/rebuttal/multistep_rollout_raw.csv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


def csv_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run(cmd: List[str], dry_run: bool) -> int:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[orch] $ {pretty}", flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


def gather_raw_rows(eval_dir: Path) -> List[dict]:
    csv_path = eval_dir / "eval_result.csv"
    if not csv_path.exists():
        print(f"[orch] WARN: missing {csv_path}", flush=True)
        return []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> int:
    args = parse_args()
    ctx_lens = csv_int_list(args.ctx_lens)
    output_root = REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    raw_csv = REPO_ROOT / args.raw_csv

    all_rows: List[dict] = []

    # ---------- DSE clean ----------
    for seed in csv_int_list(args.dse_clean_seeds):
        model = (REPO_ROOT / args.dse_clean_root /
                 f"seed{seed}" / "models" / "final_model.pth")
        data = DATA_ROOT / f"quad{seed}_n.npz"
        for C in ctx_lens:
            out = output_root / "dse" / "clean" / f"seed{seed}" / f"ctx{C}"
            cmd = [
                sys.executable, "scripts/rebuttal/evaluate_multistep_rollout_dse.py",
                "--model", str(model),
                "--data", str(data),
                "--regime", "clean",
                "--seed_label", str(seed),
                "--ctx_len", str(C),
                "--horizons", args.horizons,
                "--gamma-q", "1e-6", "--gamma-r", "1e-6",
                "--output", str(out),
                "--device", args.device,
            ]
            rc = run(cmd, args.dry_run)
            if rc != 0:
                print(f"[orch] DSE clean seed{seed} ctx{C} FAILED (rc={rc})",
                      flush=True)
                continue
            all_rows.extend(gather_raw_rows(out))

    # ---------- DSE corrupted ----------
    for seed in csv_int_list(args.dse_corrupted_seeds):
        model = (REPO_ROOT / args.dse_noisy_root /
                 f"seed{seed}" / "models" / "final_model.pth")
        data = DATA_ROOT / f"quad{seed}_y.npz"
        if not model.exists():
            print(f"[orch] WARN: DSE corrupted seed{seed} ckpt not found: {model}",
                  flush=True)
            continue
        for C in ctx_lens:
            out = output_root / "dse" / "corrupted" / f"seed{seed}" / f"ctx{C}"
            cmd = [
                sys.executable, "scripts/rebuttal/evaluate_multistep_rollout_dse.py",
                "--model", str(model),
                "--data", str(data),
                "--regime", "corrupted",
                "--seed_label", str(seed),
                "--ctx_len", str(C),
                "--horizons", args.horizons,
                "--gamma-q", "1e-3", "--gamma-r", "1e-3",
                "--output", str(out),
                "--device", args.device,
            ]
            rc = run(cmd, args.dry_run)
            if rc != 0:
                print(f"[orch] DSE corrupted seed{seed} ctx{C} FAILED (rc={rc})",
                      flush=True)
                continue
            all_rows.extend(gather_raw_rows(out))

    # ---------- NCDSSM clean ----------
    for seed in csv_int_list(args.ncdssm_clean_seeds):
        ckpt = (REPO_ROOT / args.ncdssm_root / "clean" / f"seed{seed}" /
                "ckpts" / f"model_{args.ncdssm_ckpt_step}.pt")
        for C in ctx_lens:
            out = output_root / "ncdssm" / "clean" / f"seed{seed}" / f"ctx{C}"
            cmd = [
                sys.executable, "scripts/rebuttal/evaluate_multistep_rollout_ncdssm.py",
                "--ckpt", str(ckpt),
                "--regime", "clean",
                "--data_seed", str(seed),
                "--seed_label", str(seed),
                "--ctx_len", str(C),
                "--horizons", args.horizons,
                "--output", str(out),
                "--device", args.device,
            ]
            rc = run(cmd, args.dry_run)
            if rc != 0:
                print(f"[orch] NCDSSM clean seed{seed} ctx{C} FAILED (rc={rc})",
                      flush=True)
                continue
            all_rows.extend(gather_raw_rows(out))

    # ---------- NCDSSM corrupted ----------
    for seed in csv_int_list(args.ncdssm_corrupted_seeds):
        ckpt = (REPO_ROOT / args.ncdssm_root / "corrupted" / f"seed{seed}" /
                "ckpts" / f"model_{args.ncdssm_ckpt_step}.pt")
        for C in ctx_lens:
            out = output_root / "ncdssm" / "corrupted" / f"seed{seed}" / f"ctx{C}"
            cmd = [
                sys.executable, "scripts/rebuttal/evaluate_multistep_rollout_ncdssm.py",
                "--ckpt", str(ckpt),
                "--regime", "corrupted",
                "--data_seed", str(seed),
                "--seed_label", str(seed),
                "--ctx_len", str(C),
                "--horizons", args.horizons,
                "--output", str(out),
                "--device", args.device,
            ]
            rc = run(cmd, args.dry_run)
            if rc != 0:
                print(f"[orch] NCDSSM corrupted seed{seed} ctx{C} FAILED (rc={rc})",
                      flush=True)
                continue
            all_rows.extend(gather_raw_rows(out))

    # ---------- Aggregate raw CSV ----------
    if all_rows:
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["method", "regime", "seed", "sequence_id",
                            "context_length", "horizon", "mse"],
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[orch] aggregated {len(all_rows)} rows -> {raw_csv}",
              flush=True)
    else:
        print("\n[orch] no rows to aggregate", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
