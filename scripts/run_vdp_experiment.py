#!/usr/bin/env python3
"""
VDP Oscillator Single-Trial Experiment Script

Pipeline:
1. Load data from npz
2. Apply Hankel delay embedding
3. Train DSE model (TwoStageTrainer)
4. Extract V_A transfer operator
5. Compute eigenvalue spectrum
6. Compare with reference spectrum (VDP harmonics)
7. Save results

Usage:
    python scripts/run_vdp_experiment.py \
        --data data/vdp/sig_o_0.05/trial_001.npz \
        --config configs/vdp_config.yaml \
        --output results/vdp/sig_o_0.05/trial_001 \
        --device cuda:0

    # With custom seed
    python scripts/run_vdp_experiment.py \
        --data data/vdp/sig_o_0.05/trial_001.npz \
        --config configs/vdp_config.yaml \
        --output results/vdp/sig_o_0.05/trial_001 \
        --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hankel_embedding import hankel_embed
from src.training.two_stage_trainer import TwoStageTrainer
from src.evaluation.mode_decomposition import SpectrumAnalyzer


# =============================================================================
# Reference Spectrum for VDP oscillator
# =============================================================================

def compute_vdp_fundamental_frequency(mu=2.0, ic=(2.0, 0.0), dt=0.1, n_periods=50):
    """
    Numerically estimate VDP fundamental angular frequency from a clean simulation.

    Runs a long noise-free VDP trajectory and measures the limit cycle period
    by detecting zero-crossings of x(t).

    Returns:
        omega: Fundamental angular frequency (rad/s)
        omega_dt: omega * dt (discrete angle per step)
    """
    from scripts.generate_vdp_data import rk4_step

    # Run long trajectory to reach limit cycle
    n_warmup = 5000
    n_measure = int(n_periods * 100)  # ~100 steps per period
    n_total = n_warmup + n_measure

    state = np.array(ic, dtype=np.float64)
    for _ in range(n_warmup):
        state = rk4_step(state, dt, mu)

    # Record trajectory and find zero-crossings (x crosses zero going positive)
    x_prev = state[0]
    crossing_times = []
    for step in range(n_measure):
        state = rk4_step(state, dt, mu)
        x_curr = state[0]
        if x_prev < 0 and x_curr >= 0:
            # Linear interpolation for precise crossing time
            frac = -x_prev / (x_curr - x_prev)
            crossing_times.append((step + frac) * dt)
        x_prev = x_curr

    if len(crossing_times) < 3:
        raise RuntimeError("Could not detect enough zero crossings")

    # Average period from consecutive crossings
    periods = np.diff(crossing_times)
    avg_period = np.mean(periods)
    omega = 2 * np.pi / avg_period
    omega_dt = omega * dt

    return omega, omega_dt


def get_vdp_reference_eigenvalues(omega_dt, n_harmonics=6):
    """
    Generate VDP reference eigenvalues (discrete time).

    VDP limit cycle has Koopman eigenvalues at integer multiples of
    the fundamental frequency: lambda_m = exp(i * m * omega * dt)

    Args:
        omega_dt: Fundamental angle per time step (omega * dt)
        n_harmonics: Number of harmonics in each direction (total = 2*n_harmonics + 1)

    Returns:
        reference_eigs: Complex array of reference eigenvalues
    """
    harmonics = np.arange(-n_harmonics, n_harmonics + 1)  # [-6, ..., 0, ..., 6]
    reference_eigs = np.exp(1j * harmonics * omega_dt)
    return reference_eigs


# =============================================================================
# Eigenvalue error computation
# =============================================================================

def calculate_total_eigenvalue_error(reference_eigs, estimated_eigs):
    """
    Calculate total eigenvalue error between reference and estimated eigenvalues.

    For each reference eigenvalue, find the nearest estimated eigenvalue
    and compute the absolute distance. Sum all distances.

    This is the "total eigenvalue error" metric used in the paper (Table 3).

    Args:
        reference_eigs: Complex array of reference eigenvalues (K,)
        estimated_eigs: Complex array of estimated eigenvalues (N,)

    Returns:
        total_error: Sum of minimum distances
        per_harmonic_errors: Array of per-harmonic errors (K,)
    """
    reference_eigs = np.asarray(reference_eigs)
    estimated_eigs = np.asarray(estimated_eigs)

    per_harmonic_errors = np.zeros(len(reference_eigs))
    for i, ref in enumerate(reference_eigs):
        distances = np.abs(estimated_eigs - ref)
        per_harmonic_errors[i] = np.min(distances)

    total_error = np.sum(per_harmonic_errors)
    return total_error, per_harmonic_errors


# =============================================================================
# Main experiment pipeline
# =============================================================================

def run_vdp_experiment(args):
    """Run a single VDP experiment trial."""

    # ---- Setup ----
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu')

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print(f"=== VDP Experiment ===")
    print(f"Data: {args.data}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # ---- Step 1: Load data ----
    print("\n[Step 1] Loading data...")
    data = np.load(args.data, allow_pickle=True)
    train_obs = data['train_obs']  # (3000,) for VDP
    val_obs = data['val_obs']      # (500,)

    params = data['params'].item() if data['params'].ndim == 0 else dict(data['params'])
    sig_o_sq = params.get('sig_o_sq', params.get('sig_o', 0)**2)
    print(f"  train_obs: {train_obs.shape}, val_obs: {val_obs.shape}")
    print(f"  sig_o^2 = {sig_o_sq:.4f}")

    # ---- Step 2: Hankel delay embedding ----
    print("\n[Step 2] Hankel delay embedding...")
    window = args.hankel_window
    Y_train = hankel_embed(train_obs, window)  # (T-window+1, window)
    Y_val = hankel_embed(val_obs, window)

    print(f"  Hankel window = {window}")
    print(f"  Y_train: {Y_train.shape}, Y_val: {Y_val.shape}")

    # Convert to torch tensors
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    # ---- Step 3: Train DSE model ----
    print("\n[Step 3] Training DSE model...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    start_time = time.time()
    trainer = TwoStageTrainer(config=config, device=device, output_dir=str(output_dir))
    results = trainer.train_integrated(Y_train=Y_train_t, Y_val=Y_val_t)
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.1f}s")

    # ---- Step 4: Extract V_A ----
    print("\n[Step 4] Extracting V_A transfer operator...")
    V_A = trainer.df_state.get_transfer_operator()
    print(f"  V_A shape: {V_A.shape}")

    # ---- Step 5: Spectrum analysis ----
    print("\n[Step 5] Spectrum analysis...")
    dt = args.dt
    analyzer = SpectrumAnalyzer(sampling_interval=dt)
    spectrum = analyzer.analyze_spectrum(V_A)

    eigs_discrete = spectrum['eigenvalues_discrete'].cpu().numpy()
    print(f"  Found {len(eigs_discrete)} eigenvalues")
    print(f"  Spectral radius: {spectrum['spectral_radius']:.6f}")

    # ---- Step 6: Reference spectrum ----
    print("\n[Step 6] Computing reference spectrum...")
    if args.omega_dt is not None:
        omega_dt = args.omega_dt
        omega = omega_dt / dt
        print(f"  Using provided omega*dt = {omega_dt:.10f}")
    else:
        omega, omega_dt = compute_vdp_fundamental_frequency(mu=args.mu, dt=dt)
        print(f"  Computed omega = {omega:.6f}, omega*dt = {omega_dt:.10f}")

    n_harmonics = args.n_harmonics
    reference_eigs = get_vdp_reference_eigenvalues(omega_dt, n_harmonics)
    print(f"  Reference: {2 * n_harmonics + 1} eigenvalues (harmonics -{n_harmonics} to +{n_harmonics})")

    # ---- Step 7: Calculate total eigenvalue error ----
    print("\n[Step 7] Calculating eigenvalue error...")
    total_error, per_harmonic_errors = calculate_total_eigenvalue_error(
        reference_eigs, eigs_discrete
    )
    print(f"  Total eigenvalue error: {total_error:.6f}")
    print(f"  Per-harmonic errors (max): {np.max(per_harmonic_errors):.6f}")
    print(f"  Per-harmonic errors (mean): {np.mean(per_harmonic_errors):.6f}")

    # ---- Step 8: Save results ----
    print("\n[Step 8] Saving results...")

    result_dict = {
        'total_eigenvalue_error': float(total_error),
        'per_harmonic_errors': per_harmonic_errors.tolist(),
        'spectral_radius': float(spectrum['spectral_radius']),
        'omega': float(omega_dt / dt),
        'omega_dt': float(omega_dt),
        'n_harmonics': n_harmonics,
        'sig_o_sq': float(sig_o_sq),
        'hankel_window': window,
        'train_time_sec': float(train_time),
        'seed': args.seed,
        'data_path': str(args.data),
        'config_path': str(args.config),
        'V_A_shape': list(V_A.shape),
    }

    # Save JSON summary
    json_path = output_dir / 'vdp_result.json'
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"  JSON: {json_path}")

    # Save detailed numpy data
    npz_path = output_dir / 'vdp_spectrum.npz'
    np.savez(npz_path,
             eigenvalues_discrete=eigs_discrete,
             reference_eigenvalues=reference_eigs,
             per_harmonic_errors=per_harmonic_errors,
             V_A=V_A.cpu().numpy(),
             eigenvalues_magnitude=spectrum['eigenvalues_magnitude'].cpu().numpy(),
             eigenvalues_phase=spectrum['eigenvalues_phase'].cpu().numpy())
    print(f"  NPZ: {npz_path}")

    print(f"\n=== Result: total_eigenvalue_error = {total_error:.6f} ===")
    return result_dict


def main():
    parser = argparse.ArgumentParser(description='Run VDP oscillator experiment (single trial)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to trial npz file')
    parser.add_argument('--config', type=str, default='configs/vdp_config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (cuda:0, cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Sampling interval')
    parser.add_argument('--mu', type=float, default=2.0,
                        help='VDP parameter mu (for reference frequency computation)')
    parser.add_argument('--hankel-window', type=int, default=30,
                        help='Hankel delay embedding window size')
    parser.add_argument('--omega-dt', type=float, default=None,
                        help='Override omega*dt (skip numerical frequency estimation)')
    parser.add_argument('--n-harmonics', type=int, default=6,
                        help='Number of harmonics in each direction for reference')
    args = parser.parse_args()

    run_vdp_experiment(args)


if __name__ == '__main__':
    main()
