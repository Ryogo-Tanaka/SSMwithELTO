#!/usr/bin/env python3
"""
SL (Stuart-Landau) Oscillator Data Generator

Generates time series data from the Stuart-Landau oscillator (complex SDE):
    dz/dt = (mu + i*gamma)*z - (1 + i*beta)*|z|^2*z + sqrt(var_p)*noise

Observation: y_t = [cos(theta_t), sin(theta_t)] + N(0, var_o*I_2)

Paper parameters (Table 9):
    mu = 1.0, gamma = 0.9, beta = 0.3
    IC = (r, theta) = (0.1, 0.0)
    dt = 0.1, T = 3500 steps
    Train: first 3000, Val: last 500
    var_o = 0.01 (fixed observation noise)
    var_p in {0.01, 0.02, ..., 0.09} (9 process noise conditions)
    50 trials per condition

Usage:
    # Single condition, 1 trial (smoke test)
    python scripts/generate_sl_data.py --output-dir data/sl --var-p-list 0.05 --n-trials 1

    # All 9 conditions, 50 trials each
    python scripts/generate_sl_data.py --output-dir data/sl --n-trials 50

    # Specific conditions
    python scripts/generate_sl_data.py --output-dir data/sl --var-p-list 0.01 0.05 0.09 --n-trials 50
"""

import argparse
import numpy as np
from pathlib import Path


def sl_drift(z, mu, gamma, beta):
    """Stuart-Landau drift term (complex)."""
    return (mu + 1j * gamma) * z - (1 + 1j * beta) * np.abs(z)**2 * z


def rk4_drift_step(z, dt, mu, gamma, beta):
    """Single RK4 step for the drift part only (no noise)."""
    k1 = sl_drift(z, mu, gamma, beta)
    k2 = sl_drift(z + 0.5 * dt * k1, mu, gamma, beta)
    k3 = sl_drift(z + 0.5 * dt * k2, mu, gamma, beta)
    k4 = sl_drift(z + dt * k3, mu, gamma, beta)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_sl_trajectory(mu, gamma, beta, ic_r, ic_theta, dt, n_steps, var_p, rng):
    """Generate SL trajectory using RK4(drift) + additive noise.

    Args:
        mu: Radius parameter
        gamma: Angular velocity parameter
        beta: Nonlinearity parameter
        ic_r: Initial radius
        ic_theta: Initial angle
        dt: Time step
        n_steps: Number of steps to generate
        var_p: Process noise variance (epsilon in paper)
        rng: numpy random generator

    Returns:
        states: (n_steps,) complex array of z states
    """
    states = np.zeros(n_steps, dtype=np.complex128)
    states[0] = ic_r * np.exp(1j * ic_theta)

    sqrt_dt = np.sqrt(dt)
    sqrt_var_p = np.sqrt(var_p)

    for t in range(n_steps - 1):
        # RK4 for drift
        z_drift = rk4_drift_step(states[t], dt, mu, gamma, beta)
        # Additive complex noise: sqrt(dt) * sqrt(var_p) * (N(0,1) + i*N(0,1)) / sqrt(2)
        # Using standard complex Wiener increment
        noise_real = rng.normal(0, 1)
        noise_imag = rng.normal(0, 1)
        states[t + 1] = z_drift + sqrt_dt * sqrt_var_p * (noise_real + 1j * noise_imag) / np.sqrt(2)

    return states


def generate_trial_data(mu, gamma, beta, ic_r, ic_theta, dt, n_steps, var_p, var_o, rng):
    """Generate a single trial of SL data with observation noise.

    Args:
        mu, gamma, beta: SL parameters
        ic_r, ic_theta: Initial condition in polar coordinates
        dt: Time step
        n_steps: Total number of steps
        var_p: Process noise variance
        var_o: Observation noise variance
        rng: numpy random generator

    Returns:
        dict with keys: train_obs, val_obs, train_states, val_states, params
    """
    states = generate_sl_trajectory(mu, gamma, beta, ic_r, ic_theta, dt, n_steps, var_p, rng)

    # Observation: y_t = [cos(theta_t), sin(theta_t)] + N(0, var_o*I_2)
    theta = np.angle(states)
    clean_obs = np.column_stack([np.cos(theta), np.sin(theta)])  # (T, 2)

    sig_o = np.sqrt(var_o)
    noise = rng.normal(0, sig_o, size=clean_obs.shape)
    noisy_obs = clean_obs + noise  # (T, 2)

    # Split: first 3000 train, last 500 val
    n_train = n_steps - 500
    train_obs = noisy_obs[:n_train]       # (3000, 2)
    val_obs = noisy_obs[n_train:]          # (500, 2)
    train_states = states[:n_train]        # (3000,) complex
    val_states = states[n_train:]          # (500,) complex

    params = {
        'mu': mu,
        'gamma': gamma,
        'beta': beta,
        'ic_r': ic_r,
        'ic_theta': ic_theta,
        'dt': dt,
        'var_p': var_p,
        'var_o': var_o,
        'n_steps': n_steps,
        'n_train': n_train,
    }

    return {
        'train_obs': train_obs,
        'val_obs': val_obs,
        'train_states': train_states,
        'val_states': val_states,
        'params': params,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate Stuart-Landau oscillator data')
    parser.add_argument('--output-dir', type=str, default='data/sl',
                        help='Output directory')
    parser.add_argument('--var-p-list', type=float, nargs='+', default=None,
                        help='List of process noise variance values (default: 0.01 to 0.09)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials per condition')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--mu', type=float, default=1.0,
                        help='Radius parameter')
    parser.add_argument('--gamma-param', type=float, default=0.9,
                        help='Angular velocity parameter')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Nonlinearity parameter')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step')
    parser.add_argument('--n-steps', type=int, default=3500,
                        help='Total number of time steps')
    parser.add_argument('--ic-r', type=float, default=0.1,
                        help='Initial radius')
    parser.add_argument('--ic-theta', type=float, default=0.0,
                        help='Initial angle')
    parser.add_argument('--var-o', type=float, default=0.01,
                        help='Observation noise variance (fixed)')
    args = parser.parse_args()

    # Default: 9 process noise conditions
    if args.var_p_list is None:
        var_p_list = [round(0.01 * i, 2) for i in range(1, 10)]
    else:
        var_p_list = args.var_p_list

    output_dir = Path(args.output_dir)

    total_files = len(var_p_list) * args.n_trials
    generated = 0

    for var_p in var_p_list:
        condition_dir = output_dir / f"var_p_{var_p:.2f}"
        condition_dir.mkdir(parents=True, exist_ok=True)

        for trial_idx in range(1, args.n_trials + 1):
            out_path = condition_dir / f"trial_{trial_idx:03d}.npz"

            # Deterministic seed: base_seed + condition_hash + trial
            seed = args.base_seed + int(var_p * 10000) + trial_idx
            rng = np.random.default_rng(seed)

            data = generate_trial_data(
                mu=args.mu, gamma=args.gamma_param, beta=args.beta,
                ic_r=args.ic_r, ic_theta=args.ic_theta,
                dt=args.dt, n_steps=args.n_steps,
                var_p=var_p, var_o=args.var_o, rng=rng
            )

            np.savez(out_path,
                     train_obs=data['train_obs'],
                     val_obs=data['val_obs'],
                     train_states=data['train_states'],
                     val_states=data['val_states'],
                     params=data['params'])

            generated += 1

        print(f"[{generated}/{total_files}] var_p={var_p:.2f}: "
              f"{args.n_trials} trials -> {condition_dir}")

    print(f"\nDone. Generated {generated} files in {output_dir}")


if __name__ == '__main__':
    main()
