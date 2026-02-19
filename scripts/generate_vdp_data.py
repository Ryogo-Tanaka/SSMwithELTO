#!/usr/bin/env python3
"""
VDP (Van der Pol) Oscillator Data Generator

Generates time series data from the Van der Pol oscillator:
    dx/dt = y
    dy/dt = mu * (1 - x^2) * y - x

Observation: z_t = x_t + N(0, sig_o^2)

Paper parameters (Table 7):
    mu = 2.0, IC = (2.0, 0.0), dt = 0.1, T = 3500 steps
    Train: first 3000, Val: last 500
    sig_o^2 in {0.01, 0.02, ..., 0.20} (20 conditions)
    50 trials per condition

Usage:
    # Single condition, 1 trial (smoke test)
    python scripts/generate_vdp_data.py --output-dir data/vdp --sig-o-list 0.05 --n-trials 1

    # All 20 conditions, 50 trials each
    python scripts/generate_vdp_data.py --output-dir data/vdp --n-trials 50

    # Specific conditions
    python scripts/generate_vdp_data.py --output-dir data/vdp --sig-o-list 0.05 0.10 0.15 0.20 --n-trials 50
"""

import argparse
import numpy as np
from pathlib import Path


def vdp_rhs(state, mu):
    """Van der Pol oscillator right-hand side."""
    x, y = state
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return np.array([dxdt, dydt])


def rk4_step(state, dt, mu):
    """Single RK4 integration step."""
    k1 = vdp_rhs(state, mu)
    k2 = vdp_rhs(state + 0.5 * dt * k1, mu)
    k3 = vdp_rhs(state + 0.5 * dt * k2, mu)
    k4 = vdp_rhs(state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_vdp_trajectory(mu, ic, dt, n_steps):
    """Generate VDP trajectory using RK4 integration.

    Args:
        mu: VDP parameter
        ic: Initial condition [x0, y0]
        dt: Time step
        n_steps: Number of steps to generate

    Returns:
        states: (n_steps, 2) array of [x, y] states
    """
    states = np.zeros((n_steps, 2))
    states[0] = ic
    for t in range(n_steps - 1):
        states[t + 1] = rk4_step(states[t], dt, mu)
    return states


def generate_trial_data(mu, ic, dt, n_steps, sig_o, rng):
    """Generate a single trial of VDP data with observation noise.

    Args:
        mu: VDP parameter
        ic: Initial condition
        dt: Time step
        n_steps: Total number of steps
        sig_o: Observation noise std (sig_o, NOT sig_o^2)
        rng: numpy random generator

    Returns:
        dict with keys: train_obs, val_obs, train_states, val_states, params
    """
    states = generate_vdp_trajectory(mu, ic, dt, n_steps)

    # Observation = x component + noise
    clean_obs = states[:, 0]  # scalar x_t
    noise = rng.normal(0, sig_o, size=clean_obs.shape)
    noisy_obs = clean_obs + noise

    # Split: first 3000 train, last 500 val (paper: T=3500)
    n_train = n_steps - 500
    train_obs = noisy_obs[:n_train]       # (3000,)
    val_obs = noisy_obs[n_train:]          # (500,)
    train_states = states[:n_train]        # (3000, 2)
    val_states = states[n_train:]          # (500, 2)

    params = {
        'mu': mu,
        'ic': np.array(ic),
        'dt': dt,
        'sig_o': sig_o,
        'sig_o_sq': sig_o**2,
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
    parser = argparse.ArgumentParser(description='Generate VDP oscillator data')
    parser.add_argument('--output-dir', type=str, default='data/vdp',
                        help='Output directory')
    parser.add_argument('--sig-o-list', type=float, nargs='+', default=None,
                        help='List of sig_o^2 values (default: 0.01 to 0.20 in steps of 0.01)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials per condition')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--mu', type=float, default=2.0,
                        help='VDP parameter mu')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step')
    parser.add_argument('--n-steps', type=int, default=3500,
                        help='Total number of time steps')
    parser.add_argument('--ic', type=float, nargs=2, default=[2.0, 0.0],
                        help='Initial condition [x0, y0]')
    args = parser.parse_args()

    # Default: 20 noise conditions (sig_o^2 from 0.01 to 0.20)
    if args.sig_o_list is None:
        sig_o_sq_list = [round(0.01 * i, 2) for i in range(1, 21)]
    else:
        sig_o_sq_list = args.sig_o_list

    output_dir = Path(args.output_dir)
    ic = np.array(args.ic)

    total_files = len(sig_o_sq_list) * args.n_trials
    generated = 0

    for sig_o_sq in sig_o_sq_list:
        sig_o = np.sqrt(sig_o_sq)  # Convert variance to std
        condition_dir = output_dir / f"sig_o_{sig_o_sq:.2f}"
        condition_dir.mkdir(parents=True, exist_ok=True)

        for trial_idx in range(1, args.n_trials + 1):
            out_path = condition_dir / f"trial_{trial_idx:03d}.npz"

            # Deterministic seed: base_seed + condition_hash + trial
            seed = args.base_seed + int(sig_o_sq * 10000) + trial_idx
            rng = np.random.default_rng(seed)

            data = generate_trial_data(
                mu=args.mu, ic=ic, dt=args.dt,
                n_steps=args.n_steps, sig_o=sig_o, rng=rng
            )

            np.savez(out_path,
                     train_obs=data['train_obs'],
                     val_obs=data['val_obs'],
                     train_states=data['train_states'],
                     val_states=data['val_states'],
                     params=data['params'])

            generated += 1

        print(f"[{generated}/{total_files}] sig_o^2={sig_o_sq:.2f}: "
              f"{args.n_trials} trials -> {condition_dir}")

    print(f"\nDone. Generated {generated} files in {output_dir}")


if __name__ == '__main__':
    main()
