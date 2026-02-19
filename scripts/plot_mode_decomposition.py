#!/usr/bin/env python3
"""
Mode Decomposition Plot Generator

Creates Figure 2 (VDP) and Figure 3 (SL) style plots from experiment results.

Usage:
    # VDP Figure 2
    python scripts/plot_mode_decomposition.py \
        --results-dir results/vdp \
        --experiment vdp \
        --output results/vdp/figures/figure2.pdf

    # SL Figure 3
    python scripts/plot_mode_decomposition.py \
        --results-dir results/sl \
        --experiment sl \
        --output results/sl/figures/figure3.pdf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Paper reference values
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


def load_evaluation(results_dir):
    """Load evaluation summary JSON."""
    eval_path = Path(results_dir) / 'summary' / 'evaluation.json'
    if not eval_path.exists():
        return None
    with open(eval_path) as f:
        return json.load(f)


def plot_eigenvalue_error(evaluation, experiment, paper_ref, output_path):
    """Create error bar plot matching paper Figure 2/3 style."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Skipping plot generation.")
        return

    conditions = evaluation['conditions']
    x_vals = []
    means = []
    stds = []

    for cond_key in sorted(conditions.keys()):
        x_vals.append(float(cond_key))
        means.append(conditions[cond_key]['mean'])
        stds.append(conditions[cond_key]['std'])

    x_vals = np.array(x_vals)
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Our results
    ax.errorbar(x_vals, means, yerr=stds, fmt='o-', capsize=4, capthick=1.5,
                linewidth=2, markersize=6, label='DSE (ours)', color='#2196F3')

    # Paper reference
    if paper_ref:
        paper_x = sorted(paper_ref.keys())
        paper_means = [paper_ref[x][0] for x in paper_x]
        paper_stds = [paper_ref[x][1] for x in paper_x]
        ax.errorbar(paper_x, paper_means, yerr=paper_stds, fmt='s--', capsize=4,
                    capthick=1.5, linewidth=2, markersize=6, label='DSE (paper)',
                    color='#FF9800', alpha=0.8)

    if experiment.upper() == 'VDP':
        ax.set_xlabel(r'Observation noise variance $\sigma_o^2$', fontsize=12)
        ax.set_title('VDP Oscillator: Total Eigenvalue Error', fontsize=14)
    else:
        ax.set_xlabel(r'Process noise variance $\varepsilon$', fontsize=12)
        ax.set_title('SL Oscillator: Total Eigenvalue Error', fontsize=14)

    ax.set_ylabel('Total eigenvalue error', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {output_path}")
    plt.close()


def plot_eigenvalue_spectrum(results_dir, experiment, output_path):
    """Plot estimated eigenvalues on the complex plane for one trial."""
    if not HAS_MATPLOTLIB:
        return

    results_dir = Path(results_dir)
    # Find first completed trial
    npz_file = None
    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        for trial_dir in sorted(condition_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            candidate = trial_dir / 'vdp_spectrum.npz'
            if not candidate.exists():
                candidate = trial_dir / 'sl_spectrum.npz'
            if candidate.exists():
                npz_file = candidate
                break
        if npz_file:
            break

    if npz_file is None:
        print("No spectrum data found for complex plane plot.")
        return

    data = np.load(npz_file, allow_pickle=True)
    eigs = data['eigenvalues_discrete']
    ref_eigs = data['reference_eigenvalues']

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.2, linewidth=1)

    # Reference eigenvalues
    ax.scatter(ref_eigs.real, ref_eigs.imag, marker='x', s=100, c='red',
               linewidths=2, zorder=5, label='Reference')

    # Estimated eigenvalues
    ax.scatter(eigs.real, eigs.imag, marker='o', s=30, c='#2196F3',
               alpha=0.6, zorder=4, label='Estimated')

    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    ax.set_title(f'{experiment.upper()} Eigenvalue Spectrum (sample trial)', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    spectrum_path = output_path.parent / f'{experiment}_spectrum_example.pdf'
    plt.savefig(spectrum_path, dpi=150, bbox_inches='tight')
    print(f"Spectrum plot saved: {spectrum_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot mode decomposition results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Results directory')
    parser.add_argument('--experiment', type=str, choices=['vdp', 'sl'], required=True,
                        help='Experiment type')
    parser.add_argument('--output', type=str, default=None,
                        help='Output plot path')
    parser.add_argument('--skip-spectrum', action='store_true',
                        help='Skip complex plane spectrum plot')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load evaluation summary
    evaluation = load_evaluation(results_dir)
    if evaluation is None:
        print("Error: Run evaluate_mode_decomposition.py first.")
        sys.exit(1)

    if args.experiment == 'vdp':
        paper_ref = PAPER_VDP
        default_output = results_dir / 'figures' / 'figure2_vdp_error.pdf'
    else:
        paper_ref = PAPER_SL
        default_output = results_dir / 'figures' / 'figure3_sl_error.pdf'

    output_path = Path(args.output) if args.output else default_output

    # Error bar plot
    plot_eigenvalue_error(evaluation, args.experiment, paper_ref, output_path)

    # Spectrum plot
    if not args.skip_spectrum:
        plot_eigenvalue_spectrum(results_dir, args.experiment, output_path)


if __name__ == '__main__':
    main()
