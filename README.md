# Deep Spectral Encoder (DSE)

Implementation of **"Deep Spectral Learning of Embedded Latent Transfer Operators for Stochastic Dynamical Systems"** (UAI 2026).

This repository provides the code for training and evaluating DSE models on three benchmark dynamical systems: quad-link pendulum image sequences, Van der Pol (VDP) oscillator, and Stuart-Landau (SL) oscillator. The method combines CCA-based stochastic realization with deep encoder/decoder networks and distribution-free (DF) operators for state-space modeling.

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

## Repository Structure

```
SSMwithELTO/
├── src/                           # Core source code
│   ├── models/                    #   Encoder/Decoder (factory pattern)
│   │   ├── encoder.py             #     Encoder factory
│   │   ├── decoder.py             #     Decoder factory
│   │   └── architectures/         #     Architecture implementations
│   │       ├── cnn_image.py       #       CNN for 48x48 images (Quad-link)
│   │       └── time_invariant.py  #       MLP for time series (VDP/SL)
│   ├── ssm/                       #   State-space model components
│   │   ├── realization.py         #     CCA-based stochastic realization
│   │   ├── df_state_layer.py      #     DF-A: state transfer operator
│   │   ├── df_observation_layer.py#     DF-B: observation operator
│   │   └── cross_fitting.py       #     K-fold cross-fitting
│   ├── training/
│   │   └── two_stage_trainer.py   #   Two-stage training (Phase 1 + Phase 2)
│   ├── inference/                 #   Kalman filtering inference
│   │   ├── kalman_filter.py       #     Operator-based Kalman filter (Alg. 1)
│   │   ├── state_estimator.py     #     Batch inference wrapper
│   │   └── noise_covariance.py    #     Q/R covariance estimation
│   ├── evaluation/                #   Evaluation utilities
│   │   ├── metrics.py             #     MSE, RMSE, R^2, etc.
│   │   └── mode_decomposition.py  #     Koopman spectrum analysis
│   └── utils/
│       ├── data_loader.py         #   Unified data loader
│       └── hankel_embedding.py    #   Hankel delay embedding
├── scripts/                       # Experiment runner scripts
├── configs/                       # YAML configuration files
├── data/                          # Experiment data (not included)
└── requirements.txt
```

## Quick Start

### 1. Quad-link Pendulum (Table 1)

**Training:**
```bash
python scripts/run_quadlink_experiment.py \
    --config configs/quad_image_reconstruction_config.yaml \
    --data data/rkn_quad/quad1_n.npz \
    --output results/quadlink_seed1 \
    --seed 42
```

**Kalman Filtering Evaluation:**
```bash
python scripts/evaluate_kalman_image_mse.py \
    --model results/quadlink_seed1/models/final_model.pth \
    --data data/rkn_quad/quad1_n.npz \
    --output results/kalman_eval
```

**Multi-seed Evaluation (5 seeds):**
```bash
python scripts/run_multiseed_evaluation.py \
    --condition clean-1.5k \
    --base-output results/clean_1500 \
    --config configs/quad_image_reconstruction_config.yaml
```

### 2. Van der Pol Oscillator (Table 3, Figure 2)

```bash
# Generate data
python scripts/generate_vdp_data.py \
    --output-dir data/vdp --sig-o-list 0.05 --n-trials 1

# Run single trial
python scripts/run_vdp_experiment.py \
    --data data/vdp/sig_o_0.05/trial_001.npz \
    --config configs/vdp_config.yaml \
    --output results/vdp_trial
```

### 3. Stuart-Landau Oscillator (Table 2, Figure 3)

```bash
# Generate data
python scripts/generate_sl_data.py \
    --output-dir data/sl --var-p-list 0.05 --n-trials 1

# Run single trial
python scripts/run_sl_experiment.py \
    --data data/sl/var_p_0.05/trial_001.npz \
    --config configs/sl_config.yaml \
    --output results/sl_trial
```

## Configuration

| Config file | Experiment | Paper reference |
|-------------|-----------|-----------------|
| `quad_image_reconstruction_config.yaml` | Quad-link pendulum | Tables 5-6 |
| `vdp_config.yaml` | Van der Pol oscillator | Table 8 |
| `sl_config.yaml` | Stuart-Landau oscillator | Table 10 |

## Citation

```bibtex
@inproceedings{dse2026,
    title={Deep Spectral Learning of Embedded Latent Transfer Operators for Stochastic Dynamical Systems},
    booktitle={Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI)},
    year={2026}
}
```
