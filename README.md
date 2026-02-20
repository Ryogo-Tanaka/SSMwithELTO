# Deep Spectral Encoder (DSE)

Code for the three experiments in **"Deep Spectral Learning of Embedded Latent Transfer Operators for Stochastic Dynamical Systems"** (UAI 2026): quad-link pendulum, Van der Pol oscillator, and Stuart-Landau oscillator.

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Quad-link Pendulum (Table 1)

**Training:**
```bash
python scripts/run_quadlink_experiment.py \
    --config configs/quad_image_reconstruction_config.yaml \
    --data data/rkn_quad/quad1_n.npz \
    --output results/quadlink_seed1 
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
