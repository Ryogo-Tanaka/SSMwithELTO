# Deep Spectral Encoder (DSE)

Reference implementation for the single-run experiments in the accompanying paper:
quad-link pendulum image reconstruction, Van der Pol mode decomposition, and
Stuart-Landau mode decomposition.

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

## Data

### Quad-link Pendulum

The quad-link pendulum image sequences are generated using the simulator from the RKN repository ([Becker et al., 2019](https://github.com/ALRhub/rkn_share)).
Please generate the data following the instructions in that repository and place the resulting `.npz` files under `data/rkn_quad/`.

### Van der Pol / Stuart-Landau Oscillators

VDP and SL data can be generated directly using the scripts in this repository:

```bash
# VDP: sweep observation noise variance
python scripts/generate_vdp_data.py \
    --output-dir data/vdp --n-trials 50

# SL: sweep process noise variance
python scripts/generate_sl_data.py \
    --output-dir data/sl --n-trials 50
```

See the paper appendix for the full list of simulation parameters.

## Quick Start

### 1. Quad-link Pendulum

```bash
python scripts/run_quadlink_experiment.py \
    --config configs/quad_image_reconstruction_config.yaml \
    --data data/rkn_quad/quad1_n.npz \
    --output results/quadlink_seed1
```

```bash
python scripts/evaluate_kalman_image_mse.py \
    --model results/quadlink_seed1/models/final_model.pth \
    --data data/rkn_quad/quad1_n.npz \
    --output results/kalman_eval
```

### 2. Van der Pol Oscillator

```bash
python scripts/generate_vdp_data.py \
    --output-dir data/vdp --sig-o-list 0.05 --n-trials 1

python scripts/run_vdp_experiment.py \
    --data data/vdp/sig_o_0.05/trial_001.npz \
    --config configs/vdp_config.yaml \
    --output results/vdp_trial
```

### 3. Stuart-Landau Oscillator

```bash
python scripts/generate_sl_data.py \
    --output-dir data/sl --var-p-list 0.05 --n-trials 1

python scripts/run_sl_experiment.py \
    --data data/sl/var_p_0.05/trial_001.npz \
    --config configs/sl_config.yaml \
    --output results/sl_trial
```

## Configuration

| Config file | Experiment |
|-------------|-----------|
| `quad_image_reconstruction_config.yaml` | Quad-link pendulum |
| `vdp_config.yaml` | Van der Pol oscillator |
| `sl_config.yaml` | Stuart-Landau oscillator |

See the paper appendix for detailed hyperparameter settings.
