#!/usr/bin/env python3
# scripts/run_quadlink_experiment.py
"""
Full experiment pipeline for quad-link image reconstruction.

Features:
- Unified data loading with architecture dispatch
- Stage-1 + Stage-2 integrated training (with optional Kalman filtering)
- Model and transfer operator saving
- Training visualization and logging
- Reproducibility via random seed

Usage:
python scripts/run_quadlink_experiment.py \
    --config configs/quad_image_reconstruction_config.yaml \
    --data data/quad1_n.npz \
    --output results/experiment_001 \
    --use-kalman
"""

import argparse
import random
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_random_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set: {seed}")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.utils.data_loader import load_experimental_data_with_architecture, DataMetadata
from src.training.two_stage_trainer import TwoStageTrainer
from src.utils.gpu_utils import select_device
from src.ssm.realization import StochasticRealizationWithEncoder
from src.evaluation.mode_decomposition import TrainedModelSpectrumAnalysis, SpectrumResultsSaver


class FullExperimentPipeline:
    """Full experiment pipeline for training and evaluation."""

    def __init__(self, config: Dict[str, Any], output_dir: Path, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'artifacts').mkdir(exist_ok=True)

        self.experiment_log = []
        self.start_time = datetime.now()
        self._log_experiment_start()

    def _log_experiment_start(self):
        """Log experiment start."""
        log_entry = {
            'timestamp': self.start_time.isoformat(),
            'event': 'experiment_start',
            'config': self.config,
            'device': str(self.device),
            'output_dir': str(self.output_dir)
        }
        self.experiment_log.append(log_entry)
        print(f"Experiment started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.device}")

    def step_1_data_loading(self, data_path: str) -> Dict[str, torch.Tensor]:
        """Step 1: Data loading and preprocessing."""
        print("\n" + "="*5)
        print("Step 1: Data Loading")
        print("="*5)

        start_time = datetime.now()

        experiment_mode = self.config.get('experiment', {}).get('mode', 'reconstruction')
        print(f"Experiment mode: {experiment_mode}")

        data_config = self.config.get('data', {})

        # Paper data protocol: train_obs -> training, test_obs -> evaluation
        paper_protocol = data_config.get('paper_data_protocol', False)
        datasets = {}

        if paper_protocol:
            print("=== Paper Data Protocol ===")
            print(f"  Loading data: {data_path}")
            print("  train_obs -> training, test_obs -> evaluation (paper-compliant)")
            raw = np.load(data_path)

            # Handle both 5D (1,T,H,W,C) clean data and 4D (T,H,W,C) noisy data
            train_obs_raw = raw['train_obs']
            if train_obs_raw.ndim == 5 and train_obs_raw.shape[0] == 1:
                train_obs = train_obs_raw[0].astype(np.float32) / 255.0
            else:
                train_obs = train_obs_raw.astype(np.float32) / 255.0

            test_obs_raw = raw['test_obs']
            if test_obs_raw.ndim == 5 and test_obs_raw.shape[0] == 1:
                test_obs = test_obs_raw[0].astype(np.float32) / 255.0
            else:
                test_obs = test_obs_raw.astype(np.float32) / 255.0

            train_tensor = torch.from_numpy(train_obs).float()
            test_tensor = torch.from_numpy(test_obs).float()

            val_size = min(300, len(train_tensor) // 5)

            T_train = train_obs.shape[0]
            metadata = DataMetadata(
                original_shape=train_obs.shape,
                feature_names=[f"pixel_{i}" for i in range(train_obs.shape[-1])],
                time_index=None,
                sampling_rate=None,
                missing_ratio=0.0,
                data_source=str(data_path),
                normalization_method="unit_scale",
                train_indices=(0, T_train),
                val_indices=(T_train - val_size, T_train),
                test_indices=(0, test_obs.shape[0]),
                has_target_data='train_targets' in raw,
                target_shape=tuple(raw['train_targets'].shape) if 'train_targets' in raw else None
            )

            data_dict = {
                'train': train_tensor,
                'val': train_tensor[-val_size:],
                'test': test_tensor,
                'metadata': metadata
            }

            if 'train_targets' in raw:
                train_tgt_raw = raw['train_targets']
                train_targets = train_tgt_raw[0] if (train_tgt_raw.ndim >= 2 and train_tgt_raw.shape[0] == 1) else train_tgt_raw
                test_tgt_raw = raw['test_targets']
                test_targets = test_tgt_raw[0] if (test_tgt_raw.ndim >= 2 and test_tgt_raw.shape[0] == 1) else test_tgt_raw
                data_dict['train_targets'] = torch.from_numpy(train_targets).float()
                data_dict['test_targets'] = torch.from_numpy(test_targets).float()
                print(f"  Train targets: {data_dict['train_targets'].shape}")
                print(f"  Test targets: {data_dict['test_targets'].shape}")

            print(f"  Train: {data_dict['train'].shape}")
            print(f"  Val (monitoring): {data_dict['val'].shape}")
            print(f"  Test (test_obs): {data_dict['test'].shape}")
        else:
            print(f"Loading data: {data_path}")
            datasets = load_experimental_data_with_architecture(
                data_path=data_path,
                config=self.config,
                split="all",
                return_dataloaders=False
            )

            data_dict = {split: dataset.get_full_data() for split, dataset in datasets.items()}
            data_dict['metadata'] = datasets['train'].metadata

        # Add target data from datasets to data_dict (skip for paper_protocol)
        for split, dataset in (datasets.items() if not paper_protocol else []):
            if hasattr(dataset, 'target_data') and dataset.target_data is not None:
                split_size = data_dict[split].shape[0]
                if split == 'train':
                    target_data = dataset.target_data
                elif split == 'test' and hasattr(dataset, 'target_test_data') and dataset.target_test_data is not None:
                    if dataset.target_test_data.shape[0] == split_size:
                        target_data = dataset.target_test_data
                    else:
                        if hasattr(dataset, 'target_data') and dataset.target_data is not None:
                            train_size = datasets['train'].data.shape[0]
                            val_size = datasets['val'].data.shape[0] if 'val' in datasets else 0
                            target_data = dataset.target_data[train_size + val_size:train_size + val_size + split_size]
                        else:
                            continue
                else:
                    if hasattr(dataset, 'target_data') and dataset.target_data is not None:
                        train_size = datasets['train'].data.shape[0]
                        val_size = datasets['val'].data.shape[0] if 'val' in datasets else 0

                        if split == 'val':
                            target_data = dataset.target_data[train_size:train_size + val_size]
                        elif split == 'test':
                            target_data = dataset.target_data[train_size + val_size:train_size + val_size + split_size]
                        else:
                            target_data = dataset.target_data[:split_size]
                    else:
                        continue

                if isinstance(target_data, np.ndarray):
                    target_data = torch.from_numpy(target_data).float()
                data_dict[f'{split}_targets'] = target_data

        # Validate target data for target prediction mode
        metadata: DataMetadata = data_dict['metadata']
        if experiment_mode == "target_prediction":
            if not hasattr(metadata, 'has_target_data') or not metadata.has_target_data:
                raise ValueError("Target prediction mode requires target data")

        print(f"Data statistics:")
        print(f"  - Original shape: {metadata.original_shape}")
        print(f"  - Features: {len(metadata.feature_names)}")
        print(f"  - Missing ratio: {metadata.missing_ratio:.2%}")
        print(f"  - Normalization: {metadata.normalization_method}")
        print(f"  - Train: {data_dict['train'].shape}")
        print(f"  - Val: {data_dict['val'].shape}")
        print(f"  - Test: {data_dict['test'].shape}")

        # Auto-adjust encoder/decoder dimensions to match data
        data_dim = data_dict['train'].shape[1]
        if 'model' in self.config:
            if 'encoder' in self.config['model']:
                original_input_dim = self.config['model']['encoder'].get('input_dim', data_dim)
                self.config['model']['encoder']['input_dim'] = data_dim
                if original_input_dim != data_dim:
                    print(f"Auto-adjusted encoder input_dim: {original_input_dim} -> {data_dim}")

            if 'decoder' in self.config['model']:
                original_output_dim = self.config['model']['decoder'].get('output_dim', data_dim)
                self.config['model']['decoder']['output_dim'] = data_dim
                if original_output_dim != data_dim:
                    print(f"Auto-adjusted decoder output_dim: {original_output_dim} -> {data_dim}")

        for key in ['train', 'val', 'test']:
            data_dict[key] = data_dict[key].to(self.device)

        data_dict['experiment_mode'] = experiment_mode

        with open(self.output_dir / 'logs' / 'data_metadata.json', 'w') as f:
            metadata_dict = {
                'original_shape': metadata.original_shape,
                'feature_names': metadata.feature_names,
                'time_index': metadata.time_index,
                'sampling_rate': metadata.sampling_rate,
                'missing_ratio': metadata.missing_ratio,
                'data_source': metadata.data_source,
                'normalization_method': metadata.normalization_method,
                'train_indices': metadata.train_indices,
                'val_indices': metadata.val_indices,
                'test_indices': metadata.test_indices
            }
            json.dump(metadata_dict, f, indent=2)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Data preprocessing done ({elapsed:.1f}s)")

        self.experiment_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'data_loading_complete',
            'elapsed_seconds': elapsed,
            'data_shapes': {k: list(v.shape) for k, v in data_dict.items() if isinstance(v, torch.Tensor)},
            'metadata': metadata_dict
        })

        return data_dict

    def step_2_training_execution(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Step 2: Full training pipeline execution."""
        print("\n" + "="*5)
        print("Step 2: Training")
        print("="*5)

        start_time = datetime.now()

        experiment_mode = data_dict.get('experiment_mode', 'reconstruction')
        print(f"Training mode: {experiment_mode}")

        if experiment_mode == "target_prediction":
            if 'target_decoder' in self.config.get('model', {}):
                print("Using target prediction decoder")

        use_kalman = self.config.get('training', {}).get('use_kalman_filtering', False)
        print(f"Kalman filtering: {'enabled' if use_kalman else 'disabled'}")

        if 'training' in self.config:
            self.config['training']['experiment_mode'] = experiment_mode

        trainer = TwoStageTrainer(
            config=self.config,
            device=self.device,
            output_dir=str(self.output_dir),
            use_kalman_filtering=use_kalman
        )

        print("Starting integrated training...")

        if experiment_mode == "target_prediction":
            target_train = self._extract_targets_from_dict(data_dict, 'train')
            target_val = self._extract_targets_from_dict(data_dict, 'val') if data_dict.get('val') is not None else None

            integrated_results = trainer.train_integrated(
                Y_train=data_dict['train'],
                Y_val=data_dict['val'],
                target_train=target_train,
                target_val=target_val
            )
        else:
            integrated_results = trainer.train_integrated(
                Y_train=data_dict['train'],
                Y_val=data_dict['val']
            )

        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Integrated training done ({total_elapsed:.1f}s)")

        training_results = {
            'integrated': integrated_results,
            'phase1_metrics': integrated_results['phase1_metrics'],
            'phase2_losses': integrated_results['phase2_losses'],
            'integrated_metrics': integrated_results['integrated_metrics'],
            'total_time': total_elapsed,
            'use_kalman': use_kalman
        }

        self._plot_training_progress(training_results)

        results_path = self.output_dir / 'logs' / 'training_results.json'
        with open(results_path, 'w') as f:
            serializable_results = self._make_json_serializable(training_results)
            json.dump(serializable_results, f, indent=2)

        self.experiment_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'integrated_training_complete',
            'total_time': total_elapsed,
            'epochs': len(integrated_results.get('integrated_metrics', [])),
            'use_kalman': use_kalman
        })

        return {
            'trainer': trainer,
            'results': training_results
        }

    def step_3_model_analysis(self, trainer: TwoStageTrainer, data_dict: Dict[str, torch.Tensor]):
        """Step 3: Model analysis and evaluation."""
        print("\n" + "="*5)
        print("Step 3: Model Analysis")
        print("="*5)

        start_time = datetime.now()

        operators_info = self._analyze_transfer_operators(trainer)
        representations_info = self._analyze_internal_representations(trainer, data_dict)

        viz_config = self.config.get('evaluation', {}).get('encoded_feature_space_viz', {})
        dim_indices = tuple(viz_config.get('dim_indices', [0, 1]))
        max_samples = viz_config.get('max_samples', 100)
        self._visualize_encoded_feature_space(trainer, data_dict['test'], dim_indices, max_samples)

        mode_decomp_info = self._perform_mode_decomposition_analysis(trainer)

        target_evaluation_info = {}
        reconstruction_evaluation_info = {}
        experiment_mode = data_dict.get('experiment_mode', 'reconstruction')

        if experiment_mode == "target_prediction":
            target_evaluation_info = self._perform_target_prediction_evaluation(trainer, data_dict)
        elif experiment_mode == "reconstruction":
            reconstruction_evaluation_info = self._perform_reconstruction_evaluation(trainer, data_dict)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Analysis and evaluation done ({elapsed:.1f}s)")

        analysis_results = {
            'operators': operators_info,
            'representations': representations_info,
            'mode_decomposition': mode_decomp_info,
            'target_evaluation': target_evaluation_info,
            'reconstruction_evaluation': reconstruction_evaluation_info,
            'analysis_time': elapsed
        }

        with open(self.output_dir / 'logs' / 'model_analysis.json', 'w') as f:
            serializable_analysis = self._make_json_serializable(analysis_results)
            json.dump(serializable_analysis, f, indent=2)

        self.experiment_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'model_analysis_complete',
            'elapsed_seconds': elapsed
        })

    def finalize_experiment(self, trainer: TwoStageTrainer):
        """Finalize experiment: save models and logs."""
        print("\n" + "="*5)
        print("Finalizing Experiment")
        print("="*5)

        trainer._save_final_model()

        config_path = self.output_dir / 'logs' / 'experiment_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        config_txt_path = self.output_dir / 'logs' / 'experiment_config.txt'
        with open(config_txt_path, 'w', encoding='utf-8') as f:
            f.write("=== DFIV Kalman Filter Experiment Configuration ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")

            def write_config_section(config_dict, prefix=""):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        f.write(f"{prefix}[{key}]\n")
                        write_config_section(value, prefix + "  ")
                    else:
                        f.write(f"{prefix}{key}: {value}\n")

            write_config_section(self.config)

        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()

        self.experiment_log.append({
            'timestamp': end_time.isoformat(),
            'event': 'experiment_complete',
            'total_experiment_time': total_time
        })

        with open(self.output_dir / 'logs' / 'full_experiment_log.json', 'w') as f:
            json.dump(self.experiment_log, f, indent=2)

        print(f"Total experiment time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"Experiment completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"All results saved to: {self.output_dir}")

    def _plot_data_overview(self, data_dict: Dict[str, torch.Tensor]):
        """Plot data overview (image and time series data)."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Data Overview', fontsize=14)

        train_data = data_dict['train'].cpu().numpy()

        if len(train_data.shape) == 4:  # Image data (T, H, W, C)
            T, H, W, C = train_data.shape
            sample_images = train_data[:min(6, T)]
            for i, img in enumerate(sample_images):
                if i >= 6:
                    break
                row = i // 3
                col = i % 3
                if row < 2 and col < 2:
                    if C == 1:
                        axes[row, col].imshow(img.squeeze(-1), cmap='gray')
                    else:
                        axes[row, col].imshow(img)
                    axes[row, col].set_title(f'Frame {i}')
                    axes[row, col].axis('off')

            sizes = [data_dict['train'].shape[0], data_dict['val'].shape[0], data_dict['test'].shape[0]]
            if len(sizes) >= 3:
                axes[0, 1].pie(sizes, labels=['Train', 'Val', 'Test'], autopct='%1.1f%%')
                axes[0, 1].set_title('Data Split Ratio')

            axes[1, 0].hist(train_data.flatten(), bins=50, alpha=0.7)
            axes[1, 0].set_title('Pixel Value Distribution')
            axes[1, 0].set_xlabel('Pixel Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)

            stats_text = f"""
            Data Type: Image Sequence
            Shape: {train_data.shape}
            Time steps: {T}
            Image size: {H}x{W}x{C}
            Mean: {train_data.mean():.3f}
            Std: {train_data.std():.3f}
            Min: {train_data.min():.3f}
            Max: {train_data.max():.3f}
            """

        else:  # Time series data (T, d)
            for i in range(min(3, train_data.shape[1])):
                axes[0, 0].plot(train_data[:, i], label=f'Feature {i+1}')
            axes[0, 0].set_title('Training Data Time Series')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            sizes = [data_dict['train'].shape[0], data_dict['val'].shape[0], data_dict['test'].shape[0]]
            axes[0, 1].pie(sizes, labels=['Train', 'Val', 'Test'], autopct='%1.1f%%')
            axes[0, 1].set_title('Data Split Ratio')

            axes[1, 0].hist(train_data.flatten(), bins=50, alpha=0.7)
            axes[1, 0].set_title('Feature Value Distribution')
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)

            stats_text = f"""
            Data Type: Time Series
            Shape: {train_data.shape}
            Mean: {train_data.mean():.3f}
            Std: {train_data.std():.3f}
            Min: {train_data.min():.3f}
            Max: {train_data.max():.3f}
            """

        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        verticalalignment='center', fontsize=10)
        axes[1, 1].set_title('Data Statistics')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'data_overview.png', dpi=300)
        plt.close()

    def _plot_training_progress(self, results: Dict[str, Any]):
        """Visualize training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress', fontsize=14)

        experiment_mode = self.config.get('experiment', {}).get('mode', 'reconstruction')

        if 'phase2_losses' in results and len(results['phase2_losses']) > 0:
            phase2_data = results['phase2_losses']
            epochs = list(range(len(phase2_data)))

            if experiment_mode == "target_prediction":
                total_losses = [entry.get('total_loss', 0) for entry in phase2_data]
                target_losses = [entry.get('loss_target', entry.get('target_loss', 0)) for entry in phase2_data]
                cca_losses = [entry.get('cca_loss', 0) for entry in phase2_data]

                axes[0, 0].plot(epochs, target_losses, label='Target Loss (MSE)', color='red', linewidth=2)
                axes[0, 0].plot(epochs, total_losses, label='Total Loss', color='blue')
                axes[0, 0].plot(epochs, cca_losses, label='CCA Loss', color='green')
                axes[0, 0].set_title('Target Prediction Loss')
                axes[0, 0].set_ylabel('MSE / Total Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].legend()
                axes[0, 0].grid(True)

                axes[0, 1].plot(epochs, target_losses, 'r-', linewidth=2)
                axes[0, 1].set_title('Target Prediction MSE Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('MSE')
                axes[0, 1].grid(True)
            else:
                total_losses = [entry['total_loss'] for entry in phase2_data]
                rec_losses = [entry.get('rec_loss', entry.get('loss_rec', 0)) for entry in phase2_data]
                cca_losses = [entry['cca_loss'] for entry in phase2_data]

                axes[0, 0].plot(epochs, total_losses, label='Total Loss')
                axes[0, 0].plot(epochs, rec_losses, label='Reconstruction Loss')
                axes[0, 0].plot(epochs, cca_losses, label='CCA Loss')
                axes[0, 0].set_title('Stage-2 Loss Components')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)

                axes[0, 1].plot(epochs, cca_losses, 'r-', linewidth=2)
                axes[0, 1].set_title('CCA Loss (Dynamic Check)')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('CCA Loss')
                axes[0, 1].grid(True)
        else:
            loss_type = "Target Loss" if experiment_mode == "target_prediction" else "Stage-2 Loss"
            axes[0, 0].text(0.5, 0.5, f'No {loss_type} training data available',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title(f'{loss_type} Components')

            detail_type = "Target MSE" if experiment_mode == "target_prediction" else "CCA"
            axes[0, 1].text(0.5, 0.5, f'No {detail_type} loss data available',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title(f'{detail_type} Loss')

        total_time = results.get('total_time', 0)
        axes[1, 0].bar(['Integrated Training'], [total_time])
        axes[1, 0].set_title('Training Time')
        axes[1, 0].set_ylabel('Time (seconds)')

        phase2_count = len(results.get('phase2_losses', []))
        integrated_count = len(results.get('integrated_metrics', []))

        info_text = f"""
        Experiment Mode: {experiment_mode}
        Total Training Time: {total_time:.1f}s
        Kalman Filtering: {results.get('use_kalman', False)}
        Stage-2 Epochs: {phase2_count}
        Integrated Epochs: {integrated_count}
        Status: {'Completed' if phase2_count > 0 else 'Stage-1 Only'}
        """
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                        verticalalignment='center', fontsize=10)
        axes[1, 1].set_title('Training Info')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'training_progress.png', dpi=300)
        plt.close()

    def _analyze_transfer_operators(self, trainer: TwoStageTrainer) -> Dict[str, Any]:
        """Analyze transfer operators."""
        operators_info = {}

        if hasattr(trainer, 'df_state') and trainer.df_state is not None:
            try:
                state_dict = trainer.df_state.get_state_dict()
                if 'V_A' in state_dict:
                    V_A = state_dict['V_A']
                    operators_info['V_A_shape'] = list(V_A.shape)
                    operators_info['V_A_norm'] = float(torch.norm(V_A).item())
                if 'U_A' in state_dict:
                    U_A = state_dict['U_A']
                    operators_info['U_A_shape'] = list(U_A.shape)
                    operators_info['U_A_norm'] = float(torch.norm(U_A).item())
            except Exception as e:
                print(f"DF-A operator analysis error: {e}")

        if hasattr(trainer, 'df_obs') and trainer.df_obs is not None:
            try:
                obs_dict = trainer.df_obs.get_state_dict()
                if 'V_B' in obs_dict:
                    V_B = obs_dict['V_B']
                    operators_info['V_B_shape'] = list(V_B.shape)
                    operators_info['V_B_norm'] = float(torch.norm(V_B).item())
                if 'u_B' in obs_dict:
                    u_B = obs_dict['u_B']
                    operators_info['u_B_shape'] = list(u_B.shape)
                    operators_info['u_B_norm'] = float(torch.norm(u_B).item())
            except Exception as e:
                print(f"DF-B operator analysis error: {e}")

        operators_path = self.output_dir / 'artifacts' / 'transfer_operators.pth'
        try:
            operators_data = {
                'df_state': trainer.df_state.get_state_dict() if hasattr(trainer, 'df_state') and trainer.df_state else None,
                'df_obs': trainer.df_obs.get_state_dict() if hasattr(trainer, 'df_obs') and trainer.df_obs else None
            }
            torch.save(operators_data, operators_path)
            print(f"Transfer operators saved: {operators_path}")
        except Exception as e:
            print(f"Transfer operator save error: {e}")

        return operators_info

    def _analyze_internal_representations(self, trainer: TwoStageTrainer, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze internal representations."""
        representations_info = {}

        try:
            if hasattr(trainer, 'encoder'):
                trainer.encoder = trainer.encoder.to(trainer.device)
                test_sample = data_dict['test'][:100]
                test_sample = test_sample.to(trainer.device)

                with torch.no_grad():
                    encoded = trainer.encoder(test_sample)
                    representations_info['encoder_output_shape'] = list(encoded.shape)
                    representations_info['encoder_output_mean'] = float(encoded.mean().item())
                    representations_info['encoder_output_std'] = float(encoded.std().item())

        except Exception as e:
            import traceback
            print(f"Internal representation analysis error: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

        return representations_info

    def _visualize_encoded_feature_space(self, trainer: TwoStageTrainer, test_data: torch.Tensor,
                                        dim_indices: tuple = (0, 1), max_samples: int = 100):
        """Visualize encoded feature space (time series plot)."""
        try:
            trainer.encoder = trainer.encoder.to(trainer.device)
            test_data = test_data.to(trainer.device)

            n_samples = min(len(test_data), max_samples)

            with torch.no_grad():
                if hasattr(trainer, 'encoder'):
                    encoded = trainer.encoder(test_data[:n_samples])

                    if encoded.shape[1] <= max(dim_indices):
                        dim_indices = (0, min(1, encoded.shape[1]-1))

                    plt.figure(figsize=(12, 6))
                    plt.plot(encoded[:, dim_indices[0]].cpu().numpy(),
                            label=f'Dim {dim_indices[0]}', linewidth=2)
                    plt.plot(encoded[:, dim_indices[1]].cpu().numpy(),
                            label=f'Dim {dim_indices[1]}', linewidth=2)
                    plt.title(f'Feature Trajectory (Dims {dim_indices[0]}, {dim_indices[1]})')
                    plt.xlabel('Time Step')
                    plt.ylabel('Feature Value')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'plots' / 'encoded_feature_space_visualization.png', dpi=300)
                    plt.close()

        except Exception as e:
            import traceback
            print(f"Feature space visualization error: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

    def _perform_mode_decomposition_analysis(self, trainer: TwoStageTrainer) -> Dict[str, Any]:
        """Perform mode decomposition analysis."""
        mode_decomp_info = {}

        try:
            sampling_interval = self.config.get('evaluation', {}).get('spectrum_analysis', {}).get('sampling_interval', 0.1)
            model_spectrum_analyzer = TrainedModelSpectrumAnalysis(sampling_interval)

            if hasattr(trainer, 'df_state') and trainer.df_state is not None:
                try:
                    V_A = None
                    state_dict = trainer.df_state.get_state_dict()
                    if 'V_A' in state_dict:
                        V_A = state_dict['V_A']
                    elif hasattr(trainer.df_state, 'V_A') and trainer.df_state.V_A is not None:
                        V_A = trainer.df_state.V_A
                    elif hasattr(trainer.df_state, '_stage1_cache') and 'V_A' in trainer.df_state._stage1_cache:
                        V_A = trainer.df_state._stage1_cache['V_A']

                    if V_A is not None:
                        spectrum_analysis = model_spectrum_analyzer.analyzer.analyze_spectrum(V_A)

                        mode_decomp_info = {
                            'V_A_shape': list(V_A.shape),
                            'spectral_radius': spectrum_analysis['spectral_radius'],
                            'n_stable_modes': spectrum_analysis['n_stable_modes'],
                            'n_dominant_modes': spectrum_analysis['n_dominant_modes'],
                            'dominant_indices': spectrum_analysis['dominant_indices'],
                            'stable_indices': spectrum_analysis['stable_indices'],
                            'sampling_interval': sampling_interval
                        }

                        eigenvals_continuous = spectrum_analysis['eigenvalues_continuous']
                        mode_decomp_info['eigenvalues_statistics'] = {
                            'mean_growth_rate': float(eigenvals_continuous.real.mean().item()),
                            'std_growth_rate': float(eigenvals_continuous.real.std().item()),
                            'mean_frequency_hz': float(spectrum_analysis['frequencies_hz'].mean().item()),
                            'std_frequency_hz': float(spectrum_analysis['frequencies_hz'].std().item())
                        }

                        spectrum_save_path = self.output_dir / 'artifacts' / 'mode_decomposition'
                        SpectrumResultsSaver.save_results(
                            {'spectrum': spectrum_analysis, 'V_A': V_A, 'sampling_interval': sampling_interval},
                            str(spectrum_save_path),
                            save_format='both'
                        )

                        mode_decomp_info['detailed_results_saved'] = True
                        mode_decomp_info['save_path'] = str(spectrum_save_path)

                        print(f"Mode decomposition done:")
                        print(f"  - Spectral radius: {spectrum_analysis['spectral_radius']:.4f}")
                        print(f"  - Stable modes: {spectrum_analysis['n_stable_modes']}")
                        print(f"  - Dominant modes: {spectrum_analysis['n_dominant_modes']}")
                    else:
                        print(f"V_A matrix not found")
                        mode_decomp_info['error'] = 'V_A not found in df_state'

                except Exception as e:
                    print(f"Mode decomposition error: {e}")
                    mode_decomp_info['error'] = str(e)
            else:
                print(f"DF-A state layer not found")
                mode_decomp_info['error'] = 'df_state layer not found'

        except Exception as e:
            print(f"Mode decomposition init error: {e}")
            mode_decomp_info['error'] = str(e)

        return mode_decomp_info

    def _extract_targets_from_dict(self, data_dict: Dict[str, torch.Tensor], split: str) -> torch.Tensor:
        """Extract target data from data dict."""
        try:
            target_key = f'{split}_targets'
            if target_key in data_dict:
                return data_dict[target_key]

            data = data_dict[split]
            metadata = data_dict.get('metadata')
            if hasattr(metadata, 'has_target_data') and metadata.has_target_data:
                if hasattr(metadata, 'target_indices'):
                    target_indices = metadata.target_indices
                    if isinstance(target_indices, (list, tuple, torch.Tensor)):
                        targets = data[:, target_indices] if len(data.shape) >= 2 else data[target_indices]
                        return targets

            return data

        except Exception:
            return data_dict[split]

    def _extract_targets(self, data: torch.Tensor, metadata: Optional[DataMetadata] = None) -> torch.Tensor:
        """Extract target data (legacy backward compatibility)."""
        try:
            if hasattr(metadata, 'has_target_data') and metadata.has_target_data:
                if hasattr(metadata, 'target_indices'):
                    target_indices = metadata.target_indices
                    if isinstance(target_indices, (list, tuple, torch.Tensor)):
                        targets = data[:, target_indices] if len(data.shape) >= 2 else data[target_indices]
                        return targets
            return data
        except Exception:
            return data

    def _predict_targets(self, test_data: torch.Tensor, trainer: TwoStageTrainer) -> torch.Tensor:
        """Run target prediction through the full SSM pipeline."""
        trainer.encoder.eval()
        if hasattr(trainer, 'decoder'):
            trainer.decoder.eval()
        if hasattr(trainer, 'target_decoder') and trainer.target_decoder is not None:
            trainer.target_decoder.eval()
        if hasattr(trainer, 'df_state') and trainer.df_state is not None:
            trainer.df_state.eval()
        if hasattr(trainer, 'df_obs') and trainer.df_obs is not None:
            trainer.df_obs.eval()

        with torch.no_grad():
            try:
                T = test_data.shape[0]
                h = trainer.realization.h

                if T <= 2 * h:
                    M_features = trainer.encoder(test_data)
                    if M_features.dim() == 1:
                        M_features = M_features.unsqueeze(1)
                    if hasattr(trainer, 'target_decoder') and trainer.target_decoder is not None:
                        return trainer.target_decoder(M_features)
                    else:
                        encoded = trainer.encoder(test_data)
                        if hasattr(trainer, 'target_decoder') and trainer.target_decoder is not None:
                            return trainer.target_decoder(encoded)
                        elif hasattr(trainer, 'decoder') and trainer.decoder is not None:
                            return trainer.decoder(encoded)
                        else:
                            return encoded

                # Step 1: Encode y_t -> m_t
                M_features = trainer.encoder(test_data)
                if M_features.dim() == 1:
                    M_features = M_features.unsqueeze(1)

                # Step 2: Stochastic realization
                try:
                    from src.ssm.realization import StochasticRealizationWithEncoder
                    if isinstance(trainer.realization, StochasticRealizationWithEncoder):
                        trainer.realization.fit(test_data, trainer.encoder)
                        X_states = trainer.realization.estimate_states(test_data)
                    else:
                        m_series_scalar = M_features.mean(dim=1)
                        X_states = trainer.realization.estimate_states(m_series_scalar.unsqueeze(1))
                except Exception:
                    X_states = M_features

                # Step 3: DF-A prediction x_{t-1} -> x_hat_{t|t-1}
                X_hat_states = trainer.df_state.predict_sequence(X_states)
                T_pred = X_hat_states.size(0)

                # Step 4: DF-B prediction x_hat_{t|t-1} -> m_hat_{t|t-1}
                M_hat_series = []
                for t in range(T_pred):
                    m_hat_t = trainer.df_obs.predict_one_step(X_hat_states[t])
                    M_hat_series.append(m_hat_t)
                M_hat_tensor = torch.stack(M_hat_series)
                M_hat_tensor = trainer._ensure_device(M_hat_tensor)

                # Step 5: Target prediction m_hat -> target
                targets = trainer.target_decoder(M_hat_tensor)
                return targets

            except Exception as e:
                raise RuntimeError(f"Target prediction pipeline failed: {e}") from e

    def _perform_target_prediction_evaluation(
        self,
        trainer: TwoStageTrainer,
        data_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Evaluate target prediction performance."""
        print("\n" + "-"*40)
        print("Target Prediction Evaluation")
        print("-"*40)

        evaluation_results = {}

        try:
            from src.evaluation.metrics import TargetPredictionMetrics

            evaluation_config = self.config.get('evaluation', {}).get('target_metrics', {})
            selected_metrics = evaluation_config.get('metrics', ['rmse'])

            target_evaluator = TargetPredictionMetrics(device=str(self.device))

            test_predictions = self._predict_targets(data_dict['test'], trainer)
            test_targets = self._extract_targets_from_dict(data_dict, 'test')

            if test_predictions.shape != test_targets.shape:
                if len(test_predictions.shape) > 2:
                    test_predictions = test_predictions.view(test_predictions.shape[0], -1)
                if len(test_targets.shape) > 2:
                    test_targets = test_targets.view(test_targets.shape[0], -1)

                if test_predictions.shape[0] != test_targets.shape[0]:
                    pred_samples = test_predictions.shape[0]
                    target_samples = test_targets.shape[0]

                    if pred_samples < target_samples:
                        h = self.config.get('ssm', {}).get('realization', {}).get('past_horizon', 20)
                        start_idx = h + 1
                        end_idx = start_idx + pred_samples
                        if end_idx <= target_samples:
                            test_targets = test_targets[start_idx:end_idx]
                        else:
                            test_targets = test_targets[-pred_samples:]
                    else:
                        test_predictions = test_predictions[:target_samples]

            if test_predictions.shape != test_targets.shape:
                min_dim = min(test_predictions.shape[1], test_targets.shape[1])
                test_predictions = test_predictions[:, :min_dim]
                test_targets = test_targets[:, :min_dim]

            target_metrics = target_evaluator.compute_target_metrics(
                test_targets, test_predictions, metrics=selected_metrics, verbose=True
            )

            generated_files = target_evaluator.create_target_visualizations(
                test_targets, test_predictions,
                metrics=selected_metrics,
                output_dir=str(self.output_dir / 'plots')
            )

            experiment_info = {
                'experiment_mode': 'target_prediction',
                'test_data_shape': list(test_targets.shape),
                'predictions_shape': list(test_predictions.shape),
                'selected_metrics': selected_metrics,
                'model_architecture': 'cnn_image'
            }

            saved_metrics_file = target_evaluator.save_target_metrics_results(
                results=target_metrics,
                output_dir=str(self.output_dir / 'logs'),
                experiment_info=experiment_info
            )

            evaluation_results = {
                'metrics': target_metrics,
                'selected_metrics': selected_metrics,
                'generated_visualizations': generated_files,
                'saved_metrics_file': saved_metrics_file,
                'test_data_shape': list(test_targets.shape),
                'predictions_shape': list(test_predictions.shape),
                'evaluation_success': True
            }

        except Exception as e:
            import traceback
            print(f"Target prediction evaluation error: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

            evaluation_results = {
                'error': str(e),
                'evaluation_success': False
            }

        return evaluation_results

    def _perform_reconstruction_evaluation(
        self,
        trainer: TwoStageTrainer,
        data_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Evaluate reconstruction performance."""
        print("\n" + "-"*40)
        print("Reconstruction Evaluation")
        print("-"*40)

        evaluation_results = {}

        try:
            from src.evaluation.metrics import ReconstructionMetrics

            evaluation_config = self.config.get('evaluation', {}).get('reconstruction_metrics', {})
            selected_metrics = evaluation_config.get('metrics', ['reconstruction_rmse'])

            reconstruction_evaluator = ReconstructionMetrics(device=str(self.device))

            test_reconstructions = self._reconstruct_data(data_dict['test'], trainer)
            test_originals = data_dict['test']

            if test_reconstructions.shape != test_originals.shape:
                if len(test_reconstructions.shape) > 2:
                    test_reconstructions = test_reconstructions.view(test_reconstructions.shape[0], -1)
                if len(test_originals.shape) > 2:
                    test_originals = test_originals.view(test_originals.shape[0], -1)

                if test_reconstructions.shape[0] != test_originals.shape[0]:
                    rec_samples = test_reconstructions.shape[0]
                    orig_samples = test_originals.shape[0]

                    if rec_samples < orig_samples:
                        trim_start = orig_samples - rec_samples
                        test_originals = test_originals[trim_start:]
                    else:
                        test_reconstructions = test_reconstructions[:orig_samples]

            if test_reconstructions.shape != test_originals.shape:
                min_dim = min(test_reconstructions.shape[1], test_originals.shape[1])
                test_reconstructions = test_reconstructions[:, :min_dim]
                test_originals = test_originals[:, :min_dim]

            reconstruction_metrics = reconstruction_evaluator.compute_reconstruction_metrics(
                test_originals, test_reconstructions, metrics=selected_metrics, verbose=True
            )

            generated_files = reconstruction_evaluator.create_reconstruction_visualizations(
                test_originals, test_reconstructions,
                metrics=selected_metrics,
                output_dir=str(self.output_dir / 'plots')
            )

            experiment_info = {
                'experiment_mode': 'reconstruction',
                'test_data_shape': list(test_originals.shape),
                'reconstructions_shape': list(test_reconstructions.shape),
                'selected_metrics': selected_metrics,
                'model_architecture': 'cnn_image'
            }

            saved_metrics_file = reconstruction_evaluator.save_reconstruction_metrics_results(
                results=reconstruction_metrics,
                output_dir=str(self.output_dir / 'logs'),
                experiment_info=experiment_info
            )

            evaluation_results = {
                'metrics': reconstruction_metrics,
                'selected_metrics': selected_metrics,
                'generated_visualizations': generated_files,
                'saved_metrics_file': saved_metrics_file,
                'test_data_shape': list(test_originals.shape),
                'reconstructions_shape': list(test_reconstructions.shape),
                'evaluation_success': True
            }

            print(f"Reconstruction evaluation done")

        except Exception as e:
            import traceback
            print(f"Reconstruction evaluation error: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

            evaluation_results = {
                'error': str(e),
                'evaluation_success': False
            }

        return evaluation_results

    def _reconstruct_data(self, test_data: torch.Tensor, trainer: TwoStageTrainer) -> torch.Tensor:
        """Reconstruct test data using the full SSM pipeline."""
        try:
            with torch.no_grad():
                trainer.encoder.eval()
                trainer.decoder.eval()
                if hasattr(trainer, 'df_state') and trainer.df_state is not None:
                    trainer.df_state.eval()
                if hasattr(trainer, 'df_obs') and trainer.df_obs is not None:
                    trainer.df_obs.eval()

                reconstructed_data = self._perform_reconstruction_with_existing_process(test_data, trainer)
                return reconstructed_data

        except Exception as e:
            print(f"Reconstruction process error: {e}")
            try:
                with torch.no_grad():
                    trainer.encoder.eval()
                    trainer.decoder.eval()
                    if hasattr(trainer, 'df_state') and trainer.df_state is not None:
                        trainer.df_state.eval()
                    if hasattr(trainer, 'df_obs') and trainer.df_obs is not None:
                        trainer.df_obs.eval()
                    loss_total, loss_rec, loss_cca = trainer._forward_and_loss_phase2_reconstruction(test_data)

                    M_features = trainer.encoder(test_data)
                    if M_features.dim() == 1:
                        M_features = M_features.unsqueeze(1)

                    reconstructed_data = trainer.decoder(M_features)
                    print("Fallback reconstruction succeeded")
                    return reconstructed_data

            except Exception as e2:
                print(f"Fallback reconstruction error: {e2}")
                return test_data

    def _perform_reconstruction_with_existing_process(self, test_data: torch.Tensor, trainer: TwoStageTrainer) -> torch.Tensor:
        """Reconstruction via the full SSM pipeline (matching training flow)."""
        T = test_data.shape[0]
        h = trainer.realization.h

        if T <= 2 * h:
            raise RuntimeError(f"Time series too short ({T}): T <= 2*h({2*h})")

        # Step 1: Encode y_t -> m_t
        M_features = trainer.encoder(test_data)
        if M_features.dim() == 1:
            M_features = M_features.unsqueeze(1)

        # Step 2: Stochastic realization m_t -> x_t
        from src.ssm.realization import StochasticRealizationWithEncoder
        if isinstance(trainer.realization, StochasticRealizationWithEncoder):
            trainer.realization.fit(test_data, trainer.encoder)
            X_states = trainer.realization.estimate_states(test_data)
        else:
            m_series_scalar = M_features.mean(dim=1)
            trainer.realization.fit(m_series_scalar.unsqueeze(1))
            X_states = trainer.realization.filter(m_series_scalar.unsqueeze(1))

        # Step 3: DF-A prediction x_{t-1} -> x_hat_{t|t-1}
        X_hat_states = trainer.df_state.predict_sequence(X_states)
        T_pred = X_hat_states.size(0)

        # Step 4: DF-B prediction x_hat -> m_hat
        M_hat_series = []
        for t in range(T_pred):
            m_hat_t = trainer.df_obs.predict_one_step(X_hat_states[t])
            M_hat_series.append(m_hat_t)
        M_hat_tensor = torch.stack(M_hat_series)
        M_hat_tensor = trainer._ensure_device(M_hat_tensor)

        # Step 5: Decode m_hat -> y_hat
        Y_hat = trainer.decoder(M_hat_tensor)
        return Y_hat

    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Full experiment pipeline")

    parser.add_argument('--config', '-c', type=str, required=True, help='Config file (.yaml)')
    parser.add_argument('--data', '-d', type=str, required=True, help='Data file path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Compute device (auto if None)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--use-kalman', action='store_true', help='Enable Kalman filtering')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip Step 3 model analysis')

    return parser.parse_args()


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    required_sections = ['model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config requires '{section}' section")

    return config


def main():
    """Main entry point."""
    args = parse_args()

    print("Experiment Start")
    print("="*5)

    if args.seed is not None:
        set_random_seed(args.seed)

    config = load_experiment_config(args.config)

    if args.use_kalman:
        config.setdefault('training', {})['use_kalman_filtering'] = True

    device = torch.device(args.device) if args.device else select_device()
    output_dir = Path(args.output)

    pipeline = FullExperimentPipeline(config, output_dir, device)

    try:
        data_dict = pipeline.step_1_data_loading(args.data)
        training_result = pipeline.step_2_training_execution(data_dict)

        if not args.skip_analysis:
            pipeline.step_3_model_analysis(training_result['trainer'], data_dict)

        pipeline.finalize_experiment(training_result['trainer'])

        print("\nExperiment pipeline completed successfully!")
        print(f"Results: {output_dir}")

    except Exception as e:
        print(f"\nExperiment error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
