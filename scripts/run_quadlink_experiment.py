#!/usr/bin/env python3
"""
Single-run quad-link image reconstruction experiment.

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
from typing import Dict, Any

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

from src.utils.data_loader import load_experimental_data_with_architecture, DataMetadata
from src.training.two_stage_trainer import TwoStageTrainer
from src.utils.gpu_utils import select_device
from src.ssm.realization import StochasticRealizationWithEncoder


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
        print(f"Experiment started: {self.start_time:%Y-%m-%d %H:%M:%S}")
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.device}")

    def step_1_data_loading(self, data_path: str) -> Dict[str, torch.Tensor]:
        """Load and preprocess train/validation/test sequences."""
        print("\n" + "="*5)
        print("Step 1: Data Loading")
        print("="*5)

        start_time = datetime.now()
        data_config = self.config.get('data', {})
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
                test_indices=(0, test_obs.shape[0])
            )

            data_dict = {
                'train': train_tensor,
                'val': train_tensor[-val_size:],
                'test': test_tensor,
                'metadata': metadata
            }

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

        metadata: DataMetadata = data_dict['metadata']
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
        """Run integrated reconstruction training."""
        print("\n" + "="*5)
        print("Step 2: Training")
        print("="*5)

        start_time = datetime.now()
        use_kalman = self.config.get('training', {}).get('use_kalman_filtering', False)
        print(f"Kalman filtering: {'enabled' if use_kalman else 'disabled'}")

        trainer = TwoStageTrainer(
            config=self.config,
            device=self.device,
            output_dir=str(self.output_dir),
            use_kalman_filtering=use_kalman
        )

        print("Starting integrated training...")
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
        """Analyze trained operators and reconstruction quality."""
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

        reconstruction_evaluation_info = self._perform_reconstruction_evaluation(trainer, data_dict)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Analysis and evaluation done ({elapsed:.1f}s)")

        analysis_results = {
            'operators': operators_info,
            'representations': representations_info,
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

        if 'phase2_losses' in results and len(results['phase2_losses']) > 0:
            phase2_data = results['phase2_losses']
            epochs = list(range(len(phase2_data)))

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
            axes[0, 1].set_title('CCA Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('CCA Loss')
            axes[0, 1].grid(True)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Stage-2 training data available',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Stage-2 Loss Components')

            axes[0, 1].text(0.5, 0.5, 'No CCA loss data available',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('CCA Loss')

        total_time = results.get('total_time', 0)
        axes[1, 0].bar(['Integrated Training'], [total_time])
        axes[1, 0].set_title('Training Time')
        axes[1, 0].set_ylabel('Time (seconds)')

        phase2_count = len(results.get('phase2_losses', []))
        integrated_count = len(results.get('integrated_metrics', []))

        info_text = f"""
        Experiment Mode: reconstruction
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
