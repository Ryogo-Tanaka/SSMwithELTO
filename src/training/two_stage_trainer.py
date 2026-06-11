"""
TwoStageTrainer: Two-stage training strategy implementation.

Stage-1: DF-A/DF-B Stage-1/Stage-2 alternating training
Stage-2: End-to-end fine-tuning

Training strategy:
**DF-A (State Layer)**:
for epoch in warmup_epochs:
  for t = 1 to T1:  # Stage-1
    V_A^{(-k)} = closed_form(Phi_minus, Phi_plus, phi_theta fixed)
    phi_theta <- phi_theta - alpha * grad_L1(V_A^{(-k)}, phi_theta)

  for t = 1 to T2:  # Stage-2
    U_A = closed_form(H^{(cf)}_A, X_+)  # U_A update (closed-form only)

**DF-B (Observation Layer)**:
for epoch in warmup_epochs:
  for t = 1 to T1:  # Stage-1
    V_B = closed_form(Phi_prev, Psi_curr)  # V_B (psi_omega fixed)
    phi_theta <- phi_theta - alpha * grad_L1(V_B, phi_theta)

  for t = 1 to T2:  # Stage-2
    U_B = closed_form(H^{(cf)}_B, M)    # U_B (phi_theta fixed)
    psi_omega <- psi_omega - alpha * grad_L2(U_B, psi_omega)

Stage-2: End-to-end fine-tuning
for epoch in active_epochs:
  # Fixed inference path
  x_hat_{t|t-1} = U_A^T V_A phi_theta(x_{t-1})
  m_hat_{t|t-1} = U_B^T V_B phi_theta(x_hat_{t|t-1})
  y_hat_{t|t-1} = g_alpha(m_hat_{t|t-1})

  # Loss
  L_total = L_rec + lambda_c * L_cca

  # Selective update
  (u_eta, g_alpha, phi_theta, psi_omega).backward()
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import warnings
from pathlib import Path
import json
import csv
from dataclasses import dataclass
from enum import Enum
import gc

from ..ssm.df_state_layer import DFStateLayer
from ..ssm.df_observation_layer import DFObservationLayer
from ..ssm.realization import Realization, StochasticRealizationWithEncoder, RealizationError


class TrainingPhase(Enum):
    """Training phase definitions."""
    PHASE1_DF_A = "phase1_df_a"
    PHASE1_DF_B = "phase1_df_b"
    PHASE2_E2E = "phase2_e2e"


@dataclass
class TrainingConfig:
    """Structured training configuration."""
    epochs: int = 100
    T1_iterations: int = 10
    T2_iterations: int = 5
    phase1_warmup_epochs: int = 5
    lambda_cca: float = 0.001
    update_strategy: str = "encoder_decoder_only"
    lr_phi: float = 1e-3
    lr_psi: float = 1e-3
    lr_encoder: float = 1e-3
    lr_decoder: float = 1e-3
    log_interval: int = 5
    save_interval: int = 10
    verbose: bool = True
    
    def __post_init__(self):
        """Post-init type conversion and validation."""
        # Type coercion for values loaded from YAML
        self.epochs = int(self.epochs)
        self.T1_iterations = int(self.T1_iterations)
        self.T2_iterations = int(self.T2_iterations)
        self.phase1_warmup_epochs = int(self.phase1_warmup_epochs)
        self.log_interval = int(self.log_interval)
        self.save_interval = int(self.save_interval)
        
        self.lr_phi = float(self.lr_phi)
        self.lr_psi = float(self.lr_psi)
        self.lr_encoder = float(self.lr_encoder)
        self.lr_decoder = float(self.lr_decoder)
        self.lambda_cca = float(self.lambda_cca)
        
        self.update_strategy = str(self.update_strategy)

        # Handle string "true"/"false" from YAML
        if isinstance(self.verbose, str):
            self.verbose = self.verbose.lower() in ('true', '1', 'yes', 'on')
        else:
            self.verbose = bool(self.verbose)

    @classmethod
    def from_nested_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create flat TrainingConfig from nested config dict."""
        return cls(
            epochs=config_dict.get('epochs', 100),
            T1_iterations=config_dict.get('T1_iterations', 10),
            T2_iterations=config_dict.get('T2_iterations', 5),
            phase1_warmup_epochs=config_dict.get('phase1_warmup_epochs', 5),
            lr_phi=config_dict.get('lr_phi', 1e-3),
            lr_psi=config_dict.get('lr_psi', 1e-3),
            lr_encoder=config_dict.get('lr_encoder', 1e-3),
            lr_decoder=config_dict.get('lr_decoder', 1e-3),
            lambda_cca=config_dict.get('lambda_cca', 0.001),
            update_strategy=config_dict.get('update_strategy', "all"),
            log_interval=config_dict.get('log_interval', 5),
            save_interval=config_dict.get('checkpoint', {}).get('save_every', 10),
            verbose=config_dict.get('verbose', True)
        )


class TrainingLogger:
    """Training log manager."""
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.phase1_csv_path = self.output_dir / 'phase1_training.csv'
        self.phase2_csv_path = self.output_dir / 'phase2_training.csv'
        self.canonical_correlations_csv_path = self.output_dir / 'canonical_correlations.csv'

        self.phase1_logs = []
        self.phase2_logs = []
        self.canonical_correlations_logs = []

        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Initialize CSV file headers."""
        with open(self.phase1_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'stage', 'iteration', 'loss', 
                'lr_phi', 'lr_psi'
            ])

        with open(self.phase2_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'total_loss', 'rec_loss', 'cca_loss',
                'lr_encoder', 'lr_decoder', 'lr_phi', 'lr_psi'
            ])

        with open(self.canonical_correlations_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'num_components', 'rho_sum', 'rho_min', 'rho_max', 'rho_mean',
                'rho_values'
            ])
    
    def log_phase1(self, epoch: int, phase: TrainingPhase, stage: str, 
                   iteration: int, metrics: Dict[str, float], 
                   learning_rates: Dict[str, float]):
        """Record Stage-1 log entry."""
        log_entry = {
            'epoch': epoch,
            'phase': phase.value,
            'stage': stage,
            'iteration': iteration,
            'metrics': metrics,
            'learning_rates': learning_rates
        }
        
        self.phase1_logs.append(log_entry)

        with open(self.phase1_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            loss_value = metrics.get('stage1_loss') or metrics.get('stage2_loss', '')
            writer.writerow([
                epoch, phase.value, stage, iteration, loss_value,
                learning_rates.get('lr_phi', ''),
                learning_rates.get('lr_psi', '')
            ])

    def log_phase2(self, epoch: int, total_loss: float, rec_loss: float,
                   cca_loss: float, learning_rates: Dict[str, float]):
        """Record Stage-2 log entry."""
        log_entry = {
            'epoch': epoch,
            'total_loss': total_loss,
            'rec_loss': rec_loss,
            'cca_loss': cca_loss,
            'learning_rates': learning_rates
        }

        self.phase2_logs.append(log_entry)

        with open(self.phase2_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, total_loss, rec_loss, cca_loss,
                learning_rates.get('lr_encoder', ''),
                learning_rates.get('lr_decoder', ''),
                learning_rates.get('lr_phi', ''),
                learning_rates.get('lr_psi', '')
            ])

    def log_canonical_correlations(self, epoch: int, phase: str, canonical_correlations: torch.Tensor):
        """Record canonical correlations detail log."""
        import json

        rho_values = canonical_correlations.detach().cpu().numpy()
        rho_sum = float(rho_values.sum())
        rho_min = float(rho_values.min())
        rho_max = float(rho_values.max())
        rho_mean = float(rho_values.mean())
        num_components = len(rho_values)

        log_entry = {
            'epoch': epoch,
            'phase': phase,
            'num_components': num_components,
            'rho_sum': rho_sum,
            'rho_min': rho_min,
            'rho_max': rho_max,
            'rho_mean': rho_mean,
            'rho_values': rho_values.tolist()
        }

        self.canonical_correlations_logs.append(log_entry)

        with open(self.canonical_correlations_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, phase, num_components, rho_sum, rho_min, rho_max, rho_mean,
                json.dumps(rho_values.tolist())
            ])

    def save_summary(self):
        """Save training summary as JSON."""
        summary = {
            'phase1_summary': {
                'total_epochs': len(set(log['epoch'] for log in self.phase1_logs)),
                'total_iterations': len(self.phase1_logs),
                'final_metrics': self.phase1_logs[-1]['metrics'] if self.phase1_logs else {}
            },
            'phase2_summary': {
                'total_epochs': len(self.phase2_logs),
                'final_loss': self.phase2_logs[-1]['total_loss'] if self.phase2_logs else None
            }
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


class TwoStageTrainer:
    """
    Main class for the proposed two-stage training strategy.
    
    1. Stage-1: DF-A/DF-B cooperative training
    2. Stage-2: End-to-end fine-tuning
    3. Detailed logging and visualization
    4. Time index alignment and memory efficiency
    """
    
    def __init__(self, encoder: nn.Module = None, decoder: nn.Module = None, realization: Realization = None,
                 df_state_config: Dict[str, Any] = None, df_obs_config: Dict[str, Any] = None,
                 training_config: TrainingConfig = None, device: torch.device = None, output_dir: str = None,
                 use_kalman_filtering: bool = True,
                calibration_ratio: float = 0.1,
                auto_inference_setup: bool = True,
                config: Dict[str, Any] = None):
        
        if config is not None:
            self._init_from_config(config, device, output_dir, use_kalman_filtering)
        else:
            self._init_from_args(encoder, decoder, realization, df_state_config, df_obs_config,
                               training_config, device, output_dir, use_kalman_filtering,
                               calibration_ratio, auto_inference_setup, None)
    
    def _init_from_config(self, config: Dict[str, Any], device: torch.device, output_dir: str,
                         use_kalman_filtering: bool):
        """Initialize from config dictionary."""
        self.yaml_config = config  # used by _save_final_model()

        from ..models.encoder import build_encoder
        from ..models.decoder import build_decoder

        encoder_config = config['model']['encoder'].copy()
        decoder_config = config['model']['decoder'].copy()

        if 'type' not in encoder_config:
            encoder_config['type'] = 'time_invariant'

        if encoder_config['type'] == 'time_invariant':
            if 'output_dim' not in encoder_config:
                encoder_config['output_dim'] = encoder_config.get('channels', 32)

        encoder = build_encoder(encoder_config)
        decoder = build_decoder(decoder_config)

        realization_config = config['ssm']['realization']
        if config.get('evaluation', {}).get('use_new_realization', True):
            realization_config_copy = realization_config.copy()
            feature_mapping_cfg = realization_config.get('feature_mapping', {})

            realization = StochasticRealizationWithEncoder(
                encoder=encoder,
                encoder_output_dim=realization_config_copy['encoder_output_dim'],
                past_horizon=realization_config_copy.get('past_horizon', 10),
                rank=realization_config_copy.get('rank', 8),
                ridge_param=realization_config_copy.get('ridge_param', 1e-3),
                jitter=realization_config_copy.get('jitter', 1e-8),
                device=str(device),
                feature_mapping_type=feature_mapping_cfg.get('type', 'averaging'),
                feature_mapping_hidden_dims=feature_mapping_cfg.get('hidden_dims', None),
                feature_mapping_activation=feature_mapping_cfg.get('activation', 'relu')
            )
        else:
            realization = Realization(**realization_config)

        training_config = TrainingConfig.from_nested_dict(config['training'])
        calibration_ratio = config['training'].get('calibration_ratio', 0.25)
        auto_inference_setup = config['training'].get('auto_inference_setup', True)

        self._init_from_args(encoder, decoder, realization,
                           config['ssm']['df_state'], config['ssm']['df_observation'],
                           training_config, device, output_dir, use_kalman_filtering,
                           calibration_ratio, auto_inference_setup)

    @classmethod
    def from_trained_model(cls, model_path: str, config_path: str = None,
                          device: torch.device = None, output_dir: str = None) -> 'TwoStageTrainer':
        """
        Create inference-only instance from trained model.

        Args:
            model_path: Path to trained model
            config_path: Training config file path (YAML)
                        If None, attempts to restore from checkpoint['config']
            device: Device
            output_dir: Output directory

        Returns:
            TwoStageTrainer: Inference-only instance
        """
        import yaml

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if output_dir is None:
            output_dir = 'temp_inference'

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Normalize checkpoint structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Prefer config_path; fall back to checkpoint['config']
        if config_path is not None:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            encoder_type = yaml_config.get('model', {}).get('encoder', {}).get('type')
            decoder_type = yaml_config.get('model', {}).get('decoder', {}).get('type')
            encoder_config = yaml_config.get('model', {}).get('encoder', {})
            decoder_config = yaml_config.get('model', {}).get('decoder', {})

            print(f"Loaded from YAML config: encoder={encoder_type}, decoder={decoder_type}")

            checkpoint_config = checkpoint.get('config', {})
            if checkpoint_config:
                checkpoint_encoder_type = checkpoint_config.get('model', {}).get('encoder', {}).get('type')
                if checkpoint_encoder_type and checkpoint_encoder_type != encoder_type:
                    raise ValueError(
                        f"Encoder type mismatch!\n"
                        f"  YAML config: '{encoder_type}'\n"
                        f"  checkpoint['config']: '{checkpoint_encoder_type}'\n"
                        f"Config file and checkpoint do not match. Please verify the correct pair."
                    )

        else:
            checkpoint_config = checkpoint.get('config', {})

            if not checkpoint_config:
                raise ValueError(
                    "Config not found!\n"
                    "  config_path not specified and checkpoint['config'] does not exist.\n"
                    "  Fix:\n"
                    "    1. Pass config_path argument to from_trained_model()\n"
                    "    2. Or retrain with complete checkpoint['config']"
                )

            encoder_type = checkpoint_config.get('model', {}).get('encoder', {}).get('type')
            decoder_type = checkpoint_config.get('model', {}).get('decoder', {}).get('type')
            encoder_config = checkpoint_config.get('model', {}).get('encoder', {})
            decoder_config = checkpoint_config.get('model', {}).get('decoder', {})

            print(f"Loaded from checkpoint['config']: encoder={encoder_type}, decoder={decoder_type}")

        if not encoder_type:
            raise ValueError(
                f"Cannot retrieve Encoder type!\n"
                f"Verify that 'model.encoder.type' is in the config."
            )

        if not decoder_type:
            raise ValueError(
                f"Cannot retrieve Decoder type!\n"
                f"Verify that 'model.decoder.type' is in the config."
            )

        from ..models.encoder import build_encoder
        from ..models.decoder import build_decoder

        encoder = build_encoder(encoder_config).to(device)
        decoder = build_decoder(decoder_config).to(device)

        realization = Realization(past_horizon=10, rank=3)
        df_state_config = {'feature_dim': 16}
        df_obs_config = {'obs_feature_dim': 8}
        training_config = TrainingConfig()

        instance = cls._init_from_args_direct(
            encoder, decoder, realization, df_state_config, df_obs_config,
            training_config, device, output_dir, use_kalman_filtering=False
        )

        instance.encoder.load_state_dict(state_dict.get('encoder', {}))
        instance.decoder.load_state_dict(state_dict.get('decoder', {}))

        print(f"Model loaded: {encoder_type}Encoder + {decoder_type}Decoder")

        return instance

    @classmethod
    def _detect_encoder_structure(cls, encoder_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Detect encoder structure from parameters."""
        input_dim = 6
        output_dim = 32
        architecture = 'mlp'
        hidden_dims = [64, 32]

        if not encoder_dict:
            pass
        elif 'core_net.0.weight' in encoder_dict:
            architecture = 'mlp'
            input_dim = encoder_dict['core_net.0.weight'].shape[1]

            if 'output_mean' in encoder_dict:
                output_dim = encoder_dict['output_mean'].shape[0]
            else:
                for key in encoder_dict.keys():
                    if key.startswith('core_net.') and key.endswith('.weight'):
                        output_dim = encoder_dict[key].shape[0]

            hidden_dims = []
            layer_keys = [k for k in encoder_dict.keys() if k.startswith('core_net.') and k.endswith('.weight')]
            layer_keys.sort(key=lambda x: int(x.split('.')[1]))

            for i, key in enumerate(layer_keys[:-1]):
                hidden_dims.append(encoder_dict[key].shape[0])

        elif 'layers.0.weight' in encoder_dict:
            architecture = 'mlp'
            input_dim = encoder_dict['layers.0.weight'].shape[1]
            hidden_dims = []
            max_layer_idx = -1

            for key in encoder_dict.keys():
                if key.startswith('layers.') and key.endswith('.weight'):
                    try:
                        layer_idx = int(key.split('.')[1])
                        max_layer_idx = max(max_layer_idx, layer_idx)

                        if layer_idx > 0:
                            layer_output_dim = encoder_dict[key].shape[0]
                            if layer_idx == max_layer_idx:
                                output_dim = layer_output_dim
                            else:
                                hidden_dims.append(layer_output_dim)
                    except (ValueError, IndexError):
                        continue

            if not hidden_dims:
                hidden_dims = [64, 32]

        elif any(key.startswith('conv') for key in encoder_dict.keys()):
            architecture = 'resnet'
            first_conv_keys = [k for k in encoder_dict.keys() if 'conv' in k and 'weight' in k]
            if first_conv_keys:
                first_key = sorted(first_conv_keys)[0]
                conv_shape = encoder_dict[first_key].shape
                if len(conv_shape) >= 2:
                    input_dim = conv_shape[1] if len(conv_shape) == 4 else conv_shape[1]
        else:
            weight_keys = [k for k in encoder_dict.keys() if 'weight' in k]
            if weight_keys:
                first_weight = encoder_dict[weight_keys[0]]
                if len(first_weight.shape) >= 2:
                    input_dim = first_weight.shape[1]
                    output_dim = first_weight.shape[0]

        return {
            'type': 'time_invariant',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'architecture': architecture,
            'hidden_dims': hidden_dims,
            'activation': 'GELU',
            'dropout': 0.1,
            'normalize_input': True,
            'normalize_output': True
        }

    @classmethod
    def _detect_decoder_structure(cls, decoder_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Detect decoder structure from parameters."""
        input_dim = 32
        output_dim = 6
        architecture = 'mlp'
        hidden_dims = [64, 32]

        if not decoder_dict:
            pass
        elif 'net.0.weight' in decoder_dict:
            architecture = 'mlp'
            input_dim = decoder_dict['net.0.weight'].shape[1]
            output_dim = 6
            for key in decoder_dict.keys():
                if key.startswith('net.') and key.endswith('.weight'):
                    output_dim = decoder_dict[key].shape[0]

            hidden_dims = []
            layer_keys = [k for k in decoder_dict.keys() if k.startswith('net.') and k.endswith('.weight')]
            layer_keys.sort(key=lambda x: int(x.split('.')[1]))

            for key in layer_keys[:-1]:
                hidden_dims.append(decoder_dict[key].shape[0])

        elif 'layers.0.weight' in decoder_dict:
            architecture = 'mlp'
            input_dim = decoder_dict['layers.0.weight'].shape[1]
            hidden_dims = []
            max_layer_idx = -1

            for key in decoder_dict.keys():
                if key.startswith('layers.') and key.endswith('.weight'):
                    try:
                        layer_idx = int(key.split('.')[1])
                        max_layer_idx = max(max_layer_idx, layer_idx)

                        # Collect hidden layer output dims
                        if layer_idx > 0:  # exclude first layer
                            layer_output_dim = decoder_dict[key].shape[0]
                            if layer_idx == max_layer_idx:
                                output_dim = layer_output_dim  # final layer = output dim
                            else:
                                hidden_dims.append(layer_output_dim)
                    except (ValueError, IndexError):
                        continue

            if not hidden_dims:
                hidden_dims = [64, 32]

        elif 'out_proj.weight' in decoder_dict:
            architecture = 'mlp'
            output_dim = decoder_dict['out_proj.weight'].shape[0]

            if 'takens_proj.weight' in decoder_dict:
                input_dim = decoder_dict['takens_proj.weight'].shape[1]

        elif any(key.startswith('conv') for key in decoder_dict.keys()):
            architecture = 'resnet'
            conv_keys = [k for k in decoder_dict.keys() if 'conv' in k and 'weight' in k]
            if conv_keys:
                last_key = sorted(conv_keys)[-1]
                conv_shape = decoder_dict[last_key].shape
                if len(conv_shape) >= 2:
                    output_dim = conv_shape[0] if len(conv_shape) == 4 else conv_shape[0]
        else:
            weight_keys = [k for k in decoder_dict.keys() if 'weight' in k]
            if weight_keys:
                last_weight = decoder_dict[weight_keys[-1]]
                if len(last_weight.shape) >= 2:
                    output_dim = last_weight.shape[0]
                    input_dim = last_weight.shape[1]

        return {
            'type': 'time_invariant',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'architecture': architecture,
            'hidden_dims': hidden_dims,
            'activation': 'GELU',
            'dropout': 0.1,
            'normalize_input': True,
            'normalize_output': True
        }

    @classmethod
    def _init_from_args_direct(cls, encoder, decoder, realization, df_state_config,
                              df_obs_config, training_config, device, output_dir,
                              use_kalman_filtering):
        """Direct initialization (for classmethods)."""
        instance = cls.__new__(cls)
        instance._init_from_args(encoder, decoder, realization, df_state_config,
                               df_obs_config, training_config, device, output_dir,
                               use_kalman_filtering, calibration_ratio=0.1,
                               auto_inference_setup=False)
        return instance
    
    def _init_from_args(self, encoder: nn.Module, decoder: nn.Module, realization: Realization,
                       df_state_config: Dict[str, Any], df_obs_config: Dict[str, Any],
                       training_config: TrainingConfig, device: torch.device, output_dir: str,
                       use_kalman_filtering: bool, calibration_ratio: float = 0.1,
                       auto_inference_setup: bool = True):
        """Initialize from individual arguments."""
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        # realization inherits nn.Module; move to device like encoder/decoder
        if hasattr(realization, 'to'):
            self.realization = realization.to(device)
        else:
            self.realization = realization

        self.config = training_config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._last_X_states_length = None
        self.use_kalman_filtering = use_kalman_filtering
        self.calibration_ratio = calibration_ratio
        self.auto_inference_setup = auto_inference_setup
        self.calibration_data: Optional[torch.Tensor] = None
        self.df_state_config = df_state_config
        self.df_obs_config = df_obs_config

        self.df_state = None
        self.df_obs = None
        self.optimizers = {}
        self.current_epoch = 0
        self.phase1_complete = False

        self.training_history = {
            'phase1_metrics': [],
            'phase2_losses': []
        }

        self._temp_data = {}
        self.logger = TrainingLogger(self.output_dir)

        # Guard set: prevent repeated log messages
        self._static_logs_shown = set()
        
        if 'trainer_init' not in self._static_logs_shown:
            print(f"TwoStageTrainer initialized: {device}")
            self._static_logs_shown.add('trainer_init')
    
    def _initialize_df_layers(self, X_states: torch.Tensor):
        """Initialize DF layers on GPU."""
        _, r = X_states.shape
        self.df_state = DFStateLayer(
            state_dim=r,
            feature_dim=self.df_state_config['feature_dim'],
            lambda_A=self.df_state_config['lambda_A'],
            lambda_B=self.df_state_config['lambda_B'],
            feature_net_config=self.df_state_config.get('feature_net'),
            cross_fitting_config=self.df_state_config.get('cross_fitting'),
            readout_config=self.df_state_config.get('readout')
        )
        self.df_state.phi_theta = self.df_state.phi_theta.to(self.device)
        if hasattr(self.df_state, 'readout_net'):
            self.df_state.readout_net = self.df_state.readout_net.to(self.device)

        self.df_obs = DFObservationLayer(
            df_state_layer=self.df_state,
            obs_feature_dim=self.df_obs_config['obs_feature_dim'],
            multivariate_feature_dim=self.df_obs_config['multivariate_feature_dim'],
            lambda_B=self.df_obs_config['lambda_B'],
            lambda_dB=self.df_obs_config['lambda_dB'],
            obs_net_config=self.df_obs_config.get('obs_net'),
            cross_fitting_config=self.df_obs_config.get('cross_fitting'),
            readout_config=self.df_obs_config.get('readout')
        )
        self.df_obs.psi_omega = self.df_obs.psi_omega.to(self.device)
        if hasattr(self.df_obs, 'readout_net'):
            self.df_obs.readout_net = self.df_obs.readout_net.to(self.device)
        
        if 'df_layers_init' not in self._static_logs_shown:
            print(f"DF layers initialized: state_dim={r}")
            self._static_logs_shown.add('df_layers_init')
    
    def _initialize_optimizers(self):
        """Initialize optimizers."""
        phi_params = list(self.df_state.phi_theta.parameters())
        if hasattr(self.df_state, 'readout_net'):
            phi_params += list(self.df_state.readout_net.parameters())
        self.optimizers['phi'] = torch.optim.Adam(phi_params, lr=self.config.lr_phi)

        psi_params = list(self.df_obs.psi_omega.parameters())
        if hasattr(self.df_obs, 'readout_net'):
            psi_params += list(self.df_obs.readout_net.parameters())
        self.optimizers['psi'] = torch.optim.Adam(psi_params, lr=self.config.lr_psi)

        # Stage-2 optimizer: param groups depend on update_strategy
        param_groups = [
            {'params': list(self.encoder.parameters()), 'lr': self.config.lr_encoder},
            {'params': list(self.decoder.parameters()), 'lr': self.config.lr_decoder},
        ]

        if self.config.update_strategy == "all":
            param_groups.extend([
                {'params': list(self.df_state.phi_theta.parameters()), 'lr': self.config.lr_phi},
                {'params': list(self.df_obs.psi_omega.parameters()), 'lr': self.config.lr_psi},
            ])

        self.optimizers['e2e'] = torch.optim.Adam(param_groups)
        print(f"Stage-2 optimizer: {len(param_groups)} param groups (update_strategy={self.config.update_strategy})")
        
        print("Optimizers initialized")
    
    def _prepare_data(self, Y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Data preparation (StochasticRealizationWithEncoder support).

        Returns:
            M_features: Encoder output sequence (T, m) - multivariate features
            X_states: State estimation sequence (T_eff, r)
        """
        if 'input_data_shape' not in self._static_logs_shown:
            self._static_logs_shown.add('input_data_shape')

        Y_train = self._ensure_device(Y_train)

        # 1. Encode: y_t -> m_t in R^m
        with torch.no_grad():
            self.encoder.eval()
            M_batch = self.encoder(Y_train)  # (T, d) or (T, H, W, C) -> (T, m)

            if M_batch.dim() == 1:
                M_features = M_batch.unsqueeze(1)
            elif M_batch.dim() == 2:
                M_features = M_batch
            else:
                raise ValueError(f"Invalid encoder output shape: {M_batch.shape}. Expected: (T, m)")

        if 'encoder_output_shape' not in self._static_logs_shown:
            self._static_logs_shown.add('encoder_output_shape')

        # Prepare calibration data for Kalman filtering
        if hasattr(self, 'use_kalman_filtering') and self.use_kalman_filtering:
            n_calib = int(Y_train.size(0) * getattr(self, 'calibration_ratio', 0.1))
            self.calibration_data = Y_train[:n_calib].clone()
            if self.config.verbose and 'calibration_data' not in self._static_logs_shown:
                print(f"Calibration data prepared: {self.calibration_data.shape}")
                self._static_logs_shown.add('calibration_data')

        # 2. Stochastic realization: multivariate features -> state estimates
        try:
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                self.realization.fit(Y_train, self.encoder)
                X_states = self.realization.estimate_states(Y_train)
                if 'stochastic_realization_shapes' not in self._static_logs_shown:
                                        self._static_logs_shown.add('stochastic_realization_shapes')
            else:
                # Scalar realization path.
                m_scalar = M_features.mean(dim=1)
                self.realization.fit(m_scalar.unsqueeze(1))
                X_states = self.realization.filter(m_scalar.unsqueeze(1))
                if 'traditional_realization_shapes' not in self._static_logs_shown:
                                        self._static_logs_shown.add('traditional_realization_shapes')
        except RealizationError as e:
            print(f"RealizationError: {e}")
            raise RealizationError(f"Stage-1 realization failed: {e}") from e

        self._last_X_states_length = X_states.size(0)

        # Detach X_states for Stage-1 (allows multiple backward passes)
        X_states = X_states.detach()

        if self.config.verbose and 'state_estimation_complete' not in self._static_logs_shown:
            print(f"State estimation complete: M_features={M_features.shape} -> X_states={X_states.shape}")
            self._static_logs_shown.add('state_estimation_complete')

        self._temp_data = {
            'Y_train': Y_train,
            'M_features': M_features,
            'X_states': X_states
        }

        return M_features, X_states

    def train_phase1(self, Y_train: torch.Tensor) -> Dict[str, Any]:
        """
        Stage-1: DF-A/DF-B cooperative training.
        
        Args:
            Y_train: Training observation sequence (T, d)
            
        Returns:
            Stage-1 metrics
        """
        print("\n=")

        M_features, X_states = self._prepare_data(Y_train)
        self._initialize_df_layers(X_states)
        self._initialize_optimizers()
        
        for epoch in range(self.config.phase1_epochs):
            self.current_epoch = epoch
            epoch_metrics = {}

            df_a_metrics = self._train_df_a_epoch(X_states, epoch)
            epoch_metrics.update(df_a_metrics)

            df_b_metrics = self._train_df_b_epoch(X_states, M_features, epoch)
            epoch_metrics.update(df_b_metrics)

            self.training_history['phase1_metrics'].append(epoch_metrics)

            if epoch % self.config.log_interval == 0 and self.config.verbose:
                self._print_phase1_progress(epoch, epoch_metrics)
                self.log_multivariate_training_progress(epoch, "phase1")

            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, TrainingPhase.PHASE1_DF_A)

        print("Computing final operators (V_A/V_B/U_A/U_B)...")
        self._compute_final_operators(Y_train)

        self.phase1_complete = True
        print("Stage-1 training complete")

        return self.training_history['phase1_metrics']

    def _compute_final_operators(self, Y_train: torch.Tensor):
        """Compute final operators V_A/V_B/U_A/U_B after Stage-1."""
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        if hasattr(self.df_state, 'phi_theta'):
            self.df_state.phi_theta = self.df_state.phi_theta.to(self.device)
        if hasattr(self.df_obs, 'psi_omega'):
            self.df_obs.psi_omega = self.df_obs.psi_omega.to(self.device)

        M_features, X_states = self._prepare_data(Y_train)

        with torch.no_grad():
            Phi_full = self.df_state.phi_theta(X_states)  # (T, d_A)
            Phi_minus = Phi_full[:-1]  # phi(x_{t-1})
            Phi_plus = Phi_full[1:]    # phi(x_t)
            X_plus = X_states[1:]      # x_t

        # DF-A: compute V_A, U_A
        print("  Computing DF-A operators (V_A/U_A)...")
        if hasattr(self.df_state, 'cf_config') and self.df_state.cf_config:
            self.df_state._fit_with_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose=True)
        else:
            self.df_state._fit_without_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose=True)

        if 'v_a_u_a_shapes' not in self._static_logs_shown:
            self._static_logs_shown.add('v_a_u_a_shapes')

        # DF-B: compute V_B, U_B
        if hasattr(self, 'df_obs') and self.df_obs is not None:
            print("  Computing DF-B operators (V_B/U_B)...")

            with torch.no_grad():
                # x_hat_{t|t-1} = r_xi_A(V_A phi(x_{t-1}))
                H_A = (self.df_state.V_A @ Phi_minus.T).T
                X_pred = self.df_state.apply_readout(H_A)
                Phi_pred = self.df_state.phi_theta(X_pred)

                # Account for realization time reduction: T -> T_eff = T - 2*h + 1
                if isinstance(self.realization, StochasticRealizationWithEncoder):
                    h = getattr(self.realization, 'window_length', 5)
                else:
                    h = self.realization.h
                T_states = X_states.shape[0]

                M_curr = M_features[h:h+T_states]
                Psi_curr = self.df_obs.psi_omega(M_curr)
                m_curr = M_curr

                # Align to minimum length
                min_size = min(Phi_pred.shape[0], Psi_curr.shape[0], m_curr.shape[0])
                Phi_pred = Phi_pred[:min_size]
                Psi_curr = Psi_curr[:min_size]
                m_curr = m_curr[:min_size]

            if 'df_b_learning_data' not in self._static_logs_shown:
                self._static_logs_shown.add('df_b_learning_data')
            if hasattr(self.df_obs, 'cf_config') and self.df_obs.cf_config:
                self.df_obs._fit_with_cross_fitting(Phi_pred, Psi_curr, m_curr, verbose=True)
            else:
                self.df_obs._fit_without_cross_fitting(Phi_pred, Psi_curr, m_curr, verbose=True)

            if 'v_b_u_b_shapes' not in self._static_logs_shown:
                self._static_logs_shown.add('v_b_u_b_shapes')

        print("Final operator computation complete")

        try:
            canonical_correlations = self._get_canonical_correlations_from_realization()
            if canonical_correlations is not None:
                phase1_epochs = getattr(self.config, 'phase1_epochs', self.config.epochs)
                final_epoch = getattr(self, 'current_epoch', phase1_epochs - 1)
                self.logger.log_canonical_correlations(final_epoch, "Stage-1-Complete", canonical_correlations)

        except Exception as e:
            print(f"Error logging canonical correlations at Stage-1 completion: {e}")

    def _train_df_a_epoch(self, X_states: torch.Tensor, epoch: int) -> Dict[str, float]:
        """
        DF-A (state layer) epoch training.

        Runs Stage-1 T1_iterations times and Stage-2 T2_iterations times.
        """
        metrics = {}
        opt_phi = self.optimizers['phi']
        X_states_gpu = X_states.to(self.device)

        stage1_losses = []
        stage1_pred_losses = []
        stage1_reg_losses = []
        for t in range(self.config.T1_iterations):
            stage1_metrics = self.df_state.train_stage1_with_gradients(
                X_states_gpu,
                opt_phi,
                epoch=epoch
            )
            stage1_losses.append(stage1_metrics['stage1_loss'])
            stage1_pred_losses.append(stage1_metrics['stage1_pred_loss'])
            stage1_reg_losses.append(stage1_metrics['stage1_reg_loss'])

            self.logger.log_phase1(
                epoch, TrainingPhase.PHASE1_DF_A, 'stage1', t,
                stage1_metrics, {'lr_phi': opt_phi.param_groups[0]['lr']}
            )

        metrics['df_a_stage1_loss'] = sum(stage1_losses) / len(stage1_losses)
        metrics['df_a_stage1_pred'] = sum(stage1_pred_losses) / len(stage1_pred_losses)
        metrics['df_a_stage1_reg'] = sum(stage1_reg_losses) / len(stage1_reg_losses)

        stage2_losses = []
        stage2_pred_losses = []
        stage2_reg_losses = []
        for t in range(self.config.T2_iterations):
            stage2_metrics = self.df_state.train_stage2_with_gradients(
                X_states_gpu,
                opt_phi,
                epoch=epoch
            )
            stage2_losses.append(stage2_metrics['stage2_loss'])
            stage2_pred_losses.append(stage2_metrics['stage2_pred_loss'])
            stage2_reg_losses.append(stage2_metrics['stage2_reg_loss'])

            self.logger.log_phase1(
                epoch, TrainingPhase.PHASE1_DF_A, 'stage2', t,
                stage2_metrics, {'lr_phi': opt_phi.param_groups[0]['lr']}
            )

        metrics['df_a_stage2_loss'] = sum(stage2_losses) / len(stage2_losses)
        metrics['df_a_stage2_pred'] = sum(stage2_pred_losses) / len(stage2_pred_losses)
        metrics['df_a_stage2_reg'] = sum(stage2_reg_losses) / len(stage2_reg_losses)

        return metrics
    
    def _train_df_b_epoch(self, X_states: torch.Tensor, M_features: torch.Tensor,
                        epoch: int) -> Dict[str, float]:
        """
        DF-B (observation layer) epoch training.

        Runs Stage-1 T1_iterations times and Stage-2 T2_iterations times.
        """
        metrics = {}
        opt_phi = self.optimizers['phi']
        opt_psi = self.optimizers['psi']

        X_states_gpu = X_states.to(self.device)
        M_features_gpu = M_features.to(self.device)

        # State predictions from DF-A (inference only, used as instrumental variables)
        with torch.no_grad():
            X_hat_states = self.df_state.predict_sequence(X_states_gpu)

        M_aligned = self._align_time_series_multivariate(
            X_hat_states, M_features_gpu, X_states.size(0), epoch, "DF-B"
        )

        stage1_losses = []
        stage1_pred_losses = []
        stage1_reg_losses = []
        for t in range(self.config.T1_iterations):
            stage1_metrics = self.df_obs.train_stage1_with_gradients(
                X_hat_states,
                M_aligned,
                opt_phi,
                fix_psi_omega=True,
                epoch=epoch
            )
            stage1_losses.append(stage1_metrics['stage1_loss'])
            stage1_pred_losses.append(stage1_metrics['stage1_pred_loss'])
            stage1_reg_losses.append(stage1_metrics['stage1_reg_loss'])

            self.logger.log_phase1(
                epoch, TrainingPhase.PHASE1_DF_B, 'stage1', t,
                stage1_metrics, {'lr_phi': opt_phi.param_groups[0]['lr']}
            )

        metrics['df_b_stage1_loss'] = sum(stage1_losses) / len(stage1_losses)
        metrics['df_b_stage1_pred'] = sum(stage1_pred_losses) / len(stage1_pred_losses)
        metrics['df_b_stage1_reg'] = sum(stage1_reg_losses) / len(stage1_reg_losses)

        stage2_losses = []
        stage2_pred_losses = []
        stage2_reg_losses = []
        for t in range(self.config.T2_iterations):
            stage2_metrics = self.df_obs.train_stage2_with_gradients(
                M_aligned,
                opt_psi,
                fix_phi_theta=True,
                epoch=epoch
            )
            stage2_losses.append(stage2_metrics['stage2_loss'])
            stage2_pred_losses.append(stage2_metrics['stage2_pred_loss'])
            stage2_reg_losses.append(stage2_metrics['stage2_reg_loss'])

            self.logger.log_phase1(
                epoch, TrainingPhase.PHASE1_DF_B, 'stage2', t,
                stage2_metrics, {'lr_psi': opt_psi.param_groups[0]['lr']}
            )

        metrics['df_b_stage2_loss'] = sum(stage2_losses) / len(stage2_losses)
        metrics['df_b_stage2_pred'] = sum(stage2_pred_losses) / len(stage2_pred_losses)
        metrics['df_b_stage2_reg'] = sum(stage2_reg_losses) / len(stage2_reg_losses)

        return metrics
    
    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to target device if needed."""
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def train_phase2(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Stage-2: End-to-end fine-tuning.
        
        Fixed inference path:
        x̂_{t|t-1} = U_A^T V_A φ_θ(x_{t-1})
        m̂_{t|t-1} = u_B^T V_B φ_θ(x̂_{t|t-1})
        ŷ_{t|t-1} = g_α(m̂_{t|t-1})
        
        Args:
            Y_train: Training observation sequence
            Y_val: Validation observation sequence (optional)
        """
        print("\n=")

        if 'device_allocation' not in self._static_logs_shown:
            print(f"Device ensured: encoder -> {self.device}, decoder -> {self.device}")
            self._static_logs_shown.add('device_allocation')
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        Y_train = self._ensure_device(Y_train)
        if Y_val is not None:
            Y_val = self._ensure_device(Y_val)
        
        if not self.phase1_complete:
            raise RuntimeError("Stage-1 not completed")
        
        opt_e2e = self.optimizers['e2e']

        for epoch in range(self.config.phase2_epochs):
            self.current_epoch = self.config.phase1_epochs + epoch
            
            try:
                loss_total, rec_loss, cca_loss = self._forward_and_loss_phase2(Y_train)

                opt_e2e.zero_grad()
                loss_total.backward()
                opt_e2e.step()
                
            except RealizationError as e:
                print(f"Epoch {epoch} skipped (Stage-2 realization failure): {e}")
                continue
            
            lr_dict = {f'lr_{name}': group['lr'] for name, group in
                      zip(['encoder', 'decoder', 'phi', 'psi'], opt_e2e.param_groups)}
            self.logger.log_phase2(epoch, loss_total.item(), rec_loss.item(), 
                                  cca_loss.item(), lr_dict)
            
            self.training_history['phase2_losses'].append({
                'epoch': epoch,
                'total_loss': loss_total.item(),
                'rec_loss': rec_loss.item(),
                'cca_loss': cca_loss.item()
            })

            if epoch % self.config.log_interval == 0 and self.config.verbose:
                print(f"Stage-2 Epoch {epoch}: Total={loss_total.item():.6f}, "
                      f"Rec={rec_loss.item():.6f}, CCA={cca_loss.item():.6f}")
                self.log_multivariate_training_progress(epoch, "phase2")

            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, TrainingPhase.PHASE2_E2E)
        
        print("Stage-2 training complete")
        return self.training_history['phase2_losses']

    def train_integrated(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Unified training: run Stage-1 and Stage-2 consecutively each epoch.

        Args:
            Y_train: Training observation sequence (T, d)
            Y_val: Validation observation sequence (optional)

        Returns:
            Unified training results
        """
        print(f"\n=")

        M_features, X_states = self._prepare_data(Y_train)
        self._initialize_df_layers(X_states)
        self._initialize_optimizers()

        self.training_history = {
            'phase1_metrics': [],
            'phase2_losses': [],
            'integrated_metrics': []
        }

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            try:
                # Recompute X_states after encoder update in Stage-2
                if epoch >= self.config.phase1_warmup_epochs:
                    with torch.no_grad():
                        M_features = self.encoder(Y_train)

                        if isinstance(self.realization, StochasticRealizationWithEncoder):
                            self.realization.fit(Y_train, self.encoder)
                            X_states = self.realization.estimate_states(Y_train)
                        else:
                            m_scalar = M_features.mean(dim=1)
                            self.realization.fit(m_scalar.unsqueeze(1))
                            X_states = self.realization.filter(m_scalar.unsqueeze(1))

                phase1_metrics = self._train_integrated_phase1_epoch(X_states, M_features, epoch)
                self._lightweight_operator_update(Y_train)

                phase2_metrics = {}
                if epoch >= self.config.phase1_warmup_epochs:
                    phase2_metrics = self._train_integrated_phase2_epoch(Y_train, epoch)
                else:
                    phase2_metrics = {
                        'total_loss': 0.0,
                        'rec_loss': 0.0,
                        'cca_loss': 0.0
                    }

                integrated_metrics = {
                    'epoch': epoch,
                    'phase1_loss': phase1_metrics.get('df_a_stage1_loss', 0.0),
                    'phase2_total': phase2_metrics['total_loss'],
                    'phase2_rec': phase2_metrics['rec_loss'],
                    'phase2_cca': phase2_metrics['cca_loss']
                }

                self.training_history['phase1_metrics'].append(phase1_metrics)
                self.training_history['phase2_losses'].append(phase2_metrics)
                self.training_history['integrated_metrics'].append(integrated_metrics)

                df_a_s1_pred = phase1_metrics.get('df_a_stage1_pred', 0.0)
                df_a_s1_reg = phase1_metrics.get('df_a_stage1_reg', 0.0)
                df_a_s2_pred = phase1_metrics.get('df_a_stage2_pred', 0.0)
                df_a_s2_reg = phase1_metrics.get('df_a_stage2_reg', 0.0)
                df_b_s1_pred = phase1_metrics.get('df_b_stage1_pred', 0.0)
                df_b_s1_reg = phase1_metrics.get('df_b_stage1_reg', 0.0)
                df_b_s2_pred = phase1_metrics.get('df_b_stage2_pred', 0.0)
                df_b_s2_reg = phase1_metrics.get('df_b_stage2_reg', 0.0)
                rec_loss = phase2_metrics['rec_loss']
                cca_loss = phase2_metrics['cca_loss']
                total_loss = phase2_metrics['total_loss']
                phase2_status = "active" if epoch >= self.config.phase1_warmup_epochs else "warmup"

                print(f"epoch {epoch+1:3d}, "
                      f"df-A(S1p={df_a_s1_pred:.4f},r={df_a_s1_reg:.4f}), "
                      f"df-A(S2p={df_a_s2_pred:.4f},r={df_a_s2_reg:.4f}), "
                      f"df-B(S1p={df_b_s1_pred:.4f},r={df_b_s1_reg:.4f}), "
                      f"df-B(S2p={df_b_s2_pred:.4f},r={df_b_s2_reg:.4f}), "
                      f"rec={rec_loss:.4f}, cca={cca_loss:.4f}, total={total_loss:.4f} ({phase2_status})")

                if epoch % self.config.save_interval == 0:
                    self._save_checkpoint(epoch, TrainingPhase.PHASE2_E2E)

            except Exception as e:
                import traceback
                print(f"\n{'='*60}")
                print(f"Error at epoch {epoch}: {e}")
                print(f"{'='*60}")
                print("Detailed stack trace:")
                traceback.print_exc()
                print(f"{'='*60}\n")
                continue

        print("Computing final operators (V_A/V_B/U_A/U_B)...")
        self._compute_final_operators(Y_train)

        self.phase1_complete = True
        print(f"Unified training complete ({self.config.epochs} epochs)")
        return self.training_history

    def _train_integrated_phase1_epoch(self, X_states: torch.Tensor, M_features: torch.Tensor,
                                     epoch: int) -> Dict[str, Any]:
        """
        Execute one epoch of Stage-1 in unified training.
        Same formulation-based implementation as _train_df_a_epoch, _train_df_b_epoch.
        """
        epoch_metrics = {}

        df_a_metrics = self._train_df_a_epoch(X_states, epoch)
        epoch_metrics.update(df_a_metrics)

        df_b_metrics = self._train_df_b_epoch(X_states, M_features, epoch)
        epoch_metrics.update(df_b_metrics)

        return epoch_metrics

    def _train_integrated_phase2_epoch(self, Y_train: torch.Tensor, epoch: int) -> Dict[str, float]:
        """
        Execute one epoch of Stage-2 in unified training.
        Applies same device management and error handling as train_phase2.
        """
        Y_train = self._ensure_device(Y_train)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.encoder.train()
        self.decoder.train()

        # DF layers in eval mode during Stage-2 (disable dropout)
        if hasattr(self, 'df_state') and self.df_state is not None:
            self.df_state.eval()
        if hasattr(self, 'df_obs') and self.df_obs is not None:
            self.df_obs.eval()

        if 'e2e' not in self.optimizers:
            self._initialize_phase2_optimizer()

        opt_e2e = self.optimizers['e2e']

        try:
            loss_total, rec_loss, cca_loss = self._forward_and_loss_phase2(Y_train)

            opt_e2e.zero_grad()
            loss_total.backward()

            # Gradient diagnostics at first active Stage-2 epoch
            if epoch == self.config.phase1_warmup_epochs and not hasattr(self, '_grad_diag_done'):
                enc_grad_norm = sum(
                    p.grad.norm().item() for p in self.encoder.parameters()
                    if p.grad is not None
                )
                dec_grad_norm = sum(
                    p.grad.norm().item() for p in self.decoder.parameters()
                    if p.grad is not None
                )
                enc_has_grad = any(p.grad is not None for p in self.encoder.parameters())
                print(f"[GradDiag] epoch={epoch}: encoder_grad_norm={enc_grad_norm:.6f}, "
                      f"decoder_grad_norm={dec_grad_norm:.6f}, encoder_has_grad={enc_has_grad}")
                print(f"[GradDiag] e2e optimizer param_groups: {len(opt_e2e.param_groups)}")
                self._grad_diag_done = True

            opt_e2e.step()

            group_names = ['encoder', 'decoder', 'phi', 'psi'][:len(opt_e2e.param_groups)]
            lr_dict = {f'lr_{name}': group['lr'] for name, group in
                      zip(group_names, opt_e2e.param_groups)}

            self.logger.log_phase2(epoch, loss_total.item(), rec_loss.item(),
                                  cca_loss.item(), lr_dict)

            result = {
                'epoch': epoch,
                'total_loss': loss_total.item(),
                'rec_loss': rec_loss.item(),
                'cca_loss': cca_loss.item()
            }

            return result

        except RealizationError as e:
            print(f"Epoch {epoch} Stage-2 skipped (realization failure): {e}")
            result = {
                'epoch': epoch,
                'total_loss': 0.0,
                'rec_loss': 0.0,
                'cca_loss': 0.0
            }

            return result

    def _lightweight_operator_update(self, Y_train: torch.Tensor):
        """
        Lightweight operator update after Stage-1.
        Based on _compute_final_operators pattern.
        """
        try:
            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)

            if hasattr(self.df_state, 'phi_theta'):
                self.df_state.phi_theta = self.df_state.phi_theta.to(self.device)
            if hasattr(self.df_obs, 'psi_omega'):
                self.df_obs.psi_omega = self.df_obs.psi_omega.to(self.device)

            # Refit realization without building computation graph
            with torch.no_grad():
                if hasattr(self.realization, 'fit') and isinstance(self.realization, StochasticRealizationWithEncoder):
                    self.realization.fit(Y_train, self.encoder)

        except Exception as e:
            print(f"Lightweight operator update error: {e}")

    def _initialize_phase2_optimizer(self):
        """Initialize Stage-2 optimizer with param groups based on update_strategy."""
        if 'e2e' not in self.optimizers:
            param_groups = [
                {'params': list(self.encoder.parameters()), 'lr': self.config.lr_encoder},
                {'params': list(self.decoder.parameters()), 'lr': self.config.lr_decoder}
            ]

            # Include feature mapping parameters if available
            if hasattr(self.realization, 'component_transforms') and \
               self.realization.component_transforms is not None:
                param_groups.append({
                    'params': list(self.realization.component_transforms.parameters()),
                    'lr': self.config.lr_encoder
                })

            if self.config.update_strategy == "all":
                param_groups.extend([
                    {'params': list(self.df_state.phi_theta.parameters()), 'lr': self.config.lr_phi},
                    {'params': list(self.df_obs.psi_omega.parameters()), 'lr': self.config.lr_psi}
                ])
                print("Stage-2 also updates DF layers (update_strategy='all', staged+Stage-2 DF update)")
            else:
                print("Stage-2 updates encoder/decoder only (staged training design)")

            self.optimizers['e2e'] = torch.optim.Adam(param_groups)

    def _forward_and_loss_phase2(self, Y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage-2 reconstruction forward pass and loss computation."""
        return self._forward_and_loss_phase2_reconstruction(Y_train)

    def _forward_and_loss_phase2_reconstruction(self, Y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage-2 reconstruction mode: forward pass and loss computation."""
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        T = Y_train.shape[0]
        h = self.realization.h

        if T <= 2 * h:
            raise RuntimeError(f"Time series too short: T({T}) <= 2*h({2*h})")

        # Step 1-2: Stochastic realization
        try:
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                self.realization.fit(Y_train, self.encoder)
                X_states = self.realization.estimate_states(Y_train)
            else:
                M_features = self.encoder(Y_train)
                if M_features.dim() == 1:
                    M_features = M_features.unsqueeze(1)
                m_scalar = M_features.mean(dim=1)
                self.realization.fit(m_scalar.unsqueeze(1))
                X_states = self.realization.filter(m_scalar.unsqueeze(1))
        except RealizationError as e:
            print(f"Stage-2 RealizationError: {e}")
            raise RealizationError(f"Stage-2 realization failed: {e}") from e

        # Step 3: DF-A prediction (retain gradients for end-to-end)
        X_hat_states = self.df_state.predict_sequence(X_states, training=True)
        T_pred = X_hat_states.size(0)

        # Step 4: DF-B prediction
        M_hat_series = []
        for t in range(T_pred):
            m_hat_t = self.df_obs.predict_one_step(X_hat_states[t])
            M_hat_series.append(m_hat_t)
        M_hat_tensor = torch.stack(M_hat_series)
        M_hat_tensor = self._ensure_device(M_hat_tensor)

        # Step 5: Decode m_hat -> y_hat (reconstruction)
        Y_hat = self.decoder(M_hat_tensor)

        # Step 6: Loss against corresponding ground truth
        Y_target = Y_train[h+1:h+1+T_pred]
        Y_target = self._ensure_device(Y_target)

        loss_rec = torch.norm(Y_hat - Y_target, p='fro') ** 2 / Y_target.numel()

        if self.config.lambda_cca > 0:
            loss_cca = self._compute_cca_loss()
        else:
            loss_cca = torch.tensor(0.0, requires_grad=True)

        loss_total = loss_rec + self.config.lambda_cca * loss_cca

        return loss_total, loss_rec, loss_cca

    def _compute_cca_loss(self) -> torch.Tensor:
        """CCA loss: L_cca = -sum_i rho_i (maximize canonical correlations)."""
        try:
            canonical_correlations = self._get_canonical_correlations_from_realization()

            if canonical_correlations is None:
                if self.config.verbose:
                    print("Canonical correlations unavailable. Setting CCA loss to 0.")
                return torch.tensor(0.0, requires_grad=True, device=self.device)

            cca_loss = -canonical_correlations.sum()

            if hasattr(self, 'current_epoch') and hasattr(self, 'logger'):
                epoch = getattr(self, 'current_epoch', 0)
                if epoch % self.config.log_interval == 0:
                    self.logger.log_canonical_correlations(epoch, "Stage-2", canonical_correlations)

            return cca_loss

        except Exception as e:
            warnings.warn(f"CCA loss computation error: {e}. Setting loss to 0.")
            return torch.tensor(0.0, requires_grad=True, device=self.device)

    def _get_canonical_correlations_from_realization(self) -> Optional[torch.Tensor]:
        """Get canonical correlations rho_i from the realization, or None."""
        try:
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                if hasattr(self.realization, 'canonical_correlations') and \
                   self.realization.canonical_correlations is not None:
                    correlations = self.realization.canonical_correlations
                    return correlations.to(self.device)

            elif hasattr(self.realization, '_L_vals') and self.realization._L_vals is not None:
                l_vals = self.realization._L_vals
                return l_vals.to(self.device)

            return None

        except Exception as e:
            warnings.warn(f"Error getting canonical correlations: {e}")
            return None

    def _align_time_series(self, X_hat_states: torch.Tensor, m_series: torch.Tensor,
                          T_states: int, epoch: int, component: str) -> torch.Tensor:
        """
        Unified time series alignment helper.
        
        Args:
            X_hat_states: State prediction sequence
            m_series: Original scalar feature series
            T_states: State sequence length
            epoch: Current epoch (for logging)
            component: Component name (for logging)
        
        Returns:
            torch.Tensor: Time-aligned scalar feature series
        """
        T_pred = X_hat_states.size(0)
        T_original = m_series.size(0)
        
        # Offset computation
        total_offset = self._get_time_alignment_offset(T_original, T_states, T_pred)
        
        # Extract time-aligned m_series
        if total_offset + T_pred <= T_original:
            m_aligned = m_series[total_offset:total_offset + T_pred]
        else:
            # Safety fallback: get required length from end
            m_aligned = m_series[-T_pred:]
            if self.config.verbose:
                print(f"Warning: {component} offset adjustment failed, using tail truncation")
        
        self._validate_time_alignment(X_hat_states, m_aligned, component)

        return m_aligned

    def _align_time_series_multivariate(
        self,
        X_hat_states: torch.Tensor,
        M_features: torch.Tensor,
        T_states: int,
        epoch: int,
        component: str = "unknown"
    ) -> torch.Tensor:
        """
        Time index adjustment for multivariate features (M_features in R^(T x m)).

        Args:
            X_hat_states: State prediction (T_pred, r)
            M_features: Multivariate features (T, m)
            T_states: State sequence length
            epoch: Epoch number
            component: Component name

        Returns:
            torch.Tensor: Time-aligned multivariate feature series (T_pred, m)
        """
        T_pred = X_hat_states.size(0)
        T_original = M_features.size(0)

        # Offset computation (using existing method)
        total_offset = self._get_time_alignment_offset(T_original, T_states, T_pred)

        if total_offset + T_pred <= T_original:
            M_aligned = M_features[total_offset:total_offset + T_pred]
        else:
            M_aligned = M_features[-T_pred:]
            if self.config.verbose:
                print(f"Warning: {component} offset adjustment failed, using tail truncation")

        self._validate_time_alignment_multivariate(X_hat_states, M_aligned, component)

        return M_aligned

    def _get_time_alignment_offset(self, T_original: int, T_states: int, T_pred: int) -> int:
        """
        Compute offset for time index adjustment (= h + 1).
        Realization outputs x_h,...,x_{h+T_states-1}; DF-A predicts x_hat_{h+1|h},...
        so x_hat_{h+1|h} corresponds to m_{h+1}.
        Args:
            T_original: Original series length
            T_states: State sequence length  
            T_pred: Prediction sequence length
            
        Returns:
            int: m_series offset (= h + 1)
        """
        h_candidates = ['h', 'past_horizon', 'lags', 'window_size']
        
        h = None
        for attr_name in h_candidates:
            if hasattr(self.realization, attr_name):
                h = getattr(self.realization, attr_name)
                if isinstance(h, (int, float)) and h > 0:
                    h = int(h)
                    break
        
        if h is None:
            # Back-calculate h from T_states = T_original - 2*h + 1
            h = (T_original - T_states + 1) // 2
            if self.config.verbose:
                print(f"Warning: h estimated by back-calculation: h = {h}")

        expected_T_states = T_original - 2 * h + 1
        if abs(T_states - expected_T_states) > 1:
            if self.config.verbose:
                print(f"Warning: expected T_states={expected_T_states} with h={h} does not match actual {T_states}")
        
        return h + 1
    
    def _validate_time_alignment(self, X_hat_states: torch.Tensor, m_aligned: torch.Tensor,
                               component: str = "unknown") -> None:
        """Verify time dimension match between state predictions and aligned features."""
        if X_hat_states.size(0) != m_aligned.size(0):
            raise RuntimeError(
                f"{component} time index mismatch: "
                f"X_hat={X_hat_states.shape} vs m_aligned={m_aligned.shape}"
            )

    def _validate_time_alignment_multivariate(
        self,
        X_hat_states: torch.Tensor,
        M_aligned: torch.Tensor,
        component: str = "unknown"
    ) -> None:
        """Verify time dimension match for multivariate features."""
        if X_hat_states.size(0) != M_aligned.size(0):
            raise RuntimeError(
                f"{component} time index mismatch: "
                f"X_hat={X_hat_states.shape} vs M_aligned={M_aligned.shape}"
            )

        if M_aligned.dim() != 2:
            raise RuntimeError(
                f"{component} multivariate feature shape error: "
                f"expected=(T_pred, m), actual={M_aligned.shape}"
            )

    def _clear_computation_graph(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def verify_multivariate_dimensions(self, M_features: torch.Tensor, X_states: torch.Tensor) -> Dict[str, Any]:
        """
        Verify multivariate dimension consistency.

        Args:
            M_features: Multivariate features (T, m)
            X_states: State estimates (T_eff, r)

        Returns:
            Dict: Verification results
        """
        result = {
            "status": "ok",
            "dimensions": {},
            "warnings": [],
            "errors": []
        }

        try:
            result["dimensions"] = {
                "M_features_shape": tuple(M_features.shape),
                "X_states_shape": tuple(X_states.shape),
                "encoder_output_dim": getattr(self.encoder, 'output_dim', 'unknown'),
                "expected_feature_dim": getattr(self.df_obs, 'multivariate_feature_dim', 'unknown'),
                "df_state_feature_dim": getattr(self.df_state, 'feature_dim', 'unknown'),
                "df_obs_feature_dim": getattr(self.df_obs, 'obs_feature_dim', 'unknown')
            }

            if M_features.dim() != 2:
                result["errors"].append(f"M_features shape error: expected=(T, m), actual={M_features.shape}")

            if hasattr(self.df_obs, 'multivariate_feature_dim'):
                expected_m = self.df_obs.multivariate_feature_dim
                if M_features.size(1) != expected_m:
                    result["errors"].append(
                        f"Feature dimension mismatch: M_features.size(1)={M_features.size(1)} vs "
                        f"expected={expected_m}"
                    )

            if hasattr(self.df_obs, 'U_B') and self.df_obs.U_B is not None:
                U_B_shape = self.df_obs.U_B.shape
                expected_U_B_shape = (self.df_obs.obs_feature_dim, M_features.size(1))
                if tuple(U_B_shape) != expected_U_B_shape:
                    result["errors"].append(
                        f"U_B matrix dimension error: actual={U_B_shape} vs expected={expected_U_B_shape}"
                    )

            if hasattr(self.df_obs, 'V_B') and self.df_obs.V_B is not None:
                cond_V_B = torch.linalg.cond(self.df_obs.V_B).item()
                if cond_V_B > 1e12:
                    result["warnings"].append(f"V_B condition number large: {cond_V_B:.2e}")

            if hasattr(self.df_obs, 'U_B') and self.df_obs.U_B is not None:
                U_B_gram = self.df_obs.U_B @ self.df_obs.U_B.T
                cond_U_B_gram = torch.linalg.cond(U_B_gram).item()
                if cond_U_B_gram > 1e12:
                    result["warnings"].append(f"U_B @ U_B.T condition number large: {cond_U_B_gram:.2e}")

            if result["errors"]:
                result["status"] = "error"
            elif result["warnings"]:
                result["status"] = "warning"

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Error during verification: {str(e)}")

        return result

    def log_multivariate_training_progress(self, epoch: int, phase: str):
        """Log multivariate training progress (detailed on first call per phase)."""
        if not self.config.verbose or epoch % self.config.log_interval != 0:
            return

        if not hasattr(self, '_multivariate_logged'):
            self._multivariate_logged = set()

        show_details = phase not in self._multivariate_logged

        try:
            if show_details:
                if hasattr(self, '_temp_data') and 'M_features' in self._temp_data:
                    M_features = self._temp_data['M_features']
                    print(f"Multivariate features: shape={M_features.shape}, "
                          f"mean={M_features.mean().item():.4f}, "
                          f"std={M_features.std().item():.4f}")

                if hasattr(self, '_temp_data') and 'M_features' in self._temp_data:
                    M_features = self._temp_data['M_features']
                    X_states = self._temp_data['X_states']
                    verification = self.verify_multivariate_dimensions(M_features, X_states)

                    if verification["status"] == "error":
                        print(f"Dimension error: {verification['errors']}")
                    elif verification["status"] == "warning":
                        print(f"Numerical warning: {verification['warnings']}")
                    else:
                        print(f"Dimension consistency: OK")

                self._multivariate_logged.add(phase)

        except Exception as e:
            print(f"Log output error: {e}")

    def forecast(self, Y_test: torch.Tensor, forecast_steps: int) -> torch.Tensor:
        """Execute prediction."""
        self.encoder.eval()
        self.decoder.eval()
        self.df_state.eval()
        self.df_obs.eval()
        
        with torch.no_grad():
            T_test, d = Y_test.shape
            warmup_len = min(T_test, self.realization.h + 10)
            Y_warmup = Y_test[:warmup_len]

            m_warmup = self.encoder(Y_warmup.unsqueeze(0)).squeeze()

            try:
                self.realization.fit(m_warmup.unsqueeze(1))
            except RealizationError as e:
                print(f"Warmup RealizationError: {e}")
                raise RealizationError(f"Warmup realization failed: {e}") from e
            X_warmup = self.realization.filter(m_warmup.unsqueeze(1))

            predictions = []
            x_current = X_warmup[-1]

            for step in range(forecast_steps):
                x_pred = self.df_state.predict_one_step(x_current)
                m_pred = self.df_obs.predict_one_step(x_pred)
                m_input = m_pred.unsqueeze(0).unsqueeze(0).unsqueeze(2)
                y_pred = self.decoder(m_input).squeeze()
                predictions.append(y_pred)
                x_current = x_pred

            return torch.stack(predictions)
    
    def train_full(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Execute full training (Stage-1 + Stage-2)."""
        try:
            phase1_metrics = self.train_phase1(Y_train)
            phase2_metrics = self.train_phase2(Y_train, Y_val)
            self._save_final_model()
            self.logger.save_summary()
            
            return {
                'phase1_metrics': phase1_metrics,
                'phase2_losses': phase2_metrics,
                'training_config': self.config.__dict__,
                'model_paths': {
                    'final_model': str(self.output_dir / 'final_model.pth'),
                    'logs': str(self.logger.output_dir)
                }
            }
            
        except Exception as e:
            print(f"Error during training: {e}")
            self._save_checkpoint(self.current_epoch, TrainingPhase.PHASE1_DF_A, emergency=True)
            raise
    
    def _print_phase1_progress(self, epoch: int, metrics: Dict[str, float]):
        """Display Stage-1 progress."""
        df_a_s1 = metrics.get('df_a_stage1_loss', 0)
        df_a_s2 = metrics.get('df_a_stage2_loss', 0)
        df_b_s1 = metrics.get('df_b_stage1_loss', 0)
        df_b_s2 = metrics.get('df_b_stage2_loss', 0)
        
        print(f"Stage-1 Epoch {epoch:3d}: "
              f"DF-A(S1={df_a_s1:.4f}, S2={df_a_s2:.4f}) "
              f"DF-B(S1={df_b_s1:.4f}, S2={df_b_s2:.4f})")
    
    def _save_checkpoint(self, epoch: int, phase: TrainingPhase, emergency: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'phase': phase.value,
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'training_config': self.config.__dict__,
            'training_history': self.training_history,
            'phase1_complete': self.phase1_complete
        }

        if self.df_state is not None:
            checkpoint['df_state'] = self.df_state.get_state_dict()
        if self.df_obs is not None:
            checkpoint['df_obs'] = self.df_obs.get_state_dict()

        opt_states = {}
        for name, opt in self.optimizers.items():
            if opt is not None:
                opt_states[name] = opt.state_dict()
        checkpoint['optimizer_states'] = opt_states

        if emergency:
            save_path = self.output_dir / f'emergency_checkpoint_epoch_{epoch}.pth'
        else:
            save_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, save_path)
        
        if self.config.verbose:
            print(f"Checkpoint saved: {save_path}")
    
    def _save_final_model(self):
        """Save final model."""
        try:
            complete_config = self._build_complete_config_from_training_config()
        except KeyError as e:
            print(f"Model save error: {e}")
            raise

        model_state = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'df_state': self.df_state.get_inference_state_dict() if self.df_state else None,
            'df_obs': self.df_obs.get_inference_state_dict() if self.df_obs else None,
            'realization_config': self.realization.__dict__,
            'training_config': self.config.__dict__,
            'config': complete_config
        }

        models_dir = self.output_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = models_dir / 'final_model.pth'
        torch.save(model_state, save_path)

        print(f"Final model saved: {save_path}")
        print(f"   Encoder type: '{complete_config['model']['encoder'].get('type')}'")
        print(f"   Checkpoint structure: flat format (checkpoint['df_state'])")
        print(f"   Complete configuration saved")
    
    def _build_complete_config_from_training_config(self) -> Dict[str, Any]:
        """Build complete configuration for checkpoint from yaml_config."""
        if hasattr(self, 'yaml_config') and self.yaml_config is not None:
            encoder_type = self.yaml_config.get('model', {}).get('encoder', {}).get('type')
            if not encoder_type:
                raise KeyError(
                    "yaml_config['model']['encoder']['type'] not found.\n"
                    "Check model.encoder.type in the training YAML config."
                )

            return {
                'model': self.yaml_config.get('model', {}),
                'ssm': self.yaml_config.get('ssm', {})
            }

        raise KeyError(
            "Failed to build complete configuration.\n\n"
            "Cause: yaml_config not saved.\n"
            "Fix: Initialize TwoStageTrainer with config argument.\n"
            "Example: TwoStageTrainer(config=config_dict, device=device, output_dir=output_dir)"
        )
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        phase1_epochs = getattr(self.config, 'phase1_epochs', self.config.epochs)
        phase2_epochs = getattr(self.config, 'phase2_epochs', 0)

        summary = {
            'training_complete': self.phase1_complete,
            'total_epochs': {
                'phase1': phase1_epochs,
                'phase2': phase2_epochs if self.phase1_complete else 0
            },
            'final_losses': {},
            'model_info': {
                'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
                'decoder_params': sum(p.numel() for p in self.decoder.parameters()),
                'df_state_params': sum(p.numel() for p in self.df_state.phi_theta.parameters()) if self.df_state else 0,
                'df_obs_params': sum(p.numel() for p in self.df_obs.psi_omega.parameters()) if self.df_obs else 0
            }
        }
        
        if self.training_history['phase1_metrics']:
            summary['final_losses']['phase1'] = self.training_history['phase1_metrics'][-1]
        
        if self.training_history['phase2_losses']:
            summary['final_losses']['phase2'] = self.training_history['phase2_losses'][-1]
        
        return summary
