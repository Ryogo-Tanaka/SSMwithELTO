# =
# Time alignment + computation graph isolation + helper functions

"""
TwoStageTrainer: Two-stage training strategy implementation.

Phase-1: DF-A/DF-B Stage-1/Stage-2 alternating training
Phase-2: End-to-end fine-tuning

Training strategy:
**DF-A (State Layer)**:
for epoch in Phase1:
  for t = 1 to T1:  # Stage-1
    V_A^{(-k)} = closed_form(Phi_minus, Phi_plus, phi_theta fixed)
    phi_theta <- phi_theta - alpha * grad_L1(V_A^{(-k)}, phi_theta)

  for t = 1 to T2:  # Stage-2
    U_A = closed_form(H^{(cf)}_A, X_+)  # U_A update (closed-form only)

**DF-B (Observation Layer)**:
for epoch in Phase1:
  for t = 1 to T1:  # Stage-1
    V_B = closed_form(Phi_prev, Psi_curr)  # V_B (psi_omega fixed)
    phi_theta <- phi_theta - alpha * grad_L1(V_B, phi_theta)

  for t = 1 to T2:  # Stage-2
    U_B = closed_form(H^{(cf)}_B, M)    # U_B (phi_theta fixed)
    psi_omega <- psi_omega - alpha * grad_L2(U_B, psi_omega)

Phase-2: End-to-end fine-tuning
for epoch in Phase2:
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

# Component imports
from ..ssm.df_state_layer import DFStateLayer
from ..ssm.df_observation_layer import DFObservationLayer
from ..ssm.realization import Realization, StochasticRealizationWithEncoder, RealizationError
from ..models.architectures.time_invariant import time_invariantEncoder, time_invariantDecoder


class TrainingPhase(Enum):
    """Training phase definitions."""
    PHASE1_DF_A = "phase1_df_a"
    PHASE1_DF_B = "phase1_df_b"
    PHASE2_E2E = "phase2_e2e"


@dataclass
class TrainingConfig:
    """Structured training configuration."""
    # Training settings
    epochs: int = 100                # total epochs
    T1_iterations: int = 10          # Stage-1 iterations per epoch
    T2_iterations: int = 5           # Stage-2 iterations per epoch
    phase1_warmup_epochs: int = 5    # Phase-1 warmup before Phase-2 starts
    lambda_cca: float = 0.001        # CCA loss weight
    update_strategy: str = "encoder_decoder_only"  # "encoder_decoder_only" or "all"

    # Experiment mode
    experiment_mode: str = "reconstruction"          # "reconstruction" or "target_prediction"

    # Learning rates
    lr_phi: float = 1e-3     # phi_theta (state feature) lr
    lr_psi: float = 1e-3     # psi_omega (observation feature) lr
    lr_encoder: float = 1e-3 # encoder lr
    lr_decoder: float = 1e-3 # decoder lr

    # Logging and saving
    log_interval: int = 5    # log interval (epochs)
    save_interval: int = 10  # model save interval (epochs)
    verbose: bool = True     # verbose logging
    
    def __post_init__(self):
        """Post-init type conversion and validation."""
        # Ensure numeric types (YAML loading workaround)
        self.epochs = int(self.epochs)
        self.T1_iterations = int(self.T1_iterations)
        self.T2_iterations = int(self.T2_iterations)
        self.phase1_warmup_epochs = int(self.phase1_warmup_epochs)
        self.log_interval = int(self.log_interval)
        self.save_interval = int(self.save_interval)
        
        # Learning rate type conversion
        self.lr_phi = float(self.lr_phi)
        self.lr_psi = float(self.lr_psi)
        self.lr_encoder = float(self.lr_encoder)
        self.lr_decoder = float(self.lr_decoder)
        self.lambda_cca = float(self.lambda_cca)
        
        # String type normalization
        self.update_strategy = str(self.update_strategy)
        
        # Boolean conversion (handle string "true"/"false")
        if isinstance(self.verbose, str):
            self.verbose = self.verbose.lower() in ('true', '1', 'yes', 'on')
        else:
            self.verbose = bool(self.verbose)

    @classmethod
    def from_nested_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create flat TrainingConfig from nested config dict."""
        # Support unified training settings (new format)
        return cls(
            # Training settings
            epochs=config_dict.get('epochs', 100),
            T1_iterations=config_dict.get('T1_iterations', 10),
            T2_iterations=config_dict.get('T2_iterations', 5),
            phase1_warmup_epochs=config_dict.get('phase1_warmup_epochs', 5),

            # Learning rate settings
            lr_phi=config_dict.get('lr_phi', 1e-3),
            lr_psi=config_dict.get('lr_psi', 1e-3),
            lr_encoder=config_dict.get('lr_encoder', 1e-3),
            lr_decoder=config_dict.get('lr_decoder', 1e-3),
            lambda_cca=config_dict.get('lambda_cca', 0.001),
            update_strategy=config_dict.get('update_strategy', "all"),
            
# Log/save settings (from top level)
            log_interval=config_dict.get('log_interval', 5),
            save_interval=config_dict.get('checkpoint', {}).get('save_every', 10),
            verbose=config_dict.get('verbose', True)
        )


class TrainingLogger:
    """Training log manager."""
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file paths
        self.phase1_csv_path = self.output_dir / 'phase1_training.csv'
        self.phase2_csv_path = self.output_dir / 'phase2_training.csv'
        self.canonical_correlations_csv_path = self.output_dir / 'canonical_correlations.csv'

        # Log data
        self.phase1_logs = []
        self.phase2_logs = []
        self.canonical_correlations_logs = []
        
        # Initialize CSVs
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Initialize CSV file headers."""
        # Phase-1 CSV
        with open(self.phase1_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'stage', 'iteration', 'loss', 
                'lr_phi', 'lr_psi'
            ])
        
        # Phase-2 CSV
        with open(self.phase2_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'total_loss', 'rec_loss', 'cca_loss',
                'lr_encoder', 'lr_decoder', 'lr_phi', 'lr_psi'
            ])

        # Canonical correlations detail CSV
        with open(self.canonical_correlations_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'num_components', 'rho_sum', 'rho_min', 'rho_max', 'rho_mean',
                'rho_values'  # stored as JSON string
            ])
    
    def log_phase1(self, epoch: int, phase: TrainingPhase, stage: str, 
                   iteration: int, metrics: Dict[str, float], 
                   learning_rates: Dict[str, float]):
        """Record Phase-1 log entry."""
        log_entry = {
            'epoch': epoch,
            'phase': phase.value,
            'stage': stage,
            'iteration': iteration,
            'metrics': metrics,
            'learning_rates': learning_rates
        }
        
        self.phase1_logs.append(log_entry)
        
        # Write to CSV
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
        """Record Phase-2 log entry."""
        log_entry = {
            'epoch': epoch,
            'total_loss': total_loss,
            'rec_loss': rec_loss,
            'cca_loss': cca_loss,
            'learning_rates': learning_rates
        }
        
        self.phase2_logs.append(log_entry)
        
        # Write to CSV
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

        # Convert to numpy
        rho_values = canonical_correlations.detach().cpu().numpy()

        # Compute statistics
        rho_sum = float(rho_values.sum())
        rho_min = float(rho_values.min())
        rho_max = float(rho_values.max())
        rho_mean = float(rho_values.mean())
        num_components = len(rho_values)

        # Log entry
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

        # Write to CSV
        with open(self.canonical_correlations_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, phase, num_components, rho_sum, rho_min, rho_max, rho_mean,
                json.dumps(rho_values.tolist())  # store values as JSON
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
    
    1. Phase-1: DF-A/DF-B cooperative training
    2. Phase-2: End-to-end fine-tuning
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
        
        # Initialize from config dict if provided
        if config is not None:
            self._init_from_config(config, device, output_dir, use_kalman_filtering)
        else:
            # Initialize from individual arguments (legacy)
            self._init_from_args(encoder, decoder, realization, df_state_config, df_obs_config,
                               training_config, device, output_dir, use_kalman_filtering,
                               calibration_ratio, auto_inference_setup, None)
    
    def _init_from_config(self, config: Dict[str, Any], device: torch.device, output_dir: str,
                         use_kalman_filtering: bool):
        """Initialize from config dictionary."""
        # Store original YAML config (used by _save_final_model())
        self.yaml_config = config

        # Model initialization via Factory Pattern
        from ..models.encoder import build_encoder
        from ..models.decoder import build_decoder

        encoder_config = config['model']['encoder'].copy()
        decoder_config = config['model']['decoder'].copy()

        # Backward compatibility: support legacy config format
        if 'type' not in encoder_config:
            encoder_config['type'] = 'time_invariant'
        # Handle legacy type names like time_invariantEncoder
        if encoder_config['type'] == 'time_invariantEncoder':
            encoder_config['type'] = 'time_invariant'
        if decoder_config.get('type') == 'time_invariantDecoder':
            decoder_config['type'] = 'time_invariant'

        # Special config adjustment for time_invariant
        if encoder_config['type'] == 'time_invariant':
            if 'output_dim' not in encoder_config:
                encoder_config['output_dim'] = encoder_config.get('channels', 32)

        # Create encoder/decoder via Factory Pattern
        encoder = build_encoder(encoder_config)
        decoder = build_decoder(decoder_config)

        # Create target_decoder for experiment_mode
        target_decoder = None
        experiment_mode = config.get('training', {}).get('experiment_mode', 'reconstruction')
        if experiment_mode == "target_prediction" and 'target_decoder' in config['model']:
            target_decoder_config = config['model']['target_decoder'].copy()
            target_decoder = build_decoder(target_decoder_config, experiment_mode="target_prediction")
            print(f"target_decoder created: {target_decoder_config.get('type', 'unknown')}")
        elif experiment_mode == "target_prediction":
            print(f"target_prediction mode specified but target_decoder config not found")

        # Stochastic realization (prefer new class)
        realization_config = config['ssm']['realization']
        if config.get('evaluation', {}).get('use_new_realization', True):
            # StochasticRealizationWithEncoder requires encoder argument
            realization_config_copy = realization_config.copy()

            # Load feature mapping config
            feature_mapping_cfg = realization_config.get('feature_mapping', {})

            realization = StochasticRealizationWithEncoder(
                encoder=encoder,
                encoder_output_dim=realization_config_copy['encoder_output_dim'],
                past_horizon=realization_config_copy.get('past_horizon', 10),
                rank=realization_config_copy.get('rank', 8),
                ridge_param=realization_config_copy.get('ridge_param', 1e-3),
                jitter=realization_config_copy.get('jitter', 1e-8),
                m=realization_config_copy.get('m', 500),
                device=str(device),
                # Feature mapping config
                feature_mapping_type=feature_mapping_cfg.get('type', 'averaging'),
                feature_mapping_hidden_dims=feature_mapping_cfg.get('hidden_dims', None),
                feature_mapping_activation=feature_mapping_cfg.get('activation', 'relu')
            )
        else:
            realization = Realization(**realization_config)
        
        # Config conversion
        training_config = TrainingConfig.from_nested_dict(config['training'])
        
        # Delegate to individual-argument initialization
        calibration_ratio = config['training'].get('calibration_ratio', 0.25)
        auto_inference_setup = config['training'].get('auto_inference_setup', True)

        self._init_from_args(encoder, decoder, realization,
                           config['ssm']['df_state'], config['ssm']['df_observation'],
                           training_config, device, output_dir, use_kalman_filtering,
                           calibration_ratio, auto_inference_setup, target_decoder)

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

        # Load trained model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Unify checkpoint structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Get config: config_path first, then checkpoint['config']
        if config_path is not None:
            # Load config from YAML file
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            encoder_type = yaml_config.get('model', {}).get('encoder', {}).get('type')
            decoder_type = yaml_config.get('model', {}).get('decoder', {}).get('type')
            encoder_config = yaml_config.get('model', {}).get('encoder', {})
            decoder_config = yaml_config.get('model', {}).get('decoder', {})

            print(f"Loaded from YAML config: encoder={encoder_type}, decoder={decoder_type}")

            # Verify consistency with checkpoint['config'] (if exists)
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
            # No config_path: restore from checkpoint['config']
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

        # Validate Encoder/Decoder types
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

        # Dynamic Encoder/Decoder construction
        from ..models.encoder import build_encoder
        from ..models.decoder import build_decoder

        encoder = build_encoder(encoder_config).to(device)
        decoder = build_decoder(decoder_config).to(device)

        # Initialize with minimal config
        realization = Realization(past_horizon=10, rank=3)
        df_state_config = {'feature_dim': 16}
        df_obs_config = {'obs_feature_dim': 8}
        training_config = TrainingConfig()

        # Create instance
        instance = cls._init_from_args_direct(
            encoder, decoder, realization, df_state_config, df_obs_config,
            training_config, device, output_dir, use_kalman_filtering=False
        )

        # Load weights
        instance.encoder.load_state_dict(state_dict.get('encoder', {}))
        instance.decoder.load_state_dict(state_dict.get('decoder', {}))

        print(f"Model loaded: {encoder_type}Encoder + {decoder_type}Decoder")

        return instance

    @classmethod
    def _detect_encoder_structure(cls, encoder_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Detect encoder structure from parameters."""
        input_dim = 6  # default
        output_dim = 32  # default
        architecture = 'mlp'  # default
        hidden_dims = [64, 32]  # default

        # Auto-detect structure
        if not encoder_dict:
            # Empty: use defaults
            pass
        elif 'core_net.0.weight' in encoder_dict:
            # Detect time_invariant architecture
            architecture = 'mlp'

            # Input dim (first layer of core_net)
            input_dim = encoder_dict['core_net.0.weight'].shape[1]

            # output_dim (detect from output_mean)
            if 'output_mean' in encoder_dict:
                output_dim = encoder_dict['output_mean'].shape[0]
            else:
                # Estimate from last layer of core_net
                for key in encoder_dict.keys():
                    if key.startswith('core_net.') and key.endswith('.weight'):
                        output_dim = encoder_dict[key].shape[0]

            # Detect hidden layer dimensions
            hidden_dims = []
            layer_keys = [k for k in encoder_dict.keys() if k.startswith('core_net.') and k.endswith('.weight')]
            layer_keys.sort(key=lambda x: int(x.split('.')[1]))

            for i, key in enumerate(layer_keys[:-1]):  # exclude final layer
                hidden_dims.append(encoder_dict[key].shape[0])

        elif 'layers.0.weight' in encoder_dict:
            # Detect MLP structure
            architecture = 'mlp'

            # Input dim (first linear layer)
            input_dim = encoder_dict['layers.0.weight'].shape[1]

            # Detect hidden dims and output_dim
            hidden_dims = []
            max_layer_idx = -1

            for key in encoder_dict.keys():
                if key.startswith('layers.') and key.endswith('.weight'):
                    try:
                        layer_idx = int(key.split('.')[1])
                        max_layer_idx = max(max_layer_idx, layer_idx)

                        # Collect hidden layer output dims
                        if layer_idx > 0:  # exclude first layer
                            layer_output_dim = encoder_dict[key].shape[0]
                            if layer_idx == max_layer_idx:
                                output_dim = layer_output_dim  # final layer = output dim
                            else:
                                hidden_dims.append(layer_output_dim)
                    except (ValueError, IndexError):
                        continue

            if not hidden_dims:
                hidden_dims = [64, 32]  # default

        elif any(key.startswith('tcn.') for key in encoder_dict.keys()):
            # Legacy TCN structure (backward compat) - map to time_invariant
            architecture = 'mlp'  # map to time_invariant MLP

            if 'in_proj.weight' in encoder_dict:
                input_dim = encoder_dict['in_proj.weight'].shape[1]
                output_dim = encoder_dict['in_proj.weight'].shape[0]

        elif any(key.startswith('conv') for key in encoder_dict.keys()):
            # CNN-based structure
            architecture = 'resnet'  # map to time_invariant resnet

            # Estimate input dim from first conv layer
            first_conv_keys = [k for k in encoder_dict.keys() if 'conv' in k and 'weight' in k]
            if first_conv_keys:
                first_key = sorted(first_conv_keys)[0]
                # Get input channels from conv layer
                conv_shape = encoder_dict[first_key].shape
                if len(conv_shape) >= 2:
                    input_dim = conv_shape[1] if len(conv_shape) == 4 else conv_shape[1]
        else:
            # Other structures - minimal estimation
            weight_keys = [k for k in encoder_dict.keys() if 'weight' in k]
            if weight_keys:
                # Estimate from first weight layer
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
        input_dim = 32  # default (assumes match with encoder output)
        output_dim = 6  # default
        architecture = 'mlp'  # default
        hidden_dims = [64, 32]  # default

        # Auto-detect structure
        if not decoder_dict:
            # Empty: use defaults
            pass
        elif 'net.0.weight' in decoder_dict:
            # Detect time_invariantDecoder architecture
            architecture = 'mlp'

            # Input dim (first layer of net)
            input_dim = decoder_dict['net.0.weight'].shape[1]

            # Output dim (last layer of net)
            output_dim = 6  # default
            for key in decoder_dict.keys():
                if key.startswith('net.') and key.endswith('.weight'):
                    output_dim = decoder_dict[key].shape[0]

            # Detect hidden layer dimensions
            hidden_dims = []
            layer_keys = [k for k in decoder_dict.keys() if k.startswith('net.') and k.endswith('.weight')]
            layer_keys.sort(key=lambda x: int(x.split('.')[1]))

            for key in layer_keys[:-1]:  # exclude final layer
                hidden_dims.append(decoder_dict[key].shape[0])

        elif 'layers.0.weight' in decoder_dict:
            # Detect MLP structure
            architecture = 'mlp'

            # Input dim (first linear layer)
            input_dim = decoder_dict['layers.0.weight'].shape[1]

            # Detect hidden dims and output_dim
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
                hidden_dims = [64, 32]  # default

        elif 'out_proj.weight' in decoder_dict:
            # Legacy TCN structure (backward compat) - map to time_invariant
            architecture = 'mlp'
            output_dim = decoder_dict['out_proj.weight'].shape[0]

            if 'takens_proj.weight' in decoder_dict:
                input_dim = decoder_dict['takens_proj.weight'].shape[1]

        elif any(key.startswith('conv') for key in decoder_dict.keys()):
            # CNN-based structure
            architecture = 'resnet'

            # Estimate dims from conv layers
            conv_keys = [k for k in decoder_dict.keys() if 'conv' in k and 'weight' in k]
            if conv_keys:
                # Estimate output dim from last conv layer
                last_key = sorted(conv_keys)[-1]
                conv_shape = decoder_dict[last_key].shape
                if len(conv_shape) >= 2:
                    output_dim = conv_shape[0] if len(conv_shape) == 4 else conv_shape[0]
        else:
            # Other structures - minimal estimation
            weight_keys = [k for k in decoder_dict.keys() if 'weight' in k]
            if weight_keys:
                # Estimate output dim from last weight layer
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
                       auto_inference_setup: bool = True, target_decoder: nn.Module = None):
        """Initialize from individual arguments."""
        # Basic settings
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        # target_decoder setup (experiment_mode support)
        if target_decoder is not None:
            self.target_decoder = target_decoder.to(device)
            print(f"target_decoder configured: {type(target_decoder).__name__}")
        else:
            self.target_decoder = None

        # Move all parameters including component_transforms to device
        # realization inherits nn.Module, treat like encoder/decoder
        if hasattr(realization, 'to'):
            self.realization = realization.to(device)
        else:
            self.realization = realization

        self.config = training_config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Time alignment state management
        self._last_X_states_length = None  # state sequence length cache
        
        # =
        self.use_kalman_filtering = use_kalman_filtering
        self.calibration_ratio = calibration_ratio
        self.auto_inference_setup = auto_inference_setup
        
        # =
        self.calibration_data: Optional[torch.Tensor] = None

        # Store DF layer configs
        self.df_state_config = df_state_config
        self.df_obs_config = df_obs_config
        
        # Training state
        self.df_state = None
        self.df_obs = None
        self.optimizers = {}
        self.current_epoch = 0
        self.phase1_complete = False
        
        # Training history
        self.training_history = {
            'phase1_metrics': [],
            'phase2_losses': []
        }
        
        # Temporary data storage
        self._temp_data = {}

        # Log management
        self.logger = TrainingLogger(self.output_dir)

        # Static log display management (show only once)
        self._static_logs_shown = set()
        
        if 'trainer_init' not in self._static_logs_shown:
            print(f"TwoStageTrainer initialized: {device}")
            self._static_logs_shown.add('trainer_init')
    
    def _initialize_df_layers(self, X_states: torch.Tensor):
        """Initialize DF layers (GPU-unified)."""
        # DF-A initialization
        _, r = X_states.shape
        self.df_state = DFStateLayer(
            state_dim=r,
            feature_dim=self.df_state_config['feature_dim'],
            lambda_A=self.df_state_config['lambda_A'],
            lambda_B=self.df_state_config['lambda_B'],
            feature_net_config=self.df_state_config.get('feature_net'),
            cross_fitting_config=self.df_state_config.get('cross_fitting')
        )
        
        # Move DF-A internal neural networks to GPU
        self.df_state.phi_theta = self.df_state.phi_theta.to(self.device)
        
        # DF-B initialization
        self.df_obs = DFObservationLayer(
            df_state_layer=self.df_state,
            obs_feature_dim=self.df_obs_config['obs_feature_dim'],
            multivariate_feature_dim=self.df_obs_config['multivariate_feature_dim'],
            lambda_B=self.df_obs_config['lambda_B'],
            lambda_dB=self.df_obs_config['lambda_dB'],
            obs_net_config=self.df_obs_config.get('obs_net'),
            cross_fitting_config=self.df_obs_config.get('cross_fitting')
        )
        # Move DF-B internal neural networks to GPU
        self.df_obs.psi_omega = self.df_obs.psi_omega.to(self.device)
        
        if 'df_layers_init' not in self._static_logs_shown:
            print(f"DF layers initialized: state_dim={r}")
            self._static_logs_shown.add('df_layers_init')
    
    def _initialize_optimizers(self):
        """Initialize optimizers."""
        # Phase-1 individual optimizers
        self.optimizers['phi'] = torch.optim.Adam(
            self.df_state.phi_theta.parameters(), 
            lr=self.config.lr_phi
        )
        
        self.optimizers['psi'] = torch.optim.Adam(
            self.df_obs.psi_omega.parameters(), 
            lr=self.config.lr_psi
        )
        
        # Phase-2 unified optimizer (parameter selection based on update_strategy)
        param_groups = [
            {'params': list(self.encoder.parameters()), 'lr': self.config.lr_encoder},
            {'params': list(self.decoder.parameters()), 'lr': self.config.lr_decoder},
        ]

        if self.config.update_strategy in ("all", "joint_all"):
            param_groups.extend([
                {'params': list(self.df_state.phi_theta.parameters()), 'lr': self.config.lr_phi},
                {'params': list(self.df_obs.psi_omega.parameters()), 'lr': self.config.lr_psi},
            ])

        self.optimizers['e2e'] = torch.optim.Adam(param_groups)
        print(f"Phase-2 optimizer: {len(param_groups)} param groups (update_strategy={self.config.update_strategy})")
        
        print("Optimizers initialized")
    
    def _prepare_data(self, Y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Data preparation (StochasticRealizationWithEncoder support).

        Returns:
            M_features: Encoder output sequence (T, m) - multivariate features
            X_states: State estimation sequence (T_eff, r)
        """
        # Display data shape on first call only
        if 'input_data_shape' not in self._static_logs_shown:
            # Data shape check (debug)
            
            self._static_logs_shown.add('input_data_shape')

        # Move input data to GPU
        Y_train = self._ensure_device(Y_train)

        # 1. Encode: y_t -> m_t in R^m (multivariate features)
        with torch.no_grad():
            self.encoder.eval()  # accurate inference (BatchNorm etc.)

            # Unified encoder call
            M_batch = self.encoder(Y_train)  # (T, d) or (T, H, W, C) → (T, m)

            # Multivariate feature shape check
            if M_batch.dim() == 1:
                # Scalar output -> (T, 1)
                M_features = M_batch.unsqueeze(1)
            elif M_batch.dim() == 2:
                # Normal multivariate features (T, m)
                M_features = M_batch
            else:
                raise ValueError(f"Invalid encoder output shape: {M_batch.shape}. Expected: (T, m)")

        # Display encoder output shape on first call only
        if 'encoder_output_shape' not in self._static_logs_shown:
                        self._static_logs_shown.add('encoder_output_shape')

        # 2. Process multivariate features via StochasticRealizationWithEncoder
        
        # =
        if hasattr(self, 'use_kalman_filtering') and self.use_kalman_filtering:
            n_calib = int(Y_train.size(0) * getattr(self, 'calibration_ratio', 0.1))
            self.calibration_data = Y_train[:n_calib].clone()
            if self.config.verbose and 'calibration_data' not in self._static_logs_shown:
                print(f"Calibration data prepared: {self.calibration_data.shape}")
                self._static_logs_shown.add('calibration_data')

        # 2. Stochastic realization via StochasticRealizationWithEncoder
        try:
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                # Process multivariate features directly: Y_train + encoder -> x_t
                self.realization.fit(Y_train, self.encoder)
                X_states = self.realization.estimate_states(Y_train)
                if 'stochastic_realization_shapes' not in self._static_logs_shown:
                                        self._static_logs_shown.add('stochastic_realization_shapes')
            else:
                # Legacy Realization: reduce multivariate features to scalar
                m_scalar = M_features.mean(dim=1)  # (T, m) -> (T,) scalar features
                self.realization.fit(m_scalar.unsqueeze(1))  # (T,) → (T, 1)
                X_states = self.realization.filter(m_scalar.unsqueeze(1))
                if 'traditional_realization_shapes' not in self._static_logs_shown:
                                        self._static_logs_shown.add('traditional_realization_shapes')
        except RealizationError as e:
            print(f"RealizationError: {e}")
            # Re-raise RealizationError for full epoch skip
            raise RealizationError(f"Phase1 realization failed: {e}") from e

        # Record state sequence length (for time alignment)
        self._last_X_states_length = X_states.size(0)

        # Detach X_states as fixed data for Phase-1 (allows multiple backward passes)
        
        
        
        
        
        X_states = X_states.detach()

        if self.config.verbose and 'state_estimation_complete' not in self._static_logs_shown:
            print(f"State estimation complete: M_features={M_features.shape} -> X_states={X_states.shape}")
            self._static_logs_shown.add('state_estimation_complete')

        # Temporary storage (multivariate)
        self._temp_data = {
            'Y_train': Y_train,
            'M_features': M_features,  # multivariate features
            'X_states': X_states
        }

        return M_features, X_states  # return multivariate features

    def train_phase1(self, Y_train: torch.Tensor) -> Dict[str, Any]:
        """
        Phase-1: DF-A/DF-B cooperative training.
        
        Args:
            Y_train: Training observation sequence (T, d)
            
        Returns:
            Phase-1 metrics
        """
        print("\n=")

        # Data preparation (multivariate)
        M_features, X_states = self._prepare_data(Y_train)
        
        # Initialize DF layers
        self._initialize_df_layers(X_states)
        self._initialize_optimizers()
        
        # Phase-1 training loop
        for epoch in range(self.config.phase1_epochs):
            self.current_epoch = epoch
            epoch_metrics = {}
            
            # DF-A training
            df_a_metrics = self._train_df_a_epoch(X_states, epoch)
            epoch_metrics.update(df_a_metrics)
            
            # DF-B training (starts from warmup period)
            df_b_metrics = self._train_df_b_epoch(X_states, M_features, epoch)
            epoch_metrics.update(df_b_metrics)
            
            # Log entry
            self.training_history['phase1_metrics'].append(epoch_metrics)
            
            # Log output
            if epoch % self.config.log_interval == 0 and self.config.verbose:
                self._print_phase1_progress(epoch, epoch_metrics)
                # Additional multivariate log
                self.log_multivariate_training_progress(epoch, "phase1")
            
            # Save model
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, TrainingPhase.PHASE1_DF_A)
        
        # After Phase-1: compute V_A/V_B/U_A/U_B via DFLayer fit()
        print("Computing final operators (V_A/V_B/U_A/U_B)...")
        self._compute_final_operators(Y_train)

        self.phase1_complete = True
        print("Phase-1 training complete")

        return self.training_history['phase1_metrics']

    def _compute_final_operators(self, Y_train: torch.Tensor):
        """Compute final operators V_A/V_B/U_A/U_B after Phase-1 (multivariate)."""

        # Verify/fix model device state (safety after Phase-1)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # Ensure DF layer neural networks are on GPU
        if hasattr(self.df_state, 'phi_theta'):
            self.df_state.phi_theta = self.df_state.phi_theta.to(self.device)
        if hasattr(self.df_obs, 'psi_omega'):
            self.df_obs.psi_omega = self.df_obs.psi_omega.to(self.device)

        # Data preparation (multivariate)
        M_features, X_states = self._prepare_data(Y_train)

        # Data preparation for DFStateLayer
        with torch.no_grad():
            # State feature computation: phi_theta(x_t)
            Phi_full = self.df_state.phi_theta(X_states)  # (T, d_A)

            # Time-shifted data preparation (same alignment as training)
            Phi_minus = Phi_full[:-1]  # φ(x_{t-1}): t=0,...,T-2
            Phi_plus = Phi_full[1:]    # φ(x_t): t=1,...,T-1
            X_plus = X_states[1:]      # x_{t}: t=1,...,T-1 (same as training)

        # DF-A: Compute V_A, U_A
        print("  Computing DF-A operators (V_A/U_A)...")
        if hasattr(self.df_state, 'cf_config') and self.df_state.cf_config:
            self.df_state._fit_with_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose=True)
        else:
            self.df_state._fit_without_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose=True)

        if 'v_a_u_a_shapes' not in self._static_logs_shown:
                        self._static_logs_shown.add('v_a_u_a_shapes')

        # Data preparation for DFObservationLayer (using DF-A results)
        if hasattr(self, 'df_obs') and self.df_obs is not None:
            print("  Computing DF-B operators (V_B/U_B)...")

            # DF-A one-step prediction and correct encoder output usage
            with torch.no_grad():
                # One-step prediction: x_hat_{t|t-1} = U_A^T V_A phi(x_{t-1})
                X_pred = (self.df_state.U_A.T @ (self.df_state.V_A @ Phi_minus.T)).T  # (T-1, d_x)
                # Feature-map predictions: phi_theta(x_hat_{t|t-1})
                Phi_pred = self.df_state.phi_theta(X_pred)  # (T-1, d_A)

                # Get multivariate features considering realization time reduction
                # realization: T -> T_eff = T - 2*h + 1
                if isinstance(self.realization, StochasticRealizationWithEncoder):
                    # StochasticRealizationWithEncoder case
                    h = getattr(self.realization, 'window_length', 5)  # default
                else:
                    # Legacy Realization case
                    h = self.realization.h
                T_states = X_states.shape[0]  # state sequence length after realization

                # Correct range of multivariate features (same time range as realization)
                M_curr = M_features[h:h+T_states]  # M_t: aligned with realization range (T_states, m)

                # Observation features: psi_omega(M_t) - multivariate
                Psi_curr = self.df_obs.psi_omega(M_curr)  # (T_states, d_B)

                # Concurrent multivariate features
                m_curr = M_curr  # M_t (T_states, m)

                # Align data sizes (to minimum)
                min_size = min(Phi_pred.shape[0], Psi_curr.shape[0], m_curr.shape[0])
                Phi_pred = Phi_pred[:min_size]  # φ_θ(x̂_{t|t-1})
                Psi_curr = Psi_curr[:min_size]  # ψ_ω(h_t)
                m_curr = m_curr[:min_size]     # m_t

            if 'df_b_learning_data' not in self._static_logs_shown:
                # print(f"    DF-B: Phi_pred={Phi_pred.shape}, Psi_curr={Psi_curr.shape}")
                                self._static_logs_shown.add('df_b_learning_data')

            # DF-B: Compute V_B, U_B (phi_theta(x_hat) -> psi_omega(h_t) mapping)
            if hasattr(self.df_obs, 'cf_config') and self.df_obs.cf_config:
                self.df_obs._fit_with_cross_fitting(Phi_pred, Psi_curr, m_curr, verbose=True)
            else:
                self.df_obs._fit_without_cross_fitting(Phi_pred, Psi_curr, m_curr, verbose=True)

            if 'v_b_u_b_shapes' not in self._static_logs_shown:
                                self._static_logs_shown.add('v_b_u_b_shapes')

        print("Final operator computation complete")

        # Log canonical correlations at Phase-1 completion
        try:
            canonical_correlations = self._get_canonical_correlations_from_realization()
            if canonical_correlations is not None:
                # Log at Phase-1 completion (record as final epoch)
                # Use different attributes for unified vs separate training
                phase1_epochs = getattr(self.config, 'phase1_epochs', self.config.epochs)
                final_epoch = getattr(self, 'current_epoch', phase1_epochs - 1)
                self.logger.log_canonical_correlations(final_epoch, "Phase-1-Complete", canonical_correlations)

                if self.config.verbose:
                    rho_values = canonical_correlations.detach().cpu().numpy()
                    # print(f"  CCA: sum={rho_values.sum():.4f}")
        except Exception as e:
            print(f"Error logging canonical correlations at Phase-1 completion: {e}")

    def _train_df_a_epoch(self, X_states: torch.Tensor, epoch: int) -> Dict[str, float]:
        """
        DF-A (state layer) epoch training.

        Runs Stage-1 T1_iterations times and Stage-2 T2_iterations times.
        """
        metrics = {}
        opt_phi = self.optimizers['phi']
        X_states_gpu = X_states.to(self.device)

        # Stage-1: T1_iterations calls
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

        # Stage-2: T2_iterations calls
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

        # Move input data to GPU
        X_states_gpu = X_states.to(self.device)
        M_features_gpu = M_features.to(self.device)

        # Get state predictions from DF-A (inference only, as instrumental variables)
        with torch.no_grad():
            X_hat_states = self.df_state.predict_sequence(X_states_gpu)

        # # Diagnostic: X_hat accuracy check
        # if epoch == self.config.phase1_warmup_epochs and not hasattr(self, '_xhat_accuracy_diag_logged'):
        #     with torch.no_grad():
        #         X_true = X_states_gpu[1:]
        #         if X_hat_states.size(0) == X_true.size(0):
        #             prediction_error = torch.norm(X_hat_states - X_true, p='fro') ** 2 / X_true.numel()
        #             X_norm = torch.norm(X_true, p='fro') ** 2 / X_true.numel()
        #             relative_error = (prediction_error / X_norm).item() if X_norm > 0 else 0.0
        #             print(f"[Diag] X_hat pred MSE: {prediction_error.item():.6e}, rel_err: {relative_error:.2%}")
        #         self._xhat_accuracy_diag_logged = True

        # Time index adjustment (multivariate)
        M_aligned = self._align_time_series_multivariate(
            X_hat_states, M_features_gpu, X_states.size(0), epoch, "DF-B"
        )

        # Stage-1: T1_iterations calls
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

            # Log entry
            self.logger.log_phase1(
                epoch, TrainingPhase.PHASE1_DF_B, 'stage1', t,
                stage1_metrics, {'lr_phi': opt_phi.param_groups[0]['lr']}
            )

        metrics['df_b_stage1_loss'] = sum(stage1_losses) / len(stage1_losses)
        metrics['df_b_stage1_pred'] = sum(stage1_pred_losses) / len(stage1_pred_losses)
        metrics['df_b_stage1_reg'] = sum(stage1_reg_losses) / len(stage1_reg_losses)

        # Stage-2: T2_iterations calls
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

            # Log entry
            self.logger.log_phase1(
                epoch, TrainingPhase.PHASE1_DF_B, 'stage2', t,
                stage2_metrics, {'lr_psi': opt_psi.param_groups[0]['lr']}
            )

        metrics['df_b_stage2_loss'] = sum(stage2_losses) / len(stage2_losses)
        metrics['df_b_stage2_pred'] = sum(stage2_pred_losses) / len(stage2_pred_losses)
        metrics['df_b_stage2_reg'] = sum(stage2_reg_losses) / len(stage2_reg_losses)

        return metrics
    
    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        # Move tensor to specified device (if needed)
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def train_phase2(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None,
                     target_train: Optional[torch.Tensor] = None, target_val: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Phase-2: End-to-end fine-tuning.
        
        Fixed inference path:
        x̂_{t|t-1} = U_A^T V_A φ_θ(x_{t-1})
        m̂_{t|t-1} = u_B^T V_B φ_θ(x̂_{t|t-1})
        ŷ_{t|t-1} = g_α(m̂_{t|t-1})
        
        Args:
            Y_train: Training observation sequence
            Y_val: Validation observation sequence (optional)
        """
        print("\n=")

        # Initialize device state at Phase-2 start (reset after Phase-1)
        if 'device_allocation' not in self._static_logs_shown:
            print(f"Device ensured: encoder -> {self.device}, decoder -> {self.device}")
            self._static_logs_shown.add('device_allocation')
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # Ensure GPU device consistency
        Y_train = self._ensure_device(Y_train)
        if Y_val is not None:
            Y_val = self._ensure_device(Y_val)
        
        if not self.phase1_complete:
            raise RuntimeError("Phase-1 not completed")
        
        opt_e2e = self.optimizers['e2e']
        
        # Phase-2 training loop
        for epoch in range(self.config.phase2_epochs):
            self.current_epoch = self.config.phase1_epochs + epoch
            
            try:
                # Forward inference and loss computation (experiment_mode aware)
                loss_total, rec_loss, cca_loss = self._forward_and_loss_phase2(Y_train, target_train)
                
                # Backpropagation
                opt_e2e.zero_grad()
                loss_total.backward()
                opt_e2e.step()
                
            except RealizationError as e:
                print(f"Epoch {epoch} skipped (Phase2 realization failure): {e}")
                # Fully skip this epoch and proceed to next
                continue
            
            # Log entry
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
            
            # Progress display
            if epoch % self.config.log_interval == 0 and self.config.verbose:
                print(f"Phase-2 Epoch {epoch}: Total={loss_total.item():.6f}, "
                      f"Rec={rec_loss.item():.6f}, CCA={cca_loss.item():.6f}")
                # Additional multivariate log
                self.log_multivariate_training_progress(epoch, "phase2")
            
            # Save model
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, TrainingPhase.PHASE2_E2E)
        
        print("Phase-2 training complete")
        return self.training_history['phase2_losses']

    def train_integrated(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None,
                         target_train: Optional[torch.Tensor] = None, target_val: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Unified training: run Phase-1 and Phase-2 consecutively each epoch.

        Args:
            Y_train: Training observation sequence (T, d)
            Y_val: Validation observation sequence (optional)

        Returns:
            Unified training results
        """
        print(f"\n=")

        # Initial data preparation
        M_features, X_states = self._prepare_data(Y_train)

        # Initialize DF layers (all modes unified)
        self._initialize_df_layers(X_states)
        self._initialize_optimizers()

        # Initialize unified training history
        self.training_history = {
            'phase1_metrics': [],
            'phase2_losses': [],
            'integrated_metrics': []
        }

        # Unified training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            try:
                # At active period start: recompute X_states after encoder update
                if epoch >= self.config.phase1_warmup_epochs:
                    with torch.no_grad():
                        # Reflect encoder update from Phase-2: recompute M_features
                        M_features = self.encoder(Y_train)

                        if isinstance(self.realization, StochasticRealizationWithEncoder):
                            self.realization.fit(Y_train, self.encoder)
                            X_states = self.realization.estimate_states(Y_train)
                        else:
                            # M_features already updated above
                            m_scalar = M_features.mean(dim=1)
                            self.realization.fit(m_scalar.unsqueeze(1))
                            X_states = self.realization.filter(m_scalar.unsqueeze(1))

                # Phase-1: DF training (skipped in joint_all mode)
                if self.config.update_strategy == "joint_all":
                    # joint_all mode: skip Phase-1 training
                    if epoch == 0:
                        print("[joint_all] Phase-1 skipped (joint_all mode)")
                        print("[joint_all] Computing initial operators...")

                        # Initial operator computation (from randomly initialized phi_theta/psi_omega)
                        self._compute_operators_for_joint_all(Y_train)

                        # Phase-1 completion flag (operators computed)
                        self.phase1_complete = True
                        self.df_state._is_fitted = True
                        self.df_obs._is_fitted = True
                        print("[joint_all] Initial operators computed (proceeding to Phase-2)")

                    # Phase-1 loss retrieved later from Phase-2 (after operator computation)
                    phase1_metrics = {
                        'df_a_stage1_pred': 0.0,
                        'df_a_stage1_reg': 0.0,
                        'df_a_stage2_pred': 0.0,
                        'df_a_stage2_reg': 0.0,
                        'df_b_stage1_pred': 0.0,
                        'df_b_stage1_reg': 0.0,
                        'df_b_stage2_pred': 0.0,
                        'df_b_stage2_reg': 0.0
                    }
                else:
                    # Staged training mode: execute Phase-1
                    phase1_metrics = self._train_integrated_phase1_epoch(X_states, M_features, epoch)

                    # Lightweight operator update after Phase-1
                    self._lightweight_operator_update(Y_train)

                # Phase-2: End-to-end training (joint_all: from epoch 0, others: after warmup)
                phase2_metrics = {}
                if self.config.update_strategy == "joint_all" or epoch >= self.config.phase1_warmup_epochs:
                    phase2_metrics = self._train_integrated_phase2_epoch(Y_train, epoch, target_train)

                    # joint_all mode: merge validation loss from Phase-2 into phase1_metrics
                    if self.config.update_strategy == "joint_all":
                        phase1_metrics.update({
                            'df_a_stage1_pred': phase2_metrics.get('df_a_stage1_pred', 0.0),
                            'df_a_stage1_reg': phase2_metrics.get('df_a_stage1_reg', 0.0),
                            'df_a_stage2_pred': phase2_metrics.get('df_a_stage2_pred', 0.0),
                            'df_a_stage2_reg': phase2_metrics.get('df_a_stage2_reg', 0.0),
                            'df_b_stage1_pred': phase2_metrics.get('df_b_stage1_pred', 0.0),
                            'df_b_stage1_reg': phase2_metrics.get('df_b_stage1_reg', 0.0),
                            'df_b_stage2_pred': phase2_metrics.get('df_b_stage2_pred', 0.0),
                            'df_b_stage2_reg': phase2_metrics.get('df_b_stage2_reg', 0.0)
                        })
                else:
                    # Skip Phase-2 during warmup period
                    phase2_metrics = {
                        'total_loss': 0.0,
                        'rec_loss': 0.0,
                        'cca_loss': 0.0
                    }

                # Record unified metrics
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

                # Simple one-line log (show prediction and regularization losses separately)
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

                # Save model
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

        # After unified training: compute final operators (V_A/V_B/U_A/U_B)
        print("Computing final operators (V_A/V_B/U_A/U_B)...")
        self._compute_final_operators(Y_train)

        self.phase1_complete = True
        print(f"Unified training complete ({self.config.epochs} epochs)")
        return self.training_history

    def _train_integrated_phase1_epoch(self, X_states: torch.Tensor, M_features: torch.Tensor,
                                     epoch: int) -> Dict[str, Any]:
        """
        Execute one epoch of Phase-1 in unified training.
        Same formulation-based implementation as _train_df_a_epoch, _train_df_b_epoch.
        """
        epoch_metrics = {}

        # DF-A training (reuses existing implementation)
        df_a_metrics = self._train_df_a_epoch(X_states, epoch)
        epoch_metrics.update(df_a_metrics)

        # DF-B training (starts from warmup, reuses existing implementation)
        df_b_metrics = self._train_df_b_epoch(X_states, M_features, epoch)
        epoch_metrics.update(df_b_metrics)

        return epoch_metrics

    def _train_integrated_phase2_epoch(self, Y_train: torch.Tensor, epoch: int, target_data: torch.Tensor = None) -> Dict[str, float]:
        """
        Execute one epoch of Phase-2 in unified training.
        Applies same device management and error handling as train_phase2.
        """
        # joint_all mode: recompute operators every epoch (epoch >= 1)
        validation_losses = None
        if self.config.update_strategy == "joint_all" and epoch >= 1:
            # Recompute operators since phi_theta/psi_omega were updated in previous epoch
            self._compute_operators_for_joint_all(Y_train)

        # joint_all mode: compute validation loss after operator computation (all epochs, for display)
        if self.config.update_strategy == "joint_all":
            # Get M_features and X_states (same data used in Phase-2)
            M_features = self.encoder(Y_train)
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                self.realization.fit(Y_train, self.encoder)
                X_states = self.realization.estimate_states(Y_train)
            else:
                m_scalar = M_features.mean(dim=1)
                self.realization.fit(m_scalar.unsqueeze(1))
                X_states = self.realization.filter(m_scalar.unsqueeze(1))

            # Compute validation loss (no grad, for display)
            validation_losses = self._compute_validation_losses_for_joint_all(M_features, X_states)

        # Ensure device consistency (existing pattern)
        Y_train = self._ensure_device(Y_train)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # Set Phase-2 training mode (important for CCA loss gradient flow)
        self.encoder.train()
        self.decoder.train()

        # v4: Set DF layers to eval mode in Phase-2 (disable dropout)
        if hasattr(self, 'df_state') and self.df_state is not None:
            self.df_state.eval()
        if hasattr(self, 'df_obs') and self.df_obs is not None:
            self.df_obs.eval()

        # Initialize Phase-2 optimizer if not yet done
        if 'e2e' not in self.optimizers:
            self._initialize_phase2_optimizer()

        opt_e2e = self.optimizers['e2e']

        # # Diagnostic: Phase-2 learning rate check
        # if epoch == self.config.phase1_warmup_epochs and not hasattr(self, '_phase2_lr_logged'):
        #     for idx, (name, group) in enumerate(zip(['encoder', 'decoder', 'phi', 'psi'], opt_e2e.param_groups)):
        #         print(f"[Diag] Phase-2 LR {name}: {group['lr']:.6e}")
        #     self._phase2_lr_logged = True

        try:
            # Forward inference and loss computation (experiment_mode aware)
            loss_total, rec_loss, cca_loss = self._forward_and_loss_phase2(Y_train, target_data)

            # Backpropagation (standard pattern)
            opt_e2e.zero_grad()
            loss_total.backward()

            # Gradient flow diagnostics: print encoder/decoder grad norms at first active epoch
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

            # Log entry (dynamic key generation based on param_groups count)
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

            # joint_all mode: add validation loss
            if validation_losses is not None:
                result.update(validation_losses)

            return result

        except RealizationError as e:
            print(f"Epoch {epoch} Phase-2 skipped (realization failure): {e}")
            # Error handling matching existing pattern
            result = {
                'epoch': epoch,
                'total_loss': 0.0,
                'rec_loss': 0.0,
                'cca_loss': 0.0
            }

            # joint_all mode: return 0 for validation loss on error
            if self.config.update_strategy == "joint_all":
                result.update({
                    'df_a_stage1_pred': 0.0,
                    'df_a_stage1_reg': 0.0,
                    'df_a_stage2_pred': 0.0,
                    'df_a_stage2_reg': 0.0,
                    'df_b_stage1_pred': 0.0,
                    'df_b_stage1_reg': 0.0,
                    'df_b_stage2_pred': 0.0,
                    'df_b_stage2_reg': 0.0
                })

            return result

    def _lightweight_operator_update(self, Y_train: torch.Tensor):
        """
        Lightweight operator update after Phase-1.
        Based on _compute_final_operators pattern.
        """
        try:
            # Check/fix device state (existing pattern)
            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)

            # Ensure DF layer neural networks on GPU (existing pattern)
            if hasattr(self.df_state, 'phi_theta'):
                self.df_state.phi_theta = self.df_state.phi_theta.to(self.device)
            if hasattr(self.df_obs, 'psi_omega'):
                self.df_obs.psi_omega = self.df_obs.psi_omega.to(self.device)

            # Lightweight realization update (simplified from full _compute_final_operators)
            # v4: suppress computation graph with torch.no_grad() (prevent memory/cache issues)
            with torch.no_grad():
                if hasattr(self.realization, 'fit') and isinstance(self.realization, StochasticRealizationWithEncoder):
                    self.realization.fit(Y_train, self.encoder)

        except Exception as e:
            print(f"Lightweight operator update error: {e}")

    def _compute_operators_for_joint_all(self, Y_train: torch.Tensor):
        """
        Recompute operators dynamically for joint_all mode.

        Design intent:
        - phi_theta/psi_omega are updated in Phase-2, so recompute operators every epoch
        - Same logic as _compute_final_operators(), but called every epoch
        - V_A/U_A derived from phi_theta via Ridge estimation

        Args:
            Y_train: Observation data
        """
        with torch.no_grad():
            # Ensure device state
            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)
            if hasattr(self.df_state, 'phi_theta'):
                self.df_state.phi_theta = self.df_state.phi_theta.to(self.device)
            if hasattr(self.df_obs, 'psi_omega'):
                self.df_obs.psi_omega = self.df_obs.psi_omega.to(self.device)

            # Data preparation
            M_features, X_states = self._prepare_data(Y_train)

            # Recompute DF-A operators
            Phi_full = self.df_state.phi_theta(X_states)  # (T, d_A)
            Phi_minus = Phi_full[:-1]  # (T-1, d_A)
            Phi_plus = Phi_full[1:]    # (T-1, d_A)
            X_plus = X_states[1:]      # (T-1, r)

            # Compute V_A, U_A (Ridge estimation)
            if hasattr(self.df_state, 'cf_config') and self.df_state.cf_config:
                self.df_state._fit_with_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose=False)
            else:
                self.df_state._fit_without_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose=False)

            # Recompute DF-B operators
            if hasattr(self, 'df_obs') and self.df_obs is not None:
                # DF-A one-step prediction
                X_pred = (self.df_state.U_A.T @ (self.df_state.V_A @ Phi_minus.T)).T  # (T-1, r)
                Phi_pred = self.df_state.phi_theta(X_pred)  # (T-1, d_A)

                # Account for realization time reduction
                if isinstance(self.realization, StochasticRealizationWithEncoder):
                    h = getattr(self.realization, 'window_length', 5)
                else:
                    h = self.realization.h
                T_states = X_states.shape[0]

                # Correct range of multivariate features
                M_curr = M_features[h:h+T_states]  # (T_states, m)
                Psi_curr = self.df_obs.psi_omega(M_curr)  # (T_states, d_B)

                # Data size adjustment
                min_size = min(Phi_pred.shape[0], Psi_curr.shape[0], M_curr.shape[0])
                Phi_pred = Phi_pred[:min_size]
                Psi_curr = Psi_curr[:min_size]
                m_curr = M_curr[:min_size]

                # Compute V_B, U_B
                if hasattr(self.df_obs, 'cf_config') and self.df_obs.cf_config:
                    self.df_obs._fit_with_cross_fitting(Phi_pred, Psi_curr, m_curr, verbose=False)
                else:
                    self.df_obs._fit_without_cross_fitting(Phi_pred, Psi_curr, m_curr, verbose=False)

    def _compute_validation_losses_for_joint_all(
        self,
        M_features: torch.Tensor,
        X_states: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute validation losses for joint_all mode (no gradients, for display).

        Purpose:
        - Compute losses for display only (not used in training) in joint_all mode
        - Return prediction and regularization losses separately in same format

        Design:
        - Stage-1 loss: always 0.0 (Phase-1 skipped)
        - Stage-2 loss: actual prediction and regularization losses

        Args:
            M_features: Multivariate features (T, m)
            X_states: State data (T, r)

        Returns:
            Dict with keys:
                - df_a_stage1_pred, df_a_stage1_reg: always 0.0
                - df_a_stage2_pred, df_a_stage2_reg: DF-A Stage-2 prediction/regularization loss
                - df_b_stage1_pred, df_b_stage1_reg: always 0.0
                - df_b_stage2_pred, df_b_stage2_reg: DF-B Stage-2 prediction/regularization loss
        """
        with torch.no_grad():
            # Stage-1 loss: 0.0 (Phase-1 skipped)
            losses = {
                'df_a_stage1_pred': 0.0,
                'df_a_stage1_reg': 0.0,
                'df_b_stage1_pred': 0.0,
                'df_b_stage1_reg': 0.0
            }

            # DF-A Stage-2 loss computation
            # Eq. (42a): L_Stage-2 = ||X^+ - U_A^T H||^2_F / T + lambda_B ||U_A||^2_F
            Phi_full = self.df_state.phi_theta(X_states)  # (T, d_A)
            Phi_minus = Phi_full[:-1]  # (T-1, d_A)
            X_plus = X_states[1:]      # (T-1, r)

            # H = V_A @ Phi^-
            H = self.df_state.V_A @ Phi_minus.T  # (r, T-1)

            # Prediction error: ||X^+ - U_A^T H||^2_F / T
            X_pred = (self.df_state.U_A.T @ H).T  # (T-1, r)
            prediction_error = torch.norm(X_plus - X_pred, p='fro') ** 2
            T_samples = X_plus.shape[0]
            df_a_stage2_pred = (prediction_error / T_samples).item()

            # Regularization: lambda_B ||U_A||^2_F
            df_a_stage2_reg = (self.df_state.lambda_B * torch.norm(self.df_state.U_A, p='fro') ** 2).item()

            losses['df_a_stage2_pred'] = df_a_stage2_pred
            losses['df_a_stage2_reg'] = df_a_stage2_reg

            # DF-B Stage-2 loss computation
            if hasattr(self, 'df_obs') and self.df_obs is not None:
                # DF-A one-step prediction
                X_pred = (self.df_state.U_A.T @ H).T  # (T-1, r)
                Phi_pred = self.df_state.phi_theta(X_pred)  # (T-1, d_A)

                # Adjust realization time range
                if isinstance(self.realization, StochasticRealizationWithEncoder):
                    h = getattr(self.realization, 'window_length', 5)
                else:
                    h = self.realization.h
                T_states = X_states.shape[0]

                # Correct multivariate feature range
                M_curr = M_features[h:h+T_states]  # (T_states, m)
                Psi_curr = self.df_obs.psi_omega(M_curr)  # (T_states, d_B)

                # Data size adjustment
                min_size = min(Phi_pred.shape[0], Psi_curr.shape[0], M_curr.shape[0])
                Phi_pred = Phi_pred[:min_size]
                Psi_curr = Psi_curr[:min_size]
                m_curr = M_curr[:min_size]

                # Eq. (42b): L_Stage-2 = ||M - U_B^T H_B||^2_F / T + lambda_dB ||U_B||^2_F
                # where H_B = V_B @ Phi_pred.T: (d_B, T)
                H_B = self.df_obs.V_B @ Phi_pred.T  # (d_B, T) = (25, 1010)
                m_pred = (self.df_obs.U_B.T @ H_B).T  # (m, T) → (T, m) = (1010, 50)

                # Prediction error: ||m - m_pred||^2_F / T
                prediction_error_b = torch.norm(m_curr - m_pred, p='fro') ** 2
                T_samples_b = m_curr.shape[0]
                df_b_stage2_pred = (prediction_error_b / T_samples_b).item()

                # Regularization: lambda_dB ||U_B||^2_F
                df_b_stage2_reg = (self.df_obs.lambda_dB * torch.norm(self.df_obs.U_B, p='fro') ** 2).item()

                losses['df_b_stage2_pred'] = df_b_stage2_pred
                losses['df_b_stage2_reg'] = df_b_stage2_reg
            else:
                losses['df_b_stage2_pred'] = 0.0
                losses['df_b_stage2_reg'] = 0.0

            return losses

    def _initialize_phase2_optimizer(self):
        """
        Initialize Phase-2 optimizer (for unified training).

        Design:
        - Phase-2 updates encoder/decoder only
        - DF layers (phi_theta, psi_omega) are fixed (trained in Phase-1)
        - End-to-end optimization via CCA loss adjusts encoder/decoder
        """
        if 'e2e' not in self.optimizers:
            # Phase-2 design: update encoder/decoder only
            param_groups = [
                {'params': list(self.encoder.parameters()), 'lr': self.config.lr_encoder},
                {'params': list(self.decoder.parameters()), 'lr': self.config.lr_decoder}
            ]

            # =
            # Include feature mapping parameters for StochasticRealizationWithEncoder
            if hasattr(self.realization, 'component_transforms') and \
               self.realization.component_transforms is not None:
                # Update component_transforms params with same lr as encoder
                param_groups.append({
                    'params': list(self.realization.component_transforms.parameters()),
                    'lr': self.config.lr_encoder  # same lr as encoder
                })

            # DF layer parameter handling (based on update_strategy)
            if self.config.update_strategy == "all":
                # Experimental option: also update DF layers (Phase-1 still runs)
                param_groups.extend([
                    {'params': list(self.df_state.phi_theta.parameters()), 'lr': self.config.lr_phi},
                    {'params': list(self.df_obs.psi_omega.parameters()), 'lr': self.config.lr_psi}
                ])
                print("Phase-2 also updates DF layers (update_strategy='all', staged+Phase-2 DF update)")
            elif self.config.update_strategy == "joint_all":
                # All-parameter joint training (Ablation Study baseline)
                # Train phi_theta/psi_omega in Phase-2, recompute operators every epoch
                param_groups.extend([
                    {'params': list(self.df_state.phi_theta.parameters()), 'lr': self.config.lr_phi},
                    {'params': list(self.df_obs.psi_omega.parameters()), 'lr': self.config.lr_psi}
                ])
                print("Joint all-parameter training mode (update_strategy='joint_all')")
                print("  - Training targets: encoder + decoder + phi_theta + psi_omega")
                print("  - Operators: recomputed from phi_theta/psi_omega every epoch")
            else:
                # Standard design: update encoder/decoder only
                print("Phase-2 updates encoder/decoder only (staged training design)")

            self.optimizers['e2e'] = torch.optim.Adam(param_groups)

    def _forward_and_loss_phase2(self, Y_train: torch.Tensor, target_data: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase-2 forward inference and loss computation (experiment_mode aware).
        Args:
            Y_train: Observation data
            target_data: Target data (required for target_prediction mode)
        Returns:
            (loss_total, primary_loss, loss_cca)
        """
        # Branch by experiment_mode
        if self.config.experiment_mode == "target_prediction":
            if target_data is None:
                raise ValueError("Target data required for target_prediction mode")
            return self._forward_and_loss_phase2_target(Y_train, target_data)
        else:  # reconstruction mode
            return self._forward_and_loss_phase2_reconstruction(Y_train)

    def _forward_and_loss_phase2_reconstruction(self, Y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase-2 reconstruction mode loss computation (with time alignment helpers).
        """
        # Ensure device state at Phase-2 start (safety after Phase-1)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # Get time series length (works for both image and time series data)
        T = Y_train.shape[0]
        h = self.realization.h

        if T <= 2 * h:
            # Short time series not supported
            raise RuntimeError(f"Time series too short: T({T}) <= 2*h({2*h})")

        # Step 1-2: Stochastic realization (encoder -> feature matrices -> CCA -> states)
        try:
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                # StochasticRealizationWithEncoder: calls encoder internally
                self.realization.fit(Y_train, self.encoder)
                X_states = self.realization.estimate_states(Y_train)
            else:
                # Legacy Realization: encoder -> scalar externally
                M_features = self.encoder(Y_train)  # (T, d) or (T, H, W, C) → (T, m)
                if M_features.dim() == 1:
                    M_features = M_features.unsqueeze(1)  # (T,) → (T, 1)
                m_scalar = M_features.mean(dim=1)  # (T, m) → (T,)
                self.realization.fit(m_scalar.unsqueeze(1))
                X_states = self.realization.filter(m_scalar.unsqueeze(1))
        except RealizationError as e:
            print(f"Phase2 RealizationError: {e}")
            # Re-raise RealizationError for full epoch skip
            raise RealizationError(f"Phase2 realization failed: {e}") from e

        # Step 3: DF-A prediction x_{t-1} -> x_hat_{t|t-1} (retain gradients for end-to-end)
        X_hat_states = self.df_state.predict_sequence(X_states, training=True)  # (T_pred, r)
        T_pred = X_hat_states.size(0)

        # Step 4: DF-B prediction x_hat_{t|t-1} -> m_hat_{t|t-1} (multivariate)
        M_hat_series = []
        for t in range(T_pred):
            m_hat_t = self.df_obs.predict_one_step(X_hat_states[t])  # m in R^m (multivariate)
            M_hat_series.append(m_hat_t)
        M_hat_tensor = torch.stack(M_hat_series)  # (T_pred, m)
        M_hat_tensor = self._ensure_device(M_hat_tensor)  # ensure GPU device consistency

        # Step 5: Decode m_hat_{t|t-1} -> y_hat_{t|t-1} (reconstruction)
        Y_hat = self.decoder(M_hat_tensor)  # (T_pred, n) - image reconstruction

        # Step 6: Get corresponding ground truth
        Y_target = Y_train[h+1:h+1+T_pred]  # corresponding observations
        Y_target = self._ensure_device(Y_target)  # ensure GPU device consistency

        # Loss computation (reconstruction error)
        loss_rec = torch.norm(Y_hat - Y_target, p='fro') ** 2 / Y_target.numel()

        # CCA loss (optional)
        if self.config.lambda_cca > 0:
            loss_cca = self._compute_cca_loss()
        else:
            loss_cca = torch.tensor(0.0, requires_grad=True)

        loss_total = loss_rec + self.config.lambda_cca * loss_cca

        return loss_total, loss_rec, loss_cca

    def _forward_and_loss_phase2_target(self, Y_train: torch.Tensor, target_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase-2 target prediction mode loss computation.
        Args:
            Y_train: Observation data (images)
            target_data: Target data (control states etc.)
        """
        # Ensure device state at Phase-2 start
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # Get time series length
        T = Y_train.shape[0]
        h = self.realization.h

        if T <= 2 * h:
            # Short time series not supported
            raise RuntimeError(f"Time series too short: T({T}) <= 2*h({2*h})")

        # Step 1: Encode y_t -> m_t
        M_features = self.encoder(Y_train)  # (T, m)

        if M_features.dim() == 1:
            M_features = M_features.unsqueeze(1)

        # Step 2: Stochastic realization M_t -> x_t
        try:
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                self.realization.fit(Y_train, self.encoder)
                X_states = self.realization.estimate_states(Y_train)
            else:
                m_scalar = M_features.mean(dim=1)
                self.realization.fit(m_scalar.unsqueeze(1))
                X_states = self.realization.filter(m_scalar.unsqueeze(1))
        except RealizationError as e:
            print(f"Phase2 Target RealizationError: {e}")
            raise RealizationError(f"Phase2 target realization failed: {e}") from e

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

        # Step 5: Decode m_hat -> target_pred (experiment_mode aware)
        if hasattr(self, 'target_decoder') and self.target_decoder is not None:
            target_pred = self.target_decoder(M_hat_tensor)  # target prediction mode
            print(f"Target prediction mode: using target_decoder -> {target_pred.shape}")
        else:
            target_pred = self.decoder(M_hat_tensor)  # reconstruction mode (backward compat)
            print(f"Target prediction mode: target_decoder not set, using decoder -> {target_pred.shape}")

        # (T_pred, state_dim) - target prediction

        # Step 6: Get corresponding target ground truth
        target_true = target_data[h+1:h+1+T_pred]
        target_true = self._ensure_device(target_true)

        # Loss computation (target prediction error)
        loss_target = torch.nn.functional.mse_loss(target_pred, target_true)

        # CCA loss (optional)
        if self.config.lambda_cca > 0:
            loss_cca = self._compute_cca_loss()
        else:
            loss_cca = torch.tensor(0.0, requires_grad=True)

        loss_total = loss_target + self.config.lambda_cca * loss_cca

        return loss_total, loss_target, loss_cca
    
    
    def _compute_cca_loss(self) -> torch.Tensor:
        """
        CCA loss based on canonical correlations from stochastic realization.

        Correct implementation based on formulation:
        L_cca = -sum_i rho_i (maximize sum of canonical correlations)

        Returns:
            torch.Tensor: CCA loss (negative sum of canonical correlations)
        """
        try:
            # Get canonical correlations from stochastic realization
            canonical_correlations = self._get_canonical_correlations_from_realization()

            if canonical_correlations is None:
                # Warning and fallback when canonical correlations unavailable
                if self.config.verbose:
                    print("Canonical correlations unavailable. Setting CCA loss to 0.")
                return torch.tensor(0.0, requires_grad=True, device=self.device)

            # Maximize sum of canonical correlations -> negate for minimization
            cca_loss = -canonical_correlations.sum()

            # Save canonical correlations detail to CSV
            if hasattr(self, 'current_epoch') and hasattr(self, 'logger'):
                epoch = getattr(self, 'current_epoch', 0)
                if epoch % self.config.log_interval == 0:  # detailed log at log_interval
                    self.logger.log_canonical_correlations(epoch, "Phase-2", canonical_correlations)

            # Debug logs consolidated into unified log

            return cca_loss

        except Exception as e:
            warnings.warn(f"CCA loss computation error: {e}. Setting loss to 0.")
            return torch.tensor(0.0, requires_grad=True, device=self.device)

    def _get_canonical_correlations_from_realization(self) -> Optional[torch.Tensor]:
        """
        Get canonical correlations from stochastic realization class.

        Returns:
            torch.Tensor: Canonical correlations rho_i in R^r, or None
        """
        try:
            if isinstance(self.realization, StochasticRealizationWithEncoder):
                # StochasticRealizationWithEncoder case
                if hasattr(self.realization, 'canonical_correlations') and \
                   self.realization.canonical_correlations is not None:
                    correlations = self.realization.canonical_correlations
                    # Device unification
                    return correlations.to(self.device)

            elif hasattr(self.realization, '_L_vals') and self.realization._L_vals is not None:
                # Legacy Realization: use singular values as canonical correlations
                # Assumes singular values are normalized to [0, 1]
                l_vals = self.realization._L_vals
                # Device unification
                return l_vals.to(self.device)

            return None

        except Exception as e:
            warnings.warn(f"Error getting canonical correlations: {e}")
            return None
    
    # =
    
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
        
        # Time alignment validation
        self._validate_time_alignment(X_hat_states, m_aligned, component)
        
        # Debug info (verbose only)
        # if self.config.verbose and epoch % 10 == 0:
        #     print(f"{component} time alignment - Epoch {epoch}: "
        #           f"X_hat: {X_hat_states.shape}, m_aligned: {m_aligned.shape}, "
        #           f"offset: {total_offset}")
        
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

        # Extract time-aligned multivariate features
        if total_offset + T_pred <= T_original:
            M_aligned = M_features[total_offset:total_offset + T_pred]  # (T_pred, m)
        else:
            # Safety fallback: get required length from end
            M_aligned = M_features[-T_pred:]  # (T_pred, m)
            if self.config.verbose:
                print(f"Warning: {component} offset adjustment failed, using tail truncation")

        # Time alignment validation (multivariate)
        self._validate_time_alignment_multivariate(X_hat_states, M_aligned, component)

        return M_aligned

    def _get_time_alignment_offset(self, T_original: int, T_states: int, T_pred: int) -> int:
        """
        Compute offset for time index adjustment.
        Theory: 
        - Stochastic realization output: x_h, x_{h+1}, ..., x_{h+T_states-1}
        - DF-A prediction: x_hat_{h+1|h}, x_hat_{h+2|h+1}, ..., x_hat_{h+T_pred|h+T_pred-1}
        - Correct correspondence: x_hat_{h+1|h} <-> m_{h+1}
        Args:
            T_original: Original series length
            T_states: State sequence length  
            T_pred: Prediction sequence length
            
        Returns:
            int: m_series offset (= h + 1)
        """
        # Get h value
        h_candidates = [
            'h',                    # standard Realization attribute
            'past_horizon',         # init parameter name
            'lags',                 # possible alias
            'window_size',          # possible alias
        ]
        
        h = None
        for attr_name in h_candidates:
            if hasattr(self.realization, attr_name):
                h = getattr(self.realization, attr_name)
                if isinstance(h, (int, float)) and h > 0:
                    h = int(h)
                    break
        
        # Fallback: derive from T_original and T_states
        if h is None:
            # Compute h from T_states = T_original - 2*h + 1
            h = (T_original - T_states + 1) // 2
            if self.config.verbose:
                print(f"Warning: h estimated by back-calculation: h = {h}")
        
        # Validation
        expected_T_states = T_original - 2 * h + 1
        if abs(T_states - expected_T_states) > 1:  # allow off-by-1
            if self.config.verbose:
                print(f"Warning: expected T_states={expected_T_states} with h={h} does not match actual {T_states}")
        
        return h + 1
    
    def _validate_time_alignment(self, X_hat_states: torch.Tensor, m_aligned: torch.Tensor, 
                               component: str = "unknown") -> None:
        """
        Verify time index alignment.
        
        Args:
            X_hat_states: State prediction
            m_aligned: Aligned scalar features
            component: Component name (for logging)
        """
        if X_hat_states.size(0) != m_aligned.size(0):
            raise RuntimeError(
                f"{component} time index mismatch: "
                f"X_hat={X_hat_states.shape} vs m_aligned={m_aligned.shape}"
            )
        
        # Time alignment confirmation (verbose only)
        # if self.config.verbose:
        #     print(f"{component} time alignment check: {X_hat_states.shape} <-> {m_aligned.shape}")

    def _validate_time_alignment_multivariate(
        self,
        X_hat_states: torch.Tensor,
        M_aligned: torch.Tensor,
        component: str = "unknown"
    ) -> None:
        """
        Verify time index alignment (multivariate).

        Args:
            X_hat_states: State prediction (T_pred, r)
            M_aligned: Aligned multivariate features (T_pred, m)
            component: Component name (for logging)
        """
        if X_hat_states.size(0) != M_aligned.size(0):
            raise RuntimeError(
                f"{component} time index mismatch: "
                f"X_hat={X_hat_states.shape} vs M_aligned={M_aligned.shape}"
            )

        # Multivariate feature dimension check
        if M_aligned.dim() != 2:
            raise RuntimeError(
                f"{component} multivariate feature shape error: "
                f"expected=(T_pred, m), actual={M_aligned.shape}"
            )

    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to appropriate device."""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor

    def _clear_computation_graph(self):
        """
        Explicit computation graph clearing.
        """
        # GPU memory clear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # CPU garbage collection
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
            # Basic dimension check
            result["dimensions"] = {
                "M_features_shape": tuple(M_features.shape),
                "X_states_shape": tuple(X_states.shape),
                "encoder_output_dim": getattr(self.encoder, 'output_dim', 'unknown'),
                "expected_feature_dim": getattr(self.df_obs, 'multivariate_feature_dim', 'unknown'),
                "df_state_feature_dim": getattr(self.df_state, 'feature_dim', 'unknown'),
                "df_obs_feature_dim": getattr(self.df_obs, 'obs_feature_dim', 'unknown')
            }

            # Multivariate feature shape check
            if M_features.dim() != 2:
                result["errors"].append(f"M_features shape error: expected=(T, m), actual={M_features.shape}")

            # DF-B multivariate feature dimension check
            if hasattr(self.df_obs, 'multivariate_feature_dim'):
                expected_m = self.df_obs.multivariate_feature_dim
                if M_features.size(1) != expected_m:
                    result["errors"].append(
                        f"Feature dimension mismatch: M_features.size(1)={M_features.size(1)} vs "
                        f"expected={expected_m}"
                    )

            # U_B matrix dimension check
            if hasattr(self.df_obs, 'U_B') and self.df_obs.U_B is not None:
                U_B_shape = self.df_obs.U_B.shape
                expected_U_B_shape = (self.df_obs.obs_feature_dim, M_features.size(1))
                if tuple(U_B_shape) != expected_U_B_shape:
                    result["errors"].append(
                        f"U_B matrix dimension error: actual={U_B_shape} vs expected={expected_U_B_shape}"
                    )

            # Numerical stability check
            if hasattr(self.df_obs, 'V_B') and self.df_obs.V_B is not None:
                cond_V_B = torch.linalg.cond(self.df_obs.V_B).item()
                if cond_V_B > 1e12:
                    result["warnings"].append(f"V_B condition number large: {cond_V_B:.2e}")

            if hasattr(self.df_obs, 'U_B') and self.df_obs.U_B is not None:
                # Condition number of U_B @ U_B.T
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
        """
        Log multivariate training progress (detailed on first call only).

        Args:
            epoch: Epoch number
            phase: Training phase ("phase1_df_a", "phase1_df_b", "phase2")
        """
        if not self.config.verbose or epoch % self.config.log_interval != 0:
            return

        # Show detailed info on first call only
        if not hasattr(self, '_multivariate_logged'):
            self._multivariate_logged = set()

        show_details = phase not in self._multivariate_logged

        try:
            # Show detailed info on first call
            if show_details:
                # Encoder statistics
                if hasattr(self, '_temp_data') and 'M_features' in self._temp_data:
                    M_features = self._temp_data['M_features']
                    print(f"Multivariate features: shape={M_features.shape}, "
                          f"mean={M_features.mean().item():.4f}, "
                          f"std={M_features.std().item():.4f}")

                # Numerical stability check
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
    
    # =
    
    def forecast(self, Y_test: torch.Tensor, forecast_steps: int) -> torch.Tensor:
        """Execute prediction."""
        self.encoder.eval()
        self.decoder.eval()
        self.df_state.eval()
        self.df_obs.eval()
        
        with torch.no_grad():
            # Initial state estimation
            T_test, d = Y_test.shape
            warmup_len = min(T_test, self.realization.h + 10)
            Y_warmup = Y_test[:warmup_len]
            
            # Encode
            m_warmup = self.encoder(Y_warmup.unsqueeze(0)).squeeze()
            
            # State estimation
            try:
                self.realization.fit(m_warmup.unsqueeze(1))
            except RealizationError as e:
                print(f"Warmup RealizationError: {e}")
                # Raise error from warmup to skip processing
                raise RealizationError(f"Warmup realization failed: {e}") from e
            X_warmup = self.realization.filter(m_warmup.unsqueeze(1))
            
            # Sequential prediction
            predictions = []
            x_current = X_warmup[-1]  # latest state
            
            for step in range(forecast_steps):
                # DF-A: state prediction
                x_pred = self.df_state.predict_one_step(x_current)
                
                # DF-B: feature prediction
                m_pred = self.df_obs.predict_one_step(x_pred)
                
                # Decode: observation prediction
                m_input = m_pred.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1)
                y_pred = self.decoder(m_input).squeeze()  # (d,)
                
                predictions.append(y_pred)
                x_current = x_pred  # state update
            
            return torch.stack(predictions)  # (forecast_steps, d)
    
    def train_full(self, Y_train: torch.Tensor, Y_val: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Execute full training (Phase-1 + Phase-2)."""
        try:
            # Phase-1 training
            phase1_metrics = self.train_phase1(Y_train)
            
            # Phase-2 training
            phase2_metrics = self.train_phase2(Y_train, Y_val)
            
            # Final save
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
            # Emergency save
            self._save_checkpoint(self.current_epoch, TrainingPhase.PHASE1_DF_A, emergency=True)
            raise
    
    def _print_phase1_progress(self, epoch: int, metrics: Dict[str, float]):
        """Display Phase-1 progress."""
        df_a_s1 = metrics.get('df_a_stage1_loss', 0)
        df_a_s2 = metrics.get('df_a_stage2_loss', 0)
        df_b_s1 = metrics.get('df_b_stage1_loss', 0)
        df_b_s2 = metrics.get('df_b_stage2_loss', 0)
        
        print(f"Phase-1 Epoch {epoch:3d}: "
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
        
        # DF layer states
        if self.df_state is not None:
            checkpoint['df_state'] = self.df_state.get_state_dict()
        if self.df_obs is not None:
            checkpoint['df_obs'] = self.df_obs.get_state_dict()
        
        # Optimizer states
        opt_states = {}
        for name, opt in self.optimizers.items():
            if opt is not None:
                opt_states[name] = opt.state_dict()
        checkpoint['optimizer_states'] = opt_states
        
        # Save path
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
            # Build complete config from training settings (no hardcoded defaults)
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
            'config': complete_config  # complete config used at inference
        }

        # Save in models/ subdirectory (consistent with run_full_experiment.py)
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = models_dir / 'final_model.pth'
        torch.save(model_state, save_path)

        print(f"Final model saved: {save_path}")
        print(f"   Encoder type: '{complete_config['model']['encoder'].get('type')}'")
        print(f"   Checkpoint structure: flat format (checkpoint['df_state'])")
        print(f"   Complete configuration saved")
    
    def _build_complete_config_from_training_config(self) -> Dict[str, Any]:
        """
        Build complete configuration from training settings.

        Purpose:
        - Save all information needed at inference into checkpoint
        - Eliminate hardcoded defaults; raise explicit errors for missing info

        Returns:
            Complete configuration dictionary

        Raises:
            KeyError: If required information cannot be obtained
        """
        # Use yaml_config if saved (via _init_from_config)
        if hasattr(self, 'yaml_config') and self.yaml_config is not None:
            # yaml_config already contains complete settings
            encoder_type = self.yaml_config.get('model', {}).get('encoder', {}).get('type')
            if not encoder_type:
                raise KeyError(
                    "yaml_config['model']['encoder']['type'] not found.\n"
                    "Check model.encoder.type in the training YAML config."
                )

            # Return YAML config as-is
            return {
                'model': self.yaml_config.get('model', {}),
                'ssm': self.yaml_config.get('ssm', {})
            }

        # No yaml_config (legacy individual-argument init): raise error
        raise KeyError(
            "Failed to build complete configuration.\n\n"
            "Cause: yaml_config not saved.\n"
            "Fix: Initialize TwoStageTrainer with config argument.\n"
            "Example: TwoStageTrainer(config=config_dict, device=device, output_dir=output_dir)"
        )
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        # Use different attributes for joint vs separate training
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


# Utility functions
def create_trainer_from_config(config_path: str, device: torch.device, output_dir: str) -> TwoStageTrainer:
    """
    Create trainer from config file.
    
    Args:
        config_path: YAML config file path
        device: Computation device
        output_dir: Output directory
        
    Returns:
        TwoStageTrainer: Initialized trainer
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Model initialization (using time_invariant architecture)
    encoder_config = config['model']['encoder'].copy()
    decoder_config = config['model']['decoder'].copy()

    # Config adjustment for time_invariant
    if 'output_dim' not in encoder_config:
        encoder_config['output_dim'] = encoder_config.get('channels', 32)

    encoder = time_invariantEncoder(**encoder_config)
    decoder = time_invariantDecoder(**decoder_config)

    # Stochastic realization (prefer new class)
    realization_config = config['ssm']['realization']
    if config.get('evaluation', {}).get('use_new_realization', True):
        # StochasticRealizationWithEncoder requires encoder argument
        realization_config_copy = realization_config.copy()
        realization = StochasticRealizationWithEncoder(
            encoder=encoder,
            **realization_config_copy
        )
    else:
        realization = Realization(**realization_config)
    
    # Config conversion
    training_config = TrainingConfig.from_nested_dict(config['training'])
    
    # Create trainer
    trainer = TwoStageTrainer(
        encoder=encoder,
        decoder=decoder,
        realization=realization,
        df_state_config=config['ssm']['df_state'],
        df_obs_config=config['ssm']['df_observation'],
        training_config=training_config,
        device=device,
        output_dir=output_dir
    )
    
    return trainer


def run_training_experiment(
    config_path: str,
    data_path: str,
    output_dir: str,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Execute training experiment.
    
    Args:
        config_path: Config file path
        data_path: Data file path (.npz)
        output_dir: Results output directory
        device: Computation device (auto-select if None)
        
    Returns:
        Experiment results dictionary
    """
    import yaml
    import numpy as np
    from ..utils.gpu_utils import select_device

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device setup
    if device is None:
        device = select_device()
    
    print(f"Experiment started: device={device}")
    
    # Load data
    try:
        from ..utils.data_loader import load_experimental_data
        
        # Check if data config exists
        if 'data' in config:
            print(f"Loading data with unified data loader: {data_path}")
            data_dict = load_experimental_data(data_path, config['data'])
            Y_train = data_dict['train'].to(device)
            print(f"Data shape: {Y_train.shape} (normalization: {data_dict['metadata'].normalization_method})")
        else:
            raise ImportError("No data config, using legacy method")
            
    except (ImportError, ModuleNotFoundError, Exception) as e:
        print(f"Unified data loader unavailable, using legacy: {e}")
        
        data = np.load(data_path)
        if 'Y' in data:
            Y_train = torch.tensor(data['Y'], dtype=torch.float32, device=device)
        elif 'arr_0' in data:
            Y_train = torch.tensor(data['arr_0'], dtype=torch.float32, device=device)
        else:
            available_keys = list(data.keys())
            raise ValueError(
                f"'Y' or 'arr_0' key not found in data file."
                f"Available keys: {available_keys}"
                )
    
    print(f"Data loaded: {Y_train.shape}")
    
    # Data validation
    if Y_train.dim() != 2:
        raise ValueError(f"Data must be 2D (T, d): got {Y_train.shape}")
    
    T, d = Y_train.shape
    if T < 50:
        warnings.warn(f"Time series may be too short: T={T}")
    
    # Create trainer
    try:
        trainer = create_trainer_from_config(config_path, device, output_dir)
    except Exception as e:
        raise RuntimeError(f"Trainer creation failed: {config_path}. Error: {e}")
    
    # Use train_full method
    try:
        results = trainer.train_full(Y_train)
    except Exception as e:
        print(f"Error during training: {e}")
        # Emergency save attempt
        try:
            trainer._save_checkpoint(
                trainer.current_epoch, 
                TrainingPhase.PHASE1_DF_A, 
                emergency=True
            )
            print(f"Emergency checkpoint saved: {trainer.output_dir}")
        except:
            print("Emergency save also failed")
        raise
    
    # Add summary
    try:
        experiment_summary = trainer.get_training_summary()
        results['experiment_summary'] = experiment_summary
        results['data_info'] = {
            'data_path': data_path,
            'data_shape': tuple(Y_train.shape),
            'device': str(device),
            'total_parameters': experiment_summary.get('model_info', {}).get('total_params', 0)
        }
        
        # Backup config file
        config_backup_path = Path(output_dir) / 'config_used.yaml'
        if not config_backup_path.exists():
            import shutil
            shutil.copy2(config_path, config_backup_path)
            print(f"Config file backed up: {config_backup_path}")
            
    except Exception as e:
        warnings.warn(f"Error creating summary: {e}")
        results['experiment_summary'] = {'error': str(e)}
        results['data_info'] = {
            'data_path': data_path,
            'data_shape': tuple(Y_train.shape),
            'device': str(device)
        }
    
    print(f"Experiment complete: results saved to {output_dir}")
    
    return results


def run_validation(
    trainer: TwoStageTrainer, 
    Y_test: torch.Tensor, 
    output_dir: str,
    forecast_steps: int = 96
) -> Dict[str, Any]:
    """
    Execute validation on trained model.
    
    Args:
        trainer: Trained trainer
        Y_test: Test data
        output_dir: Results output directory
        forecast_steps: Number of forecast steps
        
    Returns:
        Validation results dictionary
    """
    print("Validation started...")
    
    try:
        # Run predictions
        predictions = trainer.forecast(Y_test, forecast_steps)
        
        # Compute prediction accuracy
        if Y_test.size(0) > forecast_steps:
            Y_true = Y_test[-forecast_steps:]
            mse = torch.mean((predictions - Y_true) ** 2).item()
            mae = torch.mean(torch.abs(predictions - Y_true)).item()
            
            # Relative error
            relative_error = (torch.norm(predictions - Y_true) / torch.norm(Y_true)).item()
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': mse ** 0.5,
                'relative_error': relative_error,
                'forecast_steps': forecast_steps
            }
        else:
            warnings.warn("Test data shorter than forecast steps, skipping accuracy computation")
            metrics = {
                'forecast_steps': forecast_steps,
                'note': 'Insufficient test data for accuracy computation'
            }
        
        # Save results
        validation_results = {
            'metrics': metrics,
            'predictions_shape': tuple(predictions.shape),
            'test_data_shape': tuple(Y_test.shape),
            'model_summary': trainer.get_training_summary()
        }
        
        # Save prediction results as numpy arrays
        output_path = Path(output_dir)
        predictions_path = output_path / 'predictions.npz'
        np.savez(
            predictions_path,
            predictions=predictions.cpu().numpy(),
            Y_test=Y_test.cpu().numpy()
        )
        
        print(f"Validation complete: MSE={metrics.get('mse', 'N/A'):.6f}")
        
        return validation_results
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'test_data_shape': tuple(Y_test.shape),
            'forecast_steps': forecast_steps
        }
        print(f"Validation error: {e}")
        return error_result


def plot_training_results(output_dir: str) -> None:
    """
    Visualize training results.
    
    Args:
        output_dir: Results directory
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        output_path = Path(output_dir)
        
        # Phase-1 loss plot
        phase1_csv = output_path / 'phase1_training.csv'
        if phase1_csv.exists():
            df_phase1 = pd.read_csv(phase1_csv)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Phase-1 Training Progress')
            
            # DF-A Stage-1
            df_a_s1 = df_phase1[(df_phase1['phase'] == 'phase1_df_a') & (df_phase1['stage'] == 'stage1')]
            if not df_a_s1.empty:
                axes[0, 0].plot(df_a_s1['epoch'], df_a_s1['loss'])
                axes[0, 0].set_title('DF-A Stage-1 Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
            
            # DF-A Stage-2
            df_a_s2 = df_phase1[(df_phase1['phase'] == 'phase1_df_a') & (df_phase1['stage'] == 'stage2')]
            if not df_a_s2.empty:
                axes[0, 1].plot(df_a_s2['epoch'], df_a_s2['loss'])
                axes[0, 1].set_title('DF-A Stage-2 Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
            
            # DF-B Stage-1
            df_b_s1 = df_phase1[(df_phase1['phase'] == 'phase1_df_b') & (df_phase1['stage'] == 'stage1')]
            if not df_b_s1.empty:
                axes[1, 0].plot(df_b_s1['epoch'], df_b_s1['loss'])
                axes[1, 0].set_title('DF-B Stage-1 Loss')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
            
            # DF-B Stage-2
            df_b_s2 = df_phase1[(df_phase1['phase'] == 'phase1_df_b') & (df_phase1['stage'] == 'stage2')]
            if not df_b_s2.empty:
                axes[1, 1].plot(df_b_s2['epoch'], df_b_s2['loss'])
                axes[1, 1].set_title('DF-B Stage-2 Loss')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss')
            
            plt.tight_layout()
            plt.savefig(output_path / 'phase1_losses.png', dpi=150)
            plt.close()
        
        # Phase-2 loss plot
        phase2_csv = output_path / 'phase2_training.csv'
        if phase2_csv.exists():
            df_phase2 = pd.read_csv(phase2_csv)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Phase-2 Training Progress')
            
            axes[0].plot(df_phase2['epoch'], df_phase2['total_loss'])
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            
            axes[1].plot(df_phase2['epoch'], df_phase2['rec_loss'])
            axes[1].set_title('Reconstruction Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            
            axes[2].plot(df_phase2['epoch'], df_phase2['cca_loss'])
            axes[2].set_title('CCA Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Loss')
            
            plt.tight_layout()
            plt.savefig(output_path / 'phase2_losses.png', dpi=150)
            plt.close()
        
        print(f"Visualization complete: {output_path}")
        
    except ImportError:
        warnings.warn("matplotlib/pandas not available, skipping visualization")
    except Exception as e:
        warnings.warn(f"Visualization error: {e}")


if __name__ == "__main__":
    # Simple test code
    print("TwoStageTrainer loaded")