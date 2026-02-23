# src/inference/state_estimator.py
"""
Integrated inference class: StateEstimator.

Constructs a Kalman filtering inference engine from trained DFIV models.
Manages integration with DF-A/DF-B components, noise estimation, and inference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
import warnings
import yaml

from .kalman_filter import OperatorBasedKalmanFilter
from .utils import (
    estimate_noise_covariances,
    compute_residuals_from_operators,
    validate_kalman_inputs,
    initialize_state_data_driven,
    format_filter_results
)
from ..ssm.realization import Realization


class StateEstimator:
    """
    Integrated inference class.

    Extracts transfer operators from trained DFIV models (DF-A + DF-B) and
    performs sequential state estimation via Algorithm 1.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize from config dict.

        Args:
            config: Inference configuration dict
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        self.df_state_layer = None      # DF-A
        self.df_obs_layer = None        # DF-B
        self.encoder = None
        self.realization = None
        self.kalman_filter = None

        self.V_A: Optional[torch.Tensor] = None
        self.V_B: Optional[torch.Tensor] = None
        self.U_A: Optional[torch.Tensor] = None
        self.U_B: Optional[torch.Tensor] = None
        self.Q: Optional[torch.Tensor] = None
        self.R: Optional[Union[torch.Tensor, float]] = None

        # Nonlinear readout networks
        self.readout_A: Optional[nn.Module] = None
        self.readout_B: Optional[nn.Module] = None
        
        self.is_initialized = False
        self.calibration_data: Optional[torch.Tensor] = None

    @classmethod
    def from_trained_model(
        cls,
        model_path: Union[str, Path],
        config_path: Union[str, Path]
    ) -> 'StateEstimator':
        """
        Initialize from trained model.

        Args:
            model_path: Path to trained model
            config_path: Path to config file

        Returns:
            Initialized StateEstimator instance.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        estimator = cls(config['inference'])
        estimator.load_components(model_path)
        
        return estimator

    def load_components(self, model_path: Union[str, Path]):
        """
        Load individual components (DF-A, DF-B, encoder) and extract operators.

        Args:
            model_path: Path to trained model
        """
        model_path = Path(model_path)

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            training_config = checkpoint.get('config', {})

            encoder_type = training_config.get('model', {}).get('encoder', {}).get('type')

            self._update_config_from_checkpoint(training_config)

            # Component state dicts (both flat and nested formats supported)
            if 'df_state' in checkpoint:
                state_dict = checkpoint
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                raise KeyError("Checkpoint structure not recognized (neither flat nor nested)")

            # DF-A component
            if 'df_state' in state_dict:
                self._load_df_state_component(state_dict['df_state'])
            else:
                raise KeyError("DF-A component not found in model")

            # DF-B component
            if 'df_obs' in state_dict:
                self._load_df_obs_component(state_dict['df_obs'])
            else:
                raise KeyError("DF-B component not found in model")

            # Encoder
            if 'encoder' in state_dict:
                self._load_encoder_component(state_dict['encoder'], encoder_type=encoder_type)
            else:
                raise KeyError("Encoder component not found in model")

            # Extract transfer operators
            self._extract_operators()

            print(f"Successfully loaded components from {model_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model components: {e}")

    def _update_config_from_checkpoint(self, training_config: Dict[str, Any]):
        """Update config from checkpoint training config."""
        if not training_config:
            return

        if 'model' not in self.config:
            self.config['model'] = {}

        if 'encoder' in training_config.get('model', {}):
            self.config['model']['encoder'] = training_config['model']['encoder']

        if 'ssm' in training_config and 'df_state' in training_config['ssm']:
            if 'df_state' not in self.config.get('model', {}):
                self.config['model']['df_state'] = {}
            df_state_cfg = training_config['ssm']['df_state']
            self.config['model']['df_state'].update({
                'feature_dim': df_state_cfg.get('feature_dim'),
                'state_dim': training_config['ssm']['realization'].get('rank')
            })

        if 'ssm' in training_config and 'df_observation' in training_config['ssm']:
            if 'df_obs' not in self.config.get('model', {}):
                self.config['model']['df_obs'] = {}
            df_obs_cfg = training_config['ssm']['df_observation']
            self.config['model']['df_obs'].update({
                'obs_feature_dim': df_obs_cfg.get('obs_feature_dim'),
                'multivariate_feature_dim': df_obs_cfg.get('multivariate_feature_dim'),
                'lambda_B': df_obs_cfg.get('lambda_B'),
                'lambda_dB': df_obs_cfg.get('lambda_dB'),
                'obs_net': df_obs_cfg.get('obs_net'),
                'cross_fitting': df_obs_cfg.get('cross_fitting')
            })

    def _flatten_nested_state_dict(self, nested_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested state_dict."""
        flattened = {}
        for key, value in nested_dict.items():
            if isinstance(value, dict) and hasattr(value, 'keys'):
                for sub_key, sub_value in value.items():
                    flattened[f"{key}.{sub_key}"] = sub_value
            else:
                flattened[key] = value
        return flattened

    def _load_df_state_component(self, df_state_dict: Dict[str, Any]):
        """Load DF-A component."""
        from ..ssm.df_state_layer import DFStateLayer
        
        state_config = self.config.get('model', {}).get('df_state', {})
        
        self.df_state_layer = DFStateLayer(
            state_dim=state_config.get('state_dim', 5),
            feature_dim=state_config.get('feature_dim', 16),
            lambda_A=state_config.get('lambda_A', 1e-3),
            lambda_B=state_config.get('lambda_B', 1e-3),
            feature_net_config=state_config.get('feature_net'),
            cross_fitting_config=state_config.get('cross_fitting')
        ).to(self.device)
        
        flattened_state_dict = self._flatten_nested_state_dict(df_state_dict)
        self.df_state_layer.load_state_dict(flattened_state_dict)
        self.df_state_layer.eval()

    def _load_df_obs_component(self, df_obs_dict: Dict[str, Any]):
        """Load DF-B component."""
        from ..ssm.df_observation_layer import DFObservationLayer

        obs_config = self.config.get('model', {}).get('df_obs', {})

        self.df_obs_layer = DFObservationLayer(
            df_state_layer=self.df_state_layer,  # DF-A reference
            obs_feature_dim=obs_config.get('obs_feature_dim', 16),
            multivariate_feature_dim=obs_config.get('multivariate_feature_dim', 8),
            lambda_B=obs_config.get('lambda_B', 1e-3),
            lambda_dB=obs_config.get('lambda_dB', 1e-3),
            obs_net_config=obs_config.get('obs_net'),
            cross_fitting_config=obs_config.get('cross_fitting')
        ).to(self.device)
        
        flattened_obs_dict = self._flatten_nested_state_dict(df_obs_dict)
        # phi_theta is shared from df_state_layer, so use strict=False
        self.df_obs_layer.load_state_dict(flattened_obs_dict, strict=False)
        self.df_obs_layer.eval()

    def _load_encoder_component(self, encoder_dict: Dict[str, Any], encoder_type: str = None):
        """
        Dynamically load encoder.

        Args:
            encoder_dict: Encoder state_dict
            encoder_type: Encoder type ('cnn_image', 'time_invariant', etc.)
                          Auto-detected from state_dict if None.
        """
        if encoder_type is None:
            encoder_type = self._detect_encoder_type(encoder_dict)

        from ..models.encoder import build_encoder

        encoder_config = self._build_encoder_config_from_state_dict(
            encoder_dict, encoder_type
        )
        encoder_config['type'] = encoder_type

        self.encoder = build_encoder(encoder_config).to(self.device)

        self.encoder.load_state_dict(encoder_dict)
        self.encoder.eval()

        print(f"Loaded {encoder_type}Encoder successfully")

    def _detect_encoder_type(self, encoder_dict: Dict[str, Any]) -> str:
        """
        Auto-detect encoder type from state_dict keys.

        Supported: 'cnn_image' (CNN for images), 'time_invariant' (MLP for 1D).
        """
        keys = set(encoder_dict.keys())

        # cnn_imageEncoder: has conv layers
        if any('conv' in k for k in keys):
            return 'cnn_image'

        # time_invariantEncoder characteristic keys
        if any('core_net' in k for k in keys):
            return 'time_invariant'

        raise ValueError(
            f"Cannot auto-detect encoder type.\n"
            f"state_dict keys sample: {list(keys)[:10]}\n"
            f"Known patterns: 'conv'(cnn_image), 'core_net'(time_invariant)"
        )

    def _build_encoder_config_from_state_dict(
        self,
        encoder_dict: Dict[str, Any],
        encoder_type: str
    ) -> Dict[str, Any]:
        """
        Reconstruct encoder config from state_dict shapes.

        Supported: 'cnn_image', 'time_invariant'.
        """

        if encoder_type == 'cnn_image':
            # Estimate input_resolution from conv1.weight shape
            conv1_weight = encoder_dict.get('conv1.weight')
            if conv1_weight is not None:
                C = conv1_weight.shape[1]
                config_from_checkpoint = self.config.get('model', {}).get('encoder', {})
                input_resolution_from_cfg = config_from_checkpoint.get('input_resolution', [48, 48, 1])
                if len(input_resolution_from_cfg) >= 2:
                    H, W = input_resolution_from_cfg[:2]
                else:
                    H, W = 48, 48
                input_resolution = [H, W, C]
            else:
                input_resolution = [48, 48, 1]

            fc_out_weight = encoder_dict.get('fc_out.weight')
            if fc_out_weight is not None:
                feature_dim = fc_out_weight.shape[0]
            else:
                feature_dim = 50

            conv_channels = []
            if 'conv1.weight' in encoder_dict:
                conv_channels.append(encoder_dict['conv1.weight'].shape[0])
            if 'conv2.weight' in encoder_dict:
                conv_channels.append(encoder_dict['conv2.weight'].shape[0])

            fc1_weight = encoder_dict.get('fc1.weight')
            if fc1_weight is not None:
                hidden = fc1_weight.shape[0]
            else:
                hidden = 200

            encoder_cfg = self.config.get('model', {}).get('encoder', {})

            return {
                'input_resolution': input_resolution,
                'feature_dim': feature_dim,
                'hidden': hidden,
                'conv_channels': conv_channels if conv_channels else [32, 64],
                'activation': encoder_cfg.get('activation', 'relu'),
                'normalize_input': encoder_cfg.get('normalize_input', False),
                'normalize_output': encoder_cfg.get('normalize_output', False),
                'track_running_stats': encoder_cfg.get('track_running_stats', True)
            }

        elif encoder_type == 'time_invariant':
            input_dim = self.config.get('model', {}).get('encoder', {}).get('input_dim', 6)
            output_dim = self._detect_time_invariant_output_dim(encoder_dict)

            return {
                'input_dim': input_dim,
                'output_dim': output_dim,
                'architecture': 'mlp',
                'normalize_input': True,
                'normalize_output': True,
                'track_running_stats': True
            }

        else:
            raise ValueError(
                f"Unsupported encoder type: '{encoder_type}'\n"
                f"Currently supported: 'cnn_image' and 'time_invariant'."
            )

    def _detect_time_invariant_output_dim(self, encoder_dict: Dict[str, Any]) -> int:
        """Detect output dimension of time_invariant encoder."""
        if 'output_mean' in encoder_dict:
            return encoder_dict['output_mean'].shape[0]
        # Infer from core_net final layer
        for key in encoder_dict.keys():
            if key.endswith('.bias') and 'core_net' in key:
                return encoder_dict[key].shape[0]
        return 8

    def _extract_operators(self):
        """Extract transfer operators from learned components (DF-A: V_A, U_A/readout_A; DF-B: V_B, U_B/readout_B)."""
        if not all([self.df_state_layer, self.df_obs_layer]):
            raise RuntimeError("DF components not loaded")

        # Extract V_A from DF-A
        if hasattr(self.df_state_layer, 'V_A') and self.df_state_layer.V_A is not None:
            self.V_A = self.df_state_layer.V_A.clone().detach()
        else:
            raise RuntimeError("V_A not found in DF-A component")

        # Extract U_A or readout_A from DF-A
        readout_type_A = getattr(self.df_state_layer, 'readout_type', 'linear')
        if readout_type_A == 'nonlinear' and hasattr(self.df_state_layer, 'readout_net') and self.df_state_layer.readout_net is not None:
            self.U_A = None
            self.readout_A = self.df_state_layer.readout_net
            self.readout_A.eval()
        elif hasattr(self.df_state_layer, 'U_A') and self.df_state_layer.U_A is not None:
            self.U_A = self.df_state_layer.U_A.clone().detach()
            self.readout_A = None
        else:
            raise RuntimeError("Neither U_A nor readout_net found in DF-A component")

        # Extract V_B from DF-B
        if hasattr(self.df_obs_layer, 'V_B') and self.df_obs_layer.V_B is not None:
            self.V_B = self.df_obs_layer.V_B.clone().detach()
        else:
            raise RuntimeError("V_B not found in DF-B component")

        # Extract U_B or readout_B from DF-B
        readout_type_B = getattr(self.df_obs_layer, 'readout_type', 'linear')
        if readout_type_B == 'nonlinear' and hasattr(self.df_obs_layer, 'readout_net') and self.df_obs_layer.readout_net is not None:
            self.U_B = None
            self.readout_B = self.df_obs_layer.readout_net
            self.readout_B.eval()
        elif hasattr(self.df_obs_layer, 'U_B') and self.df_obs_layer.U_B is not None:
            self.U_B = self.df_obs_layer.U_B.clone().detach()
            self.readout_B = None
        else:
            raise RuntimeError("Neither U_B nor readout_net found in DF-B component")

        print("Operators extracted successfully:")
        print(f"  V_A: {self.V_A.shape}")
        print(f"  V_B: {self.V_B.shape}")
        print(f"  U_A: {self.U_A.shape if self.U_A is not None else 'None (nonlinear readout)'}")
        print(f"  U_B: {self.U_B.shape if self.U_B is not None else 'None (nonlinear readout)'}")
        if self.readout_A is not None:
            print(f"  readout_A: {readout_type_A}")
        if self.readout_B is not None:
            print(f"  readout_B: {readout_type_B}")

    def estimate_noise_covariances(
        self,
        calibration_data: torch.Tensor,
        method: str = "residual_based"
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, float]]:
        """
        Estimate Q, R from calibration data (Eq. 45-46).

        Args:
            calibration_data: Calibration observations (T_cal, n)
            method: Noise estimation method ("residual_based")

        Returns:
            Q: State noise covariance (dA, dA)
            R: Observation noise covariance
        """
        if method != "residual_based":
            raise ValueError(f"Unknown noise estimation method: {method}")
            
        self.calibration_data = calibration_data
        T_cal = calibration_data.size(0)
        
        print(f"Estimating noise covariances from {T_cal} calibration samples...")
        
        with torch.no_grad():
            # 1. Encode: {y_t} -> {m_t}
            m_series = self.encoder(calibration_data.unsqueeze(0)).squeeze(0)  # (T_cal, output_dim)

            # 2. State space realization: {m_t} -> {x_t}
            if self.realization is None:
                realization_config = self.config.get('ssm', {}).get('realization', {})
                past_horizon = realization_config.get('past_horizon', 10)

                # Auto-adjust past_horizon based on data length
                T_cal = m_series.size(0)
                max_horizon = (T_cal - 1) // 2
                if past_horizon > max_horizon:
                    past_horizon = max(1, max_horizon)
                    print(f"past_horizon adjusted: {realization_config.get('past_horizon', 10)} -> {past_horizon} (data length: {T_cal})")

                self.realization = Realization(
                    past_horizon=past_horizon,
                    jitter=realization_config.get('jitter', 1e-6),
                    cond_thresh=realization_config.get('cond_thresh', 1e10),
                    rank=realization_config.get('rank', 4),
                    reg_type=realization_config.get('reg_type', 'sum')
                )
                self.realization.fit(m_series)

            # Generate state sequence: (N, rank) where N = T_cal - 2*h + 1
            x_series = self.realization.filter(m_series)

            # Align m_series to the same time range as x_series
            h = self.realization.h
            m_series_aligned = m_series[h:h + x_series.size(0)]  # (N,)

            # 3. Apply feature mappings: {x_t} -> {phi_t}, {psi_t}
            phi_sequence = self._apply_state_feature_mapping(x_series)           # (N+1, dA)
            psi_sequence = self._apply_obs_feature_mapping(m_series_aligned)     # (N+1, dB)
            
            # Compute residuals
            residuals_state, residuals_obs = compute_residuals_from_operators(
                phi_sequence, psi_sequence, self.V_A, self.V_B
            )
            
            # Estimate covariances
            regularization = self.config.get('noise_estimation', {})
            Q, R = estimate_noise_covariances(
                residuals_state, residuals_obs, regularization
            )
            
        self.Q = Q
        self.R = R
        
        print(f"Noise covariances estimated:")
        print(f"  Q condition number: {torch.linalg.cond(Q).item():.2e}")
        if isinstance(R, torch.Tensor):
            print(f"  R condition number: {torch.linalg.cond(R).item():.2e}")
        else:
            print(f"  R (scalar): {R:.6f}")
            
        return Q, R

    def _apply_state_feature_mapping(self, x_series: torch.Tensor) -> torch.Tensor:
        """
        Apply state feature mapping phi_theta using the learned DF-A layer.

        Args:
            x_series: State sequence (N, rank) from realization.filter()

        Returns:
            State feature sequence (N+1, dA).
        """
        if self.df_state_layer is None or not hasattr(self.df_state_layer, 'phi_theta'):
            raise RuntimeError(
                "DF-A layer with phi_theta network is required for state feature generation. "
                "Cannot proceed without learned phi_theta(m_t) transformation."
            )

        N = x_series.size(0)

        with torch.no_grad():
            phi_sequence = []
            for t in range(N + 1):
                if t < N:
                    x_t_input = x_series[t]
                    phi_t = self.df_state_layer.phi_theta(x_t_input)  # (dA,)
                else:
                    x_t_input = x_series[-1]
                    phi_t = self.df_state_layer.phi_theta(x_t_input)

                while phi_t.dim() > 1 and phi_t.size(0) == 1:
                    phi_t = phi_t.squeeze(0)

                if phi_t.dim() != 1:
                    raise RuntimeError(
                        f"phi_theta output has unexpected shape: {phi_t.shape}. "
                        f"Expected 1D tensor (dA,) for state features."
                    )

                phi_sequence.append(phi_t)

            return torch.stack(phi_sequence)  # (T+1, dA)

    def _apply_obs_feature_mapping(self, m_series: torch.Tensor) -> torch.Tensor:
        """
        Apply observation feature mapping psi_omega using the learned DF-B layer.

        Args:
            m_series: Feature series (T,)

        Returns:
            Observation feature sequence (T+1, dB).
        """
        if self.df_obs_layer is None or not hasattr(self.df_obs_layer, 'psi_omega'):
            raise RuntimeError(
                "DF-B layer with psi_omega network is required for observation feature generation. "
                "Cannot proceed without learned psi_omega(m_t) transformation."
            )

        T = m_series.size(0)

        with torch.no_grad():
            psi_sequence = []
            for t in range(T + 1):
                if t < T:
                    m_t_input = m_series[t]
                    psi_t = self.df_obs_layer.psi_omega(m_t_input)  # (dB,)
                else:
                    m_t_input = m_series[-1]
                    psi_t = self.df_obs_layer.psi_omega(m_t_input)  # (dB,)

                while psi_t.dim() > 1 and psi_t.size(0) == 1:
                    psi_t = psi_t.squeeze(0)

                if psi_t.dim() != 1:
                    raise RuntimeError(
                        f"psi_omega output has unexpected shape: {psi_t.shape}. "
                        f"Expected 1D tensor (dB,) for observation features."
                    )

                psi_sequence.append(psi_t)

            return torch.stack(psi_sequence)  # (T+1, dB)

    def initialize_filtering(
        self,
        initial_data: Optional[torch.Tensor] = None,
        method: str = "data_driven"
    ):
        """
        Initialize before filtering (create KalmanFilter, set initial state).

        Args:
            initial_data: Data for initialization (N0, n) or None
            method: Initialization method ("data_driven" | "zero")
        """
        # Check operators: need V_A, V_B, and either U_A/U_B or readout nets
        has_state_readout = (self.U_A is not None) or (self.readout_A is not None)
        has_obs_readout = (self.U_B is not None) or (self.readout_B is not None)
        if not all([self.V_A is not None, self.V_B is not None, has_state_readout, has_obs_readout]):
            raise RuntimeError("Operators not extracted. Call load_components() first.")

        if self.Q is None or self.R is None:
            # Default noise settings
            warnings.warn("Noise covariances not estimated. Using defaults.")
            dA = int(self.V_A.size(0))
            self.Q = 0.01 * torch.eye(dA, device=self.device)
            self.R = 0.1

        # Validate inputs (only when linear readout with U_A/U_B available)
        if self.U_A is not None and self.U_B is not None:
            validation = validate_kalman_inputs(
                self.V_A, self.V_B, self.U_A, self.U_B, self.Q, self.R
            )
            if not validation["valid"]:
                raise RuntimeError(f"Invalid Kalman inputs: {validation['errors']}")
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    warnings.warn(warning)

        # Create Kalman Filter (with learned DF-B layer and optional readout nets)
        self.kalman_filter = OperatorBasedKalmanFilter(
            V_A=self.V_A,
            V_B=self.V_B,
            U_A=self.U_A,
            U_B=self.U_B,
            Q=self.Q,
            R=self.R,
            encoder=self.encoder,
            df_obs_layer=self.df_obs_layer,
            device=str(self.device),
            readout_A=self.readout_A,
            readout_B=self.readout_B
        )
        
        # Set initial state
        if initial_data is not None:
            self.kalman_filter.initialize_state(initial_data, method)
        elif self.calibration_data is not None:
            # Initialize from calibration data
            n_init = min(10, self.calibration_data.size(0))
            self.kalman_filter.initialize_state(
                self.calibration_data[:n_init], method
            )
        else:
            # Zero initialization
            self.kalman_filter.initialize_state(
                torch.zeros(1, self.encoder.input_dim, device=self.device), "zero"
            )
            
        self.is_initialized = True
        print("Kalman Filter initialized successfully")

    def filter_sequence(
        self,
        observations: torch.Tensor,
        return_likelihood: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Batch filtering over entire observation sequence.

        Args:
            observations: Observation sequence (T, n)
            return_likelihood: Whether to return likelihoods

        Returns:
            X_means: State mean sequence (T, r)
            X_covariances: State covariance sequence (T, r, r)
            likelihoods: Observation likelihood sequence (T,) [optional]
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize_filtering() first.")
            
        return self.kalman_filter.filter_sequence(observations, None, return_likelihood)

    def filter_online(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Online filtering (one observation at a time with internal state).

        Args:
            observation: Current observation (n,)

        Returns:
            x_hat: Estimated state (r,)
            Sigma_x: State covariance (r, r)
            likelihood: Observation likelihood
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Call initialize_filtering() first.")
            
        return self.kalman_filter.filter_step(observation)

    def reset_state(self):
        """Reset internal state (for new sequence)."""
        if self.kalman_filter is not None:
            self.kalman_filter.reset_state()

    def get_current_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current state estimate and covariance."""
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized")
            
        return self.kalman_filter.get_current_state()

    def predict_ahead(
        self,
        n_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        n-step-ahead prediction from current state.

        Args:
            n_steps: Number of prediction steps

        Returns:
            x_pred: Predicted states (n_steps, r)
            Sigma_pred: Predicted covariances (n_steps, r, r)
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized")
            
        mu_current, Sigma_current = self.kalman_filter.mu, self.kalman_filter.Sigma
        
        predictions = []
        covariances = []
        
        mu, Sigma = mu_current.clone(), Sigma_current.clone()
        
        for step in range(n_steps):
            # Time update only (no observation)
            mu, Sigma = self.kalman_filter.predict_step(mu, Sigma)
            
            x_pred, Sigma_x_pred = self.kalman_filter._recover_original_state(mu, Sigma)
            
            predictions.append(x_pred)
            covariances.append(Sigma_x_pred)
            
        return torch.stack(predictions), torch.stack(covariances)

    def get_filter_diagnostics(self) -> Dict[str, Any]:
        """Get filter diagnostics."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
            
        operator_shapes = {
            "V_A": self.V_A.shape,
            "V_B": self.V_B.shape,
        }
        if self.U_A is not None:
            operator_shapes["U_A"] = self.U_A.shape
        else:
            operator_shapes["readout_A"] = "nonlinear"
        if self.U_B is not None:
            operator_shapes["U_B"] = self.U_B.shape
        else:
            operator_shapes["readout_B"] = "nonlinear"

        diagnostics = {
            "initialization_status": self.is_initialized,
            "operator_shapes": operator_shapes,
            "numerical_stability": self.kalman_filter.check_numerical_stability()
        }
        
        if self.Q is not None:
            diagnostics["noise_covariances"] = {
                "Q_condition": torch.linalg.cond(self.Q).item(),
                "Q_trace": torch.trace(self.Q).item()
            }
            
        if isinstance(self.R, torch.Tensor):
            diagnostics["noise_covariances"]["R_condition"] = torch.linalg.cond(self.R).item()
        else:
            diagnostics["noise_covariances"]["R_scalar"] = self.R
            
        return diagnostics

    def export_for_deployment(self, export_path: Union[str, Path]):
        """
        Export inference-only components for deployment.

        Args:
            export_path: Output path
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized. Cannot export.")
            
        operators_dict = {
            "V_A": self.V_A.cpu(),
            "V_B": self.V_B.cpu(),
        }
        if self.U_A is not None:
            operators_dict["U_A"] = self.U_A.cpu()
        if self.U_B is not None:
            operators_dict["U_B"] = self.U_B.cpu()

        inference_params = {
            "operators": operators_dict,
            "noise_covariances": {
                "Q": self.Q.cpu(),
                "R": self.R if isinstance(self.R, (int, float)) else self.R.cpu()
            },
            "config": self.config
        }

        # Export nonlinear readout nets if present
        if self.readout_A is not None:
            inference_params["readout_A_state_dict"] = self.readout_A.state_dict()
        if self.readout_B is not None:
            inference_params["readout_B_state_dict"] = self.readout_B.state_dict()
        
        encoder_state = self.encoder.state_dict()
        
        torch.save(inference_params, export_path / "inference_params.pth")
        torch.save(encoder_state, export_path / "encoder.pth")
        
        with open(export_path / "inference_config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        print(f"Inference components exported to {export_path}")