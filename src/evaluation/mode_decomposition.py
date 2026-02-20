"""
Mode decomposition and spectrum analysis module.

Provides Koopman operator-based spectral analysis for DFIV:
- Eigenvalue decomposition and continuous-time conversion from learned V_A
- MSE evaluation against ground truth eigenvalues (when available)
- V_A extraction from trained models
- Result saving (JSON, NPZ formats)
"""

import torch
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path


class SpectrumAnalyzer:
    """
    Koopman spectrum analyzer.

    Performs eigenvalue decomposition from V_A and extracts continuous-time
    spectral characteristics, including discrete-to-continuous eigenvalue conversion.
    """

    def __init__(self, sampling_interval: float):
        """
        Args:
            sampling_interval: Sampling interval dt for continuous-time conversion.
        """
        self.dt = sampling_interval

    def analyze_spectrum(self, V_A: torch.Tensor) -> Dict[str, Any]:
        """
        Spectrum analysis from V_A.

        Args:
            V_A: Transfer operator matrix (d_A, d_A)

        Returns:
            Dict with spectrum analysis results:
                - eigenvalues_discrete: lambda in C^{d_A} (discrete-time eigenvalues)
                - eigenvalues_continuous: mu in C^{d_A} (continuous-time eigenvalues)
                - growth_rates: Re(mu) (growth/decay rates)
                - frequencies_hz: Im(mu)/(2*pi) (oscillation frequencies in Hz)
                - dominant_indices: Dominant mode indices
                - stable_indices: Stable mode indices
                - eigenvalues_magnitude: |lambda| (discrete-time magnitudes)
                - eigenvalues_phase: arg(lambda) (discrete-time phases)
        """
        with torch.no_grad():
            eigenvalues_discrete, eigenvectors = torch.linalg.eig(V_A)

            # Continuous-time conversion: mu = (1/dt) * log(lambda)
            eigenvalues_continuous = self._discrete_to_continuous_eigenvalues(
                eigenvalues_discrete
            )

            growth_rates = eigenvalues_continuous.real
            frequencies_rad = eigenvalues_continuous.imag
            frequencies_hz = frequencies_rad / (2 * np.pi)

            eigenvalues_magnitude = torch.abs(eigenvalues_discrete)
            eigenvalues_phase = torch.angle(eigenvalues_discrete)

            dominant_threshold = 0.1
            dominant_indices = self._find_dominant_modes(
                eigenvalues_magnitude, threshold=dominant_threshold
            )
            stable_indices = self._find_stable_modes(growth_rates)

            return {
                'eigenvalues_discrete': eigenvalues_discrete,
                'eigenvalues_continuous': eigenvalues_continuous,
                'eigenvectors': eigenvectors,
                'growth_rates': growth_rates,
                'frequencies_hz': frequencies_hz,
                'frequencies_rad': frequencies_rad,
                'eigenvalues_magnitude': eigenvalues_magnitude,
                'eigenvalues_phase': eigenvalues_phase,
                'dominant_indices': dominant_indices,
                'stable_indices': stable_indices,
                'n_stable_modes': len(stable_indices),
                'n_dominant_modes': len(dominant_indices),
                'spectral_radius': torch.max(eigenvalues_magnitude).item(),
                'sampling_interval': self.dt
            }

    def _discrete_to_continuous_eigenvalues(
        self,
        eigenvalues_discrete: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert discrete-time eigenvalues to continuous-time: mu = (1/dt) * log(lambda).

        Args:
            eigenvalues_discrete: Discrete-time eigenvalues lambda in C^{d_A}

        Returns:
            Continuous-time eigenvalues mu in C^{d_A}.
        """
        # Numerical stability for near-zero eigenvalues
        eigenvalues_log = torch.log(eigenvalues_discrete + 1e-12)
        eigenvalues_continuous = eigenvalues_log / self.dt

        return eigenvalues_continuous

    def _find_dominant_modes(
        self,
        eigenvalues_magnitude: torch.Tensor,
        threshold: float = 0.1
    ) -> List[int]:
        """
        Identify dominant modes.

        Args:
            eigenvalues_magnitude: Eigenvalue magnitudes |lambda|
            threshold: Threshold as ratio of spectral radius

        Returns:
            List of dominant mode indices.
        """
        spectral_radius = torch.max(eigenvalues_magnitude)
        dominant_mask = eigenvalues_magnitude > threshold * spectral_radius
        return dominant_mask.nonzero(as_tuple=True)[0].tolist()

    def _find_stable_modes(self, growth_rates: torch.Tensor) -> List[int]:
        """
        Identify stable modes (Re(mu) < 0).

        Args:
            growth_rates: Growth rates Re(mu)

        Returns:
            List of stable mode indices.
        """
        stable_mask = growth_rates < 0
        return stable_mask.nonzero(as_tuple=True)[0].tolist()

    def evaluate_against_truth(
        self,
        predicted_eigenvalues: torch.Tensor,
        true_eigenvalues: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate against ground truth eigenvalues.

        Args:
            predicted_eigenvalues: Predicted eigenvalues mu_pred in C^{d_A}
            true_eigenvalues: True eigenvalues mu_true in C^{k}

        Returns:
            Dict with mse_real, mse_imag, mse_magnitude, n_matched.
        """
        with torch.no_grad():
            # Nearest-neighbor matching
            matched_pred, matched_true = self._match_eigenvalues(
                predicted_eigenvalues, true_eigenvalues
            )

            if len(matched_pred) == 0:
                warnings.warn("Eigenvalue matching failed")
                return {
                    'mse_real': float('inf'),
                    'mse_imag': float('inf'),
                    'mse_magnitude': float('inf'),
                    'n_matched': 0
                }

            # MSE computation
            mse_real = torch.mean((matched_pred.real - matched_true.real)**2).item()
            mse_imag = torch.mean((matched_pred.imag - matched_true.imag)**2).item()
            mse_magnitude = torch.mean(
                (torch.abs(matched_pred) - torch.abs(matched_true))**2
            ).item()

            return {
                'mse_real': mse_real,
                'mse_imag': mse_imag,
                'mse_magnitude': mse_magnitude,
                'n_matched': len(matched_pred)
            }

    def _match_eigenvalues(
        self,
        pred_eigenvalues: torch.Tensor,
        true_eigenvalues: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Nearest-neighbor eigenvalue matching.

        Args:
            pred_eigenvalues: Predicted eigenvalues
            true_eigenvalues: True eigenvalues

        Returns:
            Tuple of (matched predicted eigenvalues, matched true eigenvalues).
        """
        matched_pred = []
        matched_true = []
        used_true_indices = set()

        for pred_val in pred_eigenvalues:
            distances = torch.abs(true_eigenvalues - pred_val)
            best_idx = torch.argmin(distances).item()

            # Avoid duplicate matching
            if best_idx not in used_true_indices:
                matched_pred.append(pred_val)
                matched_true.append(true_eigenvalues[best_idx])
                used_true_indices.add(best_idx)

        if len(matched_pred) > 0:
            return torch.stack(matched_pred), torch.stack(matched_true)
        else:
            return torch.tensor([]), torch.tensor([])


class TrainedModelSpectrumAnalysis:
    """
    Spectrum analysis from trained models.

    Extracts V_A matrix from trained DFIV models and performs spectrum analysis.
    """

    def __init__(self, sampling_interval: float):
        """
        Args:
            sampling_interval: Sampling interval dt.
        """
        self.sampling_interval = sampling_interval
        self.analyzer = SpectrumAnalyzer(sampling_interval)

    def extract_transfer_matrix_from_model(self, model: Any) -> torch.Tensor:
        """
        Extract V_A from a trained model.

        Args:
            model: Trained DFIV model (containing DFStateLayer)

        Returns:
            V_A matrix (d_A, d_A).
        """
        try:
            if hasattr(model, 'ssm') and hasattr(model.ssm, 'df_state_layer'):
                V_A = model.ssm.df_state_layer.get_transfer_operator()
            elif hasattr(model, 'df_state_layer'):
                V_A = model.df_state_layer.get_transfer_operator()
            elif hasattr(model, 'get_transfer_operator'):
                V_A = model.get_transfer_operator()
            elif isinstance(model, dict):
                # Extract from checkpoint dict (multiple key patterns supported)
                df_state_dict = None

                if 'model_state_dict' in model and 'df_state' in model['model_state_dict']:
                    df_state_dict = model['model_state_dict']['df_state']
                elif 'df_state' in model:
                    df_state_dict = model['df_state']
                else:
                    raise ValueError(
                        f"Invalid checkpoint structure: df_state dict not found.\n"
                        f"Expected key patterns:\n"
                        f"  - checkpoint['model_state_dict']['df_state']\n"
                        f"  - checkpoint['df_state']\n"
                        f"Actual top-level keys: {list(model.keys())}"
                    )

                if 'V_A' in df_state_dict:
                    V_A = df_state_dict['V_A']
                else:
                    raise ValueError(
                        f"V_A matrix not found in df_state dict.\n"
                        f"df_state keys: {list(df_state_dict.keys())}"
                    )
            else:
                raise ValueError("Cannot extract V_A from model. Check model structure.")

            return V_A

        except Exception as e:
            raise RuntimeError(f"V_A extraction error: {e}")

    def extract_transfer_matrix_from_path(self, model_path: str) -> torch.Tensor:
        """
        Extract V_A from a saved model file.

        Args:
            model_path: Path to the model file

        Returns:
            V_A matrix (d_A, d_A).
        """
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            if 'V_A' in checkpoint:
                return checkpoint['V_A']

            if 'state_dict' in checkpoint and 'V_A' in checkpoint['state_dict']:
                return checkpoint['state_dict']['V_A']

            if 'model' in checkpoint:
                return self.extract_transfer_matrix_from_model(checkpoint['model'])

            return self.extract_transfer_matrix_from_model(checkpoint)

        except Exception as e:
            raise RuntimeError(f"Model file loading error: {e}")

    def perform_spectrum_analysis_from_model(self, model: Any) -> Dict[str, Any]:
        """
        Run spectrum analysis from a trained model.

        Args:
            model: Trained model

        Returns:
            Dict with spectrum analysis results and V_A matrix.
        """
        V_A = self.extract_transfer_matrix_from_model(model)
        spectrum_analysis = self.analyzer.analyze_spectrum(V_A)

        return {
            'spectrum': spectrum_analysis,
            'V_A': V_A,
            'V_A_shape': V_A.shape,
            'sampling_interval': self.sampling_interval
        }

    def perform_spectrum_analysis_from_path(self, model_path: str) -> Dict[str, Any]:
        """
        Run spectrum analysis from a saved model file.

        Args:
            model_path: Model file path

        Returns:
            Dict with spectrum analysis results and V_A matrix.
        """
        V_A = self.extract_transfer_matrix_from_path(model_path)
        spectrum_analysis = self.analyzer.analyze_spectrum(V_A)

        return {
            'spectrum': spectrum_analysis,
            'V_A': V_A,
            'V_A_shape': V_A.shape,
            'model_path': model_path,
            'sampling_interval': self.sampling_interval
        }


class SpectrumResultsSaver:
    """
    Spectrum analysis result saver.

    Supports saving in JSON (config/metadata) and NPZ (numerical data) formats.
    """

    @staticmethod
    def save_results(
        results: Dict[str, Any],
        save_path: str,
        save_format: str = 'both'  # 'json', 'npz', 'both'
    ) -> None:
        """
        Save spectrum analysis results.

        Args:
            results: Analysis result dict
            save_path: Save path (without extension)
            save_format: Save format ('json', 'npz', or 'both')
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_format in ['json', 'both']:
            SpectrumResultsSaver._save_json(results, save_path.with_suffix('.json'))

        if save_format in ['npz', 'both']:
            SpectrumResultsSaver._save_npz(results, save_path.with_suffix('.npz'))

    @staticmethod
    def _save_json(results: Dict[str, Any], json_path: Path) -> None:
        """Save in JSON format (config/metadata)."""
        json_data = {}

        for key, value in results.items():
            if key == 'spectrum':
                spectrum = value
                json_spectrum = {}

                for spectrum_key, spectrum_value in spectrum.items():
                    if isinstance(spectrum_value, torch.Tensor):
                        if torch.is_complex(spectrum_value):
                            json_spectrum[f'{spectrum_key}_real'] = spectrum_value.real.tolist()
                            json_spectrum[f'{spectrum_key}_imag'] = spectrum_value.imag.tolist()
                        else:
                            json_spectrum[spectrum_key] = spectrum_value.tolist()
                    elif isinstance(spectrum_value, (int, float, str, list)):
                        json_spectrum[spectrum_key] = spectrum_value
                    else:
                        json_spectrum[spectrum_key] = str(spectrum_value)

                json_data['spectrum'] = json_spectrum

            elif isinstance(value, torch.Tensor):
                if torch.is_complex(value):
                    json_data[f'{key}_real'] = value.real.tolist()
                    json_data[f'{key}_imag'] = value.imag.tolist()
                else:
                    json_data[key] = value.tolist()

            elif isinstance(value, (int, float, str, list, tuple)):
                json_data[key] = value
            else:
                json_data[key] = str(value)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _save_npz(results: Dict[str, Any], npz_path: Path) -> None:
        """Save in NPZ format (numerical data)."""
        npz_data = {}

        def _flatten_dict(d: Dict, prefix: str = '') -> None:
            for key, value in d.items():
                full_key = f"{prefix}{key}" if prefix else key

                if isinstance(value, dict):
                    _flatten_dict(value, f"{full_key}_")
                elif isinstance(value, torch.Tensor):
                    try:
                        if torch.is_complex(value):
                            tensor_cpu = value.cpu() if value.is_cuda else value
                            npz_data[f'{full_key}_real'] = tensor_cpu.real.numpy()
                            npz_data[f'{full_key}_imag'] = tensor_cpu.imag.numpy()
                        else:
                            tensor_cpu = value.cpu() if value.is_cuda else value
                            npz_data[full_key] = tensor_cpu.numpy()
                    except Exception as e:
                        print(f"Tensor conversion error (key: {full_key}, shape: {value.shape}): {e}")
                        npz_data[f'{full_key}_error'] = f"Conversion failed: {str(e)}"
                elif isinstance(value, (list, tuple)):
                    try:
                        npz_data[full_key] = np.array(value)
                    except:
                        pass
                elif isinstance(value, (int, float)):
                    npz_data[full_key] = np.array(value)

        _flatten_dict(results)
        np.savez(npz_path, **npz_data)

    @staticmethod
    def load_results(load_path: str, load_format: str = 'json') -> Dict[str, Any]:
        """
        Load saved results.

        Args:
            load_path: File path to load
            load_format: Format ('json' or 'npz')

        Returns:
            Loaded result dict.
        """
        load_path = Path(load_path)

        if load_format == 'json':
            return SpectrumResultsSaver._load_json(load_path)
        elif load_format == 'npz':
            return SpectrumResultsSaver._load_npz(load_path)
        else:
            raise ValueError(f"Unsupported load format: {load_format}")

    @staticmethod
    def _load_json(json_path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _load_npz(npz_path: Path) -> Dict[str, Any]:
        """Load NPZ file."""
        data = np.load(npz_path)
        return dict(data)


def compute_eigenvalue_mse(
    predicted_eigenvalues: torch.Tensor,
    true_eigenvalues: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate eigenvalue MSE (standalone function).

    Args:
        predicted_eigenvalues: Predicted eigenvalues mu_pred in C^{d_A}
        true_eigenvalues: True eigenvalues mu_true in C^{k}

    Returns:
        Dict with MSE evaluation results.
    """
    analyzer = SpectrumAnalyzer(sampling_interval=1.0)
    return analyzer.evaluate_against_truth(predicted_eigenvalues, true_eigenvalues)


def create_spectrum_analyzer(sampling_interval: float) -> SpectrumAnalyzer:
    """Create a spectrum analyzer."""
    return SpectrumAnalyzer(sampling_interval)


def create_model_spectrum_analyzer(sampling_interval: float) -> TrainedModelSpectrumAnalysis:
    """Create a model spectrum analyzer."""
    return TrainedModelSpectrumAnalysis(sampling_interval)