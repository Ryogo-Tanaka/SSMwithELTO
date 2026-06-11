"""
Koopman spectrum analysis from learned transfer operators.
"""

from typing import Any, Dict, List

import numpy as np
import torch


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
            Dict with discrete and continuous eigenvalue summaries.
        """
        with torch.no_grad():
            eigenvalues_discrete, eigenvectors = torch.linalg.eig(V_A)

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
        """Convert discrete-time eigenvalues to continuous-time: mu = log(lambda)/dt."""
        eigenvalues_log = torch.log(eigenvalues_discrete + 1e-12)
        return eigenvalues_log / self.dt

    def _find_dominant_modes(
        self,
        eigenvalues_magnitude: torch.Tensor,
        threshold: float = 0.1
    ) -> List[int]:
        """Identify dominant modes by spectral-radius-relative magnitude."""
        spectral_radius = torch.max(eigenvalues_magnitude)
        dominant_mask = eigenvalues_magnitude > threshold * spectral_radius
        return dominant_mask.nonzero(as_tuple=True)[0].tolist()

    def _find_stable_modes(self, growth_rates: torch.Tensor) -> List[int]:
        """Identify stable modes (Re(mu) < 0)."""
        stable_mask = growth_rates < 0
        return stable_mask.nonzero(as_tuple=True)[0].tolist()
