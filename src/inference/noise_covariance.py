import torch
import warnings
from typing import Union, Tuple, Dict, Any


class ObservationNoiseCovarianceEstimator:
    """
    Multivariate observation noise covariance estimator.

    Estimates sample covariance R in R^{d_B x d_B} from residuals
    rho_t := psi_omega(m_t) - V_B phi_theta(x_t) in R^{d_B},
    with regularization for numerical stability.
    """

    def __init__(
        self,
        d_B: int,
        regularization: float = 1e-3,
        min_eigenvalue: float = 1e-6,
        max_condition_number: float = 1e8
    ):
        """
        Args:
            d_B: Observation feature dimension
            regularization: Regularization parameter gamma_R
            min_eigenvalue: Minimum eigenvalue floor
            max_condition_number: Maximum condition number
        """
        self.d_B = d_B
        self.gamma_R = regularization
        self.min_eigenvalue = min_eigenvalue
        self.max_condition_number = max_condition_number

    def estimate_covariance(
        self,
        residuals: torch.Tensor,
        return_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Estimate sample covariance from residuals.

        Args:
            residuals: Residual sequence (T, d_B)
            return_stats: Whether to also return statistics

        Returns:
            R: Regularized observation noise covariance (d_B, d_B)
            stats: Statistics (optional)
        """
        if residuals.dim() != 2:
            raise ValueError(f"residuals must be 2D tensor (T, d_B), got shape: {residuals.shape}")

        T, d_B_actual = residuals.shape
        if d_B_actual != self.d_B:
            raise ValueError(f"Feature dimension mismatch: expected {self.d_B}, got {d_B_actual}")

        if T < 2:
            raise ValueError(f"Too few samples for covariance estimation: T={T}")

        residuals_centered = residuals - residuals.mean(dim=0, keepdim=True)  # (T, d_B)
        R_sample = (residuals_centered.T @ residuals_centered) / (T - 1)  # (d_B, d_B)

        R_regularized = self.regularize_covariance(R_sample)

        if return_stats:
            stats = self._compute_stats(R_sample, R_regularized, T)
            return R_regularized, stats
        return R_regularized

    def regularize_covariance(self, R_sample: torch.Tensor) -> torch.Tensor:
        """
        Regularize covariance matrix.

        Args:
            R_sample: Sample covariance (d_B, d_B)

        Returns:
            R_regularized: Regularized covariance (d_B, d_B)
        """
        # Ensure symmetry
        R_sample = (R_sample + R_sample.T) / 2

        # Diagonal regularization: R = R_sample + gamma_R * I_{d_B}
        R_regularized = R_sample + self.gamma_R * torch.eye(
            self.d_B, device=R_sample.device, dtype=R_sample.dtype
        )

        # Additional regularization via eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(R_regularized)

            # Clip negative/small eigenvalues
            eigenvalues_clipped = torch.clamp(eigenvalues, min=self.min_eigenvalue)

            # Condition number control
            max_eigenvalue = eigenvalues_clipped.max()
            min_allowed = max_eigenvalue / self.max_condition_number
            eigenvalues_final = torch.clamp(eigenvalues_clipped, min=min_allowed)

            R_final = eigenvectors @ torch.diag(eigenvalues_final) @ eigenvectors.T

        except torch.linalg.LinAlgError:
            # Fallback: stronger diagonal regularization
            warnings.warn("Eigenvalue decomposition failed, using stronger regularization")
            stronger_reg = max(self.gamma_R * 10, 1e-2)
            R_final = R_sample + stronger_reg * torch.eye(
                self.d_B, device=R_sample.device, dtype=R_sample.dtype
            )

        return R_final

    def _compute_stats(
        self,
        R_sample: torch.Tensor,
        R_regularized: torch.Tensor,
        T: int
    ) -> Dict[str, float]:
        """Compute diagnostic statistics."""
        try:
            eigenvals_sample = torch.linalg.eigvals(R_sample).real
            cond_sample = eigenvals_sample.max() / eigenvals_sample.min()
            det_sample = torch.det(R_sample)

            eigenvals_reg = torch.linalg.eigvals(R_regularized).real
            cond_reg = eigenvals_reg.max() / eigenvals_reg.min()
            det_reg = torch.det(R_regularized)

            return {
                'sample_size': T,
                'condition_number_sample': cond_sample.item(),
                'condition_number_regularized': cond_reg.item(),
                'determinant_sample': det_sample.item(),
                'determinant_regularized': det_reg.item(),
                'min_eigenvalue_sample': eigenvals_sample.min().item(),
                'min_eigenvalue_regularized': eigenvals_reg.min().item(),
                'regularization_applied': self.gamma_R,
                'numerical_stable': cond_reg < self.max_condition_number
            }
        except Exception as e:
            return {'error': str(e), 'sample_size': T}

    def estimate_from_sequences(
        self,
        psi_obs: torch.Tensor,
        psi_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate directly from observation and prediction features.

        Args:
            psi_obs: Observation feature sequence (T, d_B)
            psi_pred: Predicted feature sequence (T, d_B)

        Returns:
            R: Observation noise covariance (d_B, d_B)
        """
        if psi_obs.shape != psi_pred.shape:
            raise ValueError(f"Shape mismatch: psi_obs {psi_obs.shape} vs psi_pred {psi_pred.shape}")

        residuals = psi_obs - psi_pred  # (T, d_B)
        return self.estimate_covariance(residuals)

    def adaptive_estimation(
        self,
        residuals: torch.Tensor,
        window_size: int = 50,
        overlap: float = 0.5
    ) -> torch.Tensor:
        """
        Adaptive covariance estimation (sliding window).

        Args:
            residuals: Residual sequence (T, d_B)
            window_size: Window size
            overlap: Overlap ratio

        Returns:
            R_adaptive: Adaptive observation noise covariance (d_B, d_B)
        """
        T = residuals.size(0)
        if T < window_size:
            return self.estimate_covariance(residuals)

        step_size = max(1, int(window_size * (1 - overlap)))
        covariance_estimates = []

        for start in range(0, T - window_size + 1, step_size):
            end = start + window_size
            window_residuals = residuals[start:end]
            R_window = self.estimate_covariance(window_residuals)
            covariance_estimates.append(R_window)

        # Average estimates for more stable estimation
        R_adaptive = torch.stack(covariance_estimates).mean(dim=0)
        return self.regularize_covariance(R_adaptive)