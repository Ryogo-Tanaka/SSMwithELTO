# src/inference/utils.py
"""
Inference utility functions.

Provides noise covariance estimation (Eq. 45-46), numerical stability checks,
and filter performance evaluation helpers.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import warnings


def estimate_noise_covariances(
    residuals_state: torch.Tensor,
    residuals_obs: torch.Tensor,
    regularization: Dict[str, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate Q, R from residuals (Eq. 45-46).

    Args:
        residuals_state: State residuals (T, dA)
        residuals_obs: Observation residuals (T, dB)
        regularization: {"gamma_Q": float, "gamma_R": float}

    Returns:
        Q: State noise covariance (dA, dA)
        R: Observation noise covariance (dB, dB)
    """
    T_state, dA = residuals_state.shape
    T_obs, dB = residuals_obs.shape
    
    dA = int(dA)
    dB = int(dB)
    gamma_Q = regularization.get("gamma_Q", 1e-6)
    gamma_R = regularization.get("gamma_R", 1e-6)
    
    # Eq. 45: Q = (1/T) sum_t epsilon_t epsilon_t^T + gamma_Q * I_{dA}
    Q = torch.mean(
        torch.einsum('ti,tj->tij', residuals_state, residuals_state), 
        dim=0
    )  # (dA, dA)
    Q += gamma_Q * torch.eye(dA, device=residuals_state.device)
    
    # Eq. 46: R = (1/(T+1)) sum_t rho_t rho_t^T + gamma_R * I_{dB}
    R = torch.mean(
        torch.einsum('ti,tj->tij', residuals_obs, residuals_obs),
        dim=0
    )  # (dB, dB)
    R += gamma_R * torch.eye(dB, device=residuals_obs.device)
    
    Q = regularize_covariance(Q)
    R = regularize_covariance(R)

    return Q, R


def compute_residuals_from_operators(
    phi_sequence: torch.Tensor,
    psi_sequence: torch.Tensor,
    V_A: torch.Tensor,
    V_B: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute residuals from transfer operators (preprocessing for noise estimation).

    Args:
        phi_sequence: State feature sequence (T+1, dA)
        psi_sequence: Observation feature sequence (T+1, dB)
        V_A: State transfer operator (dA, dA)
        V_B: Observation transfer operator (dB, dA)

    Returns:
        residuals_state: State residuals (T, dA)
        residuals_obs: Observation residuals (T+1, dB)
    """
    T = phi_sequence.size(0) - 1  # T+1 → T
    
    # State residuals
    phi_pred = phi_sequence[:-1] @ V_A.T  # (T, dA)
    residuals_state = phi_sequence[1:] - phi_pred  # (T, dA)
    
    # Observation residuals
    psi_pred = phi_sequence @ V_B.T  # (T+1, dB)
    residuals_obs = psi_sequence - psi_pred  # (T+1, dB)
    
    return residuals_state, residuals_obs


def check_numerical_stability(
    matrix: torch.Tensor,
    name: str = "Matrix",
    condition_threshold: float = 1e12,
    min_eigenvalue: float = 1e-8
) -> Dict[str, Any]:
    """
    Check numerical stability of a matrix (condition number, eigenvalues).

    Args:
        matrix: Matrix to check (d, d)
        name: Matrix name (for logging)
        condition_threshold: Condition number threshold
        min_eigenvalue: Minimum eigenvalue threshold

    Returns:
        Dict of diagnostic results.
    """
    try:
        condition_number = torch.linalg.cond(matrix).item()
        
        eigenvalues = torch.linalg.eigvals(matrix).real
        min_eig = eigenvalues.min().item()
        max_eig = eigenvalues.max().item()
        
        is_symmetric = torch.allclose(matrix, matrix.T, atol=1e-6)
        
        is_positive_definite = min_eig > 0
        
        is_stable = (
            condition_number < condition_threshold and
            min_eig > min_eigenvalue and
            is_symmetric and
            is_positive_definite
        )
        
        return {
            "matrix_name": name,
            "shape": matrix.shape,
            "condition_number": condition_number,
            "eigenvalues": {
                "min": min_eig,
                "max": max_eig,
                "range": max_eig - min_eig
            },
            "properties": {
                "symmetric": is_symmetric,
                "positive_definite": is_positive_definite
            },
            "stability": {
                "is_stable": is_stable,
                "condition_ok": condition_number < condition_threshold,
                "eigenvals_ok": min_eig > min_eigenvalue
            }
        }
        
    except Exception as e:
        return {
            "matrix_name": name,
            "error": str(e),
            "is_stable": False
        }


def regularize_covariance(
    cov_matrix: torch.Tensor,
    min_eigenvalue: float = 1e-8,
    jitter: float = 1e-6
) -> torch.Tensor:
    """
    Regularize covariance matrix to ensure positive definiteness.

    Args:
        cov_matrix: Covariance matrix (d, d)
        min_eigenvalue: Minimum eigenvalue floor
        jitter: Jitter term for fallback

    Returns:
        Regularized covariance matrix (d, d).
    """
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalue)
        
        cov_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        
    except Exception:
        warnings.warn(f"Eigendecomposition failed, adding jitter: {jitter}")
        cov_matrix += jitter * torch.eye(cov_matrix.size(0), device=cov_matrix.device)
        
    return cov_matrix


def initialize_state_data_driven(
    feature_samples: torch.Tensor,
    method: str = "empirical"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Data-driven state initialization (Eq. 47-48).

    Args:
        feature_samples: Feature samples (N0, dA)
        method: Initialization method ("empirical" | "robust")

    Returns:
        mu_0: Initial state mean (dA,)
        Sigma_0: Initial state covariance (dA, dA)
    """
    N0, dA = feature_samples.shape
    
    if method == "empirical":
        # Eq. 47: mu_0 = (1/N0) sum_{i=1}^{N0} phi_theta(x_0^{(i)})
        mu_0 = torch.mean(feature_samples, dim=0)  # (dA,)

        # Eq. 48: Sigma_0 = (1/(N0-1)) sum_{i=1}^{N0} (phi_theta(x_0^{(i)}) - mu_0)(phi_theta(x_0^{(i)}) - mu_0)^T
        centered = feature_samples - mu_0.unsqueeze(0)  # (N0, dA)
        Sigma_0 = (centered.T @ centered) / (N0 - 1)  # (dA, dA)
        
    elif method == "robust":
        # Robust estimation (median-based)
        mu_0 = torch.median(feature_samples, dim=0)[0]  # (dA,)

        # MAD (Median Absolute Deviation) based covariance
        centered = feature_samples - mu_0.unsqueeze(0)
        mad = torch.median(torch.abs(centered), dim=0)[0]  # (dA,)
        Sigma_0 = torch.diag(mad ** 2)  # (dA, dA)
        
    else:
        raise ValueError(f"Unknown initialization method: {method}")
        
    # Ensure positive definiteness
    Sigma_0 = regularize_covariance(Sigma_0)
    
    return mu_0, Sigma_0


def validate_kalman_inputs(
    V_A: torch.Tensor,
    V_B: torch.Tensor,
    U_A: torch.Tensor,
    u_B: torch.Tensor,
    Q: torch.Tensor,
    R: Union[torch.Tensor, float]
) -> Dict[str, Any]:
    """
    Validate Kalman filter input parameters.

    Checks dimension consistency and numerical properties.

    Args:
        V_A: State transfer operator (dA, dA)
        V_B: Observation transfer operator (dB, dA)
        U_A: State readout matrix (dA, r)
        u_B: Observation readout vector (dB,)
        Q: State noise covariance (dA, dA)
        R: Observation noise variance

    Returns:
        Validation results dict.
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "dimension_check": {},
        "numerical_check": {}
    }
    
    try:
        # Dimension check
        dA_A, dA_A2 = V_A.shape
        dB, dA_B = V_B.shape
        dA_U, r = U_A.shape
        dB_u = u_B.shape[0]
        
        validation_results["dimension_check"] = {
            "V_A_square": dA_A == dA_A2,
            "dimension_consistency": dA_A == dA_B == dA_U,
            "observation_consistency": dB == dB_u,
            "dimensions": {
                "dA": dA_A,
                "dB": dB,
                "r": r
            }
        }
        
        # Dimension consistency check
        if dA_A != dA_A2:
            validation_results["errors"].append("V_A is not square")
            validation_results["valid"] = False
            
        if not (dA_A == dA_B == dA_U):
            validation_results["errors"].append("Feature dimension mismatch")
            validation_results["valid"] = False
            
        if dB != dB_u:
            validation_results["errors"].append("Observation dimension mismatch")
            validation_results["valid"] = False
            
        # Numerical property check
        if validation_results["valid"]:
            # Positive definiteness of Q
            Q_check = check_numerical_stability(Q, "Q")
            validation_results["numerical_check"]["Q"] = Q_check
            if not Q_check.get("stability", {}).get("is_stable", False):
                validation_results["warnings"].append("Q matrix numerically unstable")
                
            # V_A stability (eigenvalues within unit circle)
            try:
                eigenvals_A = torch.linalg.eigvals(V_A)
                max_eigenval_A = torch.abs(eigenvals_A).max().item()
                
                validation_results["numerical_check"]["V_A"] = {
                    "max_eigenvalue_magnitude": max_eigenval_A,
                    "stable": max_eigenval_A <= 1.0
                }
                
                if max_eigenval_A > 1.0:
                    validation_results["warnings"].append(
                        f"V_A may be unstable (max eigenvalue: {max_eigenval_A:.3f})"
                    )
                    
            except Exception as e:
                validation_results["warnings"].append(f"V_A eigenvalue check failed: {e}")
                
            # Positive definiteness of R
            if isinstance(R, torch.Tensor):
                if R.dim() == 0:  # scalar
                    R_positive = R.item() > 0
                elif R.dim() == 2:  # matrix
                    R_check = check_numerical_stability(R, "R")
                    validation_results["numerical_check"]["R"] = R_check
                    R_positive = R_check.get("stability", {}).get("is_stable", False)
                else:
                    R_positive = False
            else:  # float/int
                R_positive = R > 0
                
            validation_results["numerical_check"]["R_positive"] = R_positive
            if not R_positive:
                validation_results["errors"].append("R is not positive")
                validation_results["valid"] = False
                
    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Validation failed: {e}")
        
    return validation_results


def format_filter_results(
    X_means: torch.Tensor,
    X_covariances: torch.Tensor,
    likelihoods: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Format filter results for visualization and storage.

    Args:
        X_means: State mean sequence (T, r)
        X_covariances: State covariance sequence (T, r, r)
        likelihoods: Observation likelihood sequence (T,) [optional]

    Returns:
        Formatted results dict.
    """
    T, r = X_means.shape
    
    # Basic statistics
    results = {
        "summary": {
            "sequence_length": T,
            "state_dimension": r,
            "mean_trajectory": X_means.cpu().numpy(),
            "covariance_trajectory": X_covariances.cpu().numpy()
        },
        "statistics": {
            "state_means": {
                "temporal_mean": torch.mean(X_means, dim=0).cpu().numpy(),
                "temporal_std": torch.std(X_means, dim=0).cpu().numpy()
            },
            "uncertainty": {
                "mean_trace": torch.mean(torch.diagonal(X_covariances, dim1=1, dim2=2), dim=0).cpu().numpy(),
                "mean_determinant": torch.mean(torch.det(X_covariances)).item()
            }
        }
    }
    
    # Likelihood statistics
    if likelihoods is not None:
        results["statistics"]["likelihood"] = {
            "total_log_likelihood": torch.sum(likelihoods).item(),
            "mean_log_likelihood": torch.mean(likelihoods).item(),
            "likelihood_trajectory": likelihoods.cpu().numpy()
        }
        
    # Confidence intervals (+-2 sigma)
    std_devs = torch.sqrt(torch.diagonal(X_covariances, dim1=1, dim2=2))  # (T, r)
    results["confidence_intervals"] = {
        "lower_2sigma": (X_means - 2 * std_devs).cpu().numpy(),
        "upper_2sigma": (X_means + 2 * std_devs).cpu().numpy()
    }
    
    return results
