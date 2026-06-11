"""
Inference utility functions for Kalman image evaluation.
"""

import warnings
from typing import Dict, Tuple

import torch


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
    _, dA = residuals_state.shape
    _, dB = residuals_obs.shape

    gamma_Q = regularization.get("gamma_Q", 1e-6)
    gamma_R = regularization.get("gamma_R", 1e-6)

    Q = torch.mean(
        torch.einsum('ti,tj->tij', residuals_state, residuals_state),
        dim=0
    )
    Q += gamma_Q * torch.eye(int(dA), device=residuals_state.device)

    R = torch.mean(
        torch.einsum('ti,tj->tij', residuals_obs, residuals_obs),
        dim=0
    )
    R += gamma_R * torch.eye(int(dB), device=residuals_obs.device)

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
    Compute residuals from transfer operators.

    Args:
        phi_sequence: State feature sequence (T+1, dA)
        psi_sequence: Observation feature sequence (T+1, dB)
        V_A: State transfer operator (dA, dA)
        V_B: Observation transfer operator (dB, dA)

    Returns:
        residuals_state: State residuals (T, dA)
        residuals_obs: Observation residuals (T+1, dB)
    """
    phi_pred = phi_sequence[:-1] @ V_A.T
    residuals_state = phi_sequence[1:] - phi_pred

    psi_pred = phi_sequence @ V_B.T
    residuals_obs = psi_sequence - psi_pred

    return residuals_state, residuals_obs


def regularize_covariance(
    cov_matrix: torch.Tensor,
    min_eigenvalue: float = 1e-8,
    jitter: float = 1e-6
) -> torch.Tensor:
    """
    Regularize covariance matrix to ensure positive definiteness.
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
