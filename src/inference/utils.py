# src/inference/utils.py
"""
Inference utility functions.

Provides noise covariance estimation (Eq. 45-46), numerical stability checks,
and filter performance evaluation helpers.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
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


def compute_innovation_residuals(
    predictions: torch.Tensor,
    observations: torch.Tensor
) -> torch.Tensor:
    """
    Compute innovation residuals for filter performance evaluation.

    Args:
        predictions: Prediction sequence (T,)
        observations: Observation sequence (T,)

    Returns:
        Innovation residuals (T,).
    """
    return observations - predictions


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


def compute_log_likelihood(
    innovations: torch.Tensor,
    innovation_covariances: torch.Tensor
) -> torch.Tensor:
    """
    Compute log-likelihood from innovation sequence (Gaussian assumption).

    Args:
        innovations: Innovation sequence (T,)
        innovation_covariances: Innovation covariance sequence (T,)

    Returns:
        Log-likelihood sequence (T,).
    """
    log_likelihoods = -0.5 * (
        torch.log(2 * torch.pi * innovation_covariances) +
        (innovations ** 2) / innovation_covariances
    )
    
    return log_likelihoods


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


def create_test_operators(
    dA: int = 10,
    dB: int = 5,
    r: int = 3,
    m: int = 8,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, float]]:
    """
    Generate test operators (multivariate-compatible).

    Creates stable operators for Kalman filter testing and debugging.

    Args:
        dA: Feature-space state dimension
        dB: Feature-space observation dimension
        r: Original state dimension
        m: Multivariate feature dimension
        device: Compute device

    Returns:
        V_A: State transfer operator (dA, dA)
        V_B: Observation transfer operator (dB, dA)
        U_A: State readout matrix (dA, r)
        U_B: Observation readout matrix (dB, m)
        Q: State noise covariance (dA, dA)
        R: Observation noise covariance (m, m) or scalar
    """
    device = torch.device(device)
    
    # Generate stable V_A (eigenvalues constrained within unit circle)
    V_A = torch.randn(dA, dA, device=device) * 0.1
    eigenvals, eigenvecs = torch.linalg.eig(V_A)
    eigenvals = eigenvals * 0.9 / torch.abs(eigenvals).max()  # Limit max eigenvalue to 0.9
    V_A = torch.real(eigenvecs @ torch.diag(eigenvals) @ torch.linalg.inv(eigenvecs))

    # V_B: Random matrix
    V_B = torch.randn(dB, dA, device=device) * 0.5

    # U_A: Random readout matrix
    U_A = torch.randn(dA, r, device=device) * 0.7

    # U_B: Random readout matrix (multivariate)
    U_B = torch.randn(dB, m, device=device) * 0.5

    # Q: Positive definite covariance matrix
    Q_half = torch.randn(dA, dA, device=device) * 0.1
    Q = Q_half @ Q_half.T + 0.01 * torch.eye(dA, device=device)

    # R: Observation noise covariance (multivariate)
    if m == 1:
        R = 0.1  # Scalar (backward compatibility)
    else:
        R_half = torch.randn(m, m, device=device) * 0.05
        R = R_half @ R_half.T + 0.01 * torch.eye(m, device=device)

    return V_A, V_B, U_A, U_B, Q, R


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


# Multivariate utility functions

def create_multivariate_kalman_filter(
    df_state_layer,
    df_obs_layer,
    encoder,
    initial_R: Union[float, torch.Tensor] = 1e-2,
    Q_regularization: float = 1e-3,
    device: str = 'cpu'
):
    """
    Factory function for multivariate Kalman filter.

    Args:
        df_state_layer: Trained DF-A layer
        df_obs_layer: Trained DF-B layer
        encoder: Encoder
        initial_R: Initial observation noise variance
        Q_regularization: State noise regularization
        device: Compute device

    Returns:
        Configured OperatorBasedKalmanFilter.
    """
    from .kalman_filter import OperatorBasedKalmanFilter

    # Retrieve trained parameters
    V_A = df_state_layer.get_state_operator()  # (d_A, d_A)
    U_A = df_state_layer.get_readout_matrix()  # (d_A, r)
    V_B = df_obs_layer.get_observation_operator()  # (d_B, d_A)
    U_B = df_obs_layer.get_readout_matrix()  # (d_B, m)

    # Get dimensions
    d_A = V_A.size(0)
    d_B = V_B.size(0)

    # State noise covariance (diagonal)
    Q = Q_regularization * torch.eye(d_A, device=device)

    # Observation noise covariance (multivariate)
    if isinstance(initial_R, (int, float)):
        R = initial_R * torch.eye(d_B, device=device)
    else:
        R = initial_R.to(device)

    # Initialize Kalman filter
    kalman_filter = OperatorBasedKalmanFilter(
        V_A=V_A,
        V_B=V_B,
        U_A=U_A,
        U_B=U_B,
        Q=Q,
        R=R,
        encoder=encoder,
        df_obs_layer=df_obs_layer,
        device=device
    )

    return kalman_filter


def estimate_multivariate_observation_noise_covariance(
    kalman_filter,
    observations: torch.Tensor,
    states: Optional[torch.Tensor] = None,
    regularization: float = 1e-3
) -> torch.Tensor:
    """
    Estimate multivariate observation noise covariance from observation sequence.

    Args:
        kalman_filter: Trained Kalman filter
        observations: Observation sequence (T, n)
        states: State sequence (T, r) - estimated via filtering if None
        regularization: Regularization parameter

    Returns:
        R_estimated: Estimated observation noise covariance (d_B, d_B)
    """
    from .noise_covariance import ObservationNoiseCovarianceEstimator

    T = observations.size(0)

    # Obtain or estimate state sequence
    if states is None:
        # Estimate states via Kalman filtering
        with torch.no_grad():
            # Initialization
            kalman_filter.initialize_state(observations[:min(10, T)])

            # Run filtering
            X_means, _, _ = kalman_filter.filter_sequence(observations)
            states = X_means  # (T, r)

    # Compute observation features
    with torch.no_grad():
        # Obtain multivariate features via encoder
        m_features = []
        for t in range(T):
            obs_t = observations[t].unsqueeze(0).unsqueeze(0)  # (1, 1, n)
            m_t = kalman_filter.encoder(obs_t).squeeze()  # (m,)
            m_features.append(m_t)
        m_features = torch.stack(m_features)  # (T, m)

        # Generate observation features
        psi_obs = []
        for t in range(T):
            psi_t = kalman_filter._generate_obs_features_from_multivariate(m_features[t])
            psi_obs.append(psi_t)
        psi_obs = torch.stack(psi_obs)  # (T, d_B)

        # Compute predicted features
        psi_pred = []
        for t in range(T):
            if t == 0:
                # Initial prediction
                x_hat_prev = states[0]  # Use initial state
            else:
                x_hat_prev = states[t-1]

            # State features
            phi_prev = kalman_filter.df_obs_layer.phi_theta(x_hat_prev.unsqueeze(0)).squeeze()  # (d_A,)

            # Observation prediction
            psi_pred_t = kalman_filter.V_B @ phi_prev  # (d_B,)
            psi_pred.append(psi_pred_t)
        psi_pred = torch.stack(psi_pred)  # (T, d_B)

    # Use covariance estimator
    d_B = psi_obs.size(1)
    estimator = ObservationNoiseCovarianceEstimator(
        d_B=d_B,
        regularization=regularization
    )

    R_estimated = estimator.estimate_from_sequences(psi_obs, psi_pred)
    return R_estimated


def create_adaptive_multivariate_kalman_filter(
    df_state_layer,
    df_obs_layer,
    encoder,
    initial_observations: torch.Tensor,
    device: str = 'cpu',
    **kwargs
):
    """
    Create Kalman filter with adaptively estimated observation noise covariance.

    Args:
        df_state_layer: Trained DF-A layer
        df_obs_layer: Trained DF-B layer
        encoder: Encoder
        initial_observations: Initial observation data (T_init, n)
        device: Compute device
        **kwargs: Additional parameters

    Returns:
        Adaptively configured OperatorBasedKalmanFilter.
    """
    # Create initial Kalman filter
    kalman_filter = create_multivariate_kalman_filter(
        df_state_layer=df_state_layer,
        df_obs_layer=df_obs_layer,
        encoder=encoder,
        device=device,
        **kwargs
    )

    # Estimate observation noise covariance from initial observations
    regularization = kwargs.get('noise_regularization', 1e-3)
    R_estimated = estimate_multivariate_observation_noise_covariance(
        kalman_filter=kalman_filter,
        observations=initial_observations,
        regularization=regularization
    )

    # Update filter with estimated covariance
    kalman_filter.R = R_estimated

    return kalman_filter


def batch_multivariate_filtering(
    df_state_layer,
    df_obs_layer,
    encoder,
    observation_sequences: torch.Tensor,
    adaptive_noise: bool = True,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Multivariate Kalman filtering over batch observation sequences.

    Args:
        df_state_layer: Trained DF-A layer
        df_obs_layer: Trained DF-B layer
        encoder: Encoder
        observation_sequences: Observation sequences (batch_size, T, n)
        adaptive_noise: Whether to use adaptive noise estimation
        device: Compute device

    Returns:
        Dict containing:
            - filtered_states: (batch_size, T, r)
            - state_covariances: (batch_size, T, r, r)
            - estimated_noise_covariances: (batch_size, d_B, d_B) if adaptive_noise
    """
    batch_size, T, n = observation_sequences.shape

    # Get dimension info
    r = df_state_layer.get_readout_matrix().size(1)  # Original state dimension
    d_B = df_obs_layer.get_observation_operator().size(0)  # Observation feature dimension

    # Result storage
    results = {
        'filtered_states': torch.zeros(batch_size, T, r, device=device),
        'state_covariances': torch.zeros(batch_size, T, r, r, device=device)
    }

    if adaptive_noise:
        results['estimated_noise_covariances'] = torch.zeros(batch_size, d_B, d_B, device=device)

    # Run filtering for each sequence
    for b in range(batch_size):
        obs_seq = observation_sequences[b]  # (T, n)

        if adaptive_noise:
            # Create adaptive filter
            kalman_filter = create_adaptive_multivariate_kalman_filter(
                df_state_layer=df_state_layer,
                df_obs_layer=df_obs_layer,
                encoder=encoder,
                initial_observations=obs_seq,
                device=device
            )
            results['estimated_noise_covariances'][b] = kalman_filter.R
        else:
            # Create standard filter
            kalman_filter = create_multivariate_kalman_filter(
                df_state_layer=df_state_layer,
                df_obs_layer=df_obs_layer,
                encoder=encoder,
                device=device
            )

        # Run filtering
        X_means, X_covariances, _ = kalman_filter.filter_sequence(obs_seq)

        results['filtered_states'][b] = X_means
        results['state_covariances'][b] = X_covariances

    return results


def validate_multivariate_kalman_setup(
    df_state_layer,
    df_obs_layer,
    encoder,
    sample_observation: torch.Tensor
) -> Dict[str, Any]:
    """
    Validate multivariate Kalman filter setup.

    Args:
        df_state_layer: DF-A layer
        df_obs_layer: DF-B layer
        encoder: Encoder
        sample_observation: Sample observation (n,)

    Returns:
        Validation results dict.
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "dimension_info": {},
        "numerical_stability": {}
    }

    try:
        # Get dimensions
        V_A = df_state_layer.get_state_operator()
        V_B = df_obs_layer.get_observation_operator()
        U_A = df_state_layer.get_readout_matrix()
        U_B = df_obs_layer.get_readout_matrix()

        d_A = V_A.size(0)
        d_B = V_B.size(0)
        r = U_A.size(1)
        m = U_B.size(1)

        validation_results["dimension_info"] = {
            "d_A": d_A,
            "d_B": d_B,
            "r": r,
            "m": m,
            "observation_dim": sample_observation.numel()
        }

        # Dimension consistency check
        if V_A.shape != (d_A, d_A):
            validation_results["errors"].append(f"V_A is not square: {V_A.shape}")
            validation_results["valid"] = False

        if V_B.shape != (d_B, d_A):
            validation_results["errors"].append(f"V_B dimension mismatch: expected ({d_B}, {d_A}), got {V_B.shape}")
            validation_results["valid"] = False

        if U_A.shape != (d_A, r):
            validation_results["errors"].append(f"U_A dimension mismatch: expected ({d_A}, {r}), got {U_A.shape}")
            validation_results["valid"] = False

        if U_B.shape != (d_B, m):
            validation_results["errors"].append(f"U_B dimension mismatch: expected ({d_B}, {m}), got {U_B.shape}")
            validation_results["valid"] = False

        # Encoder output dimension check
        with torch.no_grad():
            sample_obs_batch = sample_observation.unsqueeze(0).unsqueeze(0)  # (1, 1, n)
            encoder_output = encoder(sample_obs_batch).squeeze()
            actual_m = encoder_output.numel()

            if actual_m != m:
                validation_results["errors"].append(f"Encoder output dimension mismatch: expected {m}, got {actual_m}")
                validation_results["valid"] = False

        # Check psi_omega existence in DF-B layer
        if not hasattr(df_obs_layer, 'psi_omega'):
            validation_results["errors"].append("DF-B layer missing psi_omega network")
            validation_results["valid"] = False
        else:
            # psi_omega input/output dimension check
            try:
                with torch.no_grad():
                    dummy_m = torch.randn(m)
                    psi_output = df_obs_layer.psi_omega(dummy_m)
                    actual_d_B = psi_output.numel()

                    if actual_d_B != d_B:
                        validation_results["errors"].append(f"psi_omega output dimension mismatch: expected {d_B}, got {actual_d_B}")
                        validation_results["valid"] = False
            except Exception as e:
                validation_results["errors"].append(f"psi_omega test failed: {e}")
                validation_results["valid"] = False

        # Numerical stability check (warnings, not errors)
        if validation_results["valid"]:
            # V_A eigenvalue check
            try:
                eigenvals_A = torch.linalg.eigvals(V_A)
                max_eigenval = torch.abs(eigenvals_A).max().item()
                validation_results["numerical_stability"]["V_A_max_eigenvalue"] = max_eigenval

                if max_eigenval > 1.0:
                    validation_results["warnings"].append(f"V_A potentially unstable: max eigenvalue {max_eigenval:.3f}")
            except Exception as e:
                validation_results["warnings"].append(f"V_A eigenvalue check failed: {e}")

            # Matrix condition number check
            for name, matrix in [("V_A", V_A), ("V_B", V_B), ("U_A", U_A), ("U_B", U_B)]:
                try:
                    cond_num = torch.linalg.cond(matrix).item()
                    validation_results["numerical_stability"][f"{name}_condition_number"] = cond_num

                    if cond_num > 1e8:
                        validation_results["warnings"].append(f"{name} ill-conditioned: condition number {cond_num:.2e}")
                except Exception as e:
                    validation_results["warnings"].append(f"{name} condition number check failed: {e}")

    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Validation failed: {e}")

    return validation_results