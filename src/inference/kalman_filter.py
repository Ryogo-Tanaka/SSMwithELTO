# src/inference/kalman_filter.py
"""
Algorithm 1: Operator-based Kalman filtering with Galerkin-projected
conditional covariance operators.

Input:
- Learned operators: VA in R^(dA x dA), VB in R^(dB x dA), UB in R^(dB x m), UA^T in R^(r x dA)
- Noise covariances: Q in R^(dA x dA), R in R^(m x m)
- Encoder: u_eta : R^n -> R^m
- Initial values: (mu_0, Sigma_0)

Output:
- Filtered states: x_hat_t in R^r
- Covariances: Sigma_t^(x) in R^(r x r) at each t
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings


class OperatorBasedKalmanFilter:
    """
    Implementation of Algorithm 1.

    Sequential state estimation via operator-based Kalman updates.
    Performs Kalman updates in feature space, then recovers original state space.
    """
    
    def __init__(
        self,
        V_A: torch.Tensor,
        V_B: torch.Tensor,
        U_A: Optional[torch.Tensor],
        U_B: Optional[torch.Tensor],
        Q: torch.Tensor,
        R: Union[torch.Tensor, float],
        encoder: nn.Module,
        df_obs_layer: nn.Module = None,
        device: str = 'cpu',
        readout_A: Optional[nn.Module] = None,
        readout_B: Optional[nn.Module] = None,
        state_dim_r: Optional[int] = None
    ):
        """
        Initialize Algorithm 1.

        Args:
            V_A: Learned state transfer operator (dA, dA)
            V_B: Learned observation transfer operator (dB, dA)
            U_A: Learned state readout matrix (dA, r), or None if nonlinear
            U_B: Learned observation readout matrix (dB, m), or None if nonlinear
            Q: State noise covariance (dA, dA)
            R: Observation noise (scalar or matrix)
            encoder: Encoder network (frozen)
            df_obs_layer: DF-B layer (with learned psi_omega)
            device: Computation device
            readout_A: Nonlinear state readout network r_{xi_A}: R^{dA} -> R^r
            readout_B: Nonlinear observation readout network r_{xi_B}: R^{dB} -> R^m
            state_dim_r: Explicit state dimension r (required when U_A is None and readout_A is provided)
        """
        self.device = torch.device(device)

        self.V_A = V_A.to(self.device)
        self.V_B = V_B.to(self.device)
        self.U_A = U_A.to(self.device) if U_A is not None else None
        self.U_B = U_B.to(self.device) if U_B is not None else None

        self.Q = Q.to(self.device)
        if isinstance(R, (int, float)):
            self.R = torch.tensor(R, dtype=torch.float32, device=self.device)
        else:
            self.R = R.to(self.device)

        # Encoder (no gradients during inference)
        self.encoder = encoder
        self.encoder.eval()

        # DF-B layer (contains learned psi_omega observation feature network)
        self.df_obs_layer = df_obs_layer
        if self.df_obs_layer is not None:
            self.df_obs_layer.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Nonlinear readout networks (Phase II)
        self.readout_A = readout_A
        self.readout_B = readout_B
        self.use_nonlinear_readout_A = (readout_A is not None)
        self.use_nonlinear_readout_B = (readout_B is not None)

        if self.readout_A is not None:
            self.readout_A = self.readout_A.to(self.device)
            self.readout_A.eval()
            for param in self.readout_A.parameters():
                param.requires_grad = False

        if self.readout_B is not None:
            self.readout_B = self.readout_B.to(self.device)
            self.readout_B.eval()
            for param in self.readout_B.parameters():
                param.requires_grad = False

        # Dimensions
        self.dA = int(V_A.size(0))
        self.dB = int(V_B.size(0))

        # Determine r (state dimension in original space)
        if state_dim_r is not None:
            self.r = state_dim_r
        elif U_A is not None:
            self.r = int(U_A.size(1))
        elif self.readout_A is not None:
            # Probe readout_A to determine output dimension
            with torch.no_grad():
                probe_input = torch.zeros(self.dA, device=self.device)
                probe_output = self.readout_A(probe_input)
                self.r = int(probe_output.size(0))
        else:
            raise ValueError("Cannot determine state dimension r: provide U_A, readout_A, or state_dim_r")

        # Internal state for sequential processing
        self.mu: Optional[torch.Tensor] = None
        self.Sigma: Optional[torch.Tensor] = None
        self.is_initialized = False

        # Numerical stability parameters
        self.min_eigenvalue = 1e-6
        self.condition_threshold = 1e12
        self.jitter = 1e-5

    def initialize_state(
        self,
        initial_observations: torch.Tensor,
        method: str = "data_driven"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize mu_0, Sigma_0 (Eq. 47-48).

        Args:
            initial_observations: Initial observation samples (N0, n)
            method: Initialization method ("data_driven" | "zero")

        Returns:
            mu_0: Initial state mean (dA,)
            Sigma_0: Initial state covariance (dA, dA)
        """
        N0 = initial_observations.size(0)
        
        if method == "data_driven":
            # Eq. 47-48: Data-driven initialization
            with torch.no_grad():
                if initial_observations.dim() == 2:
                    # (N0, n) -> (N0,) scalar features
                    m_initial = self.encoder(initial_observations.unsqueeze(0)).squeeze()
                else:
                    m_initial = self.encoder(initial_observations)


                # Approximate feature mapping (simplified; real implementation uses DF-B psi_omega)
                phi_samples = self._approximate_feature_mapping(m_initial)  # (N0, dA)
                
                mu_0 = torch.mean(phi_samples, dim=0)  # (dA,)

                centered = phi_samples - mu_0.unsqueeze(0)  # (N0, dA)
                Sigma_0 = (centered.T @ centered) / (N0 - 1)  # (dA, dA)

        elif method == "zero":
            mu_0 = torch.zeros(self.dA, device=self.device)
            Sigma_0 = torch.eye(self.dA, device=self.device)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
            
        # Ensure positive definiteness
        Sigma_0 = self._regularize_covariance(Sigma_0)

        self.mu = mu_0.clone()
        self.Sigma = Sigma_0.clone()
        self.is_initialized = True
        
        return mu_0, Sigma_0

    def _approximate_feature_mapping(self, m_series: torch.Tensor) -> torch.Tensor:
        """
        Approximate feature mapping from multivariate features to dA dimensions.

        Note: This is a simplified version. The actual implementation should use
        the learned phi_theta from DF-A/DF-B.

        Args:
            m_series: Multivariate feature series (T, m) or (T,)

        Returns:
            Approximate features (T, dA).
        """
        T = m_series.size(0)

        if m_series.dim() == 1:
            m_series = m_series.unsqueeze(1)  # (T, 1)

        m = m_series.size(1)

        # Simplified feature expansion (delay embedding + padding)
        features = []

        max_delay = min(5, T)
        for t in range(T):
            delayed = []
            for d in range(max_delay):
                if t - d >= 0:
                    delayed.append(m_series[t - d])  # (m,)
                else:
                    delayed.append(torch.zeros_like(m_series[0]))  # (m,)
            delayed_features = torch.cat(delayed, dim=0)  # (max_delay * m,)
            features.append(delayed_features)

        feature_matrix = torch.stack(features)  # (T, max_delay * m)

        # Expand to dA dimensions via linear projection
        if feature_matrix.size(1) < self.dA:
            padding = torch.randn(T, self.dA - feature_matrix.size(1), device=self.device) * 0.1
            feature_matrix = torch.cat([feature_matrix, padding], dim=1)
        elif feature_matrix.size(1) > self.dA:
            feature_matrix = feature_matrix[:, :self.dA]
            
        return feature_matrix

    def predict_step(
        self, 
        mu_prev: torch.Tensor, 
        Sigma_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Time update (prediction step).

        Args:
            mu_prev: Previous state mean (dA,)
            Sigma_prev: Previous state covariance (dA, dA)

        Returns:
            mu_minus: Predicted state mean (dA,)
            Sigma_minus: Predicted state covariance (dA, dA)
        """
        # µ⁻ₜ = V_A µ⁺ₜ₋₁
        mu_minus = self.V_A @ mu_prev  # (dA,)
        
        # Σ⁻ₜ = V_A Σ⁺ₜ₋₁ V_A^T + Q
        Sigma_minus = self.V_A @ Sigma_prev @ self.V_A.T + self.Q  # (dA, dA)
        
        Sigma_minus = self._regularize_covariance(Sigma_minus)
        
        return mu_minus, Sigma_minus

    def update_step(
        self,
        mu_minus: torch.Tensor,
        Sigma_minus: torch.Tensor,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Observation update (Innovation update) - Algorithm 1.

        Args:
            mu_minus: Predicted state mean (dA,)
            Sigma_minus: Predicted state covariance (dA, dA)
            observation: Current observation y_t (n,) or (1, n)

        Returns:
            mu_plus: Updated state mean (dA,)
            Sigma_plus: Updated state covariance (dA, dA)
            likelihood: Dummy value (likelihood not used in non-Gaussian formulation)
        """
        # 1. Encode current observation: m_t = u_eta(y_t) in R^m
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
            elif observation.dim() == 2:
                observation = observation.unsqueeze(1)  # (B, 1, d)
            m_t = self.encoder(observation).squeeze()  # (m,)

            # 1.5. Generate observation features: psi_omega(m_t) in R^{d_B} (Eq. 54)
            psi_omega_m_t = self._generate_obs_features_from_multivariate(m_t)  # (dB,)

        # 2. Observation prediction (Eq. 51)
        # hat{psi}^{-}_{t} = V_B mu^-_{t} in R^{d_B}
        psi_pred = self.V_B @ mu_minus  # (dB,)

        # 3. Innovation covariance (Eq. 52)
        # S_t = V_B Σ^-_t V_B^T + R ∈ ℝ^{d_B × d_B}
        S = self.V_B @ Sigma_minus @ self.V_B.T  # (dB, dB)

        # Add observation noise covariance (multivariate)
        if isinstance(self.R, torch.Tensor) and self.R.dim() == 2:
            S = S + self.R
        elif isinstance(self.R, (int, float)) or (isinstance(self.R, torch.Tensor) and self.R.dim() == 0):
            R_scalar = float(self.R) if isinstance(self.R, torch.Tensor) else self.R
            S = S + R_scalar * torch.eye(self.dB, device=S.device, dtype=S.dtype)
        else:
            raise ValueError(f"Unsupported observation noise covariance type: {type(self.R)}, shape: {self.R.shape if hasattr(self.R, 'shape') else 'N/A'}")

        # Numerical stability check
        try:
            eigenvalues = torch.linalg.eigvals(S).real
            if torch.any(eigenvalues <= 0):
                warnings.warn(f"Non-positive definite innovation covariance matrix detected")
                # Add jitter to diagonal for regularization
                S = S + self.jitter * torch.eye(S.size(0), device=self.device)
        except Exception as e:
            warnings.warn(f"Failed to check positive definiteness: {e}")
            S = S + self.jitter * torch.eye(S.size(0), device=self.device)

        # 4. Kalman gain (Eq. 53)
        # K_t = Σ^-_t V_B^T S_t^{-1} ∈ ℝ^{d_A × d_B}
        try:
            S_inv = torch.linalg.inv(S)  # (dB, dB)
            K = Sigma_minus @ self.V_B.T @ S_inv  # (dA, dB)
        except Exception as e:
            warnings.warn(f"Failed to invert innovation covariance: {e}")
            # Use pseudo-inverse as fallback
            S_pinv = torch.linalg.pinv(S)
            K = Sigma_minus @ self.V_B.T @ S_pinv  # (dA, dB)
        
        # 5. Innovation update (Eq. 54)
        # mu^+_t = mu^-_t + K_t(psi_omega(m_t) - hat{psi}^{-}_t) in R^{d_A}
        innovation = psi_omega_m_t - psi_pred  # (dB,)
        mu_plus = mu_minus + K @ innovation  # (dA,) = (dA, dB) @ (dB,)
        
        # 6. Covariance update (Eq. 55)
        # Σ^+_t = Σ^-_t - K_t V_B Σ^-_t ∈ ℝ^{d_A × d_A}
        Sigma_plus = Sigma_minus - K @ self.V_B @ Sigma_minus  # (dA, dA)
        
        Sigma_plus = self._regularize_covariance(Sigma_plus)

        # Likelihood computation disabled (non-Gaussian formulation)
        likelihood = 0.0

        return mu_plus, Sigma_plus, likelihood

    def filter_step(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        One-step filtering (sequential processing with internal state).

        Args:
            observation: Current observation y_t (n,)
            state: External state (mu, Sigma) or None (use internal state)

        Returns:
            x_hat_t: Estimated state (r,)
            Sigma_x_t: State covariance (r, r)
            likelihood: Dummy value (non-Gaussian formulation)
        """
        if state is not None:
            mu_prev, Sigma_prev = state
        else:
            if not self.is_initialized:
                raise RuntimeError("Filter not initialized. Call initialize_state() first.")
            mu_prev, Sigma_prev = self.mu, self.Sigma
            
        # Time update
        mu_minus, Sigma_minus = self.predict_step(mu_prev, Sigma_prev)

        # Observation update
        mu_plus, Sigma_plus, likelihood = self.update_step(
            mu_minus, Sigma_minus, observation
        )
        
        # Recover original state space
        x_hat_t, Sigma_x_t = self._recover_original_state(mu_plus, Sigma_plus)

        # Update internal state
        if state is None:
            self.mu = mu_plus
            self.Sigma = Sigma_plus
            
        return x_hat_t, Sigma_x_t, likelihood

    def filter_sequence(
        self,
        observations: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_likelihood: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Batch filtering over entire observation sequence.

        Args:
            observations: Observation sequence (T, n)
            initial_state: Initial state (mu_0, Sigma_0) or None
            return_likelihood: Whether to return likelihoods

        Returns:
            X_means: State mean sequence (T, r)
            X_covariances: State covariance sequence (T, r, r)
            likelihoods: Dummy values (T,) [optional]
        """
        T = observations.size(0)
        
        if initial_state is not None:
            self.mu, self.Sigma = initial_state
            self.is_initialized = True
        elif not self.is_initialized:
            n_init = min(10, T)
            self.initialize_state(observations[:n_init])
            
        X_means = torch.zeros(T, self.r, device=self.device)
        X_covariances = torch.zeros(T, self.r, self.r, device=self.device)
        if return_likelihood:
            likelihoods = torch.zeros(T, device=self.device)
        
        for t in range(T):
            x_hat_t, Sigma_x_t, likelihood = self.filter_step(observations[t])
            
            X_means[t] = x_hat_t
            X_covariances[t] = Sigma_x_t
            if return_likelihood:
                likelihoods[t] = likelihood
                
        if return_likelihood:
            return X_means, X_covariances, likelihoods
        return X_means, X_covariances

    def _recover_original_state(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recover original state space from feature-space filtered state.

        Linear readout: x_hat = U_A^T mu, Sigma_x = U_A^T Sigma U_A
        Nonlinear readout: x_hat = r_{xi_A}(mu), Sigma_x ~ J Sigma J^T (Jacobian linearization)

        Args:
            mu: Feature-space state mean (dA,)
            Sigma: Feature-space state covariance (dA, dA)

        Returns:
            x_hat: State estimate in original space (r,)
            Sigma_x: Covariance in original space (r, r)
        """
        if self.use_nonlinear_readout_A:
            # Nonlinear readout: x_hat = r_{xi_A}(mu)
            with torch.no_grad():
                x_hat = self.readout_A(mu)  # (r,)

            # Covariance via Jacobian linearization: Sigma_x ~ J Sigma J^T
            try:
                J = self._compute_readout_jacobian(mu)  # (r, dA)
                Sigma_x = J @ Sigma @ J.T  # (r, r)
            except Exception:
                # Fallback: diagonal covariance scaled by trace
                Sigma_x = torch.eye(self.r, device=self.device) * Sigma.diag().mean()

            Sigma_x = self._regularize_covariance(Sigma_x)
            return x_hat, Sigma_x
        else:
            # Linear readout: x_hat = U_A^T mu
            x_hat = self.U_A.T @ mu  # (r,)
            Sigma_x = self.U_A.T @ Sigma @ self.U_A  # (r, r)
            Sigma_x = self._regularize_covariance(Sigma_x)
            return x_hat, Sigma_x

    def _compute_readout_jacobian(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of readout_A at mu using torch.func.jacrev.

        Args:
            mu: Feature-space state mean (dA,)

        Returns:
            J: Jacobian matrix (r, dA) where J[i,j] = d(readout_A_i)/d(mu_j)
        """
        mu_input = mu.detach().clone().requires_grad_(True)
        J = torch.func.jacrev(self.readout_A)(mu_input)
        return J.detach()

    def _regularize_covariance(self, cov_matrix: torch.Tensor) -> torch.Tensor:
        """
        Regularize covariance matrix to ensure positive definiteness.

        Args:
            cov_matrix: Covariance matrix (d, d)

        Returns:
            Regularized covariance matrix (d, d).
        """
        # Ensure symmetry
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        d = cov_matrix.size(0)
        eye = torch.eye(d, device=self.device)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        except Exception:
            # Retry with jitter
            cov_matrix = cov_matrix + self.jitter * eye
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            except Exception:
                # Final fallback: use only diagonal elements
                diag_vals = torch.clamp(torch.diag(cov_matrix), min=self.min_eigenvalue)
                return torch.diag(diag_vals)

        # Clip negative eigenvalues
        eigenvalues = torch.clamp(eigenvalues, min=self.min_eigenvalue)

        # Reconstruct
        cov_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        
        return cov_matrix

    def _compute_likelihood(self, innovation: torch.Tensor, S: torch.Tensor) -> float:
        """
        Observation likelihood (deprecated).

        The formulation does not assume a Gaussian distribution, so likelihood
        computation is not theoretically applicable. Returns a dummy value.
        """
        return 0.0

    def get_current_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current state estimate and covariance.

        Returns:
            x_hat: Current state estimate (r,)
            Sigma_x: Current state covariance (r, r)
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized")
            
        return self._recover_original_state(self.mu, self.Sigma)

    def reset_state(self):
        """Reset internal state (for new sequence)."""
        self.mu = None
        self.Sigma = None
        self.is_initialized = False

    def check_numerical_stability(self) -> Dict[str, Any]:
        """Numerical stability diagnostics."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
            
        # Condition number check
        try:
            cond_Q = torch.linalg.cond(self.Q).item()
            cond_Sigma = torch.linalg.cond(self.Sigma).item()
            
            # Eigenvalue check
            eigenvals_Q = torch.linalg.eigvals(self.Q).real
            eigenvals_Sigma = torch.linalg.eigvals(self.Sigma).real
            
            return {
                "status": "ok",
                "condition_numbers": {
                    "Q": cond_Q,
                    "Sigma": cond_Sigma
                },
                "min_eigenvalues": {
                    "Q": eigenvals_Q.min().item(),
                    "Sigma": eigenvals_Sigma.min().item()
                },
                "numerical_stable": (
                    cond_Q < self.condition_threshold and 
                    cond_Sigma < self.condition_threshold and
                    eigenvals_Q.min() > 0 and 
                    eigenvals_Sigma.min() > 0
                )
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _generate_obs_features_from_multivariate(self, m_t: torch.Tensor) -> torch.Tensor:
        """
        Generate observation features from multivariate input: psi_omega(m_t) in R^{d_B}.

        Required for Eq. 54. Uses the learned psi_omega network from DF-B layer.

        Args:
            m_t: Multivariate features (encoder output) in R^m - (m,) or (1, m)

        Returns:
            psi_omega_m_t: Observation feature vector (dB,)
        """
        with torch.no_grad():
            if self.df_obs_layer is None or not hasattr(self.df_obs_layer, 'psi_omega'):
                raise RuntimeError(
                    "DF-B layer with psi_omega network is required for observation feature generation. "
                    "Cannot proceed without learned psi_omega(m_t) transformation."
                )

            # Adjust m_t shape for psi_omega input
            if m_t.dim() == 1:
                m_t_input = m_t
            elif m_t.dim() == 2 and m_t.size(0) == 1:
                m_t_input = m_t.squeeze(0)
            else:
                raise ValueError(f"m_t must have shape (m,) or (1, m), got: {m_t.shape}")

            psi_omega_m_t = self.df_obs_layer.psi_omega(m_t_input)  # (dB,) or (1, dB)

            while psi_omega_m_t.dim() > 1 and psi_omega_m_t.size(0) == 1:
                psi_omega_m_t = psi_omega_m_t.squeeze(0)

            if psi_omega_m_t.dim() != 1:
                raise RuntimeError(
                    f"psi_omega output has unexpected shape: {psi_omega_m_t.shape}. "
                    f"Expected 1D tensor (dB,) for observation features."
                )

            return psi_omega_m_t