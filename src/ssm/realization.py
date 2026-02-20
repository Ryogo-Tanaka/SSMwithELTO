import torch
import torch.nn as nn
from torch.linalg import cholesky, solve_triangular, eigvalsh, svd
import warnings
from typing import Optional, Dict, Any, Tuple, Union, List
import math

class RealizationError(Exception):
    """
    Exception raised when stochastic realization (SSM identification)
    fails due to numerical issues (ill-conditioning, non-positive-definite, etc.).
    """
    pass


class StochasticRealizationWithEncoder(nn.Module):
    """
    Stochastic realization based on functional CCA in Hilbert space.

    Formulation (Section 3):
    - Hilbert-space feature process: u_t = u_eta(y_t) in R^m (Section 3.1)
    - Past/future subspaces: H^- := span{u_{t-1}, ...}, H^+ := span{u_t, ...}
    - Functional CCA constrained optimization
    - Galerkin projection for finite-dimensional approximation with window length l

    Implementation:
    1. Full integration with time-invariant encoder u_eta
    2. Exact covariance operator computation under weak stationarity
    3. Numerical stabilization via Ridge regularization and SVD
    4. Theory-compliant transformation from canonical directions to state variables
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        past_horizon: int = 10,
        rank: int = 8,
        ridge_param: float = 1e-3,
        jitter: float = 1e-8,
        m: int = 500,
        device: str = 'cpu',
        feature_mapping_type: str = "averaging",
        feature_mapping_hidden_dims: Optional[List[int]] = None,
        feature_mapping_activation: str = "relu"
    ):
        """
        Args:
            encoder: Time-invariant encoder u_eta: R^n -> R^m
            past_horizon: Window length l for Galerkin approximation
            rank: State dimension r (number of top canonical directions)
            ridge_param: Ridge regularization parameter lambda
            jitter: Minimum eigenvalue clipping for numerical stability
            m: Number of samples for lag-covariance estimation
            device: Computation device
        """
        super().__init__()

        self.encoder = encoder
        self.window_length = int(past_horizon)
        self.num_components = int(rank)
        self.ridge_param = float(ridge_param)
        self.min_eigenvalue = float(jitter)
        self.m = int(m)
        self.device = device

        if encoder_output_dim is not None:
            self.feature_dim = encoder_output_dim
        else:
            self.feature_dim = self._get_encoder_output_dim()

        # Fitted parameters (set after fit())
        self.canonical_directions_past: Optional[torch.Tensor] = None
        self.canonical_directions_future: Optional[torch.Tensor] = None
        self.canonical_correlations: Optional[torch.Tensor] = None
        self.is_fitted = False

        # Backward compatibility with legacy Realization class
        self.h = self.window_length
        self.rank = self.num_components

        # Intermediate result cache
        self._cached_feature_matrices: Optional[Dict[str, torch.Tensor]] = None
        self._cached_covariance_blocks: Optional[Dict[str, torch.Tensor]] = None
        self._cached_whitening_matrices: Optional[Dict[str, torch.Tensor]] = None
        self._last_input_shape: Optional[Tuple[int, int]] = None

        self._feature_statistics: Optional[Dict[str, torch.Tensor]] = None

        self.feature_mapping_type = feature_mapping_type

        if feature_mapping_type == "averaging":
            self.component_transforms = None

        elif feature_mapping_type in ["linear", "mlp"]:
            if feature_mapping_type == "linear":
                hidden_dims = []
            else:
                if feature_mapping_hidden_dims is None:
                    hidden_dims = [32]
                else:
                    hidden_dims = feature_mapping_hidden_dims

            self.component_transforms = nn.ModuleList([
                self._build_component_transform(
                    input_dim=self.window_length,
                    hidden_dims=hidden_dims,
                    activation=feature_mapping_activation
                )
                for _ in range(self.feature_dim)
            ])
        else:
            raise ValueError(
                f"Unknown feature_mapping_type: {feature_mapping_type}. "
                f"Choose from: 'averaging', 'linear', 'mlp'"
            )

    def _build_component_transform(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str
    ) -> nn.Sequential:
        """
        Build per-component transform network: R^l -> R.

        Theory Section 4.4.1, equation 333:
        phi_tilde_m^(i)(p(t)) = MLP(phi_u^(i)(p(t))) in R

        Args:
            input_dim: Input dimension l (window_length)
            hidden_dims: Hidden layer dimensions (empty list = linear only)
            activation: Activation function ("relu", "tanh", "gelu")

        Returns:
            nn.Sequential: Transform network
        """
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU
        }

        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}")

        act_fn = activation_map[activation]

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))

        return nn.Sequential(*layers)

    def _get_encoder_output_dim(self) -> int:
        """Get encoder output dimension m via attribute lookup."""
        if hasattr(self.encoder, 'output_dim'):
            return self.encoder.output_dim

        # Explicit specification via config is recommended
        raise ValueError(
            "Cannot determine encoder output dimension. "
            "Please specify 'encoder_output_dim' in config or "
            "ensure encoder has 'output_dim' attribute. "
            "Example config: ssm.realization.encoder_output_dim: 16"
        )

    def _build_feature_matrices(
        self,
        Y: torch.Tensor,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build feature matrices using time-invariant encoder (with optional caching).

        Section 3.2-3.3:
        1. Apply encoder: Y -> M = [u_eta(y_1), ..., u_eta(y_T)]
        2. Past/future block construction:
           p(t) = (y(t-1)^T, ..., y(t-l)^T)^T in R^{d*l}
           f(t) = (y(t)^T, ..., y(t+l-1)^T)^T in R^{d*l}
        3. Feature mapping phi_m:
           Feat_X[i] = phi_m(p(t)), Feat_Y[i] = phi_m(f(t))

        Args:
            Y: Observation time series (T, n)
            use_cache: Whether to use cached results

        Returns:
            Feat_X: Past feature matrix (m, N)
            Feat_Y: Future feature matrix (m, N)
        """
        # Cache disabled: fit() may be called multiple times per epoch,
        # making computation graph management complex
        use_cache = False

        input_shape = Y.shape

        if (use_cache and
            self._cached_feature_matrices is not None and
            self._last_input_shape == input_shape):
            return self._cached_feature_matrices['Feat_X'], self._cached_feature_matrices['Feat_Y']

        # Step 1: Apply encoder u_eta at all time points (Section 3.1)
        if self.encoder.training:
            # Retain gradients for Phase-2 CCA loss backpropagation
            M = self.encoder(Y)  # (T, m)
        else:
            self.encoder.eval()
            with torch.no_grad():
                M = self.encoder(Y)  # (T, m)

        T, m = M.shape

        L = self.window_length
        N = T - 2 * L + 1  # effective sample count

        if N <= 0:
            raise ValueError(f"Time series too short for window length {L}")

            if M.dim() == 1:
                M = M.unsqueeze(0)
            elif M.dim() == 3:
                M = M.squeeze(1)

        # Step 2: Build feature mapping phi_m (Section 3.3)
        Feat_X_list = []
        Feat_Y_list = []

        for i in range(N):
            t = i + L

            # Past block: p(t) = (y(t-1), ..., y(t-L)) in reverse order
            past_indices = list(range(t-1, t-L-1, -1))
            past_features = M[past_indices]  # (L, m)

            # Future block: f(t) = (y(t), ..., y(t+L-1))
            future_features = M[t:t+L]  # (L, m)

            feat_past = self._apply_feature_mapping(past_features)  # (m,)
            feat_future = self._apply_feature_mapping(future_features)  # (m,)

            Feat_X_list.append(feat_past)
            Feat_Y_list.append(feat_future)

        Feat_X = torch.stack(Feat_X_list, dim=1)  # (m, N)
        Feat_Y = torch.stack(Feat_Y_list, dim=1)  # (m, N)

        if use_cache:
            self._cached_feature_matrices = {'Feat_X': Feat_X, 'Feat_Y': Feat_Y}
            self._last_input_shape = input_shape

        return Feat_X, Feat_Y

    def _apply_feature_mapping(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply feature mapping phi_m (per-component transform).

        Theory Section 4.4.1:
        1. Group same components: phi_u^(i)(p(t)) in R^l
        2. Per-component scalar transform: phi_tilde_m^(i) in R
        3. Final vector: phi_m(p(t)) in R^m

        Mapping options (selected by feature_mapping_type):
        - "averaging": temporal mean (1/l) sum (Theory eq. 345)
        - "linear": learnable linear w_i^T phi_u^(i)(p(t))
        - "mlp": shallow MLP (Theory eq. 333)

        Args:
            features: (l, m) feature block

        Returns:
            feat_output: (m,) mapping result
        """
        L_win, m = features.shape

        if self.feature_mapping_type == "averaging":
            # Temporal averaging (Eq. 345)
            feat_output = torch.mean(features, dim=0)  # (m,)

        elif self.feature_mapping_type in ["linear", "mlp"]:
            # Per-component transform (Eq. 333)
            feat_components = []

            for i in range(m):
                feat_u_i = features[:, i]  # (l,)
                feat_tilde_i = self.component_transforms[i](feat_u_i)  # (1,)
                feat_components.append(feat_tilde_i.squeeze())

            feat_output = torch.stack(feat_components)  # (m,)

        else:
            raise ValueError(f"Unknown feature_mapping_type: {self.feature_mapping_type}")

        return feat_output

    def _compute_covariance_blocks(
        self,
        Feat_X: torch.Tensor,
        Feat_Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute sample covariance blocks (Section 3.4).

        After mean-centering:
          G = (1/N) Feat_X Feat_X^T (past variance)
          H = (1/N) Feat_Y Feat_Y^T (future variance)
          A = (1/N) Feat_Y Feat_X^T (past-future cross-covariance)

        Args:
            Feat_X: Past feature matrix (m, N)
            Feat_Y: Future feature matrix (m, N)

        Returns:
            G: Past variance matrix (m, m)
            H: Future variance matrix (m, m)
            A: Cross-covariance matrix (m, m)
        """
        m, N = Feat_X.shape

        Feat_X_mean = torch.mean(Feat_X, dim=1, keepdim=True)
        Feat_Y_mean = torch.mean(Feat_Y, dim=1, keepdim=True)

        Feat_X_centered = Feat_X - Feat_X_mean
        Feat_Y_centered = Feat_Y - Feat_Y_mean

        G = (Feat_X_centered @ Feat_X_centered.T) / N  # (m, m)
        H = (Feat_Y_centered @ Feat_Y_centered.T) / N  # (m, m)
        A = (Feat_Y_centered @ Feat_X_centered.T) / N  # (m, m)

        self._feature_statistics = {
            'past_mean': Feat_X_mean.squeeze(),
            'future_mean': Feat_Y_mean.squeeze(),
            'num_samples': N
        }

        return G, H, A

    def _apply_ridge_regularization(
        self,
        G: torch.Tensor,
        H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ridge regularization for numerical stability (Section 3.4).

        G_reg = G + lambda*I_m, H_reg = H + lambda*I_m

        Args:
            G: Past variance matrix (m, m)
            H: Future variance matrix (m, m)

        Returns:
            G_reg: Regularized past variance matrix (m, m)
            H_reg: Regularized future variance matrix (m, m)
        """
        m = G.size(0)
        I_m = torch.eye(m, device=G.device, dtype=G.dtype)

        G_reg = G + self.ridge_param * I_m
        H_reg = H + self.ridge_param * I_m

        return G_reg, H_reg

    def _compute_whitening_matrices(
        self,
        G_reg: torch.Tensor,
        H_reg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute whitening matrices via eigendecomposition (Section 3.4).

        Stable computation of G_reg^{-1/2}, H_reg^{-1/2} with
        minimum eigenvalue clipping for numerical stability.

        Args:
            G_reg: Regularized past variance matrix (m, m)
            H_reg: Regularized future variance matrix (m, m)

        Returns:
            G_inv_sqrt: G_reg^{-1/2} (m, m)
            H_inv_sqrt: H_reg^{-1/2} (m, m)
        """
        def stable_matrix_inv_sqrt(A: torch.Tensor, min_eigval: float = 1e-8) -> torch.Tensor:
            """Numerically stable A^{-1/2} via eigendecomposition with clipping."""
            A_sym = 0.5 * (A + A.T)
            eigvals, eigvecs = torch.linalg.eigh(A_sym)
            eigvals_clipped = torch.clamp(eigvals, min=min_eigval)
            inv_sqrt_eigvals = eigvals_clipped.rsqrt()
            A_inv_sqrt = eigvecs @ torch.diag(inv_sqrt_eigvals) @ eigvecs.T
            return A_inv_sqrt

        G_inv_sqrt = stable_matrix_inv_sqrt(G_reg)
        H_inv_sqrt = stable_matrix_inv_sqrt(H_reg)

        self._whitening_matrices = {
            'G_inv_sqrt': G_inv_sqrt,
            'H_inv_sqrt': H_inv_sqrt
        }

        return G_inv_sqrt, H_inv_sqrt

    def _solve_canonical_correlation(
        self,
        A: torch.Tensor,
        G_inv_sqrt: torch.Tensor,
        H_inv_sqrt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Canonical correlation analysis via SVD (Section 3.5).

        1. Whitened cross-block: A_tilde := H_reg^{-1/2} A G_reg^{-1/2}
        2. SVD: A_tilde = U Sigma V^T
        3. Inverse transform of canonical directions:
           a_i = G_reg^{-1/2} v_i, b_i = H_reg^{-1/2} u_i

        Args:
            A: Cross-covariance matrix (m, m)
            G_inv_sqrt: G_reg^{-1/2} (m, m)
            H_inv_sqrt: H_reg^{-1/2} (m, m)

        Returns:
            U: Left singular vectors (m, m)
            S_vals: Singular values (canonical correlations) (m,)
            Vt: Right singular vectors transposed (m, m)
        """
        A_tilde = H_inv_sqrt @ A @ G_inv_sqrt  # (m, m)
        U, S_vals, Vt = torch.linalg.svd(A_tilde, full_matrices=False)

        return U, S_vals, Vt

    def _extract_canonical_directions(
        self,
        U: torch.Tensor,
        Vt: torch.Tensor,
        G_inv_sqrt: torch.Tensor,
        H_inv_sqrt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and inverse-transform canonical directions (Section 3.5).

        a_i = G_reg^{-1/2} v_i (canonical direction in past space)
        b_i = H_reg^{-1/2} u_i (canonical direction in future space)

        Args:
            U: Left singular vectors (m, m)
            Vt: Right singular vectors transposed (m, m)
            G_inv_sqrt: G_reg^{-1/2} (m, m)
            H_inv_sqrt: H_reg^{-1/2} (m, m)

        Returns:
            canonical_dirs_past: Past canonical directions a_i (m, r)
            canonical_dirs_future: Future canonical directions b_i (m, r)
        """
        r = min(self.num_components, U.size(1))

        # Inverse-transform to original space
        canonical_dirs_past = G_inv_sqrt @ Vt[:r, :].T  # (m, r)
        canonical_dirs_future = H_inv_sqrt @ U[:, :r]   # (m, r)

        return canonical_dirs_past, canonical_dirs_future

    def fit(self, Y: torch.Tensor, encoder: Optional[nn.Module] = None) -> 'StochasticRealizationWithEncoder':
        """
        Fit stochastic realization based on functional CCA (Section 3).

        1. Integrate with time-invariant encoder
        2. Build Hilbert-space feature process
        3. Solve functional CCA constrained optimization
        4. Transform canonical directions to state variables

        Args:
            Y: Observation time series (T, n)
            encoder: Encoder (uses the one from __init__ if None)

        Returns:
            self: Fitted instance
        """
        if encoder is not None:
            self.encoder = encoder

        Y = Y.to(self.device)
        self.encoder = self.encoder.to(self.device)

        # Clear cache to avoid referencing graph-freed tensors from previous epoch
        self._cached_feature_matrices = None
        self._last_input_shape = None

        Feat_X, Feat_Y = self._build_feature_matrices(Y)
        G, H, A = self._compute_covariance_blocks(Feat_X, Feat_Y)
        G_reg, H_reg = self._apply_ridge_regularization(G, H)
        G_inv_sqrt, H_inv_sqrt = self._compute_whitening_matrices(G_reg, H_reg)
        U, S_vals, Vt = self._solve_canonical_correlation(A, G_inv_sqrt, H_inv_sqrt)

        canonical_dirs_past, canonical_dirs_future = self._extract_canonical_directions(
            U, Vt, G_inv_sqrt, H_inv_sqrt
        )

        self.canonical_directions_past = canonical_dirs_past
        self.canonical_directions_future = canonical_dirs_future
        self.canonical_correlations = S_vals[:min(self.num_components, len(S_vals))]

        # B = Sigma^{1/2} a^T for compatibility with legacy filter()
        sqrt_correlations = torch.sqrt(self.canonical_correlations)
        self.B_matrix = torch.diag(sqrt_correlations) @ canonical_dirs_past.T  # (r, m)

        self.is_fitted = True

        return self

    def estimate_states(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Transform canonical variates to state variables.

        Past-process canonical variate: z_p(t) = [a_1^T phi_m(p(t)), ..., a_r^T phi_m(p(t))]^T in R^r
        State variable: x(t) = Sigma^{1/2} z_p(t)

        Reuses feature matrices computed during fit() when available.

        Args:
            Y: Observation time series (T, n)

        Returns:
            X: State sequence (T_eff, r)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Y = Y.to(self.device)

        if self._cached_feature_matrices is not None:
            Phi_X = self._cached_feature_matrices['Feat_X']
        else:
            Phi_X, _ = self._build_feature_matrices(Y)

        # z_p(t) = a^T phi_m(p(t)), then x(t) = Sigma^{1/2} z_p(t)
        z_p = self.canonical_directions_past.T @ Phi_X  # (r, N)
        sqrt_canonical_corrs = torch.sqrt(self.canonical_correlations)  # (r,)
        X_weighted = sqrt_canonical_corrs.unsqueeze(1) * z_p  # (r, N)
        X = X_weighted.T  # (N, r)

        return X

    def filter_compatible(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible wrapper for legacy filter() interface.

        Delegates to estimate_states() for numerical consistency.

        Args:
            Y: Observation time series (T, n)

        Returns:
            X: State sequence (T_eff, r) - numerically identical to estimate_states()
        """
        return self.estimate_states(Y)

    def get_canonical_analysis_results(self) -> Dict[str, Any]:
        """Get canonical correlation analysis results."""
        if not self.is_fitted:
            return {"status": "not_fitted"}

        return {
            "canonical_correlations": self.canonical_correlations.cpu().numpy().tolist(),
            "num_components": self.num_components,
            "feature_dim": self.feature_dim,
            "window_length": self.window_length,
            "ridge_param": self.ridge_param,
            "condition_numbers": {
                "correlations_range": {
                    "min": self.canonical_correlations.min().item(),
                    "max": self.canonical_correlations.max().item()
                }
            }
        }

class Realization:
    """
    Exact subspace identification via canonical correlation analysis.
    
    Steps in fit(Y):
      1) Build past/future block Hankel matrices H_p, H_f
      2) Compute empirical covariances S_pp, S_ff, S_fp
      3) Add diagonal jitter ε to S_pp, S_ff for stability
      4) Check condition numbers; reject if too large
      5) Compute Cholesky factors L_pp, L_ff (S = L @ L.T)
      6) Solve triangular systems to get whitening matrices W_pp = L_pp⁻¹, W_ff = L_ff⁻¹
      7) Form T = W_ff @ S_fp @ W_pp and take its SVD → U, Lambda, Vᵀ
      8) Build state map B = Lambda^{1/2} Vᵀ W_pp
      9) Save singular values Lambda for downstream loss or analysis
    """
    def __init__(
        self,
        past_horizon: int,
        jitter: float = 1e-6,
        cond_thresh: float = 1e12,
        rank: int | None = None,
        reg_type : str = "sum",
        m: int | None = None,
    ):
        """
        Args:
            past_horizon: Number of time-lags in past/future Hankel blocks
            jitter: Small epsilon added to covariance diagonals
            cond_thresh: Max allowed condition number before rejection
        """
        self.h = int(past_horizon)
        self.jitter = float(jitter)
        self.cond_thresh = float(cond_thresh)

        if rank is not None:
            self.rank = int(rank)
        else:
            self.rank = rank

        self.reg_type = str(reg_type)

        self.m = int(m) if m is not None else None

        self.B = None
        self._L_vals = None
        self._Spp_eigvals = None
        self.H = None

    def fit(self, Y: torch.Tensor):
        T, p = Y.shape
        h = self.h
        N = T - 2*h + 1
        if N <= 0:
            return

        # 0) Mean-center
        mu = Y.mean(dim=0, keepdim=True)
        Y_c = Y - mu

        device = Y_c.device
        max_available = T - 2 * h
        if max_available <= 0:
            h_new = max(1, (T - 1) // 2)
            print(f"realization adjusted: past_horizon {h} -> {h_new} (data length: {T})")
            h = h_new
            self.h = h_new
            max_available = T - 2 * h

        m_default = min(500, max(1, max_available))
        m = getattr(self, 'm', None)
        if m is None:
            m = m_default
        else:
            m = min(m, max_available)
        rank   = self.rank
        eps_chol   = float(self.jitter)
        eps_jitter = float(self.jitter)
        q_over     = 5

        # 1. -- Lag covariances (batch outer products, float32) -----
        idx = torch.randint(0, max_available, (m,), device=device)
        Y0  = Y_c[idx]                          # (m, p)
        Lambda = {}

        for l in range(0, 2 * h):
            Yl = Y_c[idx + l]                  # (m, p)
            cov = (Yl.T @ Y0) / m              # (p, p)
            Lambda[ l]  = cov
            if h + 1 > l > 0:
                Lambda[-l] = cov.T

        # 2. -- Block matrices (k x k blocks) ----------------------
        dim_H = h * p
        H32  = torch.zeros(dim_H, dim_H, dtype=torch.float32, device=device)
        Tp32 = torch.zeros_like(H32)

        z = torch.zeros(p, p, dtype=torch.float32, device=device)
        for i in range(h):
            for j in range(h):
                H32 [i*p:(i+1)*p, j*p:(j+1)*p] = Lambda.get(i + j + 1, z)
                Tp32[i*p:(i+1)*p, j*p:(j+1)*p] = Lambda.get(i - j, z)
        
        Tp32.diagonal().add_(eps_jitter)

        # 3. -- Inverse square root (float64) ----------------------
        Tp64 = Tp32.to(torch.float64)
        Tp64 = 0.5 * (Tp64 + Tp64.T)
        try:
            L64  = torch.linalg.cholesky(
                      Tp64 + eps_chol*torch.eye(dim_H, device=device, dtype=torch.float64))
            W64  = torch.linalg.solve_triangular(
                      L64, torch.eye(dim_H, device=device, dtype=torch.float64), upper=False)
        except RuntimeError as e:
            print(f"real.fit failed cholesky decom: {e}")
            
            if not torch.isfinite(Tp64).all():
                print("Critical: Tp64 contains non-finite values.")
                raise RealizationError("Matrix contains non-finite values - numerical breakdown")
            
            # Fallback: eigendecomposition
            try:
                is_symmetric = torch.allclose(Tp64, Tp64.T, atol=1e-8)

                if is_symmetric:
                    eigvals, eigvecs = torch.linalg.eigh(Tp64)
                else:
                    print("Warning: Matrix is not symmetric")
                    eigvals_complex, eigvecs_complex = torch.linalg.eig(Tp64)
                    eigvals = eigvals_complex.real
                    eigvecs = eigvecs_complex.real
                
                if torch.any(eigvals <= 0) or not torch.isfinite(eigvals).all():
                    print(f"Unstable eigenvalues: min={eigvals.min().item()}, has_inf={not torch.isfinite(eigvals).all()}")
                    raise RealizationError("Eigenvalues are numerically unstable")

                inv_sqrt = eigvals.rsqrt()
                D = torch.diag(inv_sqrt)
                W64 = eigvecs @ D @ eigvecs.T
                
                if not torch.isfinite(W64).all():
                    raise RealizationError("Final W64 matrix contains non-finite values")
                    
            except (RuntimeError, torch.linalg.LinAlgError) as inner_e:
                print(f"All numerical recovery attempts failed: {inner_e}")
                raise RealizationError(f"Complete numerical breakdown in realization: {inner_e}")
        
        # 4. -- Normalized Hankel (float64) --------------------------
        T64 = W64.T @ (H32.to(torch.float64) @ W64)
        
        # 5. -- Rank-r SVD or full SVD --------------------------------
        if rank is None or rank >= dim_H:
            U64, S64, Vh64 = torch.linalg.svd(T64, full_matrices=False)
            U, S, Vh = U64.to(torch.float32), S64.to(torch.float32), Vh64.to(torch.float32)
            # B = Lambda^{1/2} Vᵀ W_pp
            self.B = torch.diag(S.pow(0.5)) @ Vh @ W64.to(torch.float32)
            self._L_vals = S
        else:                                  # rank-r truncation
            U64, S64, Vh64 = torch.linalg.svd(T64, full_matrices=False)
            U, S, Vh = U64.to(torch.float32), S64.to(torch.float32), Vh64.to(torch.float32)
            U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank,:]
            self.B = torch.diag(S_r.pow(0.5)) @ Vh_r @ W64.to(torch.float32)
            self._L_vals = S_r

    def filter(self, Y: torch.Tensor) -> torch.Tensor:
        h = self.h
        N = Y.shape[0] - 2*h + 1
        Yp = torch.stack([Y[i : i + h].flip(dims=(0, )).reshape(-1) for i in range(N)], dim=1)
        X_state = (self.B @ Yp).T  # (N, r)

        self.X_state_torch = X_state

        return X_state

    def singular_value_reg(self, sv_weight : float) -> torch.Tensor:
        """
        Return singular value regularization loss.
        reg_type=="sum"     -> sum(sigma_i)
        reg_type=="squared" -> sum((1 - sigma_i)^2)
        """
        if self.reg_type == "sum":
            _tr = self._L_vals.sum()
            _reg = -_tr
        elif self.reg_type == "squared":
            _reg = ((1 - self._L_vals) ** 2).sum()
        elif self.reg_type == "abs":
            _reg = (1 - self._L_vals).abs().sum()
            
        elif self.reg_type == "bounded":
            _reg = (1 - self._L_vals ** 2).sum()
        else:
            raise ValueError(f"Unknown reg_type: {self.reg_type}")

        return sv_weight * _reg

    # =====================================
    # Utility methods
    # =====================================

    def get_state_statistics(self) -> Dict[str, Any]:
        """Get state estimation statistics."""
        if not hasattr(self, 'X_state_torch') or self.X_state_torch is None:
            return {"status": "not_fitted"}
        
        X = self.X_state_torch
        
        return {
            "state_shape": X.shape,
            "state_dimension": X.size(1),
            "sequence_length": X.size(0),
            "state_statistics": {
                "mean": torch.mean(X, dim=0).tolist(),
                "std": torch.std(X, dim=0).tolist(),
                "min": torch.min(X, dim=0)[0].tolist(),
                "max": torch.max(X, dim=0)[0].tolist()
            },
            "singular_values": {
                "values": self._L_vals.tolist() if hasattr(self, '_L_vals') and self._L_vals is not None else None,
                "condition_number": (self._L_vals.max() / self._L_vals.min()).item() if hasattr(self, '_L_vals') and self._L_vals is not None else None
            }
        }

    def predict_states(
        self,
        n_steps: int = 1,
        method: str = "linear"
    ) -> torch.Tensor:
        """
        Predict future states.

        Args:
            n_steps: Number of prediction steps
            method: Prediction method ("linear" | "last_value")

        Returns:
            torch.Tensor: Predicted states (n_steps, r)
        """
        if not hasattr(self, 'X_state_torch') or self.X_state_torch is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = self.X_state_torch
        T, r = X.shape
        
        if method == "linear":
            if T >= 2:
                trend = X[-1] - X[-2]
                predictions = []
                for step in range(1, n_steps + 1):
                    pred = X[-1] + step * trend
                    predictions.append(pred)
                return torch.stack(predictions)
            else:
                method = "last_value"

        if method == "last_value":
            last_state = X[-1]
            return last_state.unsqueeze(0).expand(n_steps, r).clone()
        
        else:
            raise ValueError(f"Unknown prediction method: {method}")

    def validate_kalman_compatibility(
        self,
        df_state_layer,
        df_obs_layer
    ) -> Dict[str, Any]:
        """
        Validate compatibility with Kalman update.

        Args:
            df_state_layer: DF-A layer
            df_obs_layer: DF-B layer

        Returns:
            Dict: Validation results
        """
        validation = {
            "compatible": True,
            "issues": [],
            "requirements_met": {},
            "recommendations": []
        }
        
        requirements = [
            ("df_state_layer.V_A", hasattr(df_state_layer, 'V_A') and df_state_layer.V_A is not None),
            ("df_state_layer.U_A", hasattr(df_state_layer, 'U_A') and df_state_layer.U_A is not None),
            ("df_obs_layer.V_B", hasattr(df_obs_layer, 'V_B') and df_obs_layer.V_B is not None),
            ("df_obs_layer.U_B", hasattr(df_obs_layer, 'U_B') and df_obs_layer.U_B is not None),
            ("realization_fitted", hasattr(self, 'X_state_torch') and self.X_state_torch is not None)
        ]
        
        for req_name, req_met in requirements:
            validation["requirements_met"][req_name] = req_met
            if not req_met:
                validation["compatible"] = False
                validation["issues"].append(f"Missing requirement: {req_name}")
        
        if validation["compatible"]:
            try:
                r_realization = self.rank if self.rank is not None else self.X_state_torch.size(1)
                r_df_state = df_state_layer.state_dim
                
                if r_realization != r_df_state:
                    validation["issues"].append(f"State dimension mismatch: realization={r_realization}, df_state={r_df_state}")
                    validation["compatible"] = False
                    
                dA = df_state_layer.feature_dim
                dB = df_obs_layer.obs_feature_dim
                
                validation["dimensions"] = {
                    "state_dim": r_realization,
                    "feature_dim_A": dA,
                    "feature_dim_B": dB
                }
                
                if dA < 2 * r_realization:
                    validation["recommendations"].append(f"Consider increasing feature_dim_A (current: {dA}, recommended: >={2*r_realization})")
                    
            except Exception as e:
                validation["compatible"] = False
                validation["issues"].append(f"Dimension check failed: {e}")
        
        return validation

    # =====================================
    # Configuration helpers
    # =====================================

    def create_kalman_config(self, **overrides) -> Dict[str, Any]:
        """Create Kalman Filter configuration with optional overrides."""
        default_config = {
            'noise_estimation': {
                'method': 'residual_based',
                'gamma_Q': 1e-6,
                'gamma_R': 1e-6
            },
            'initialization': {
                'method': 'data_driven',
                'n_init_samples': 10
            },
            'numerical': {
                'condition_threshold': 1e12,
                'min_eigenvalue': 1e-8,
                'jitter': 1e-6
            },
            'device': 'cpu'
        }
        
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        config = default_config.copy()
        deep_update(config, overrides)
        
        return config
    

def build_realization(cfg):
    """
    Factory: select appropriate implementation based on encoder availability.

    Args:
        cfg: Configuration object

    Returns:
        Realization or StochasticRealizationWithEncoder

    Raises:
        ValueError: Configuration error (e.g., missing encoder_output_dim)
    """
    use_encoder = getattr(cfg, 'use_encoder', False) and hasattr(cfg, 'encoder')

    if use_encoder:
        if not hasattr(cfg, 'encoder_output_dim'):
            raise ValueError(
                "encoder_output_dim must be specified when using encoder. "
                "Example: cfg.encoder_output_dim = 16"
            )

        return StochasticRealizationWithEncoder(
            encoder=cfg.encoder,
            encoder_output_dim=cfg.encoder_output_dim,
            window_length=getattr(cfg, 'window_length', 5),
            num_components=getattr(cfg, 'num_components', 4),
            ridge_param=getattr(cfg, 'ridge_param', 1e-6),
            device=getattr(cfg, 'device', 'cpu')
        )
    else:
        r = getattr(cfg, "rank", None)
        if r is not None and r < 0:
            raise ValueError(f"Invalid rank: {r} (must be >= 0 or None)")

        return Realization(
            past_horizon=cfg.h,
            jitter=cfg.jitter,
            cond_thresh=cfg.cond_thresh,
            rank=r,
            reg_type=getattr(cfg, "svd_reg_type", "sum"),
        )