# src/ssm/observation.py

# src/ssm/observation.py

import torch
from torch.linalg import cholesky, solve_triangular, eigvalsh
from typing import Optional

def _compute_kernel_matrix(
    X_r: torch.Tensor,
    Y_r: torch.Tensor,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Cross-kernel matrix K: X[M,r], Y[N,r] -> K[M,N].
    Supports 'linear' and 'rbf' kernels.
    """
    X = X_r.to(dtype)
    Y = Y_r.to(dtype)
    if kernel == "linear":
        return X @ Y.T
    elif kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / X.size(1)
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        sq_dists = X_norm - 2 * (X @ Y.T) + Y_norm.T
        return torch.exp(-gamma * sq_dists)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")


class RealizationError(Exception):
    """Raised when CME observation fails."""
    pass


class CMEObservation:
    """
    Conditional Mean Embedding (CME) decoder: ŷ_t = Y_feats[1:] · (Φ_f (K_pp + nλI)^{-1} Φ_p^T φ(x_{t-1}))
    """

    def __init__(
        self,
        past_horizon: int,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        reg_lambda: float = 1e-3,
    ):
        self.h = past_horizon
        self.kernel = kernel
        self.gamma = gamma
        self.reg_lambda = reg_lambda

        # Placeholders
        self.L: Optional[torch.Tensor] = None
        self.X: Optional[torch.Tensor] = None
        self.Y_mat: Optional[torch.Tensor] = None
        self.n_states: Optional[int] = None

        self.gram_X : Optional[torch.Tensor] = None

        # debug
        self.K_past = None
        self.eigvals = None


    def fit(self, X_states: torch.Tensor, Y_feats: torch.Tensor):
        """
        Precompute factors for one-step decoding.

        Args:
          X_states: [N, r] state sequence (time: h, ..., h+N-1)
          Y_feats: [N, p] encoder features (time: 0, ..., T)
        """
        self.X = X_states
        self.n_states = int(X_states.size(0))
        n_feats = int(Y_feats.size(0))

        self.Y_mat = Y_feats[self.h : n_feats-self.h+1].T
        K_X = _compute_kernel_matrix(
            X_states, X_states,
            kernel=self.kernel, gamma=self.gamma, dtype = torch.float64
        )
        K_X = 0.5 * (K_X + K_X.T)  # Symmetrize
        self.gram_X = K_X

        return self

    def decode(self):
        K_past = self.gram_X[:-1, :-1]
        I_past = torch.eye(self.n_states-1, device=K_past.device).to(K_past.dtype)
        K_past_reg = K_past + self.reg_lambda * (self.n_states - 1) * I_past

        self.eigvals = torch.linalg.eigvalsh(K_past_reg)

        C_past = cholesky(K_past_reg)
        inv_K_past = torch.cholesky_inverse(C_past)
        self.gram_X = self.gram_X.float(); inv_K_past = inv_K_past.float(); K_past = K_past.float()

        k_x_pred_mat = self.gram_X[:-1, 1:] @ inv_K_past @ K_past

        Y_pred = self.Y_mat @ inv_K_past @ k_x_pred_mat

        return Y_pred.T  # shape: (T̃, p)

    def decode_pred(self, forget_flag=True):
        K_past = self.gram_X[:-1, :-1]
        I_past = torch.eye(self.n_states-1, device=K_past.device).to(K_past.dtype)
        K_past_reg = K_past + self.reg_lambda * (self.n_states - 1) * I_past

        C_past = cholesky(K_past_reg)
        inv_K_past = torch.cholesky_inverse(C_past)
        inv_K_past = inv_K_past.float()

        self.gram_X = self.gram_X.float(); inv_K_past = inv_K_past.float(); K_past = K_past.float()

        k_x_pred = self.gram_X[:-1, 1:] @ inv_K_past @ self.gram_X[:-1, -1]

        Y_pred = self.Y_mat @ inv_K_past @ k_x_pred

        # Update gram matrix
        _last_sc_kx = self.gram_X[-1, 1:] @ inv_K_past @ self.gram_X[:-1, -1]
        _last_kx = _last_sc_kx.unsqueeze(0)
        _kx = torch.cat([k_x_pred, _last_kx], dim=0)

        _sc_kernel_new = self.gram_X[:-1, -1] @ inv_K_past @ self.gram_X[1:, 1:] @ inv_K_past @ self.gram_X[1:, -1]

        # Construct gram matrix and Y_mat
        n_gram = self.gram_X.size(0)
        _new_gram = self.gram_X.new_empty((n_gram+1, n_gram+1))
        _new_gram[:n_gram, :n_gram] = self.gram_X
        _new_gram[:n_gram, -1] = _kx
        _new_gram[-1, :n_gram] = _kx
        _new_gram[-1, -1] = _sc_kernel_new

        n_row, n_col = self.Y_mat.size()
        _new_Y = self.Y_mat.new_empty((n_row, n_col+1))
        _new_Y[:, :n_col] = self.Y_mat
        _new_Y[:, -1] = Y_pred

        if not forget_flag:
            self.gram_X = _new_gram
            self.Y_mat = _new_Y
        else:
            self.gram_X = _new_gram[1:, 1:]
            self.Y_mat = _new_Y[:, 1:]

        return Y_pred  # 1D Tensor
