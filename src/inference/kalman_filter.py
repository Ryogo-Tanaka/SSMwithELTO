# src/inference/kalman_filter.py
"""
Algorithm 1: Operator-based Kalman filtering

学習済みV_A, V_Bを用いた逐次状態推定。Section 2.7.2実装。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings


class OperatorBasedKalmanFilter:
    """
    Algorithm 1実装: 作用素ベースKalman更新

    特徴空間で更新後、元状態空間へ復元。
    """
    
    def __init__(
        self,
        V_A: torch.Tensor,           # 状態転送作用素 (dA, dA)
        V_B: torch.Tensor,           # 観測転送作用素 (dB, dA)
        U_A: torch.Tensor,           # 状態読み出し行列 (dA, r)
        U_B: torch.Tensor,           # 観測読み出し行列 (dB, m)
        Q: torch.Tensor,             # 状態ノイズ共分散 (dA, dA)
        R: Union[torch.Tensor, float], # 観測ノイズ分散
        encoder: nn.Module,          # エンコーダ（frozen）
        df_obs_layer: nn.Module = None,  # DF-B層（学習済みψ_ω含む）
        device: str = 'cpu'
    ):
        """
        Args:
            V_A, V_B, U_A, U_B: 学習済み演算子
            Q, R: ノイズ共分散
            encoder: エンコーダ（frozen）
            df_obs_layer: DF-B層（学習済みψ_ω含む）
            device: 計算デバイス
        """
        self.device = torch.device(device)
        
        # 学習済みパラメータ
        self.V_A = V_A.to(self.device)
        self.V_B = V_B.to(self.device)
        self.U_A = U_A.to(self.device)
        self.U_B = U_B.to(self.device)
        
        # ノイズ共分散
        self.Q = Q.to(self.device)
        if isinstance(R, (int, float)):
            self.R = torch.tensor(R, dtype=torch.float32, device=self.device)
        else:
            self.R = R.to(self.device)
            
        # エンコーダ
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # DF-B層
        self.df_obs_layer = df_obs_layer
        if self.df_obs_layer is not None:
            self.df_obs_layer.eval()
            
        # 次元
        self.dA = int(V_A.size(0))
        self.dB = int(V_B.size(0))
        self.r = int(U_A.size(1))

        # 内部状態
        self.mu: Optional[torch.Tensor] = None      # µ ∈ R^dA
        self.Sigma: Optional[torch.Tensor] = None   # Σ ∈ R^(dA×dA)
        self.is_initialized = False
        
        # 数値安定性パラメータ
        self.min_eigenvalue = 1e-6
        self.condition_threshold = 1e12
        self.jitter = 1e-5

    def initialize_state(
        self,
        initial_observations: torch.Tensor,
        method: str = "data_driven"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        µ₀, Σ₀初期化（式47-48）

        Args:
            initial_observations: 初期観測 (N0, n)
            method: "data_driven" | "zero"
        """
        N0 = initial_observations.size(0)
        
        if method == "data_driven":
            with torch.no_grad():
                if initial_observations.dim() == 2:
                    m_initial = self.encoder(initial_observations.unsqueeze(0)).squeeze()
                else:
                    m_initial = self.encoder(initial_observations)

                phi_samples = self._approximate_feature_mapping(m_initial)  # (N0, dA)

                # 式47-48
                mu_0 = torch.mean(phi_samples, dim=0)
                centered = phi_samples - mu_0.unsqueeze(0)
                Sigma_0 = (centered.T @ centered) / (N0 - 1)
                
        elif method == "zero":
            mu_0 = torch.zeros(self.dA, device=self.device)
            Sigma_0 = torch.eye(self.dA, device=self.device)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        Sigma_0 = self._regularize_covariance(Sigma_0)
        self.mu = mu_0.clone()
        self.Sigma = Sigma_0.clone()
        self.is_initialized = True

        return mu_0, Sigma_0

    def _approximate_feature_mapping(self, m_series: torch.Tensor) -> torch.Tensor:
        """
        多変量特徴からdA次元特徴への近似写像

        注意: 簡易版。実際は学習済みφ_θ使用。

        Args:
            m_series: 多変量特徴 (T, m) or (T,)
        """
        T = m_series.size(0)
        if m_series.dim() == 1:
            m_series = m_series.unsqueeze(1)
        m = m_series.size(1)

        # 遅延埋め込み
        features = []
        max_delay = min(5, T)
        for t in range(T):
            delayed = []
            for d in range(max_delay):
                if t - d >= 0:
                    delayed.append(m_series[t - d])
                else:
                    delayed.append(torch.zeros_like(m_series[0]))
            delayed_features = torch.cat(delayed, dim=0)
            features.append(delayed_features)
        feature_matrix = torch.stack(features)

        # dA次元に調整
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
        時間更新: µ⁻ₜ = V_A µ⁺ₜ₋₁, Σ⁻ₜ = V_A Σ⁺ₜ₋₁ V_A^T + Q
        """
        mu_minus = self.V_A @ mu_prev
        Sigma_minus = self.V_A @ Sigma_prev @ self.V_A.T + self.Q
        Sigma_minus = self._regularize_covariance(Sigma_minus)
        return mu_minus, Sigma_minus

    def update_step(
        self,
        mu_minus: torch.Tensor,
        Sigma_minus: torch.Tensor,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        観測更新: 式51-55のKalman innovation update

        Args:
            mu_minus, Sigma_minus: 予測状態 (dA,), (dA, dA)
            observation: 現在観測 (n,) or (1, n)
        """
        # エンコード: m_t ← u_η(y_t)
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0).unsqueeze(0)
            elif observation.dim() == 2:
                observation = observation.unsqueeze(1)
            m_t = self.encoder(observation).squeeze()
            psi_omega_m_t = self._generate_obs_features_from_multivariate(m_t)

        # 式51: 観測予測 ハット{ψ}^{-}_t = V_B μ^-_t
        psi_pred = self.V_B @ mu_minus

        # 式52: イノベーション共分散 S_t = V_B Σ^-_t V_B^T + R
        S = self.V_B @ Sigma_minus @ self.V_B.T
        if isinstance(self.R, torch.Tensor) and self.R.dim() == 2:
            S = S + self.R
        elif isinstance(self.R, (int, float)) or (isinstance(self.R, torch.Tensor) and self.R.dim() == 0):
            R_scalar = float(self.R) if isinstance(self.R, torch.Tensor) else self.R
            S = S + R_scalar * torch.eye(self.dB, device=S.device, dtype=S.dtype)
        else:
            raise ValueError(f"Unsupported R type: {type(self.R)}")

        # 正定値性確保
        try:
            eigenvalues = torch.linalg.eigvals(S).real
            if torch.any(eigenvalues <= 0):
                warnings.warn("Non-positive definite S")
                S = S + self.jitter * torch.eye(S.size(0), device=self.device)
        except Exception as e:
            warnings.warn(f"S eigenvalue check failed: {e}")
            S = S + self.jitter * torch.eye(S.size(0), device=self.device)

        # 式53: Kalmanゲイン K_t = Σ^-_t V_B^T S_t^{-1}
        try:
            S_inv = torch.linalg.inv(S)
            K = Sigma_minus @ self.V_B.T @ S_inv
        except Exception as e:
            warnings.warn(f"S inversion failed: {e}")
            S_pinv = torch.linalg.pinv(S)
            K = Sigma_minus @ self.V_B.T @ S_pinv

        # 式54: μ^+_t = μ^-_t + K_t(ψ_ω(m_t) - ハット{ψ}^{-}_t)
        innovation = psi_omega_m_t - psi_pred
        mu_plus = mu_minus + K @ innovation

        # 式55: Σ^+_t = Σ^-_t - K_t V_B Σ^-_t
        Sigma_plus = Sigma_minus - K @ self.V_B @ Sigma_minus
        Sigma_plus = self._regularize_covariance(Sigma_plus)

        likelihood = 0.0  # 正規分布非仮定のためダミー
        return mu_plus, Sigma_plus, likelihood

    def filter_step(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        1ステップフィルタリング（逐次処理用）

        Args:
            observation: 現在観測 (n,)
            state: 外部状態 (mu, Sigma) or None
        """
        if state is not None:
            mu_prev, Sigma_prev = state
        else:
            if not self.is_initialized:
                raise RuntimeError("Filter not initialized")
            mu_prev, Sigma_prev = self.mu, self.Sigma

        mu_minus, Sigma_minus = self.predict_step(mu_prev, Sigma_prev)
        mu_plus, Sigma_plus, likelihood = self.update_step(mu_minus, Sigma_minus, observation)
        x_hat_t, Sigma_x_t = self._recover_original_state(mu_plus, Sigma_plus)

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
        バッチフィルタリング

        Args:
            observations: 観測系列 (T, n)
            initial_state: 初期状態 or None
            return_likelihood: 尤度返却フラグ
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
        元状態空間復元: x̂_t ← U_A^T µ, Σ_t^(x) ← U_A^T Σ U_A
        """
        x_hat = self.U_A.T @ mu
        Sigma_x = self.U_A.T @ Sigma @ self.U_A
        Sigma_x = self._regularize_covariance(Sigma_x)
        return x_hat, Sigma_x

    def _regularize_covariance(self, cov_matrix: torch.Tensor) -> torch.Tensor:
        """
        共分散正則化: 正定値性確保
        """
        cov_matrix = (cov_matrix + cov_matrix.T) / 2

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        except:
            cov_matrix += self.jitter * torch.eye(cov_matrix.size(0), device=self.device)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        eigenvalues = torch.clamp(eigenvalues, min=self.min_eigenvalue)
        cov_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        return cov_matrix

    def _compute_likelihood(self, innovation: torch.Tensor, S: torch.Tensor) -> float:
        """
        観測尤度計算（廃止: 正規分布非仮定のため）
        """
        _ = innovation
        _ = S
        return 0.0

    def get_current_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """現在状態取得"""
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized")
        return self._recover_original_state(self.mu, self.Sigma)

    def reset_state(self):
        """内部状態リセット"""
        self.mu = None
        self.Sigma = None
        self.is_initialized = False

    def check_numerical_stability(self) -> Dict[str, Any]:
        """数値安定性診断"""
        if not self.is_initialized:
            return {"status": "not_initialized"}

        try:
            cond_Q = torch.linalg.cond(self.Q).item()
            cond_Sigma = torch.linalg.cond(self.Sigma).item()
            eigenvals_Q = torch.linalg.eigvals(self.Q).real
            eigenvals_Sigma = torch.linalg.eigvals(self.Sigma).real

            return {
                "status": "ok",
                "condition_numbers": {"Q": cond_Q, "Sigma": cond_Sigma},
                "min_eigenvalues": {"Q": eigenvals_Q.min().item(), "Sigma": eigenvals_Sigma.min().item()},
                "numerical_stable": (
                    cond_Q < self.condition_threshold and cond_Sigma < self.condition_threshold and
                    eigenvals_Q.min() > 0 and eigenvals_Sigma.min() > 0
                )
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _generate_obs_features_from_multivariate(self, m_t: torch.Tensor) -> torch.Tensor:
        """
        多変量特徴から観測特徴生成: ψ_ω(m_t) ∈ ℝ^{d_B}

        式54で使用。学習済みψ_ωネットワーク必須。

        Args:
            m_t: 多変量特徴 (m,) or (1, m)
        """
        with torch.no_grad():
            if self.df_obs_layer is None or not hasattr(self.df_obs_layer, 'psi_omega'):
                raise RuntimeError("DF-B layer with psi_omega required")

            if m_t.dim() == 1:
                m_t_input = m_t
            elif m_t.dim() == 2 and m_t.size(0) == 1:
                m_t_input = m_t.squeeze(0)
            else:
                raise ValueError(f"m_t must be (m,) or (1, m), got {m_t.shape}")

            psi_omega_m_t = self.df_obs_layer.psi_omega(m_t_input)
            while psi_omega_m_t.dim() > 1 and psi_omega_m_t.size(0) == 1:
                psi_omega_m_t = psi_omega_m_t.squeeze(0)

            if psi_omega_m_t.dim() != 1:
                raise RuntimeError(f"psi_omega output must be 1D, got {psi_omega_m_t.shape}")

            return psi_omega_m_t