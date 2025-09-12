# src/inference/kalman_filter.py
"""
Algorithm 1: Operator-based Kalman filtering with Galerkin-projected conditional covariance operators

資料Section 2.7.2の完全実装。学習済み転送作用素VA, VBを用いた逐次状態推定。

Input: 
- 学習済み演算子: VA ∈ R^(dA×dA), VB ∈ R^(dB×dA), uB ∈ R^dB, UA^T ∈ R^(r×dA)
- ノイズ共分散: Q ∈ R^(dA×dA), R ∈ R≥0
- エンコーダ: uη : R^n→R
- 初期値: (µ0, Σ0)

Output:
- フィルタリング状態: x̂t ∈ R^r
- 共分散: Σt^(x) ∈ R^(r×r) at each t
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings


class OperatorBasedKalmanFilter:
    """
    Algorithm 1の完全実装
    
    作用素ベースKalman更新による逐次状態推定。
    特徴空間でのKalman更新後、元状態空間へ復元。
    """
    
    def __init__(
        self,
        V_A: torch.Tensor,           # 状態転送作用素 (dA, dA)
        V_B: torch.Tensor,           # 観測転送作用素 (dB, dA)
        U_A: torch.Tensor,           # 状態読み出し行列 (dA, r)
        u_B: torch.Tensor,           # 観測読み出しベクトル (dB,)
        Q: torch.Tensor,             # 状態ノイズ共分散 (dA, dA)
        R: Union[torch.Tensor, float], # 観測ノイズ分散（スカラーまたは行列）
        encoder: nn.Module,          # エンコーダネットワーク（frozen）
        device: str = 'cpu'
    ):
        """
        Algorithm 1の初期化
        
        Args:
            V_A: 学習済み状態転送作用素 (dA, dA)
            V_B: 学習済み観測転送作用素 (dB, dA)  
            U_A: 学習済み状態読み出し行列 (dA, r)
            u_B: 学習済み観測読み出しベクトル (dB,)
            Q: 状態ノイズ共分散 (dA, dA)
            R: 観測ノイズ分散（スカラーまたは行列）
            encoder: エンコーダネットワーク（frozen）
            device: 計算デバイス
        """
        self.device = torch.device(device)
        
        # 学習済みパラメータ
        self.V_A = V_A.to(self.device)
        self.V_B = V_B.to(self.device)
        self.U_A = U_A.to(self.device)
        self.u_B = u_B.to(self.device)
        
        # ノイズ共分散
        self.Q = Q.to(self.device)
        if isinstance(R, (int, float)):
            self.R = torch.tensor(R, dtype=torch.float32, device=self.device)
        else:
            self.R = R.to(self.device)
            
        # エンコーダ（推論時は勾配なし）
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 次元
        self.dA = int(V_A.size(0))  # 特徴空間状態次元
        self.dB = int(V_B.size(0))  # 特徴空間観測次元
        self.r = int(U_A.size(1))   # 元状態次元
        
        # 前計算: H ← V_B^T u_B ∈ R^dA
        self.H = self.V_B.T @ self.u_B  # (dA,)
        
        # 内部状態（逐次処理用）
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
        µ₀, Σ₀の初期化 (式47-48)
        
        Args:
            initial_observations: 初期観測サンプル (N0, n)
            method: 初期化方法 ("data_driven" | "zero")
            
        Returns:
            µ₀: 初期状態平均 (dA,)
            Σ₀: 初期状態共分散 (dA, dA)
        """
        N0 = initial_observations.size(0)
        
        if method == "data_driven":
            # 式47-48: データ駆動初期化
            with torch.no_grad():
                # エンコード
                if initial_observations.dim() == 2:
                    # (N0, n) → (N0,) スカラー特徴量
                    m_initial = self.encoder(initial_observations.unsqueeze(0)).squeeze()
                else:
                    m_initial = self.encoder(initial_observations)
                    
                # ψ_ω による特徴変換（簡易版：線形変換で近似）
                # 注意: 実際の実装では DF-B の ψ_ω を使用
                phi_samples = self._approximate_feature_mapping(m_initial)  # (N0, dA)
                
                # µ₀ = (1/N₀) Σ_{i=1}^{N₀} φ_θ(x₀^{(i)})
                mu_0 = torch.mean(phi_samples, dim=0)  # (dA,)
                
                # Σ₀ = (1/(N₀-1)) Σ_{i=1}^{N₀} (φ_θ(x₀^{(i)}) - µ₀)(φ_θ(x₀^{(i)}) - µ₀)^T
                centered = phi_samples - mu_0.unsqueeze(0)  # (N0, dA)
                Sigma_0 = (centered.T @ centered) / (N0 - 1)  # (dA, dA)
                
        elif method == "zero":
            # ゼロ初期化
            mu_0 = torch.zeros(self.dA, device=self.device)
            Sigma_0 = torch.eye(self.dA, device=self.device)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
            
        # 正定値性確保
        Sigma_0 = self._regularize_covariance(Sigma_0)
        
        # 内部状態設定
        self.mu = mu_0.clone()
        self.Sigma = Sigma_0.clone()
        self.is_initialized = True
        
        return mu_0, Sigma_0

    def _approximate_feature_mapping(self, m_series: torch.Tensor) -> torch.Tensor:
        """
        スカラー特徴量からdA次元特徴への近似写像
        
        注意: この関数は簡易版。実際の実装では学習済みDF-A/DF-Bの
        特徴写像 φ_θ を使用する必要がある。
        
        Args:
            m_series: スカラー特徴系列 (T,)
            
        Returns:
            torch.Tensor: 近似特徴 (T, dA)
        """
        T = m_series.size(0)
        
        # 簡易的な特徴拡張（実際はφ_θを使用）
        # 例: 遅延埋め込み + ランダムフーリエ特徴
        features = []
        
        # 遅延埋め込み
        max_delay = min(5, T)
        for t in range(T):
            delayed = []
            for d in range(max_delay):
                if t - d >= 0:
                    delayed.append(m_series[t - d])
                else:
                    delayed.append(torch.zeros_like(m_series[0]))
            features.append(torch.stack(delayed))
            
        feature_matrix = torch.stack(features)  # (T, max_delay)
        
        # 線形変換でdA次元に拡張
        if feature_matrix.size(1) < self.dA:
            # パディング
            padding = torch.randn(T, self.dA - feature_matrix.size(1), device=self.device) * 0.1
            feature_matrix = torch.cat([feature_matrix, padding], dim=1)
        elif feature_matrix.size(1) > self.dA:
            # 切り取り
            feature_matrix = feature_matrix[:, :self.dA]
            
        return feature_matrix

    def predict_step(
        self, 
        mu_prev: torch.Tensor, 
        Sigma_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        時間更新 (Time update)
        
        µ⁻ₜ = V_A µ⁺ₜ₋₁
        Σ⁻ₜ = V_A Σ⁺ₜ₋₁ V_A^T + Q
        
        Args:
            mu_prev: 前時刻状態平均 µ⁺ₜ₋₁ (dA,)
            Sigma_prev: 前時刻状態共分散 Σ⁺ₜ₋₁ (dA, dA)
            
        Returns:
            mu_minus: 予測状態平均 µ⁻ₜ (dA,)
            Sigma_minus: 予測状態共分散 Σ⁻ₜ (dA, dA)
        """
        # µ⁻ₜ = V_A µ⁺ₜ₋₁
        mu_minus = self.V_A @ mu_prev  # (dA,)
        
        # Σ⁻ₜ = V_A Σ⁺ₜ₋₁ V_A^T + Q
        Sigma_minus = self.V_A @ Sigma_prev @ self.V_A.T + self.Q  # (dA, dA)
        
        # 正定値性確保
        Sigma_minus = self._regularize_covariance(Sigma_minus)
        
        return mu_minus, Sigma_minus

    def update_step(
        self,
        mu_minus: torch.Tensor,
        Sigma_minus: torch.Tensor,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        観測更新 (Innovation update)
        
        Algorithm 1のKalman innovation update実装
        
        Args:
            mu_minus: 予測状態平均 µ⁻ₜ (dA,)
            Sigma_minus: 予測状態共分散 Σ⁻ₜ (dA, dA)
            observation: 現在観測 y_t (n,) または (1, n)
            
        Returns:
            mu_plus: 更新状態平均 µ⁺ₜ (dA,)
            Sigma_plus: 更新状態共分散 Σ⁺ₜ (dA, dA)
            likelihood: 観測尤度
        """
        # 1. 現在観測のエンコード: m_t ← u_η(y_t) ∈ R
        with torch.no_grad():
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)  # (1, n)
            m_t = self.encoder(observation).squeeze()  # scalar
            
        # 2. イノベーション共分散: S ← H^T Σ⁻ H + R ∈ R
        S = self.H.mT @ Sigma_minus @ self.H + self.R  # scalar
        
        # 数値安定性チェック
        if S <= 0:
            warnings.warn(f"Non-positive innovation covariance: S = {S}")
            S = torch.tensor(self.jitter, device=self.device)
            
        # 3. Kalmanゲイン: K ← Σ⁻ H S^(-1) ∈ R^(dA×1)
        K = (Sigma_minus @ self.H) / S  # (dA,)
        
        # 4. 状態更新: µ ← µ⁻ + K (m_t - H^T µ⁻)
        innovation = m_t - self.H.mT @ mu_minus  # scalar
        mu_plus = mu_minus + K * innovation  # (dA,)
        
        # 5. 共分散更新: Σ ← Σ⁻ - K H^T Σ⁻
        Sigma_plus = Sigma_minus - torch.outer(K, self.H.mT @ Sigma_minus)  # (dA, dA)
        
        # 正定値性確保
        Sigma_plus = self._regularize_covariance(Sigma_plus)
        
        # 6. 観測尤度計算
        likelihood = self._compute_likelihood(innovation, S)
        
        return mu_plus, Sigma_plus, likelihood

    def filter_step(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        1ステップフィルタリング（逐次処理用）
        
        内部状態を保持してストリーミング対応
        
        Args:
            observation: 現在観測 y_t (n,)
            state: 外部状態 (mu, Sigma) or None（内部状態使用）
            
        Returns:
            x_hat_t: 推定状態 x̂_t (r,)
            Sigma_x_t: 状態共分散 Σ_t^(x) (r, r)
            likelihood: 観測尤度
        """
        if state is not None:
            mu_prev, Sigma_prev = state
        else:
            if not self.is_initialized:
                raise RuntimeError("Filter not initialized. Call initialize_state() first.")
            mu_prev, Sigma_prev = self.mu, self.Sigma
            
        # 時間更新
        mu_minus, Sigma_minus = self.predict_step(mu_prev, Sigma_prev)
        
        # 観測更新
        mu_plus, Sigma_plus, likelihood = self.update_step(
            mu_minus, Sigma_minus, observation
        )
        
        # 元状態空間での復元
        x_hat_t, Sigma_x_t = self._recover_original_state(mu_plus, Sigma_plus)
        
        # 内部状態更新
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
        バッチフィルタリング（従来型）
        
        観測系列全体を一度に処理
        
        Args:
            observations: 観測系列 (T, n)
            initial_state: 初期状態 (µ₀, Σ₀) or None
            return_likelihood: 尤度も返すかどうか
            
        Returns:
            X_means: 状態平均系列 (T, r)
            X_covariances: 状態共分散系列 (T, r, r)
            likelihoods: 観測尤度系列 (T,) [optional]
        """
        T = observations.size(0)
        
        # 初期化
        if initial_state is not None:
            self.mu, self.Sigma = initial_state
            self.is_initialized = True
        elif not self.is_initialized:
            # データ駆動初期化（最初の数サンプル使用）
            n_init = min(10, T)
            self.initialize_state(observations[:n_init])
            
        # 結果格納用
        X_means = torch.zeros(T, self.r, device=self.device)
        X_covariances = torch.zeros(T, self.r, self.r, device=self.device)
        if return_likelihood:
            likelihoods = torch.zeros(T, device=self.device)
        
        # 逐次フィルタリング
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
        元状態空間での復元
        
        x̂_t ← U_A^T µ ∈ R^r
        Σ_t^(x) ← U_A^T Σ U_A ∈ R^(r×r)
        
        Args:
            mu: 特徴空間状態平均 (dA,)
            Sigma: 特徴空間状態共分散 (dA, dA)
            
        Returns:
            x_hat: 元状態空間での状態推定 (r,)
            Sigma_x: 元状態空間での共分散 (r, r)
        """
        # x̂_t ← U_A^T µ
        x_hat = self.U_A.T @ mu  # (r,)
        
        # Σ_t^(x) ← U_A^T Σ U_A  
        Sigma_x = self.U_A.T @ Sigma @ self.U_A  # (r, r)
        
        # 正定値性確保
        Sigma_x = self._regularize_covariance(Sigma_x)
        
        return x_hat, Sigma_x

    def _regularize_covariance(self, cov_matrix: torch.Tensor) -> torch.Tensor:
        """
        共分散行列の正則化
        
        正定値性確保と数値安定性向上
        
        Args:
            cov_matrix: 共分散行列 (d, d)
            
        Returns:
            torch.Tensor: 正則化済み共分散行列 (d, d)
        """
        # 対称性確保
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # 固有値分解
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        except:
            # 失敗時はjitter追加で対応
            cov_matrix += self.jitter * torch.eye(cov_matrix.size(0), device=self.device)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            
        # 負の固有値をクリップ
        eigenvalues = torch.clamp(eigenvalues, min=self.min_eigenvalue)
        
        # 再構成
        cov_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        
        return cov_matrix

    def _compute_likelihood(self, innovation: torch.Tensor, S: torch.Tensor) -> float:
        """
        観測尤度計算
        
        Args:
            innovation: イノベーション (scalar)
            S: イノベーション共分散 (scalar)
            
        Returns:
            float: 対数尤度
        """
        # 正規分布の対数確率密度
        log_likelihood = -0.5 * (
            torch.log(2 * torch.pi * S) + 
            (innovation ** 2) / S
        )
        
        return log_likelihood.item()

    def get_current_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        現在の状態推定値と信頼区間を取得
        
        Returns:
            x_hat: 現在状態推定 (r,)
            Sigma_x: 現在状態共分散 (r, r)
        """
        if not self.is_initialized:
            raise RuntimeError("Filter not initialized")
            
        return self._recover_original_state(self.mu, self.Sigma)

    def reset_state(self):
        """
        内部状態リセット（新しい系列開始時）
        """
        self.mu = None
        self.Sigma = None
        self.is_initialized = False

    def check_numerical_stability(self) -> Dict[str, Any]:
        """
        数値安定性診断
        
        Returns:
            Dict: 診断結果
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
            
        # 条件数チェック
        try:
            cond_Q = torch.linalg.cond(self.Q).item()
            cond_Sigma = torch.linalg.cond(self.Sigma).item()
            
            # 固有値チェック
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