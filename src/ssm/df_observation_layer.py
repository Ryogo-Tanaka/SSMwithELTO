# src/ssm/df_observation_layer.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import warnings

from .cross_fitting import CrossFittingManager, TwoStageCrossFitter, CrossFittingError
from .df_state_layer import DFStateLayer


class ObservationFeatureNet(nn.Module):
    """
    観測特徴写像 ψ_ω: R → R^{d_B}
    
    スカラー特徴量（エンコーダ出力）を高次元特徴空間に写像する。
    """
    
    def __init__(
        self, 
        input_dim: int = 1,  # スカラー特徴量
        output_dim: int = 16,
        hidden_sizes: list[int] = [32, 32],
        activation: str = "ReLU",
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: 入力次元（通常1）
            output_dim: 出力特徴次元 d_B
            hidden_sizes: 中間層のユニット数リスト
            activation: 活性化関数名
            dropout: ドロップアウト率
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 中間層
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(nn, activation)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 出力層
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # 初期化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m: スカラー特徴量 (batch_size, 1) または (1,) または scalar
            
        Returns:
            torch.Tensor: 観測特徴量 (batch_size, d_B) または (d_B,)
        """
        # スカラー入力を(1,)形状に正規化
        if m.dim() == 0:  # scalar
            m = m.unsqueeze(0).unsqueeze(0)  # (1, 1)
            return self.net(m).squeeze(0)  # (d_B,)
        elif m.dim() == 1:
            if m.size(0) == 1:  # (1,)
                m = m.unsqueeze(0)  # (1, 1)
                return self.net(m).squeeze(0)  # (d_B,)
            else:  # (batch_size,)
                m = m.unsqueeze(1)  # (batch_size, 1)
                return self.net(m)  # (batch_size, d_B)
        else:  # (batch_size, 1)
            return self.net(m)


class DFObservationLayer:
    """
    DF-B: Deep Feature Instrumental Variable for Observation Process
    
    資料のSection 1.4.2に対応。DF-Aで得られた状態予測を操作変数として、
    スカラー特徴量の1ステップ予測を2SLSで実現。
    
    計算フロー:
    1. DF-Aから状態予測 x̂_{t|t-1} を受け取る
    2. ϕ_θ(x̂_{t|t-1}) を操作変数として使用
    3. ψ_ω(m_t) で観測特徴量を計算
    4. Stage-1: V_B推定, Stage-2: u_B推定
    5. 予測: m̂_{t|t-1} = u_B^T V_B ϕ_θ(x̂_{t|t-1})
    """
    
    def __init__(
        self,
        df_state_layer: DFStateLayer,
        obs_feature_dim: int = 16,
        lambda_B: float = 1e-3,
        lambda_dB: float = 1e-3,
        obs_net_config: Optional[Dict[str, Any]] = None,
        cross_fitting_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            df_state_layer: 学習済みDFStateLayerインスタンス
            obs_feature_dim: 観測特徴次元 d_B
            lambda_B: Stage-1正則化パラメータ λ_B
            lambda_dB: Stage-2正則化パラメータ λ_{dB}
            obs_net_config: ObservationFeatureNetの設定
            cross_fitting_config: CrossFittingManagerの設定
        """
        if not df_state_layer._is_fitted:
            raise RuntimeError("df_state_layerは学習済みである必要があります")
        
        self.df_state = df_state_layer
        self.obs_feature_dim = obs_feature_dim
        self.lambda_B = lambda_B
        self.lambda_dB = lambda_dB
        
        # 共有特徴ネットワーク（DF-Aから直接参照）
        self.phi_theta = df_state_layer.phi_theta
        
        # 観測特徴ネットワーク
        obs_config = obs_net_config or {}
        self.psi_omega = ObservationFeatureNet(
            input_dim=1,
            output_dim=obs_feature_dim,
            **obs_config
        )
        
        # クロスフィッティング設定
        self.cf_config = cross_fitting_config or {'n_blocks': 5, 'min_block_size': 10}
        
        # 学習済みパラメータ
        self.V_B: Optional[torch.Tensor] = None  # 観測転送作用素 (d_B, d_A)
        self.u_B: Optional[torch.Tensor] = None  # 観測読み出しベクトル (d_B,)
        self._is_fitted = False
    
    def _ridge_stage1_vb(
        self, 
        Phi_instrument: torch.Tensor, 
        Psi_treatment: torch.Tensor, 
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-1 Ridge回帰: 観測転送作用素推定
        
        V_B = (Ψ^T Φ)(Φ^T Φ + λI)^{-1}
        
        Args:
            Phi_instrument: 操作変数特徴量 (N, d_A)
            Psi_treatment: 観測特徴量 (N, d_B)
            reg_lambda: 正則化パラメータ λ_B
            
        Returns:
            torch.Tensor: 観測転送作用素 V_B (d_B, d_A)
        """
        N, d_A = Phi_instrument.shape
        N_t, d_B = Psi_treatment.shape
        
        if N != N_t:
            raise ValueError(f"操作変数と観測の サンプル数不一致: {N} vs {N_t}")
        
        if N < max(d_A, d_B):
            warnings.warn(f"サンプル数 {N} < 特徴次元 max({d_A}, {d_B})。数値不安定の可能性")
        
        # グラム行列 + 正則化
        PhiTPhi = Phi_instrument.T @ Phi_instrument  # (d_A, d_A)
        PhiTPhi_reg = PhiTPhi + reg_lambda * torch.eye(d_A, device=Phi_instrument.device, dtype=Phi_instrument.dtype)
        
        # クロス共分散
        PsiTPhi = Psi_treatment.T @ Phi_instrument  # (d_B, d_A)
        
        # 逆行列計算（数値安定化）
        try:
            PhiTPhi_inv = torch.linalg.inv(PhiTPhi_reg)
            V_B = PsiTPhi @ PhiTPhi_inv
        except torch.linalg.LinAlgError:
            # Cholesky分解 fallback
            try:
                L = torch.linalg.cholesky(PhiTPhi_reg)
                PhiTPhi_inv = torch.cholesky_inverse(L)
                V_B = PsiTPhi @ PhiTPhi_inv
            except torch.linalg.LinAlgError:
                # SVD fallback
                U, S, Vh = torch.linalg.svd(PhiTPhi_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                PhiTPhi_inv = (Vh.T * S_inv) @ Vh
                V_B = PsiTPhi @ PhiTPhi_inv
        
        return V_B
    
    def _ridge_stage2_ub(
        self, 
        H_instrument: torch.Tensor, 
        m_target: torch.Tensor, 
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-2 Ridge回帰: 観測読み出しベクトル推定
        
        u_B = (H H^T + λI)^{-1} H m
        
        Args:
            H_instrument: クロスフィット操作変数特徴量 (N, d_B)
            m_target: 目標スカラー特徴量 (N,)
            reg_lambda: 正則化パラメータ λ_{dB}
            
        Returns:
            torch.Tensor: 観測読み出しベクトル u_B (d_B,)
        """
        N, d_B = H_instrument.shape
        
        if m_target.size(0) != N:
            raise ValueError(f"操作変数とターゲットのサンプル数不一致: {N} vs {m_target.size(0)}")
        
        # グラム行列 + 正則化
        HHt = H_instrument.T @ H_instrument  # (d_B, d_B)
        HHt_reg = HHt + reg_lambda * torch.eye(d_B, device=H_instrument.device, dtype=H_instrument.dtype)
        
        # クロス項
        Hm = H_instrument.T @ m_target  # (d_B,)
        
        # 逆行列計算（数値安定化）
        try:
            HHt_inv = torch.linalg.inv(HHt_reg)
            u_B = HHt_inv @ Hm
        except torch.linalg.LinAlgError:
            # Cholesky分解 fallback
            try:
                L = torch.linalg.cholesky(HHt_reg)
                HHt_inv = torch.cholesky_inverse(L)
                u_B = HHt_inv @ Hm
            except torch.linalg.LinAlgError:
                # SVD fallback
                U, S, Vh = torch.linalg.svd(HHt_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                HHt_inv = (Vh.T * S_inv) @ Vh
                u_B = HHt_inv @ Hm
        
        return u_B
    
    def fit_two_stage(
        self, 
        X_hat_states: torch.Tensor,  # DF-Aからの状態予測
        m_features: torch.Tensor,    # エンコーダからのスカラー特徴量
        use_cross_fitting: bool = True,
        verbose: bool = False
    ) -> 'DFObservationLayer':
        """
        2段階クロスフィッティング学習
        
        資料の式(21)(22)とDF-B cross-fittingに対応。
        
        Args:
            X_hat_states: DF-Aからの状態予測 (T, r)
            m_features: スカラー特徴量 (T,) または (T, 1)
            use_cross_fitting: クロスフィッティングを使用するか
            verbose: 詳細ログ出力
            
        Returns:
            self
        """
        T_x, r = X_hat_states.shape
        
        # スカラー特徴量の次元調整
        if m_features.dim() == 2 and m_features.size(1) == 1:
            m_features = m_features.squeeze(1)  # (T, 1) → (T,)
        elif m_features.dim() != 1:
            raise ValueError(f"スカラー特徴量は1次元であるべき: got shape {m_features.shape}")
        
        T_m = m_features.size(0)
        
        if T_x != T_m:
            raise ValueError(f"状態予測と特徴量の時系列長不一致: {T_x} vs {T_m}")
        
        if T_x < 2:
            raise ValueError(f"時系列が短すぎます: T={T_x}")
        
        # 操作変数特徴量（状態予測から）
        with torch.no_grad():
            Phi_instrument = self.phi_theta(X_hat_states)  # (T, d_A)
        
        # 観測特徴量
        with torch.no_grad():
            Psi_obs = self.psi_omega(m_features)  # (T, d_B)
        
        # 時間合わせ: 現時刻の観測を次時刻で予測
        # 操作変数: t-1 時刻の状態予測特徴量
        # 目標: t 時刻の観測特徴量
        Phi_prev = Phi_instrument[:-1]  # (T-1, d_A)
        Psi_curr = Psi_obs[1:]          # (T-1, d_B)
        m_curr = m_features[1:]         # (T-1,)
        
        if use_cross_fitting and T_x >= 20:
            self._fit_with_cross_fitting(Phi_prev, Psi_curr, m_curr, verbose)
        else:
            self._fit_without_cross_fitting(Phi_prev, Psi_curr, m_curr, verbose)
        
        self._is_fitted = True
        return self
    
    def _fit_with_cross_fitting(
        self, 
        Phi_prev: torch.Tensor, 
        Psi_curr: torch.Tensor, 
        m_curr: torch.Tensor,
        verbose: bool
    ):
        """クロスフィッティング付き学習"""
        T_eff = Phi_prev.size(0)  # T-1
        
        # クロスフィッティング管理
        cf_manager = CrossFittingManager(T_eff, **self.cf_config)
        cf_fitter = TwoStageCrossFitter(cf_manager)
        
        if verbose:
            print(f"DF-B クロスフィッティング: T={T_eff}, n_blocks={cf_manager.n_blocks}")
        
        # Stage-1: 観測転送作用素推定（クロスフィッティング）
        VB_list = cf_fitter.cross_fit_stage1(
            Phi_prev, Psi_curr,
            self._ridge_stage1_vb,
            reg_lambda=self.lambda_B
        )
        
        # 平均観測転送作用素（最終的な V_B）
        self.V_B = torch.stack(VB_list).mean(dim=0)
        
        # Out-of-fold操作変数特徴量計算
        H_cf = cf_fitter.compute_out_of_fold_features(Phi_prev, VB_list)
        
        # Stage-2: 観測読み出しベクトル推定
        self.u_B = cf_fitter.cross_fit_stage2(
            H_cf, m_curr,
            self._ridge_stage2_ub,
            detach_features=True,
            reg_lambda=self.lambda_dB
        )
        
        if verbose:
            print(f"V_B shape: {self.V_B.shape}, u_B shape: {self.u_B.shape}")
    
    def _fit_without_cross_fitting(
        self, 
        Phi_prev: torch.Tensor, 
        Psi_curr: torch.Tensor, 
        m_curr: torch.Tensor,
        verbose: bool
    ):
        """クロスフィッティングなし学習（小データ用）"""
        if verbose:
            print("DF-B クロスフィッティングなしで学習")
        
        # Stage-1: 直接推定
        self.V_B = self._ridge_stage1_vb(Phi_prev, Psi_curr, self.lambda_B)
        
        # 中間特徴量
        H = (self.V_B @ Phi_prev.T).T  # (T-1, d_B)
        
        # Stage-2: 読み出し推定
        self.u_B = self._ridge_stage2_ub(H, m_curr, self.lambda_dB)
        
        if verbose:
            print(f"V_B shape: {self.V_B.shape}, u_B shape: {self.u_B.shape}")
    
    def predict_one_step(self, x_hat_prev: torch.Tensor) -> torch.Tensor:
        """
        1ステップ特徴量予測: m̂_{t|t-1} = u_B^T V_B ϕ_θ(x̂_{t|t-1})
        
        Args:
            x_hat_prev: 前時刻の状態予測 (r,) または (batch, r)
            
        Returns:
            torch.Tensor: 予測スカラー特徴量 () または (batch,)
        """
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        
        # 状態特徴写像
        phi_prev = self.phi_theta(x_hat_prev)
        
        # 観測転送作用素適用 + 読み出し
        if phi_prev.dim() == 1:
            h_pred = self.V_B @ phi_prev  # (d_B,)
            return self.u_B @ h_pred      # scalar
        else:
            h_pred = (self.V_B @ phi_prev.T).T  # (batch, d_B)
            return (h_pred * self.u_B).sum(dim=1)  # (batch,)
    
    def predict_sequence(
        self, 
        X_hat_states: torch.Tensor
    ) -> torch.Tensor:
        """
        系列予測: 各時刻でのone-step-ahead特徴量予測
        
        Args:
            X_hat_states: 状態予測系列 (T, r)
            
        Returns:
            torch.Tensor: 予測特徴量系列 (T-1,)
        """
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        
        T = X_hat_states.size(0)
        predictions = []
        
        for t in range(T - 1):
            m_pred = self.predict_one_step(X_hat_states[t])
            predictions.append(m_pred)
        
        return torch.stack(predictions)
    
    def get_observation_operator(self) -> torch.Tensor:
        """観測転送作用素 V_B を取得"""
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        return self.V_B.clone()
    
    def get_readout_vector(self) -> torch.Tensor:
        """観測読み出しベクトル u_B を取得"""
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        return self.u_B.clone()
    
    def get_state_dict(self) -> Dict[str, Any]:
        """学習済みパラメータを辞書で取得"""
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        
        return {
            'psi_omega': self.psi_omega.state_dict(),
            'V_B': self.V_B,
            'u_B': self.u_B,
            'config': {
                'obs_feature_dim': self.obs_feature_dim,
                'lambda_B': self.lambda_B,
                'lambda_dB': self.lambda_dB
            }
        }