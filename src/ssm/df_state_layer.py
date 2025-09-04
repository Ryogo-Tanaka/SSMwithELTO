# src/ssm/df_state_layer.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import warnings

from .cross_fitting import CrossFittingManager, TwoStageCrossFitter, CrossFittingError


class StateFeatureNet(nn.Module):
    """
    状態特徴写像 ϕ_θ: R^r → R^{d_A}
    
    状態変数を高次元特徴空間に写像するニューラルネットワーク。
    DF-AとDF-Bで共有される。
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_sizes: list[int] = [64, 64],
        activation: str = "ReLU",
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: 状態次元 r
            output_dim: 特徴次元 d_A
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 状態 (batch_size, r) または (r,)
            
        Returns:
            torch.Tensor: 特徴 (batch_size, d_A) または (d_A,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.net(x).squeeze(0)
        return self.net(x)


class DFStateLayer:
    """
    DF-A: Deep Feature Instrumental Variable for State Process
    
    資料のSection 1.4.1に対応。状態系列からの1ステップ予測を
    2段階回帰（2SLS）とクロスフィッティングで実現。
    
    計算フロー:
    1. ϕ_θ(x_t) で状態を特徴空間に写像
    2. Stage-1: V_A推定（転送作用素）
    3. Stage-2: U_A推定（読み出し行列） 
    4. 予測: x̂_{t|t-1} = U_A^T V_A ϕ_θ(x_{t-1})
    """
    
    def __init__(
        self,
        state_dim: int,
        feature_dim: int,
        lambda_A: float = 1e-3,
        lambda_B: float = 1e-3,
        feature_net_config: Optional[Dict[str, Any]] = None,
        cross_fitting_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            state_dim: 状態次元 r
            feature_dim: 特徴次元 d_A
            lambda_A: Stage-1正則化パラメータ λ_A
            lambda_B: Stage-2正則化パラメータ λ_B
            feature_net_config: StateFeatureNetの設定
            cross_fitting_config: CrossFittingManagerの設定
        """
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        
        # 特徴ネットワーク
        feature_config = feature_net_config or {}
        self.phi_theta = StateFeatureNet(
            input_dim=state_dim,
            output_dim=feature_dim,
            **feature_config
        )
        
        # クロスフィッティング設定
        self.cf_config = cross_fitting_config or {'n_blocks': 5, 'min_block_size': 10}
        
        # 学習済みパラメータ
        self.V_A: Optional[torch.Tensor] = None  # 転送作用素 (d_A, d_A)
        self.U_A: Optional[torch.Tensor] = None  # 読み出し行列 (d_A, r)
        self._is_fitted = False
    
    def _ridge_stage1(
        self, 
        X_features: torch.Tensor, 
        Y_targets: torch.Tensor, 
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-1 Ridge回帰: 転送作用素推定
        
        V = (Y^T X)(X^T X + λI)^{-1}
        
        Args:
            X_features: 入力特徴量 Φ^-_{-k} (N, d_A)
            Y_targets: 目的特徴量 Φ^+_{-k} (N, d_A)  
            reg_lambda: 正則化パラメータ λ_A
            
        Returns:
            torch.Tensor: 転送作用素 V (d_A, d_A)
        """
        N, d_A = X_features.shape
        
        if N < d_A:
            warnings.warn(f"サンプル数 {N} < 特徴次元 {d_A}。数値不安定の可能性")
        
        # グラム行列 + 正則化
        XtX = X_features.T @ X_features  # (d_A, d_A)
        XtX_reg = XtX + reg_lambda * torch.eye(d_A, device=X_features.device, dtype=X_features.dtype)
        
        # クロス共分散
        YtX = Y_targets.T @ X_features  # (d_A, d_A)
        
        # 逆行列計算（数値安定化）
        try:
            XtX_inv = torch.linalg.inv(XtX_reg)
            V = YtX @ XtX_inv
        except torch.linalg.LinAlgError:
            # Cholesky分解 fallback
            try:
                L = torch.linalg.cholesky(XtX_reg)
                XtX_inv = torch.cholesky_inverse(L)
                V = YtX @ XtX_inv
            except torch.linalg.LinAlgError:
                # SVD fallback
                U, S, Vh = torch.linalg.svd(XtX_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                XtX_inv = (Vh.T * S_inv) @ Vh
                V = YtX @ XtX_inv
        
        return V
    
    def _ridge_stage2(
        self, 
        H_features: torch.Tensor, 
        X_targets: torch.Tensor, 
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-2 Ridge回帰: 読み出し行列推定
        
        U = (H H^T + λI)^{-1} H X^T
        
        Args:
            H_features: クロスフィット特徴量 H^{(cf)}_A (N, d_A)
            X_targets: 目標状態 X^+ (N, r)
            reg_lambda: 正則化パラメータ λ_B
            
        Returns:
            torch.Tensor: 読み出し行列 U (d_A, r)
        """
        N, d_A = H_features.shape
        N_t, r = X_targets.shape
        
        if N != N_t:
            raise ValueError(f"特徴量とターゲットのサンプル数不一致: {N} vs {N_t}")
        
        # グラム行列 + 正則化
        HHt = H_features.T @ H_features  # (d_A, d_A)
        HHt_reg = HHt + reg_lambda * torch.eye(d_A, device=H_features.device, dtype=H_features.dtype)
        
        # クロス項
        HXt = H_features.T @ X_targets  # (d_A, r)
        
        # 逆行列計算（数値安定化）
        try:
            HHt_inv = torch.linalg.inv(HHt_reg)
            U = HHt_inv @ HXt
        except torch.linalg.LinAlgError:
            # Cholesky分解 fallback
            try:
                L = torch.linalg.cholesky(HHt_reg)
                HHt_inv = torch.cholesky_inverse(L)
                U = HHt_inv @ HXt
            except torch.linalg.LinAlgError:
                # SVD fallback
                U_svd, S, Vh = torch.linalg.svd(HHt_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                HHt_inv = (Vh.T * S_inv) @ Vh
                U = HHt_inv @ HXt
        
        return U
    
    def fit_two_stage(
        self, 
        X_states: torch.Tensor, 
        use_cross_fitting: bool = True,
        verbose: bool = False
    ) -> 'DFStateLayer':
        """
        2段階クロスフィッティング学習
        
        資料の式(13)(14)と(17)-(20)に対応。
        
        Args:
            X_states: 状態系列 (T, r)
            use_cross_fitting: クロスフィッティングを使用するか
            verbose: 詳細ログ出力
            
        Returns:
            self
            
        Raises:
            CrossFittingError: フィッティングに失敗した場合
        """
        T, r = X_states.shape
        
        if r != self.state_dim:
            raise ValueError(f"状態次元不一致: expected {self.state_dim}, got {r}")
        
        if T < 2:
            raise ValueError(f"時系列が短すぎます: T={T}")
        
        # 特徴量計算
        with torch.no_grad():
            phi_seq = self.phi_theta(X_states)  # (T, d_A)
        
        # 過去/未来特徴量
        Phi_minus = phi_seq[:-1]  # (T-1, d_A)
        Phi_plus = phi_seq[1:]    # (T-1, d_A)
        X_plus = X_states[1:]     # (T-1, r)
        
        if use_cross_fitting and T >= 20:  # 最小限のサンプルサイズ
            self._fit_with_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose)
        else:
            self._fit_without_cross_fitting(Phi_minus, Phi_plus, X_plus, verbose)
        
        self._is_fitted = True
        return self
    
    def _fit_with_cross_fitting(
        self, 
        Phi_minus: torch.Tensor, 
        Phi_plus: torch.Tensor, 
        X_plus: torch.Tensor,
        verbose: bool
    ):
        """クロスフィッティング付き学習"""
        T_eff = Phi_minus.size(0)  # T-1
        
        # クロスフィッティング管理
        cf_manager = CrossFittingManager(T_eff, **self.cf_config)
        cf_fitter = TwoStageCrossFitter(cf_manager)
        
        if verbose:
            print(f"クロスフィッティング: T={T_eff}, n_blocks={cf_manager.n_blocks}")
        
        # Stage-1: 転送作用素推定（クロスフィッティング）
        V_list = cf_fitter.cross_fit_stage1(
            Phi_minus, Phi_plus,
            self._ridge_stage1,
            reg_lambda=self.lambda_A
        )
        
        # 平均転送作用素（最終的な V_A）
        self.V_A = torch.stack(V_list).mean(dim=0)
        
        # Out-of-fold特徴量計算
        H_cf = cf_fitter.compute_out_of_fold_features(Phi_minus, V_list)
        
        # Stage-2: 読み出し行列推定
        self.U_A = cf_fitter.cross_fit_stage2(
            H_cf, X_plus,
            self._ridge_stage2,
            detach_features=True,
            reg_lambda=self.lambda_B
        )
        
        if verbose:
            print(f"V_A shape: {self.V_A.shape}, U_A shape: {self.U_A.shape}")
    
    def _fit_without_cross_fitting(
        self, 
        Phi_minus: torch.Tensor, 
        Phi_plus: torch.Tensor, 
        X_plus: torch.Tensor,
        verbose: bool
    ):
        """クロスフィッティングなし学習（小データ用）"""
        if verbose:
            print("クロスフィッティングなしで学習")
        
        # Stage-1: 直接推定
        self.V_A = self._ridge_stage1(Phi_minus, Phi_plus, self.lambda_A)
        
        # 中間特徴量
        H = (self.V_A @ Phi_minus.T).T  # (T-1, d_A)
        
        # Stage-2: 読み出し推定
        self.U_A = self._ridge_stage2(H, X_plus, self.lambda_B)
        
        if verbose:
            print(f"V_A shape: {self.V_A.shape}, U_A shape: {self.U_A.shape}")
    
    def apply_transfer_operator(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        転送作用素の適用: φ̂_{t|t-1} = V_A φ_{t-1}
        
        Args:
            phi_prev: 前時刻の特徴量 (d_A,) または (batch, d_A)
            
        Returns:
            torch.Tensor: 予測特徴量 (d_A,) または (batch, d_A)
        """
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        
        if phi_prev.dim() == 1:
            return self.V_A @ phi_prev
        else:
            return (self.V_A @ phi_prev.T).T
    
    def predict_one_step(self, x_prev: torch.Tensor) -> torch.Tensor:
        """
        1ステップ状態予測: x̂_{t|t-1} = U_A^T V_A ϕ_θ(x_{t-1})
        
        Args:
            x_prev: 前時刻の状態 (r,) または (batch, r)
            
        Returns:
            torch.Tensor: 予測状態 (r,) または (batch, r)
        """
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        
        # 特徴写像
        phi_prev = self.phi_theta(x_prev)
        
        # 転送作用素適用
        phi_pred = self.apply_transfer_operator(phi_prev)
        
        # 状態空間に戻す
        if phi_pred.dim() == 1:
            return self.U_A.T @ phi_pred
        else:
            return (self.U_A.T @ phi_pred.T).T
    
    def predict_sequence(
        self, 
        X_states: torch.Tensor, 
        return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        系列予測: 各時刻でのone-step-ahead予測
        
        Args:
            X_states: 状態系列 (T, r)
            return_features: 特徴量も返すかどうか
            
        Returns:
            torch.Tensor: 予測系列 (T-1, r)
            Optional[torch.Tensor]: 特徴量系列 (T-1, d_A)
        """
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        
        T = X_states.size(0)
        predictions = []
        features = []
        
        for t in range(T - 1):
            x_pred = self.predict_one_step(X_states[t])
            predictions.append(x_pred)
            
            if return_features:
                phi_prev = self.phi_theta(X_states[t])
                phi_pred = self.apply_transfer_operator(phi_prev)
                features.append(phi_pred)
        
        pred_tensor = torch.stack(predictions)
        
        if return_features:
            feat_tensor = torch.stack(features)
            return pred_tensor, feat_tensor
        
        return pred_tensor
    
    def get_transfer_operator(self) -> torch.Tensor:
        """転送作用素 V_A を取得"""
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        return self.V_A.clone()
    
    def get_readout_matrix(self) -> torch.Tensor:
        """読み出し行列 U_A を取得"""
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        return self.U_A.clone()
    
    def get_state_dict(self) -> Dict[str, Any]:
        """学習済みパラメータを辞書で取得"""
        if not self._is_fitted:
            raise RuntimeError("fit_two_stage() を先に実行してください")
        
        return {
            'phi_theta': self.phi_theta.state_dict(),
            'V_A': self.V_A,
            'U_A': self.U_A,
            'config': {
                'state_dim': self.state_dim,
                'feature_dim': self.feature_dim,
                'lambda_A': self.lambda_A,
                'lambda_B': self.lambda_B
            }
        }