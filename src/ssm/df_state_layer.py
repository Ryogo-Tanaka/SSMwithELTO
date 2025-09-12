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


class DFStateLayer(nn.Module):
    """
    DF-A: Deep Feature Instrumental Variable for State Process
    
    資料のSection 1.4.1に対応。状態系列からの1ステップ予測を
    2段階回帰（2SLS）とクロスフィッティングで実現。
    
    **修正点**: クロスフィッティングの一貫性向上
    
    計算フロー:
    1. ϕ_θ(x_t) で状態を特徴空間に写像
    2. Stage-1: V_A^{(-k)}推定（クロスフィッティング）+ ϕ_θ勾配更新
    3. Stage-2: U_A推定（閉形式解のみ）
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
        super().__init__()
        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        self.lambda_A = float(lambda_A)  # 文字列対応
        self.lambda_B = float(lambda_B)  # 文字列対応
        
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
        
        # **修正**: Phase-1学習用の内部状態管理（クロスフィッティング対応）
        self._stage1_cache = {}  # V_A計算結果をキャッシュ
        self._stage2_cache = {}  # U_A計算結果をキャッシュ
        self._cf_manager: Optional[CrossFittingManager] = None  # クロスフィッティング管理
    

    
    def _ridge_stage1(
        self, 
        X_features: torch.Tensor, 
        Y_targets: torch.Tensor, 
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-1 Ridge回帰: 転送作用素推定
        
        V = (Y^T X)(X^T X + λI)^{-1}
        """

        N, d_A = X_features.shape
        N_t, d_A_t = Y_targets.shape
        
        if N != N_t:
            raise ValueError(f"特徴量とターゲットのサンプル数不一致: {N} vs {N_t}")
        
        if d_A != d_A_t:
            raise ValueError(f"特徴量とターゲットの次元不一致: {d_A} vs {d_A_t}")
        
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
    
    def _initialize_cross_fitting(self, T_eff: int) -> CrossFittingManager:
        """
        **新機能**: クロスフィッティング管理の初期化
        
        Args:
            T_eff: 有効時系列長
            
        Returns:
            CrossFittingManager: 初期化されたクロスフィッティング管理
        """
        # データサイズに応じてクロスフィッティング設定を調整
        cf_config = self.cf_config.copy()
        
        # 最小ブロックサイズの確保
        min_block_size = cf_config.get('min_block_size', 10)
        max_blocks = T_eff // min_block_size
        n_blocks = min(cf_config.get('n_blocks', 5), max_blocks)
        
        if n_blocks < 2:
            # データが小さすぎる場合は、非クロスフィッティング
            warnings.warn(f"データサイズ {T_eff} が小さすぎるため、クロスフィッティングを無効化")
            return None
        
        cf_config['n_blocks'] = n_blocks
        
        return CrossFittingManager(T_eff, **cf_config)
    
    def _compute_crossfit_stage1_loss(
        self, 
        X_states: torch.Tensor, 
        use_simple_fallback: bool = False
    ) -> torch.Tensor:
        """
        **修正版**: クロスフィッティング対応のStage-1損失計算
        
        Args:
            X_states: 状態系列 (T, r)
            use_simple_fallback: 簡易版を使用するかどうか
            
        Returns:
            torch.Tensor: Stage-1損失（スカラー）
        """
        T, r = X_states.shape
        
        # 特徴量計算
        phi_seq = self.phi_theta(X_states)  # (T, d_A)
        
        # 過去/未来特徴量分割
        phi_minus = phi_seq[:-1]  # (T-1, d_A)
        phi_plus = phi_seq[1:]    # (T-1, d_A)
        
        T_eff = phi_minus.size(0)
        
        # **修正**: クロスフィッティングの実装
        if use_simple_fallback or T_eff < 20:
            # 小データまたは簡易版：全データで推定（従来の実装）
            V_A = self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            loss = torch.norm(phi_pred - phi_plus, p='fro') ** 2
            
            # **追加**: 非クロスフィッティング用キャッシュクリア
            self._stage1_cache.pop('V_A_list', None)
            self._stage1_cache.pop('cf_manager', None)
        else:
            # **修正**: 真のクロスフィッティング実装
            cf_manager = self._initialize_cross_fitting(T_eff)
            
            if cf_manager is None:
                # フォールバック：全データ使用
                V_A = self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)
                phi_pred = (V_A @ phi_minus.T).T
                loss = torch.norm(phi_pred - phi_plus, p='fro') ** 2
                
                # **追加**: 非クロスフィッティング用キャッシュクリア
                self._stage1_cache.pop('V_A_list', None)
                self._stage1_cache.pop('cf_manager', None)
            else:
                # クロスフィッティングによる損失計算
                cf_fitter = TwoStageCrossFitter(cf_manager)
                
                # Stage-1: V_A^{(-k)} 推定（勾配なし）
                with torch.no_grad():
                    V_list = cf_fitter.cross_fit_stage1(
                        phi_minus, phi_plus,  # **修正**: detach()削除
                        self._ridge_stage1,
                        reg_lambda=self.lambda_A
                    )
                
                # Out-of-fold予測誤差の計算（勾配あり）
                total_loss = 0.0
                for k in range(cf_manager.n_blocks):
                    # ブロックkのインデックス
                    block_indices = cf_manager.get_block_indices(k)
                    
                    # ブロックkでの予測（勾配あり）
                    phi_minus_k = phi_minus[block_indices]  # 勾配あり
                    phi_plus_k = phi_plus[block_indices]    # 勾配あり
                    
                    # V_A^{(-k)}による予測
                    V_k = V_list[k]  # 勾配なし（detach済み）
                    phi_pred_k = (V_k @ phi_minus_k.T).T
                    
                    # ブロックkの損失
                    loss_k = torch.norm(phi_pred_k - phi_plus_k, p='fro') ** 2
                    total_loss += loss_k
                
                loss = total_loss / cf_manager.n_blocks
                
                # キャッシュ更新（平均転送作用素）
                self._stage1_cache['V_A_list'] = V_list
                self._stage1_cache['cf_manager'] = cf_manager
        
        return loss
    
    def train_stage1_with_gradients(
        self,
        X_states: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        T1_iterations: int = 1
    ) -> Dict[str, float]:
        """
        **修正版**: Stage-1学習 + φ_θ勾配更新（計算グラフ分離対応）
        
        修正内容:
        - retain_graph=True による複数回backward対応
        - 最後の反復のみ完全グラフ解放
        - 反復回数の動的制御
        
        Args:
            X_states: 状態系列 (T, r)  
            optimizer_phi: φ_θ用オプティマイザ
            T1_iterations: Stage-1反復回数
            
        Returns:
            Dict[str, float]: 損失メトリクス
        """
        if X_states.size(0) < 2:
            raise ValueError(f"状態系列が短すぎます: T={int(X_states.size(0))}")
        
        total_loss = 0.0
        
        # **修正**: 反復回数制御とretain_graph管理
        for t in range(T1_iterations):
            optimizer_phi.zero_grad()
            
            # 特徴量計算（各反復で新しい計算グラフ）
            phi_seq = self.phi_theta(X_states)  # (T, d_A)
            
            # 過去/未来分割
            phi_minus = phi_seq[:-1]  # (T-1, d_A)
            phi_plus = phi_seq[1:]    # (T-1, d_A)
            
            # Stage-1: V_A推定（閉形式解）
            with torch.no_grad():
                V_A = self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)
            
            # 予測誤差計算
            phi_pred = (V_A @ phi_minus.T).T  # (T-1, d_A)
            loss_stage1 = torch.norm(phi_pred - phi_plus, p='fro') ** 2
            
            total_loss += loss_stage1.item()
            
            # **修正**: 計算グラフ管理
            if t < T1_iterations - 1:
                # 最後の反復以外: retain_graph=True
                loss_stage1.backward(retain_graph=True)
            else:
                # 最後の反復: 完全解放
                loss_stage1.backward()
            
            optimizer_phi.step()
            
            # **追加**: 反復間のメモリクリア（安全性向上）
            if t < T1_iterations - 1:
                # 中間反復では変数をデタッチ（メモリ効率向上）
                phi_seq = phi_seq.detach()
        
        # Stage-1結果をキャッシュ（Stage-2用）
        with torch.no_grad():
            phi_seq_final = self.phi_theta(X_states)
            phi_minus_final = phi_seq_final[:-1]
            phi_plus_final = phi_seq_final[1:]
            V_A_final = self._ridge_stage1(phi_minus_final, phi_plus_final, self.lambda_A)
            
            self._stage1_cache = {
                'V_A': V_A_final.detach(),
                'phi_minus': phi_minus_final.detach(),
                'phi_plus': phi_plus_final.detach(),
                'X_plus': X_states[1:].detach()
            }
    
        return {
            'stage1_loss': total_loss / T1_iterations,
            'iterations_completed': T1_iterations,
            'final_loss': loss_stage1.item()
        }
    
    def train_stage2_closed_form(self) -> Dict[str, float]:
        """
        Stage-2学習（閉形式解のみ、勾配なし）
        
        資料の学習戦略に対応:
        for t = 1 to T2:  # Stage-2
            U_A = 閉形式解(H^{(cf)}_A, X_+)        # U_A更新（閉形式解のみ）
        
        Returns:
            Dict[str, float]: 損失メトリクス
        """
        if 'V_A' not in self._stage1_cache:
            raise RuntimeError("Stage-1が先に実行されている必要があります")
        
        # Stage-1からの結果を取得
        V_A = self._stage1_cache['V_A']
        phi_minus = self._stage1_cache['phi_minus']
        X_plus = self._stage1_cache['X_plus']
        
        # **修正**: クロスフィッティングを考慮したStage-2
        with torch.no_grad():
            if 'cf_manager' in self._stage1_cache and 'V_A_list' in self._stage1_cache:
                # クロスフィッティング使用時：out-of-fold特徴量を使用
                cf_manager = self._stage1_cache['cf_manager']
                V_A_list = self._stage1_cache['V_A_list']
                cf_fitter = TwoStageCrossFitter(cf_manager)
                
                # Out-of-fold特徴量計算
                H_cf = cf_fitter.compute_out_of_fold_features(phi_minus, V_A_list)
                
                # U_A推定
                U_A = self._ridge_stage2(H_cf, X_plus, self.lambda_B)
                
                # 損失計算（参考用）
                X_pred = (U_A.T @ H_cf.T).T
                loss_stage2 = torch.norm(X_pred - X_plus, p='fro') ** 2
            else:
                # 非クロスフィッティング時：直接計算
                H_cf = (V_A @ phi_minus.T).T
                U_A = self._ridge_stage2(H_cf, X_plus, self.lambda_B)
                
                # 損失計算（参考用）
                X_pred = (U_A.T @ H_cf.T).T
                loss_stage2 = torch.norm(X_pred - X_plus, p='fro') ** 2
        
        # U_A をキャッシュ
        self._stage2_cache['U_A'] = U_A
        
        return {'stage2_loss': loss_stage2.item()}
    
    def fit_two_stage(
        self, 
        X_states: torch.Tensor, 
        use_cross_fitting: bool = True,
        verbose: bool = False
    ) -> 'DFStateLayer':
        """
        従来の2段階クロスフィッティング学習（変更なし）
        
        **注意**: これは既存の学習メソッドです。
        新しいPhase-1学習では train_stage1_with_gradients と 
        train_stage2_closed_form を使用してください。
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
        T_eff = int(Phi_minus.size(0))  # T-1)
        
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
            # **修正**: キャッシュされた結果も使用可能に
            if 'V_A' in self._stage1_cache:
                V_A = self._stage1_cache['V_A']
            else:
                raise RuntimeError("fit_two_stage() または train_stage1_with_gradients() を先に実行してください")
        else:
            V_A = self.V_A
        
        if phi_prev.dim() == 1:
            return V_A @ phi_prev
        else:
            return (V_A @ phi_prev.T).T
    
    def predict_one_step(self, x_prev: torch.Tensor) -> torch.Tensor:
        """
        1ステップ状態予測: x̂_{t|t-1} = U_A^T V_A ϕ_θ(x_{t-1})
        
        Args:
            x_prev: 前時刻の状態 (r,) または (batch, r)
            
        Returns:
            torch.Tensor: 予測状態 (r,) または (batch, r)
        """
        V_A = None
        U_A = None
        
        if self._is_fitted:
            # 完全学習済み
            V_A = self.V_A
            U_A = self.U_A
        elif 'V_A' in self._stage1_cache and 'U_A' in self._stage2_cache:
            # Phase-1学習済み
            V_A = self._stage1_cache['V_A']
            U_A = self._stage2_cache['U_A']
        elif 'V_A' in self._stage1_cache:
            # **新規**: Stage-1のみ完了の場合の対応
            V_A = self._stage1_cache['V_A']
            # 簡易的なU_A推定
            if 'phi_minus' in self._stage1_cache and 'X_plus' in self._stage1_cache:
                with torch.no_grad():
                    phi_minus = self._stage1_cache['phi_minus']
                    X_plus = self._stage1_cache['X_plus']
                    H_simple = (V_A @ phi_minus.T).T
                    U_A = self._ridge_stage2(H_simple, X_plus, self.lambda_B)
                    # キャッシュに保存
                    self._stage2_cache['U_A'] = U_A.detach()
            else:
                raise RuntimeError("Stage-1は完了していますが、Stage-2実行に必要なデータが不足しています")
        else:
            raise RuntimeError("学習が完了していません。fit_two_stage() または train_stage1_with_gradients() を先に実行してください")
        
        # 特徴写像
        phi_prev = self.phi_theta(x_prev)
        
        # 転送作用素適用
        phi_pred = self.apply_transfer_operator(phi_prev)
        
        # 状態空間に戻す
        if phi_pred.dim() == 1:
            return U_A.T @ phi_pred
        else:
            return (U_A.T @ phi_pred.T).T
    
    def predict_sequence(
        self, 
        X_states: torch.Tensor, 
        return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        系列予測: 各時刻でのone-step-ahead予測（推論専用モード）
        
        Args:
            X_states: 状態系列 (T, r)
            return_features: 特徴量も返すかどうか
            
        Returns:
            torch.Tensor: 予測系列 (T-1, r)
            Optional[torch.Tensor]: 特徴量系列 (T-1, d_A)
        """
        if not self._is_fitted and 'V_A' not in self._stage1_cache:
            raise RuntimeError("学習が完了していません")
        
        T = X_states.size(0)
        predictions = []
        features = []
        
        # 推論専用モードで実行（勾配グラフを切断）
        with torch.no_grad():
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
        if self._is_fitted:
            return self.V_A.clone()
        elif 'V_A' in self._stage1_cache:
            return self._stage1_cache['V_A'].clone()
        else:
            raise RuntimeError("学習が完了していません")
    
    def get_readout_matrix(self) -> torch.Tensor:
        """読み出し行列 U_A を取得"""
        if self._is_fitted:
            return self.U_A.clone()
        elif 'U_A' in self._stage2_cache:
            return self._stage2_cache['U_A'].clone()
        else:
            raise RuntimeError("学習が完了していません")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """学習済みパラメータを辞書で取得"""
        state_dict = {
            'phi_theta': self.phi_theta.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'feature_dim': self.feature_dim,
                'lambda_A': self.lambda_A,
                'lambda_B': self.lambda_B
            }
        }
        
        if self._is_fitted:
            state_dict.update({
                'V_A': self.V_A,
                'U_A': self.U_A,
            })
        
        if self._stage1_cache:
            state_dict['stage1_cache'] = self._stage1_cache.copy()
        if self._stage2_cache:
            state_dict['stage2_cache'] = self._stage2_cache.copy()
            
        return state_dict