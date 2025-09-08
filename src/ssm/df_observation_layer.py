# src/ssm/df_observation_layer.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import warnings

from .cross_fitting import CrossFittingManager, TwoStageCrossFitter, CrossFittingError


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
    
    **修正点**: 勾配制御の明確化
    
    計算フロー:
    1. DF-Aから状態予測 x̂_{t|t-1} と共有特徴写像 φ_θ を受け取る
    2. φ_θ(x̂_{t|t-1}) を操作変数として使用
    3. ψ_ω(m_t) で観測特徴量を計算
    4. Stage-1: V_B推定 + φ_θ更新（ψ_ω完全固定）
    5. Stage-2: u_B推定 + ψ_ω更新（φ_θ完全固定）
    6. 予測: m̂_{t|t-1} = u_B^T V_B φ_θ(x̂_{t|t-1})
    """
    
    def __init__(
        self,
        df_state_layer,  # DFStateLayerインスタンス（必須）
        obs_feature_dim: int = 16,
        lambda_B: float = 1e-3,
        lambda_dB: float = 1e-3,
        obs_net_config: Optional[Dict[str, Any]] = None,
        cross_fitting_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            df_state_layer: **必須** 学習済みDFStateLayerインスタンス
            obs_feature_dim: 観測特徴次元 d_B
            lambda_B: Stage-1正則化パラメータ λ_B
            lambda_dB: Stage-2正則化パラメータ λ_{dB}
            obs_net_config: ObservationFeatureNetの設定
            cross_fitting_config: CrossFittingManagerの設定
        """
        if df_state_layer is None:
            raise ValueError("df_state_layerは必須です。DFStateLayerインスタンスを渡してください。")
        
        self.df_state = df_state_layer
        self.obs_feature_dim = obs_feature_dim
        self.lambda_B = lambda_B
        self.lambda_dB = lambda_dB
        
        # **重要**: 共有特徴ネットワーク（DF-Aから直接参照）
        self.phi_theta = df_state_layer.phi_theta
        
        # 観測特徴ネットワーク ψ_ω
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
        
        # **新機能**: Phase-1学習用の内部状態管理
        self._stage1_cache = {}  # V_B計算結果をキャッシュ
        self._stage2_cache = {}  # u_B計算結果をキャッシュ
    
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

        N = int(N.item() if hasattr(N, 'item') else N)
        d_A = int(d_A.item() if hasattr(d_A, 'item') else d_A)
        N_t = int(N_t.item() if hasattr(N_t, 'item') else N_t)
        d_B = int(d_B.item() if hasattr(d_B, 'item') else d_B)
        
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
        N = int(N.item() if hasattr(N, 'item') else N)
        d_B = int(d_B.item() if hasattr(d_B, 'item') else d_B)
        
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
    
    def _freeze_parameters(self, module: nn.Module) -> Dict[str, bool]:
        """
        パラメータを完全に固定（requires_grad=False）
        
        Args:
            module: 固定するモジュール
            
        Returns:
            Dict: 元のrequires_grad状態（復元用）
        """
        original_states = {}
        for name, param in module.named_parameters():
            original_states[name] = param.requires_grad
            param.requires_grad = False
        return original_states
    
    def _restore_parameters(self, module: nn.Module, original_states: Dict[str, bool]):
        """
        パラメータのrequires_grad状態を復元
        
        Args:
            module: 復元するモジュール
            original_states: _freeze_parametersで取得した元の状態
        """
        for name, param in module.named_parameters():
            if name in original_states:
                param.requires_grad = original_states[name]
    
    # **修正**: Phase-1学習用メソッドの勾配制御を明確化
    def train_stage1_with_gradients(
        self, 
        X_hat_states: torch.Tensor,
        m_features: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        fix_psi_omega: bool = True
    ) -> Dict[str, float]:
        """
        **修正版**: Stage-1学習 + φ_θ勾配更新（ψ_ω完全固定）
        
        資料の学習戦略に対応:
        V_B = 閉形式解(Φ_prev, Ψ_curr)       # V_B計算（ψ_ω固定）
        φ_θ ← φ_θ - α∇L1(V_B, φ_θ)         # φ_θ更新（ψ_ω固定）
        
        Args:
            X_hat_states: DF-Aからの状態予測 (T-1, r)
            m_features: スカラー特徴量 (T,)
            optimizer_phi: φ_θ用オプティマイザ
            fix_psi_omega: ψ_ω を完全固定するか
            
        Returns:
            Dict[str, float]: 損失メトリクス
        """
        if not self.df_state._is_fitted and 'V_A' not in self.df_state._stage1_cache:
            raise RuntimeError("DF-Aが学習済みである必要があります")
        
        # **修正**: ψ_ω パラメータの完全固定
        psi_original_states = {}
        if fix_psi_omega:
            psi_original_states = self._freeze_parameters(self.psi_omega)
        
        try:
            # 時間合わせ
            T_x = X_hat_states.size(0)  # T-1
            m_curr = m_features[1:T_x+1]  # (T-1,)
            
            # 観測特徴量の計算
            if fix_psi_omega:
                # パラメータ固定済みなので、通常の前向き計算
                psi_curr = self.psi_omega(m_curr.unsqueeze(1))  # (T-1, d_B)
            else:
                # ψ_ω の勾配も計算
                psi_curr = self.psi_omega(m_curr.unsqueeze(1))  # (T-1, d_B)
            
            # φ_θ 勾配ありで操作変数特徴量計算
            phi_prev = self.phi_theta(X_hat_states)  # (T-1, d_A)
            
            # V_B 推定
            V_B = self._ridge_stage1_vb(phi_prev, psi_curr, self.lambda_B)
            
            # Stage-1 損失: ||Ψ - V_B Φ||_F^2
            psi_pred = (V_B @ phi_prev.T).T  # (T-1, d_B)
            loss_stage1 = torch.norm(psi_pred - psi_curr, p='fro') ** 2
            
            # φ_θ 更新（ψ_ω は固定済み）
            optimizer_phi.zero_grad()
            loss_stage1.backward()
            
            # **修正**: 固定されたパラメータの勾配をゼロクリア（安全のため）
            if fix_psi_omega:
                for param in self.psi_omega.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            
            optimizer_phi.step()
            
            # V_B をキャッシュ（Stage-2で使用）
            self._stage1_cache['V_B'] = V_B.detach()
            self._stage1_cache['phi_prev'] = phi_prev.detach()
            
            return {'stage1_loss': loss_stage1.item()}
            
        finally:
            # **修正**: ψ_ω パラメータの復元
            if fix_psi_omega and psi_original_states:
                self._restore_parameters(self.psi_omega, psi_original_states)
        
    def train_stage2_with_gradients(
        self,
        m_features: torch.Tensor,
        optimizer_psi: torch.optim.Optimizer,
        fix_phi_theta: bool = True
    ) -> Dict[str, float]:
        """
        **修正版**: Stage-2学習 + ψ_ω勾配更新（φ_θ完全固定）
        
        資料の学習戦略に対応:
        u_B = 閉形式解(H^{(cf)}_B, m)        # u_B計算（φ_θ固定）
        ψ_ω ← ψ_ω - α∇L2(u_B, ψ_ω)         # ψ_ω更新（φ_θ固定）
        
        Args:
            m_features: スカラー特徴量 (T,)
            optimizer_psi: ψ_ω用オプティマイザ
            fix_phi_theta: φ_θ を完全固定するか
            
        Returns:
            Dict[str, float]: 損失メトリクス
        """
        if 'V_B' not in self._stage1_cache:
            raise RuntimeError("Stage-1が先に実行されている必要があります")
        
        # **修正**: φ_θ パラメータの完全固定
        phi_original_states = {}
        if fix_phi_theta:
            phi_original_states = self._freeze_parameters(self.phi_theta)
        
        try:
            # Stage-1からの結果を取得
            V_B = self._stage1_cache['V_B']
            phi_prev = self._stage1_cache['phi_prev']
            T_eff = phi_prev.size(0)
            m_curr = m_features[1:T_eff+1]  # (T-1,)
            
            # 操作変数特徴量の計算
            if fix_phi_theta:
                # パラメータ固定済みなので、通常の前向き計算
                # ただし、V_Bは既にdetach済みなので勾配は流れない
                H = (V_B @ phi_prev.T).T  # (T-1, d_B)
            else:
                # φ_θ の勾配も計算（この場合は不適切だが、フラグに従う）
                phi_prev_grad = self.phi_theta(self._stage1_cache.get('X_hat', phi_prev))
                H = (V_B @ phi_prev_grad.T).T  # (T-1, d_B)
            
            # u_B 推定
            u_B = self._ridge_stage2_ub(H, m_curr, self.lambda_dB)
            
            # Stage-2 損失: ||m - H u_B||_2^2
            m_pred = (H * u_B).sum(dim=1)  # (T-1,)
            loss_stage2 = torch.norm(m_pred - m_curr, p=2) ** 2
            
            # ψ_ω 更新（φ_θ は固定済み）
            optimizer_psi.zero_grad()
            loss_stage2.backward()
            
            # **修正**: 固定されたパラメータの勾配をゼロクリア（安全のため）
            if fix_phi_theta:
                for param in self.phi_theta.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            
            optimizer_psi.step()
            
            # u_B をキャッシュ
            self._stage2_cache['u_B'] = u_B.detach()
            
            return {'stage2_loss': loss_stage2.item()}
            
        finally:
            # **修正**: φ_θ パラメータの復元
            if fix_phi_theta and phi_original_states:
                self._restore_parameters(self.phi_theta, phi_original_states)
    
    def fit_two_stage(
        self, 
        X_hat_states: torch.Tensor,  # DF-Aからの状態予測
        m_features: torch.Tensor,    # エンコーダからのスカラー特徴量
        use_cross_fitting: bool = True,
        verbose: bool = False
    ) -> 'DFObservationLayer':
        """
        従来の2段階クロスフィッティング学習（変更なし）
        
        **注意**: これは既存の学習メソッドです。
        新しいPhase-1学習では train_stage1_with_gradients と 
        train_stage2_with_gradients を使用してください。
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
        1ステップ特徴量予測: m̂_{t|t-1} = u_B^T V_B φ_θ(x̂_{t|t-1})
        
        Args:
            x_hat_prev: 前時刻の状態予測 (r,) または (batch, r)
            
        Returns:
            torch.Tensor: 予測スカラー特徴量 () または (batch,)
        """
        if not self._is_fitted:
            # **修正**: キャッシュされた結果も使用可能に
            if 'V_B' in self._stage1_cache and 'u_B' in self._stage2_cache:
                V_B = self._stage1_cache['V_B']
                u_B = self._stage2_cache['u_B']
            else:
                raise RuntimeError("fit_two_stage() または train_stage1/2_with_gradients() を先に実行してください")
        else:
            V_B = self.V_B
            u_B = self.u_B
        
        # 状態特徴写像（共有φ_θ）
        phi_prev = self.phi_theta(x_hat_prev)
        
        # 観測転送作用素適用 + 読み出し
        if phi_prev.dim() == 1:
            h_pred = V_B @ phi_prev  # (d_B,)
            return u_B @ h_pred      # scalar
        else:
            h_pred = (V_B @ phi_prev.T).T  # (batch, d_B)
            return (h_pred * u_B).sum(dim=1)  # (batch,)
    
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
        if not self._is_fitted and 'V_B' not in self._stage1_cache:
            raise RuntimeError("学習が完了していません")
        
        T = X_hat_states.size(0)
        predictions = []
        
        for t in range(T - 1):
            m_pred = self.predict_one_step(X_hat_states[t])
            predictions.append(m_pred)
        
        return torch.stack(predictions)
    
    def get_observation_operator(self) -> torch.Tensor:
        """観測転送作用素 V_B を取得"""
        if self._is_fitted:
            return self.V_B.clone()
        elif 'V_B' in self._stage1_cache:
            return self._stage1_cache['V_B'].clone()
        else:
            raise RuntimeError("学習が完了していません")
    
    def get_readout_vector(self) -> torch.Tensor:
        """観測読み出しベクトル u_B を取得"""
        if self._is_fitted:
            return self.u_B.clone()
        elif 'u_B' in self._stage2_cache:
            return self._stage2_cache['u_B'].clone()
        else:
            raise RuntimeError("学習が完了していません")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """学習済みパラメータを辞書で取得"""
        state_dict = {
            'psi_omega': self.psi_omega.state_dict(),
            'config': {
                'obs_feature_dim': self.obs_feature_dim,
                'lambda_B': self.lambda_B,
                'lambda_dB': self.lambda_dB
            }
        }
        
        if self._is_fitted:
            state_dict.update({
                'V_B': self.V_B,
                'u_B': self.u_B,
            })
        
        if self._stage1_cache:
            state_dict['stage1_cache'] = self._stage1_cache.copy()
        if self._stage2_cache:
            state_dict['stage2_cache'] = self._stage2_cache.copy()
            
        return state_dict