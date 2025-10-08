# src/ssm/df_state_layer.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import warnings

from .cross_fitting import CrossFittingManager, TwoStageCrossFitter, CrossFittingError


class StateFeatureNet(nn.Module):
    """状態特徴写像 ϕ_θ: R^r → R^{d_A}"""

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
            input_dim: 状態次元r
            output_dim: 特徴次元d_A
            hidden_sizes: 中間層ユニット数
            activation: 活性化関数名
            dropout: ドロップアウト率
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(nn, activation)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

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
            x: 状態 (batch_size, r) or (r,)
        Returns:
            特徴 (batch_size, d_A) or (d_A,)
        """
        if hasattr(self.net, 'to'):
            self.net = self.net.to(x.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.net(x).squeeze(0)
        return self.net(x)


class DFStateLayer(nn.Module):
    """
    DF-A: Deep Feature Instrumental Variable for State Process
    状態系列の1ステップ予測を2SLS+クロスフィッティングで実現。Section 1.4.1対応。
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
            state_dim: 状態次元r
            feature_dim: 特徴次元d_A
            lambda_A: Stage-1正則化λ_A
            lambda_B: Stage-2正則化λ_B
            feature_net_config: StateFeatureNet設定
            cross_fitting_config: CrossFittingManager設定
        """
        super().__init__()
        self.state_dim = int(state_dim)
        self.feature_dim = int(feature_dim)
        self.lambda_A = float(lambda_A)
        self.lambda_B = float(lambda_B)

        feature_config = feature_net_config or {}
        self.phi_theta = StateFeatureNet(
            input_dim=state_dim,
            output_dim=feature_dim,
            **feature_config
        )

        self.cf_config = cross_fitting_config or {'n_blocks': 5, 'min_block_size': 10}

        # 学習済みパラメータ
        self.V_A: Optional[torch.Tensor] = None  # 転送作用素 (d_A, d_A)
        self.U_A: Optional[torch.Tensor] = None  # 読み出し行列 (d_A, r)
        self._is_fitted = False

        # Phase-1学習用内部状態
        self._stage1_cache = {}
        self._stage2_cache = {}
        self._cf_manager: Optional[CrossFittingManager] = None


    def _ridge_stage1(
        self,
        X_features: torch.Tensor,
        Y_targets: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-1 Ridge回帰: V = (Y^T X)(X^T X + λI)^{-1}
        """

        N, d_A = X_features.shape
        N_t, d_A_t = Y_targets.shape

        N = int(N.item() if hasattr(N, 'item') else N)
        d_A = int(d_A.item() if hasattr(d_A, 'item') else d_A)
        N_t = int(N_t.item() if hasattr(N_t, 'item') else N_t)
        d_A_t = int(d_A_t.item() if hasattr(d_A_t, 'item') else d_A_t)

        if N != N_t:
            raise ValueError(f"特徴量とターゲットのサンプル数不一致: {N} vs {N_t}")

        if d_A != d_A_t:
            raise ValueError(f"特徴量とターゲットの次元不一致: {d_A} vs {d_A_t}")

        if N < d_A:
            warnings.warn(f"サンプル数 {N} < 特徴次元 {d_A}。数値不安定の可能性")

        XtX = X_features.T @ X_features
        XtX_reg = XtX + reg_lambda * torch.eye(d_A).type_as(XtX).to(XtX.device)

        YtX = Y_targets.T @ X_features

        original_device = X_features.device
        try:
            XtX_inv = torch.linalg.inv(XtX_reg)
            V = YtX @ XtX_inv
        except torch.linalg.LinAlgError:
            try:
                L = torch.linalg.cholesky(XtX_reg)
                XtX_inv = torch.cholesky_inverse(L)
                V = YtX @ XtX_inv
            except torch.linalg.LinAlgError:
                U, S, Vh = torch.linalg.svd(XtX_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                XtX_inv = (Vh.T * S_inv) @ Vh
                V = YtX @ XtX_inv

        V = V.to(original_device)
        return V

    def _ridge_stage1_with_grad(
        self,
        X_features: torch.Tensor,
        Y_targets: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """Stage-1 Ridge回帰(勾配あり): φ_θ更新用"""
        N, d_A = X_features.shape
        N_t, d_A_t = Y_targets.shape

        if N != N_t:
            raise ValueError(f"特徴量とターゲットのサンプル数不一致: {N} vs {N_t}")

        if d_A != d_A_t:
            raise ValueError(f"特徴量とターゲットの次元不一致: {d_A} vs {d_A_t}")

        XtX = X_features.T @ X_features
        XtX_reg = XtX + reg_lambda * torch.eye(d_A, device=X_features.device, dtype=X_features.dtype)

        YtX = Y_targets.T @ X_features

        original_device = X_features.device
        try:
            XtX_inv = torch.linalg.inv(XtX_reg)
            V = YtX @ XtX_inv
        except torch.linalg.LinAlgError:
            XtX_inv = torch.linalg.pinv(XtX_reg)
            V = YtX @ XtX_inv

        V = V.to(original_device)
        return V

    def _ridge_stage2(
        self,
        H_features: torch.Tensor,
        X_targets: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-2 Ridge回帰: U = (H H^T + λI)^{-1} H X^T

        Args:
            H_features: クロスフィット特徴量 (N, d_A)
            X_targets: 目標状態 (N, r)
            reg_lambda: 正則化パラメータλ_B

        Returns:
            読み出し行列U (d_A, r)
        """
        N, d_A = H_features.shape
        N_t, r = X_targets.shape

        if N != N_t:
            raise ValueError(f"特徴量とターゲットのサンプル数不一致: {N} vs {N_t}")

        HHt = H_features.T @ H_features
        HHt_reg = HHt + reg_lambda * torch.eye(d_A, device=H_features.device, dtype=H_features.dtype)

        HXt = H_features.T @ X_targets

        original_device = H_features.device
        try:
            HHt_inv = torch.linalg.inv(HHt_reg)
            U = HHt_inv @ HXt
        except torch.linalg.LinAlgError:
            try:
                L = torch.linalg.cholesky(HHt_reg)
                HHt_inv = torch.cholesky_inverse(L)
                U = HHt_inv @ HXt
            except torch.linalg.LinAlgError:
                U_svd, S, Vh = torch.linalg.svd(HHt_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                HHt_inv = (Vh.T * S_inv) @ Vh
                U = HHt_inv @ HXt

        U = U.to(original_device)
        return U

    def _initialize_cross_fitting(self, T_eff: int) -> CrossFittingManager:
        """クロスフィッティング管理初期化"""
        cf_config = self.cf_config.copy()

        min_block_size = cf_config.get('min_block_size', 10)
        max_blocks = T_eff // min_block_size
        n_blocks = min(cf_config.get('n_blocks', 5), max_blocks)

        if n_blocks < 2:
            warnings.warn(f"データサイズ {T_eff} が小さすぎるため、クロスフィッティングを無効化")
            return None

        cf_config['n_blocks'] = n_blocks

        return CrossFittingManager(T_eff, **cf_config)

    def _compute_crossfit_stage1_loss(
        self,
        X_states: torch.Tensor,
        use_simple_fallback: bool = False
    ) -> torch.Tensor:
        """クロスフィッティング対応Stage-1損失計算"""
        T, r = X_states.shape

        phi_seq = self.phi_theta(X_states)

        phi_minus = phi_seq[:-1]
        phi_plus = phi_seq[1:]

        T_eff = phi_minus.size(0)

        if use_simple_fallback or T_eff < 20:
            V_A = self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            loss = torch.norm(phi_pred - phi_plus, p='fro') ** 2 / phi_plus.numel()

            self._stage1_cache.pop('V_A_list', None)
            self._stage1_cache.pop('cf_manager', None)
        else:
            cf_manager = self._initialize_cross_fitting(T_eff)

            if cf_manager is None:
                V_A = self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)
                phi_pred = (V_A @ phi_minus.T).T
                loss = torch.norm(phi_pred - phi_plus, p='fro') ** 2 / phi_plus.numel()

                self._stage1_cache.pop('V_A_list', None)
                self._stage1_cache.pop('cf_manager', None)
            else:
                cf_fitter = TwoStageCrossFitter(cf_manager)

                with torch.no_grad():
                    V_list = cf_fitter.cross_fit_stage1(
                        phi_minus, phi_plus,
                        self._ridge_stage1,
                        reg_lambda=self.lambda_A
                    )

                total_loss = 0.0
                for k in range(cf_manager.n_blocks):
                    block_indices = cf_manager.get_block_indices(k)

                    phi_minus_k = phi_minus[block_indices]
                    phi_plus_k = phi_plus[block_indices]

                    V_k = V_list[k]
                    phi_pred_k = (V_k @ phi_minus_k.T).T

                    loss_k = torch.norm(phi_pred_k - phi_plus_k, p='fro') ** 2 / phi_plus_k.numel()
                    total_loss += loss_k

                loss = total_loss / cf_manager.n_blocks

                self._stage1_cache['V_A_list'] = V_list
                self._stage1_cache['cf_manager'] = cf_manager

        return loss

    def train_stage1_with_gradients(
        self,
        X_states: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        T1_iterations: int = 1
    ) -> Dict[str, float]:
        """Stage-1学習 + φ_θ勾配更新(retain_graph対応)"""
        if X_states.size(0) < 2:
            raise ValueError(f"状態系列が短すぎます: T={X_states.size(0)}")

        total_loss = 0.0

        for t in range(T1_iterations):
            optimizer_phi.zero_grad()

            phi_seq = self.phi_theta(X_states)

            phi_minus = phi_seq[:-1]
            phi_plus = phi_seq[1:]

            phi_pred, V_A = self._compute_cross_fitting_prediction(phi_minus, phi_plus)

            prediction_loss = torch.norm(phi_pred - phi_plus, p='fro') ** 2 / phi_plus.numel()
            regularization_loss = self.lambda_A * torch.norm(V_A, p='fro') ** 2
            loss_stage1 = prediction_loss + regularization_loss

            total_loss += loss_stage1.item()

            if t < T1_iterations - 1:
                loss_stage1.backward(retain_graph=True)
            else:
                loss_stage1.backward()

            optimizer_phi.step()

            if t < T1_iterations - 1:
                phi_seq = phi_seq.detach()

        # Stage-1結果キャッシュ
        with torch.no_grad():
            phi_seq_final = self.phi_theta(X_states)
            phi_minus_final = phi_seq_final[:-1]
            phi_plus_final = phi_seq_final[1:]
            _, V_A_final = self._compute_cross_fitting_prediction(phi_minus_final, phi_plus_final)

            self._stage1_cache = {
                'V_A': V_A_final.detach(),
                'phi_minus': phi_minus_final.detach(),
                'phi_plus': phi_plus_final.detach(),
                'X_plus': X_states[1:].detach()
            }

            if hasattr(self, '_cross_fitting_cache'):
                self._stage1_cache.update({
                    'V_A_list': [V.detach() for V in self._cross_fitting_cache['V_A_list']],
                    'cf_manager': self._cross_fitting_cache['cf_manager']
                })

        return {
            'stage1_loss': total_loss / T1_iterations,
            'iterations_completed': T1_iterations,
            'final_loss': loss_stage1.item()
        }

    def _compute_cross_fitting_prediction_ua_matrix(
        self,
        H_features: torch.Tensor,
        X_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """U_A用クロスフィッティングでout-of-fold予測計算"""
        T_eff = H_features.size(0)

        n_blocks = self.cf_config.get('n_blocks', 6)
        min_block_size = self.cf_config.get('min_block_size', 20)

        if T_eff < max(n_blocks * min_block_size, 100):
            U_A = self._ridge_stage2(H_features, X_targets, self.lambda_B)
            X_pred = (U_A.T @ H_features.T).T
            return X_pred, U_A

        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            U_A_list = []
            for k in range(cf_manager.n_blocks):
                oof_indices = cf_manager.get_out_of_fold_indices(k)
                H_oof = H_features[oof_indices]
                X_oof = X_targets[oof_indices]

                U_A_k = self._ridge_stage2(H_oof, X_oof, self.lambda_B)
                U_A_list.append(U_A_k)

            X_pred_cf = torch.zeros_like(X_targets)
            for k in range(cf_manager.n_blocks):
                block_indices = cf_manager.get_block_indices(k)
                H_block = H_features[block_indices]

                X_pred_cf[block_indices] = (U_A_list[k].T @ H_block.T).T

            U_A_final = self._ridge_stage2(H_features, X_targets, self.lambda_B)

            return X_pred_cf, U_A_final

        except Exception as e:
            print(f"クロスフィッティング失敗、従来方式を使用: {e}")
            U_A = self._ridge_stage2(H_features, X_targets, self.lambda_B)
            X_pred = (U_A.T @ H_features.T).T
            return X_pred, U_A

    def train_stage2_with_gradients(
        self,
        X_states: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Stage-2学習 + φ_θ勾配更新"""
        if 'X_plus' not in self._stage1_cache:
            raise RuntimeError("Stage-1が先に実行されている必要があります")

        X_plus = self._stage1_cache['X_plus']

        phi_seq = self.phi_theta(X_states)
        phi_minus = phi_seq[:-1]
        phi_plus = phi_seq[1:]

        V_A_current = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)

        H = (V_A_current @ phi_minus.T).T

        X_pred, U_A = self._compute_cross_fitting_prediction_ua_matrix(H, X_plus)

        prediction_loss = torch.norm(X_pred - X_plus, p='fro') ** 2 / X_plus.numel()
        regularization_loss = self.lambda_B * torch.norm(U_A, p='fro') ** 2
        loss_stage2 = prediction_loss + regularization_loss

        optimizer_phi.zero_grad()
        loss_stage2.backward()
        optimizer_phi.step()

        self._stage2_cache['U_A'] = U_A.detach()

        return {'stage2_loss': loss_stage2.item()}

    def fit_two_stage(
        self,
        X_states: torch.Tensor,
        use_cross_fitting: bool = True,
        verbose: bool = False
    ) -> 'DFStateLayer':
        """従来の2段階クロスフィッティング学習"""
        T, r = X_states.shape

        if r != self.state_dim:
            raise ValueError(f"状態次元不一致: expected {self.state_dim}, got {r}")

        if T < 2:
            raise ValueError(f"時系列が短すぎます: T={T}")

        with torch.no_grad():
            phi_seq = self.phi_theta(X_states)

        Phi_minus = phi_seq[:-1]
        Phi_plus = phi_seq[1:]
        X_plus = X_states[1:]

        if use_cross_fitting and T >= 20:
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
        T_eff = int(Phi_minus.size(0))

        cf_manager = CrossFittingManager(T_eff, **self.cf_config)
        cf_fitter = TwoStageCrossFitter(cf_manager)

        if verbose:
            print(f"クロスフィッティング: T={T_eff}, n_blocks={cf_manager.n_blocks}")

        V_list = cf_fitter.cross_fit_stage1(
            Phi_minus, Phi_plus,
            self._ridge_stage1,
            reg_lambda=self.lambda_A
        )

        self.V_A = torch.stack(V_list).mean(dim=0)

        H_cf = cf_fitter.compute_out_of_fold_features(Phi_minus, V_list)

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
        """クロスフィッティングなし学習(小データ用)"""
        if verbose:
            print("クロスフィッティングなしで学習")

        self.V_A = self._ridge_stage1(Phi_minus, Phi_plus, self.lambda_A)

        H = (self.V_A @ Phi_minus.T).T

        self.U_A = self._ridge_stage2(H, X_plus, self.lambda_B)

        if verbose:
            print(f"V_A shape: {self.V_A.shape}, U_A shape: {self.U_A.shape}")

    def _compute_cross_fitting_prediction(
        self,
        phi_minus: torch.Tensor,
        phi_plus: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """クロスフィッティングでout-of-fold予測計算"""
        T_eff = phi_minus.size(0)

        n_blocks = self.cf_config.get('n_blocks', 6)
        min_block_size = self.cf_config.get('min_block_size', 20)

        if T_eff < max(n_blocks * min_block_size, 100):
            V_A = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            return phi_pred, V_A

        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            V_A_list = cf_fitter.cross_fit_stage1(
                phi_minus, phi_plus,
                stage1_estimator=lambda X, Y: self._ridge_stage1_with_grad(X, Y, self.lambda_A)
            )

            phi_pred_cf = cf_fitter.compute_out_of_fold_features(phi_minus, V_A_list)

            V_A_final = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)

            if not hasattr(self, '_cross_fitting_cache'):
                self._cross_fitting_cache = {}
            self._cross_fitting_cache.update({
                'V_A_list': V_A_list,
                'cf_manager': cf_manager
            })

            return phi_pred_cf, V_A_final

        except ImportError:
            V_A = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            return phi_pred, V_A
        except Exception as e:
            print(f"クロスフィッティング失敗、従来方式を使用: {e}")
            V_A = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            return phi_pred, V_A

    def _compute_V_A_with_cross_fitting(
        self,
        phi_minus: torch.Tensor,
        phi_plus: torch.Tensor
    ) -> torch.Tensor:
        """クロスフィッティングV_A計算(閉形式解)"""
        T_eff = phi_minus.size(0)

        n_blocks = self.cf_config.get('n_blocks', 6)
        min_block_size = self.cf_config.get('min_block_size', 20)

        if T_eff < max(n_blocks * min_block_size, 100):
            return self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)

        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            V_A_list = cf_fitter.cross_fit_stage1(
                phi_minus, phi_plus,
                stage1_estimator=lambda X, Y: self._ridge_stage1(X, Y, self.lambda_A)
            )

            V_A = self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)

            if not hasattr(self, '_cross_fitting_cache'):
                self._cross_fitting_cache = {}
            self._cross_fitting_cache.update({
                'V_A_list': V_A_list,
                'cf_manager': cf_manager
            })

            return V_A

        except ImportError:
            return self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)
        except Exception as e:
            print(f"クロスフィッティング失敗、従来方式を使用: {e}")
            return self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)

    def apply_transfer_operator(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """転送作用素適用: φ̂_{t|t-1} = V_A φ_{t-1}"""
        if not self._is_fitted:
            if 'V_A' in self._stage1_cache:
                V_A = self._stage1_cache['V_A']
            else:
                raise RuntimeError("fit_two_stage() または train_stage1_with_gradients() を先に実行してください")
        else:
            V_A = self.V_A

        V_A = V_A.to(phi_prev.device)

        if phi_prev.dim() == 1:
            return V_A @ phi_prev
        else:
            return (V_A @ phi_prev.T).T

    def predict_one_step(self, x_prev: torch.Tensor) -> torch.Tensor:
        """1ステップ状態予測: x̂_{t|t-1} = U_A^T V_A ϕ_θ(x_{t-1})"""
        V_A = None
        U_A = None

        if self._is_fitted:
            V_A = self.V_A
            U_A = self.U_A
        elif 'V_A' in self._stage1_cache and 'U_A' in self._stage2_cache:
            V_A = self._stage1_cache['V_A']
            U_A = self._stage2_cache['U_A']
        elif 'V_A' in self._stage1_cache:
            V_A = self._stage1_cache['V_A']
            if 'phi_minus' in self._stage1_cache and 'X_plus' in self._stage1_cache:
                with torch.no_grad():
                    phi_minus = self._stage1_cache['phi_minus']
                    X_plus = self._stage1_cache['X_plus']
                    H_simple = (V_A @ phi_minus.T).T
                    U_A = self._ridge_stage2(H_simple, X_plus, self.lambda_B)
                    self._stage2_cache['U_A'] = U_A.detach()
            else:
                raise RuntimeError("Stage-1は完了していますが、Stage-2実行に必要なデータが不足しています")
        else:
            raise RuntimeError("学習が完了していません。fit_two_stage() または train_stage1_with_gradients() を先に実行してください")

        phi_prev = self.phi_theta(x_prev)

        phi_pred = self.apply_transfer_operator(phi_prev)

        U_A = U_A.to(phi_pred.device)
        if phi_pred.dim() == 1:
            return U_A.T @ phi_pred
        else:
            return (U_A.T @ phi_pred.T).T

    def predict_sequence(
        self,
        X_states: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """系列予測: 各時刻one-step-ahead予測"""
        if not self._is_fitted and 'V_A' not in self._stage1_cache:
            raise RuntimeError("学習が完了していません")

        T = X_states.size(0)
        predictions = []
        features = []

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
        """転送作用素V_A取得"""
        if self._is_fitted:
            return self.V_A.clone()
        elif 'V_A' in self._stage1_cache:
            return self._stage1_cache['V_A'].clone()
        else:
            raise RuntimeError("学習が完了していません")

    def get_readout_matrix(self) -> torch.Tensor:
        """読み出し行列U_A取得"""
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

    def get_inference_state_dict(self) -> Dict[str, Any]:
        """推論用state_dict取得(filtering評価用にV_A/U_A含む)"""
        state_dict = {
            'phi_theta': self.phi_theta.state_dict(),
        }

        if hasattr(self, 'V_A') and self.V_A is not None:
            state_dict['V_A'] = self.V_A
        if hasattr(self, 'U_A') and self.U_A is not None:
            state_dict['U_A'] = self.U_A

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """カスタムload_state_dict: V_A/U_Aも適切に設定"""
        v_a = state_dict.pop('V_A', None)
        u_a = state_dict.pop('U_A', None)

        super().load_state_dict(state_dict, strict=strict)

        if v_a is not None:
            self.V_A = v_a.to(self.device) if hasattr(self, 'device') else v_a
        if u_a is not None:
            self.U_A = u_a.to(self.device) if hasattr(self, 'device') else u_a
