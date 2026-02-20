# src/ssm/df_state_layer.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import warnings

from .cross_fitting import CrossFittingManager, TwoStageCrossFitter, CrossFittingError


class StateFeatureNet(nn.Module):
    """
    State feature map phi_theta: R^r -> R^{d_A}.

    Neural network mapping state variables to a high-dimensional feature space.
    Shared between DF-A and DF-B.
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
            input_dim: State dimension r
            output_dim: Feature dimension d_A
            hidden_sizes: List of hidden layer widths
            activation: Activation function name
            dropout: Dropout rate
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
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State (batch_size, r) or (r,)

        Returns:
            torch.Tensor: Features (batch_size, d_A) or (d_A,)
        """
        if hasattr(self.net, 'to'):
            self.net = self.net.to(x.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.net(x).squeeze(0)
        return self.net(x)


class DFStateLayer(nn.Module):
    """
    DF-A: Deep Feature Instrumental Variable for State Process.

    Corresponds to Section 1.4.1 in the paper. Implements one-step prediction
    from state sequences via two-stage regression (2SLS) with cross-fitting.

    Computation flow:
    1. Map states to feature space via phi_theta(x_t)
    2. Stage-1: Estimate V_A^{(-k)} (cross-fitting) + update phi_theta gradients
    3. Stage-2: Estimate U_A (closed-form solution only)
    4. Predict: x_hat_{t|t-1} = U_A^T V_A phi_theta(x_{t-1})
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
            state_dim: State dimension r
            feature_dim: Feature dimension d_A
            lambda_A: Stage-1 regularization parameter lambda_A
            lambda_B: Stage-2 regularization parameter lambda_B
            feature_net_config: Configuration for StateFeatureNet
            cross_fitting_config: Configuration for CrossFittingManager
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

        # Fitted parameters
        self.V_A: Optional[torch.Tensor] = None  # Transfer operator (d_A, d_A)
        self.U_A: Optional[torch.Tensor] = None  # Readout matrix (d_A, r)
        self._is_fitted = False

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
        Stage-1 Ridge regression: transfer operator estimation.

        V = (Y^T X)(X^T X + lambda I)^{-1}
        """
        N, d_A = X_features.shape
        N_t, d_A_t = Y_targets.shape

        N = int(N.item() if hasattr(N, 'item') else N)
        d_A = int(d_A.item() if hasattr(d_A, 'item') else d_A)
        N_t = int(N_t.item() if hasattr(N_t, 'item') else N_t)
        d_A_t = int(d_A_t.item() if hasattr(d_A_t, 'item') else d_A_t)

        if N != N_t:
            raise ValueError(f"Feature-target sample count mismatch: {N} vs {N_t}")

        if d_A != d_A_t:
            raise ValueError(f"Feature-target dimension mismatch: {d_A} vs {d_A_t}")

        if N < d_A:
            warnings.warn(f"Sample count {N} < feature dimension {d_A}. May be numerically unstable")

        # Gram matrix + regularization
        XtX = X_features.T @ X_features
        XtX_reg = XtX + reg_lambda * torch.eye(d_A).type_as(XtX).to(XtX.device)

        # Cross-covariance
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
        """
        Stage-1 Ridge regression (with gradients): for phi_theta updates.

        V = (Y^T X)(X^T X + lambda I)^{-1}
        """
        N, d_A = X_features.shape
        N_t, d_A_t = Y_targets.shape

        if N != N_t:
            raise ValueError(f"Feature-target sample count mismatch: {N} vs {N_t}")

        if d_A != d_A_t:
            raise ValueError(f"Feature-target dimension mismatch: {d_A} vs {d_A_t}")

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
        Stage-2 Ridge regression: readout matrix estimation.

        U = (H H^T + lambda I)^{-1} H X^T

        Args:
            H_features: Cross-fitted features H^{(cf)}_A (N, d_A)
            X_targets: Target states X^+ (N, r)
            reg_lambda: Regularization parameter lambda_B

        Returns:
            torch.Tensor: Readout matrix U (d_A, r)
        """
        N, d_A = H_features.shape
        N_t, r = X_targets.shape

        if N != N_t:
            raise ValueError(f"Feature-target sample count mismatch: {N} vs {N_t}")

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
        """
        Initialize cross-fitting manager.

        Args:
            T_eff: Effective time series length

        Returns:
            CrossFittingManager: Initialized cross-fitting manager, or None if data is too small
        """
        cf_config = self.cf_config.copy()

        min_block_size = cf_config.get('min_block_size', 10)
        max_blocks = T_eff // min_block_size
        n_blocks = min(cf_config.get('n_blocks', 5), max_blocks)

        if n_blocks < 2:
            warnings.warn(f"Data size {T_eff} too small; disabling cross-fitting")
            return None
        
        cf_config['n_blocks'] = n_blocks
        
        return CrossFittingManager(T_eff, **cf_config)
    
    def _compute_crossfit_stage1_loss(
        self,
        X_states: torch.Tensor,
        use_simple_fallback: bool = False
    ) -> torch.Tensor:
        """
        Compute Stage-1 loss with cross-fitting support.

        Args:
            X_states: State sequence (T, r)
            use_simple_fallback: Whether to use simplified (non-cross-fitting) version

        Returns:
            torch.Tensor: Stage-1 loss (scalar)
        """
        T, r = X_states.shape

        phi_seq = self.phi_theta(X_states)

        phi_minus = phi_seq[:-1]
        phi_plus = phi_seq[1:]

        T_eff = phi_minus.size(0)

        if use_simple_fallback or T_eff < 20:
            # Small data or simple fallback: estimate on all data
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

                # Estimate V_A^{(-k)} without gradients
                with torch.no_grad():
                    V_list = cf_fitter.cross_fit_stage1(
                        phi_minus, phi_plus,
                        self._ridge_stage1,
                        reg_lambda=self.lambda_A
                    )

                # Compute out-of-fold prediction error (with gradients)
                total_loss = 0.0
                for k in range(cf_manager.n_blocks):
                    block_indices = cf_manager.get_block_indices(k)

                    phi_minus_k = phi_minus[block_indices]
                    phi_plus_k = phi_plus[block_indices]

                    V_k = V_list[k]  # No gradient (detached)
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
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Stage-1 training (one block per epoch).

        Theory (Equation 42a):
        L_Stage-1(theta) = sum_{t in B_k} ||phi_theta(x_t) - V_A^{(-k)} phi_theta(x_{t-1})||^2 + lambda_A ||V_A^{(-k)}||^2_F

        - Epoch e processes block k = (e mod K)
        - One epoch = one parameter update
        - K epochs complete one full pass over all blocks

        Args:
            X_states: State sequence (T, r)
            optimizer_phi: Optimizer for phi_theta
            epoch: Epoch number

        Returns:
            Dict[str, float]: Loss metrics
        """
        if X_states.size(0) < 2:
            raise ValueError(f"State sequence too short: T={X_states.size(0)}")

        optimizer_phi.zero_grad()

        phi_seq = self.phi_theta(X_states)

        phi_minus = phi_seq[:-1]
        phi_plus = phi_seq[1:]

        T_eff = phi_minus.size(0)

        n_blocks = self.cf_config.get('n_blocks', 5)
        min_block_size = self.cf_config.get('min_block_size', 20)

        if T_eff < max(n_blocks * min_block_size, 100):
            # Small data: no cross-fitting (full-data Ridge)
            V_A = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T

            prediction_loss = torch.norm(phi_pred - phi_plus, p='fro') ** 2 / phi_plus.size(0)
            regularization_loss = self.lambda_A * torch.norm(V_A, p='fro') ** 2
            loss_stage1 = prediction_loss + regularization_loss

            loss_stage1.backward()

            optimizer_phi.step()

            # Update cache
            with torch.no_grad():
                self._stage1_cache = {
                    'V_A': V_A,
                    'phi_minus': phi_minus,
                    'phi_plus': phi_plus,
                    'X_plus': X_states[1:]
                }

            return {
                'stage1_loss': loss_stage1.item(),
                'stage1_pred_loss': prediction_loss.item(),
                'stage1_reg_loss': regularization_loss.item(),
                'n_blocks': 0,
                'mode': 'no_crossfitting'
            }

        # Cross-fitting execution
        from .cross_fitting import CrossFittingManager

        cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)

        k = epoch % cf_manager.n_blocks
        optimizer_phi.zero_grad()

        # Compute features with current phi_theta
        phi_seq = self.phi_theta(X_states)
        phi_minus = phi_seq[:-1]
        phi_plus = phi_seq[1:]

        # Out-of-fold indices
        oof_indices = cf_manager.get_out_of_fold_indices(k)
        phi_minus_oof = phi_minus[oof_indices]
        phi_plus_oof = phi_plus[oof_indices]

        # Compute V_A^{(-k)}
        V_A_k = self._ridge_stage1_with_grad(phi_minus_oof, phi_plus_oof, self.lambda_A)

        # In-fold indices
        block_indices = cf_manager.get_block_indices(k)
        phi_minus_block = phi_minus[block_indices]
        phi_plus_block = phi_plus[block_indices]

        phi_pred_block = (V_A_k @ phi_minus_block.T).T

        # Block k loss
        prediction_loss_k = torch.norm(phi_pred_block - phi_plus_block, p='fro') ** 2 / phi_plus_block.size(0)
        regularization_loss_k = self.lambda_A * torch.norm(V_A_k, p='fro') ** 2
        loss_k = prediction_loss_k + regularization_loss_k

        loss_k.backward()
        optimizer_phi.step()

        # Update cache
        with torch.no_grad():
            phi_seq_final = self.phi_theta(X_states)
            phi_minus_final = phi_seq_final[:-1]
            phi_plus_final = phi_seq_final[1:]
            V_A_final = self._ridge_stage1(phi_minus_final, phi_plus_final, self.lambda_A)

            self._stage1_cache = {
                'V_A': V_A_final,
                'phi_minus': phi_minus_final,
                'phi_plus': phi_plus_final,
                'X_plus': X_states[1:]
            }

        return {
            'stage1_loss': loss_k.item(),
            'stage1_pred_loss': prediction_loss_k.item(),
            'stage1_reg_loss': regularization_loss_k.item(),
            'current_block': k,
            'n_blocks': cf_manager.n_blocks
        }
    
    def _compute_cross_fitting_prediction_ua_matrix(
        self,
        H_features: torch.Tensor,
        X_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute out-of-fold predictions for U_A via cross-fitting.

        Args:
            H_features: Instrumental variable features (T-1, d_A)
            X_targets: State targets (T-1, r)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X_pred_cf, U_A_final)
                - X_pred_cf: Out-of-fold predictions (T-1, r)
                - U_A_final: Final U_A matrix (d_A, r)
        """
        T_eff = H_features.size(0)

        n_blocks = self.cf_config.get('n_blocks', 6)
        min_block_size = self.cf_config.get('min_block_size', 20)

        if T_eff < max(n_blocks * min_block_size, 100):
            # Insufficient data: full-data Ridge regression
            U_A = self._ridge_stage2(H_features, X_targets, self.lambda_B)
            X_pred = (U_A.T @ H_features.T).T
            return X_pred, U_A

        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            # Build U_A^{(-k)} list (out-of-fold)
            U_A_list = []
            for k in range(cf_manager.n_blocks):
                oof_indices = cf_manager.get_out_of_fold_indices(k)
                H_oof = H_features[oof_indices]
                X_oof = X_targets[oof_indices]

                U_A_k = self._ridge_stage2(H_oof, X_oof, self.lambda_B)
                U_A_list.append(U_A_k)

            # Out-of-fold prediction (with gradients)
            X_pred_cf = torch.zeros_like(X_targets)
            for k in range(cf_manager.n_blocks):
                block_indices = cf_manager.get_block_indices(k)
                H_block = H_features[block_indices]

                X_pred_cf[block_indices] = (U_A_list[k].T @ H_block.T).T

            # Final U_A: full-data estimate
            U_A_final = self._ridge_stage2(H_features, X_targets, self.lambda_B)

            return X_pred_cf, U_A_final

        except Exception as e:
            print(f"Cross-fitting failed, using standard method: {e}")
            U_A = self._ridge_stage2(H_features, X_targets, self.lambda_B)
            X_pred = (U_A.T @ H_features.T).T
            return X_pred, U_A

    def train_stage2_with_gradients(
        self,
        X_states: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Stage-2 training (one block per epoch).

        Theory (Equation 42b):
        L_Stage-2(theta) = sum_{t in B_k} ||x_t - U_A^{(-k)}^T H_k||^2 + lambda_B ||U_A^{(-k)}||^2_F

        - Epoch e processes block k = (e mod K)
        - V_A is dynamically recomputed (gradients flow through phi_theta)
        - One epoch = one parameter update

        Args:
            X_states: State sequence (T, r)
            optimizer_phi: Optimizer for phi_theta
            epoch: Epoch number

        Returns:
            Dict[str, float]: Loss metrics
        """
        if 'X_plus' not in self._stage1_cache:
            raise RuntimeError("Stage-1 must be executed first")

        X_plus = self._stage1_cache['X_plus']  # (T-1, r)

        optimizer_phi.zero_grad()

        # Dynamically compute features with gradients
        phi_seq = self.phi_theta(X_states)
        phi_minus = phi_seq[:-1]
        phi_plus = phi_seq[1:]

        # Recompute V_A dynamically (gradients flow through phi_theta)
        V_A_current = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)

        H = (V_A_current @ phi_minus.T).T
        T_eff = H.size(0)

        n_blocks = self.cf_config.get('n_blocks', 5)
        min_block_size = self.cf_config.get('min_block_size', 20)

        if T_eff < max(n_blocks * min_block_size, 100):
            # Small data: no cross-fitting
            U_A = self._ridge_stage2(H, X_plus, self.lambda_B)
            X_pred = (U_A.T @ H.T).T

            prediction_loss = torch.norm(X_pred - X_plus, p='fro') ** 2 / X_plus.size(0)
            regularization_loss = self.lambda_B * torch.norm(U_A, p='fro') ** 2
            loss_stage2 = prediction_loss + regularization_loss

            loss_stage2.backward()
            optimizer_phi.step()

            self._stage2_cache['U_A'] = U_A.detach()
            return {
                'stage2_loss': loss_stage2.item(),
                'stage2_pred_loss': prediction_loss.item(),
                'stage2_reg_loss': regularization_loss.item(),
                'n_blocks': 0,
                'mode': 'no_crossfitting'
            }

        # Cross-fitting execution
        from .cross_fitting import CrossFittingManager
        cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)

        k = epoch % cf_manager.n_blocks

        optimizer_phi.zero_grad()

        # Compute H with current phi_theta
        phi_seq = self.phi_theta(X_states)
        phi_minus = phi_seq[:-1]
        phi_plus = phi_seq[1:]
        V_A_current = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
        H = (V_A_current @ phi_minus.T).T

        # Compute U_A^{(-k)} on out-of-fold data
        oof_indices = cf_manager.get_out_of_fold_indices(k)
        U_A_k = self._ridge_stage2(H[oof_indices], X_plus[oof_indices], self.lambda_B)

        # Predict on in-fold block
        block_indices = cf_manager.get_block_indices(k)
        X_pred_k = (U_A_k.T @ H[block_indices].T).T

        # Block k loss
        pred_loss_k = torch.norm(X_pred_k - X_plus[block_indices], p='fro') ** 2 / X_plus[block_indices].size(0)
        reg_loss_k = self.lambda_B * torch.norm(U_A_k, p='fro') ** 2
        loss_k = pred_loss_k + reg_loss_k

        loss_k.backward()

        optimizer_phi.step()

        # Update inference cache (recompute on full data to reflect phi_theta updates)
        with torch.no_grad():
            phi_seq_final = self.phi_theta(X_states)
            phi_minus_final = phi_seq_final[:-1]
            phi_plus_final = phi_seq_final[1:]
            V_A_final = self._ridge_stage1(phi_minus_final, phi_plus_final, self.lambda_A)
            H_final = (V_A_final @ phi_minus_final.T).T
            U_A_final = self._ridge_stage2(H_final, X_plus, self.lambda_B)
            self._stage2_cache['U_A'] = U_A_final

        return {
            'stage2_loss': loss_k.item(),
            'stage2_pred_loss': pred_loss_k.item(),
            'stage2_reg_loss': reg_loss_k.item(),
            'current_block': k,
            'n_blocks': cf_manager.n_blocks
        }
    
    def fit_two_stage(
        self,
        X_states: torch.Tensor,
        use_cross_fitting: bool = True,
        verbose: bool = False
    ) -> 'DFStateLayer':
        """
        Legacy two-stage cross-fitting training.

        Note: For Phase-1 training, use train_stage1_with_gradients
        and train_stage2_with_gradients instead.
        """
        T, r = X_states.shape

        if r != self.state_dim:
            raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {r}")

        if T < 2:
            raise ValueError(f"Time series too short: T={T}")

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
        """Training with cross-fitting."""
        T_eff = int(Phi_minus.size(0))

        cf_manager = CrossFittingManager(T_eff, **self.cf_config)
        cf_fitter = TwoStageCrossFitter(cf_manager)

        if verbose:
            print(f"Cross-fitting: T={T_eff}, n_blocks={cf_manager.n_blocks}")

        # Stage-1: Transfer operator estimation (cross-fitting)
        V_list = cf_fitter.cross_fit_stage1(
            Phi_minus, Phi_plus,
            self._ridge_stage1,
            reg_lambda=self.lambda_A
        )
        
        # Average transfer operator (final V_A)
        self.V_A = torch.stack(V_list).mean(dim=0)

        # Out-of-fold feature computation
        H_cf = cf_fitter.compute_out_of_fold_features(Phi_minus, V_list)
        
        # Stage-2: Readout matrix estimation
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
        """Training without cross-fitting (for small data)."""
        if verbose:
            print("Training without cross-fitting")

        # Stage-1: Direct estimation
        self.V_A = self._ridge_stage1(Phi_minus, Phi_plus, self.lambda_A)
        
        # Intermediate features
        H = (self.V_A @ Phi_minus.T).T

        # Stage-2: Readout estimation
        self.U_A = self._ridge_stage2(H, X_plus, self.lambda_B)
        
        if verbose:
            print(f"V_A shape: {self.V_A.shape}, U_A shape: {self.U_A.shape}")
    
    def _compute_cross_fitting_prediction(
        self,
        phi_minus: torch.Tensor,
        phi_plus: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute out-of-fold predictions via cross-fitting.

        Args:
            phi_minus: Past features (T-1, d_A)
            phi_plus: Future features (T-1, d_A)

        Returns:
            tuple: (phi_pred_cf, V_A_final)
                - phi_pred_cf: Out-of-fold predictions (T-1, d_A)
                - V_A_final: Final V_A matrix (d_A, d_A)
        """
        T_eff = phi_minus.size(0)

        n_blocks = self.cf_config.get('n_blocks', 6)
        min_block_size = self.cf_config.get('min_block_size', 20)

        if T_eff < max(n_blocks * min_block_size, 100):
            # Insufficient data: full-data Ridge regression (with gradients)
            V_A = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            return phi_pred, V_A

        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            # V_A estimation (cross-fitting) - with gradients for theta updates
            V_A_list = cf_fitter.cross_fit_stage1(
                phi_minus, phi_plus,
                stage1_estimator=lambda X, Y: self._ridge_stage1_with_grad(X, Y, self.lambda_A)
            )

            # Out-of-fold prediction (with gradients)
            phi_pred_cf = cf_fitter.compute_out_of_fold_features(phi_minus, V_A_list)

            # Final V_A: full-data estimate (with gradients)
            V_A_final = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)

            # Cache results
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
            print(f"Cross-fitting failed, using standard method: {e}")
            V_A = self._ridge_stage1_with_grad(phi_minus, phi_plus, self.lambda_A)
            phi_pred = (V_A @ phi_minus.T).T
            return phi_pred, V_A

    def _compute_V_A_with_cross_fitting(
        self,
        phi_minus: torch.Tensor,
        phi_plus: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute V_A using cross-fitting (closed-form solution).

        Args:
            phi_minus: Past features (T-1, d_A)
            phi_plus: Future features (T-1, d_A)

        Returns:
            torch.Tensor: V_A matrix (d_A, d_A)
        """
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

            # Cache V_A list for Stage-2
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
            print(f"Cross-fitting failed, using standard method: {e}")
            return self._ridge_stage1(phi_minus, phi_plus, self.lambda_A)

    def apply_transfer_operator(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        Apply transfer operator: phi_hat_{t|t-1} = V_A phi_{t-1}.

        Args:
            phi_prev: Previous-step features (d_A,) or (batch, d_A)

        Returns:
            torch.Tensor: Predicted features (d_A,) or (batch, d_A)
        """
        if not self._is_fitted:
            if 'V_A' in self._stage1_cache:
                V_A = self._stage1_cache['V_A']
            else:
                raise RuntimeError("Call fit_two_stage() or train_stage1_with_gradients() first")
        else:
            V_A = self.V_A

        V_A = V_A.to(phi_prev.device)

        if phi_prev.dim() == 1:
            return V_A @ phi_prev
        else:
            return (V_A @ phi_prev.T).T
    
    def predict_one_step(self, x_prev: torch.Tensor) -> torch.Tensor:
        """
        One-step state prediction: x_hat_{t|t-1} = U_A^T V_A phi_theta(x_{t-1}).

        Args:
            x_prev: Previous-step state (r,) or (batch, r)

        Returns:
            torch.Tensor: Predicted state (r,) or (batch, r)
        """
        V_A = None
        U_A = None
        
        if self._is_fitted:
            V_A = self.V_A
            U_A = self.U_A
        elif 'V_A' in self._stage1_cache and 'U_A' in self._stage2_cache:
            V_A = self._stage1_cache['V_A']
            U_A = self._stage2_cache['U_A']
        elif 'V_A' in self._stage1_cache:
            # Stage-1 only: compute U_A on the fly
            V_A = self._stage1_cache['V_A']
            if 'phi_minus' in self._stage1_cache and 'X_plus' in self._stage1_cache:
                with torch.no_grad():
                    phi_minus = self._stage1_cache['phi_minus']
                    X_plus = self._stage1_cache['X_plus']
                    H_simple = (V_A @ phi_minus.T).T
                    U_A = self._ridge_stage2(H_simple, X_plus, self.lambda_B)
                    self._stage2_cache['U_A'] = U_A.detach()
            else:
                raise RuntimeError("Stage-1 completed but data for Stage-2 is missing")
        else:
            raise RuntimeError("Not fitted. Call fit_two_stage() or train_stage1_with_gradients() first")

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
        return_features: bool = False,
        training: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequence prediction: one-step-ahead prediction at each time step.

        Args:
            X_states: State sequence (T, r)
            return_features: Whether to also return features
            training: If True, retain gradients (for Phase-2 end-to-end training)

        Returns:
            torch.Tensor: Prediction sequence (T-1, r)
            Optional[torch.Tensor]: Feature sequence (T-1, d_A)
        """
        if not self._is_fitted and 'V_A' not in self._stage1_cache:
            raise RuntimeError("Not fitted")

        T = X_states.size(0)
        predictions = []
        features = []

        def _predict_loop():
            for t in range(T - 1):
                x_pred = self.predict_one_step(X_states[t])
                predictions.append(x_pred)

                if return_features:
                    phi_prev = self.phi_theta(X_states[t])
                    phi_pred = self.apply_transfer_operator(phi_prev)
                    features.append(phi_pred)

        if training:
            # Phase-2: retain gradients for encoder->DF-A->decoder path
            _predict_loop()
        else:
            # Inference: save memory with no_grad
            with torch.no_grad():
                _predict_loop()

        pred_tensor = torch.stack(predictions)

        if return_features:
            feat_tensor = torch.stack(features)
            return pred_tensor, feat_tensor

        return pred_tensor
    
    def get_transfer_operator(self) -> torch.Tensor:
        """Get transfer operator V_A."""
        if self._is_fitted:
            return self.V_A.clone()
        elif 'V_A' in self._stage1_cache:
            return self._stage1_cache['V_A'].clone()
        else:
            raise RuntimeError("Not fitted")

    def get_readout_matrix(self) -> torch.Tensor:
        """Get readout matrix U_A."""
        if self._is_fitted:
            return self.U_A.clone()
        elif 'U_A' in self._stage2_cache:
            return self._stage2_cache['U_A'].clone()
        else:
            raise RuntimeError("Not fitted")

    def get_state_dict(self) -> Dict[str, Any]:
        """Get fitted parameters as a dictionary."""
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
        """Get state_dict for inference (includes V_A/U_A for filtering evaluation)."""
        state_dict = {
            'phi_theta': self.phi_theta.state_dict(),
        }

        # Include fitted V_A, U_A needed for filtering evaluation
        if hasattr(self, 'V_A') and self.V_A is not None:
            state_dict['V_A'] = self.V_A
        if hasattr(self, 'U_A') and self.U_A is not None:
            state_dict['U_A'] = self.U_A

        # Config excluded (loaded from config file at inference)
        # Caches excluded (not needed for inference)

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Custom load_state_dict: also sets V_A/U_A properly."""
        v_a = state_dict.pop('V_A', None)
        u_a = state_dict.pop('U_A', None)

        super().load_state_dict(state_dict, strict=strict)

        # Set V_A, U_A
        if v_a is not None:
            self.V_A = v_a.to(self.device) if hasattr(self, 'device') else v_a
        if u_a is not None:
            self.U_A = u_a.to(self.device) if hasattr(self, 'device') else u_a