import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import warnings

from .cross_fitting import CrossFittingManager, TwoStageCrossFitter, CrossFittingError
from .readout_net import ReadoutNet


class ObservationFeatureNet(nn.Module):
    """
    Observation feature map psi_omega: R^m -> R^{d_B}.
    Maps encoder output to a high-dimensional feature space.
    """

    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 16,
        hidden_sizes: list[int] = [32, 32],
        activation: str = "ReLU",
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Input dimension m (multivariate features)
            output_dim: Output feature dimension d_B
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
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m: Multivariate features (batch_size, m) or (m,)

        Returns:
            torch.Tensor: Observation features (batch_size, d_B) or (d_B,)
        """
        if m.dim() == 1:
            m = m.unsqueeze(0)
            return self.net(m).squeeze(0)
        elif m.dim() == 2:
            return self.net(m)
        else:
            raise ValueError(f"Unsupported input shape: {m.shape}. Expected (m,) or (batch_size, m)")


class DFObservationLayer(nn.Module):
    """
    DF-B: Deep Feature Instrumental Variable for Observation Process.

    Corresponds to Section 1.4.2 in the paper. Uses state predictions from DF-A
    as instrumental variables for one-step feature prediction via 2SLS.

    Computation flow:
    1. Receive state predictions x_hat_{t|t-1} and shared feature map phi_theta from DF-A
    2. Use phi_theta(x_hat_{t|t-1}) as instrumental variables
    3. Compute observation features via psi_omega(m_t)
    4. Stage-1: Estimate V_B + update phi_theta (psi_omega frozen)
    5. Stage-2: Estimate u_B + update psi_omega (phi_theta frozen)
    6. Predict: m_hat_{t|t-1} = u_B^T V_B phi_theta(x_hat_{t|t-1})
    """

    def __init__(
        self,
        df_state_layer,
        obs_feature_dim: int = 16,
        multivariate_feature_dim: int = 8,
        lambda_B: float = 1e-3,
        lambda_dB: float = 1e-3,
        obs_net_config: Optional[Dict[str, Any]] = None,
        cross_fitting_config: Optional[Dict[str, Any]] = None,
        readout_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            df_state_layer: Required DFStateLayer instance
            obs_feature_dim: Observation feature dimension d_B
            multivariate_feature_dim: Multivariate feature dimension m
            lambda_B: Stage-1 regularization parameter lambda_B
            lambda_dB: Stage-2 regularization parameter lambda_{dB}
            obs_net_config: Configuration for ObservationFeatureNet
            cross_fitting_config: Configuration for CrossFittingManager
            readout_config: Configuration for readout map (type, hidden_dims, activation)
        """
        if df_state_layer is None:
            raise ValueError("df_state_layer is required. Pass a DFStateLayer instance.")

        super().__init__()
        self.df_state = df_state_layer
        self.obs_feature_dim = obs_feature_dim
        self.multivariate_feature_dim = multivariate_feature_dim
        self.lambda_B = float(lambda_B)
        self.lambda_dB = float(lambda_dB)

        # Shared feature network from DF-A
        self.phi_theta = df_state_layer.phi_theta

        obs_config = obs_net_config or {}
        self.psi_omega = ObservationFeatureNet(
            input_dim=multivariate_feature_dim,
            output_dim=obs_feature_dim,
            **obs_config
        )

        self.cf_config = cross_fitting_config or {'n_blocks': 5, 'min_block_size': 10}

        # Readout map configuration
        self.readout_config = readout_config or {}
        self.readout_type = self.readout_config.get('type', 'linear')
        if self.readout_type == 'nonlinear':
            self.readout_net = ReadoutNet(
                input_dim=obs_feature_dim,
                output_dim=multivariate_feature_dim,
                hidden_dims=self.readout_config.get('hidden_dims', [32]),
                activation=self.readout_config.get('activation', 'ReLU')
            )

        self.V_B: Optional[torch.Tensor] = None  # Observation transfer operator (d_B, d_A)
        self.U_B: Optional[torch.Tensor] = None  # Observation readout matrix (d_B, m) — linear only
        self._is_fitted = False

        self._stage1_cache = {}
        self._stage2_cache = {}
    
    def _ridge_stage1_vb(
        self,
        Phi_instrument: torch.Tensor,
        Psi_treatment: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-1 Ridge regression: observation transfer operator estimation.

        V_B = (Psi^T Phi)(Phi^T Phi + lambda I)^{-1}

        Args:
            Phi_instrument: Instrumental variable features (N, d_A)
            Psi_treatment: Observation features (N, d_B)
            reg_lambda: Regularization parameter lambda_B

        Returns:
            torch.Tensor: Observation transfer operator V_B (d_B, d_A)
        """
        N, d_A = Phi_instrument.shape
        N_t, d_B = Psi_treatment.shape

        if N != N_t:
            raise ValueError(f"Instrument-observation sample count mismatch: {N} vs {N_t}")

        if N < max(d_A, d_B):
            warnings.warn(f"Sample count {N} < feature dimension max({d_A}, {d_B}). May be numerically unstable")

        PhiTPhi = Phi_instrument.T @ Phi_instrument
        PhiTPhi_reg = PhiTPhi + reg_lambda * torch.eye(d_A, device=Phi_instrument.device, dtype=Phi_instrument.dtype)

        Psi_treatment = Psi_treatment.to(Phi_instrument.device)
        PsiTPhi = Psi_treatment.T @ Phi_instrument

        try:
            PhiTPhi_inv = torch.linalg.inv(PhiTPhi_reg)
            V_B = PsiTPhi @ PhiTPhi_inv
        except torch.linalg.LinAlgError:
            try:
                L = torch.linalg.cholesky(PhiTPhi_reg)
                PhiTPhi_inv = torch.cholesky_inverse(L)
                V_B = PsiTPhi @ PhiTPhi_inv
            except torch.linalg.LinAlgError:
                U, S, Vh = torch.linalg.svd(PhiTPhi_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                PhiTPhi_inv = (Vh.T * S_inv) @ Vh
                V_B = PsiTPhi @ PhiTPhi_inv
        
        return V_B

    def _ridge_stage1_vb_with_grad(
        self,
        Phi_instrument: torch.Tensor,
        Psi_treatment: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-1 Ridge regression (with gradients): for phi_theta updates.

        V_B = (Psi^T Phi)(Phi^T Phi + lambda I)^{-1}
        """
        N, d_A = Phi_instrument.shape
        N_t, d_B = Psi_treatment.shape

        if N != N_t:
            raise ValueError(f"Feature-target sample count mismatch: {N} vs {N_t}")
        PhiPhi = Phi_instrument.T @ Phi_instrument
        PhiPhi_reg = PhiPhi + reg_lambda * torch.eye(d_A, device=Phi_instrument.device, dtype=Phi_instrument.dtype)

        Psi_treatment = Psi_treatment.to(Phi_instrument.device)
        PsiPhi = Psi_treatment.T @ Phi_instrument

        try:
            PhiPhi_inv = torch.linalg.inv(PhiPhi_reg)
            V_B = PsiPhi @ PhiPhi_inv
        except torch.linalg.LinAlgError:
            PhiPhi_inv = torch.linalg.pinv(PhiPhi_reg)
            V_B = PsiPhi @ PhiPhi_inv

        return V_B

    def _compute_cross_fitting_prediction_vb(
        self,
        phi_prev: torch.Tensor,
        psi_curr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute out-of-fold predictions for V_B via cross-fitting.

        Args:
            phi_prev: Instrumental variable features (T-1, d_A)
            psi_curr: Observation features (T-1, d_B)

        Returns:
            tuple: (psi_pred_cf, V_B_final)
                - psi_pred_cf: Out-of-fold predictions (T-1, d_B)
                - V_B_final: Final V_B matrix (d_B, d_A)
        """
        T_eff = phi_prev.size(0)

        n_blocks = getattr(self, 'cf_config', {}).get('n_blocks', 6)
        min_block_size = getattr(self, 'cf_config', {}).get('min_block_size', 20)

        if T_eff < max(n_blocks * min_block_size, 100):
            V_B = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr, self.lambda_B)
            psi_pred = (V_B @ phi_prev.T).T
            return psi_pred, V_B

        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            # psi_curr computed with no_grad (omega frozen)
            V_B_list = cf_fitter.cross_fit_stage1(
                phi_prev, psi_curr,
                stage1_estimator=lambda X, Y: self._ridge_stage1_vb_with_grad(X, Y, self.lambda_B)
            )

            psi_pred_cf = cf_fitter.compute_out_of_fold_features(phi_prev, V_B_list)
            V_B_final = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr, self.lambda_B)

            if not hasattr(self, '_cross_fitting_cache'):
                self._cross_fitting_cache = {}
            self._cross_fitting_cache.update({
                'V_B_list': V_B_list,
                'cf_manager': cf_manager
            })

            return psi_pred_cf, V_B_final

        except ImportError:
            V_B = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr, self.lambda_B)
            psi_pred = (V_B @ phi_prev.T).T
            return psi_pred, V_B
        except Exception as e:
            print(f"V_B cross-fitting failed, using standard method: {e}")
            V_B = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr, self.lambda_B)
            psi_pred = (V_B @ phi_prev.T).T
            return psi_pred, V_B

    def _ridge_stage2_ub_matrix(
        self,
        H_instrument: torch.Tensor,
        M_target: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-2 Ridge regression: U_B = (H^T H + lambda I)^{-1} H^T M.

        Args:
            H_instrument: Cross-fitted features (N, d_B)
            M_target: Target multivariate features (N, m)
            reg_lambda: Regularization parameter lambda_{dB}

        Returns:
            torch.Tensor: Observation readout matrix U_B (d_B, m)
        """
        N, d_B = H_instrument.shape
        N_t, m = M_target.shape

        if N != N_t:
            raise ValueError(f"Instrument-target sample count mismatch: {N} vs {N_t}")

        HHt = H_instrument.T @ H_instrument
        HHt_reg = HHt + reg_lambda * torch.eye(d_B, device=H_instrument.device, dtype=H_instrument.dtype)

        M_target = M_target.to(H_instrument.device)
        HM = H_instrument.T @ M_target
        try:
            HHt_inv = torch.linalg.inv(HHt_reg)
            U_B = HHt_inv @ HM
        except torch.linalg.LinAlgError:
            try:
                L = torch.linalg.cholesky(HHt_reg)
                HHt_inv = torch.cholesky_inverse(L)
                U_B = HHt_inv @ HM
            except torch.linalg.LinAlgError:
                U, S, Vh = torch.linalg.svd(HHt_reg)
                S_inv = torch.where(S > 1e-10, 1.0 / S, 0.0)
                HHt_inv = (Vh.T * S_inv) @ Vh
                U_B = HHt_inv @ HM

        return U_B

    def _ridge_stage2_ub_matrix_with_grad(
        self,
        H_instrument: torch.Tensor,
        M_target: torch.Tensor,
        reg_lambda: float
    ) -> torch.Tensor:
        """
        Stage-2 Ridge regression (with gradients): for psi_omega updates (multivariate).

        U_B = (H H^T + lambda I)^{-1} H M^T
        """
        N, d_B = H_instrument.shape
        N_t, m = M_target.shape

        if N != N_t:
            raise ValueError(f"Feature-target sample count mismatch: {N} vs {N_t}")

        HHt = H_instrument.T @ H_instrument
        HHt_reg = HHt + reg_lambda * torch.eye(d_B, device=H_instrument.device, dtype=H_instrument.dtype)

        M_target = M_target.to(H_instrument.device)
        HM = H_instrument.T @ M_target

        try:
            HHt_inv = torch.linalg.inv(HHt_reg)
            U_B = HHt_inv @ HM
        except torch.linalg.LinAlgError:
            HHt_inv = torch.linalg.pinv(HHt_reg)
            U_B = HHt_inv @ HM

        return U_B

    def _train_readout_net_on_data(
        self,
        H_features: torch.Tensor,
        targets: torch.Tensor,
        n_steps: int = 200
    ):
        """Train nonlinear readout net on given data (for fit_two_stage)."""
        opt = torch.optim.Adam(
            self.readout_net.parameters(),
            lr=1e-3,
            weight_decay=self.lambda_dB
        )
        H_detached = H_features.detach()
        targets_detached = targets.detach()
        with torch.enable_grad():
            for _ in range(n_steps):
                opt.zero_grad()
                pred = self.readout_net(H_detached)
                loss = F.mse_loss(pred, targets_detached)
                loss.backward()
                opt.step()

    def apply_readout(self, h: torch.Tensor) -> torch.Tensor:
        """Apply readout map (linear or nonlinear).

        Args:
            h: Feature-space input (*, d_B)

        Returns:
            torch.Tensor: Readout output (*, m)
        """
        if self.readout_type == 'nonlinear':
            return self.readout_net(h)
        else:
            U_B = self._get_readout_matrix()
            U_B = U_B.to(h.device)
            if h.dim() == 1:
                return U_B.T @ h
            else:
                return h @ U_B

    def _get_readout_matrix(self) -> torch.Tensor:
        """Get linear readout matrix U_B (linear mode only)."""
        if self._is_fitted and self.U_B is not None:
            return self.U_B
        elif 'U_B' in self._stage2_cache:
            return self._stage2_cache['U_B']
        else:
            raise RuntimeError("Readout matrix not available")

    def _compute_cross_fitting_prediction_ub_matrix(
        self,
        H_features: torch.Tensor,
        M_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute out-of-fold predictions for U_B via cross-fitting (multivariate).

        Args:
            H_features: Instrumental variable features (T-1, d_B)
            M_target: Observation target (T-1, m)

        Returns:
            tuple: (M_pred_cf, U_B_final)
                - M_pred_cf: Out-of-fold predictions (T-1, m)
                - U_B_final: Final U_B matrix (d_B, m)
        """
        T_eff = H_features.size(0)

        n_blocks = getattr(self, 'cf_config', {}).get('n_blocks', 6)
        min_block_size = getattr(self, 'cf_config', {}).get('min_block_size', 20)

        if T_eff < max(n_blocks * min_block_size, 100):
            U_B = self._ridge_stage2_ub_matrix_with_grad(H_features, M_target, self.lambda_dB)
            M_pred = (H_features @ U_B)
            return M_pred, U_B

        try:
            from .cross_fitting import CrossFittingManager, TwoStageCrossFitter

            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)
            cf_fitter = TwoStageCrossFitter(cf_manager)

            U_B_list = cf_fitter.cross_fit_stage2_matrix(
                H_features, M_target,
                stage2_estimator=lambda H, M: self._ridge_stage2_ub_matrix_with_grad(H, M, self.lambda_dB)
            )

            M_pred_cf = torch.zeros_like(M_target)
            for k in range(cf_manager.n_blocks):
                block_indices = cf_manager.get_block_indices(k)
                H_block = H_features[block_indices]
                M_pred_cf[block_indices] = H_block @ U_B_list[k]

            U_B_final = self._ridge_stage2_ub_matrix_with_grad(H_features, M_target, self.lambda_dB)

            return M_pred_cf, U_B_final

        except ImportError:
            U_B = self._ridge_stage2_ub_matrix_with_grad(H_features, M_target, self.lambda_dB)
            M_pred = H_features @ U_B
            return M_pred, U_B
        except Exception as e:
            print(f"U_B cross-fitting failed, using standard method: {e}")
            U_B = self._ridge_stage2_ub_matrix_with_grad(H_features, M_target, self.lambda_dB)
            M_pred = H_features @ U_B
            return M_pred, U_B

    def _freeze_parameters(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        """Temporarily freeze parameters and save original requires_grad state."""
        original_states = {}
        for name, param in module.named_parameters():
            original_states[name] = param.requires_grad
            param.requires_grad = False
        return original_states

    def _restore_parameters(self, module: nn.Module, original_states: Dict[str, torch.Tensor]):
        """Restore requires_grad state of parameters."""
        for name, param in module.named_parameters():
            if name in original_states:
                param.requires_grad = original_states[name]
    
    def train_stage1_with_gradients(
        self,
        X_hat_states: torch.Tensor,
        m_features: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        fix_psi_omega: bool = True,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Stage-1 training (one block per epoch).

        Theory:
        L_Stage-1(theta) = sum_{t in B_k} ||psi_omega(m_t) - V_B^{(-k)} phi_theta(x_hat_{t|t-1})||^2 + lambda_B ||V_B^{(-k)}||^2_F

        Note: psi_omega is frozen (fix_psi_omega=True).

        Args:
            X_hat_states: State prediction sequence (T, r)
            m_features: Multivariate features (T, m)
            optimizer_phi: Optimizer for phi_theta
            fix_psi_omega: Whether to freeze psi_omega
            epoch: Epoch number

        Returns:
            Dict[str, float]: Loss metrics
        """
        T_x = X_hat_states.size(0)

        if m_features.dim() != 2:
            raise ValueError(f"Multivariate features must be 2D (T, m): got {m_features.shape}")

        if m_features.size(1) != self.multivariate_feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.multivariate_feature_dim}, got {m_features.size(1)}")

        if T_x != m_features.size(0):
            error_details = {
                "X_hat_states_shape": tuple(X_hat_states.shape),
                "m_features_shape": tuple(m_features.shape),
                "size_diff": T_x - m_features.size(0)
            }
            raise ValueError(
                f"Sequence length mismatch: X_hat_states={T_x} vs m_features={m_features.size(0)}. "
                f"Details: {error_details}. "
                f"Hint: Time index adjustment may be needed."
            )

        if T_x < 2:
            raise ValueError(f"Sequence too short: T={T_x}")

        psi_original_states = {}
        if fix_psi_omega:
            psi_original_states = self._freeze_parameters(self.psi_omega)

        try:
            optimizer_phi.zero_grad()

            phi_instrument = self.phi_theta(X_hat_states)

            with torch.no_grad():
                psi_obs = self.psi_omega(m_features)

            phi_prev = phi_instrument[:-1]
            psi_curr = psi_obs[1:]

            T_eff = phi_prev.size(0)
            n_blocks = self.cf_config.get('n_blocks', 5)
            min_block_size = self.cf_config.get('min_block_size', 20)

            if T_eff < max(n_blocks * min_block_size, 100):
                V_B = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr, self.lambda_B)
                psi_pred = (V_B @ phi_prev.T).T

                prediction_loss = torch.norm(psi_pred - psi_curr, p='fro') ** 2 / psi_curr.size(0)
                regularization_loss = self.lambda_B * torch.norm(V_B, p='fro') ** 2
                loss_stage1 = prediction_loss + regularization_loss

                loss_stage1.backward()
                optimizer_phi.step()

                with torch.no_grad():
                    self._stage1_cache = {
                        'V_B': V_B,
                        'phi_prev': phi_prev,
                        'psi_curr': psi_curr,
                        'X_hat': X_hat_states
                    }

                return {
                    'stage1_loss': loss_stage1.item(),
                    'stage1_pred_loss': prediction_loss.item(),
                    'stage1_reg_loss': regularization_loss.item(),
                    'n_blocks': 0,
                    'mode': 'no_crossfitting'
                }

            from .cross_fitting import CrossFittingManager

            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)

            k = epoch % cf_manager.n_blocks

            optimizer_phi.zero_grad()

            phi_instrument = self.phi_theta(X_hat_states)
            phi_prev = phi_instrument[:-1]

            with torch.no_grad():
                psi_obs = self.psi_omega(m_features)
                psi_curr = psi_obs[1:]

            oof_indices = cf_manager.get_out_of_fold_indices(k)
            phi_prev_oof = phi_prev[oof_indices]
            psi_curr_oof = psi_curr[oof_indices]

            V_B_k = self._ridge_stage1_vb_with_grad(phi_prev_oof, psi_curr_oof, self.lambda_B)

            block_indices = cf_manager.get_block_indices(k)
            phi_prev_block = phi_prev[block_indices]
            psi_curr_block = psi_curr[block_indices]

            psi_pred_block = (V_B_k @ phi_prev_block.T).T

            prediction_loss_k = torch.norm(psi_pred_block - psi_curr_block, p='fro') ** 2 / psi_curr_block.size(0)
            regularization_loss_k = self.lambda_B * torch.norm(V_B_k, p='fro') ** 2
            loss_k = prediction_loss_k + regularization_loss_k

            loss_k.backward()
            optimizer_phi.step()

            with torch.no_grad():
                phi_final = self.phi_theta(X_hat_states)
                psi_final = self.psi_omega(m_features)
                phi_prev_final = phi_final[:-1]
                psi_curr_final = psi_final[1:]
                V_B_final = self._ridge_stage1_vb(phi_prev_final, psi_curr_final, self.lambda_B)

                self._stage1_cache = {
                    'V_B': V_B_final,
                    'phi_prev': phi_prev_final,
                    'psi_curr': psi_curr_final,
                    'X_hat': X_hat_states
                }

            return {
                'stage1_loss': loss_k.item(),
                'stage1_pred_loss': prediction_loss_k.item(),
                'stage1_reg_loss': regularization_loss_k.item(),
                'current_block': k,
                'n_blocks': cf_manager.n_blocks
            }

        finally:
            if fix_psi_omega and psi_original_states:
                self._restore_parameters(self.psi_omega, psi_original_states)

    def train_stage2_with_gradients(
            self,
            M_features: torch.Tensor,
            optimizer_psi: torch.optim.Optimizer,
            fix_phi_theta: bool = True,
            epoch: int = 0
        ) -> Dict[str, float]:
        """
        Stage-2 training (one block per epoch).

        Theory:
        L_Stage-2(omega) = sum_{t in B_k} ||M_t - U_B^{(-k)}^T H_k||^2 + lambda_{dB} ||U_B^{(-k)}||^2_F

        Note: phi_theta is frozen (fix_phi_theta=True).

        Args:
            M_features: Multivariate features (T, m)
            optimizer_psi: Optimizer for psi_omega
            fix_phi_theta: Whether to freeze phi_theta
            epoch: Epoch number

        Returns:
            Dict[str, float]: Loss metrics
        """
        if 'V_B' not in self._stage1_cache:
            raise RuntimeError("Stage-1 must be executed first")

        phi_original_states = {}
        if fix_phi_theta:
            phi_original_states = self._freeze_parameters(self.phi_theta)

        try:
            phi_prev = self._stage1_cache['phi_prev']
            T_eff = phi_prev.size(0)

            if M_features.size(0) < T_eff + 1:
                raise RuntimeError(f"M_features too short: required {T_eff+1}, got {M_features.size(0)}")
            M_curr = M_features[1:T_eff+1]

            optimizer_psi.zero_grad()

            psi_obs_current = self.psi_omega(M_features)
            psi_curr_grad = psi_obs_current[1:T_eff+1]

            V_B_current = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr_grad, self.lambda_B)

            H = (V_B_current @ phi_prev.T).T
            n_blocks = self.cf_config.get('n_blocks', 5)
            min_block_size = self.cf_config.get('min_block_size', 20)

            if self.readout_type == 'nonlinear':
                # Nonlinear readout: forward through MLP, backprop to readout_net and psi_omega
                M_pred = self.readout_net(H)
                prediction_loss = torch.norm(M_pred - M_curr, p='fro') ** 2 / M_curr.size(0)
                loss_stage2 = prediction_loss

                loss_stage2.backward()
                optimizer_psi.step()

                return {
                    'stage2_loss': loss_stage2.item(),
                    'stage2_pred_loss': prediction_loss.item(),
                    'stage2_reg_loss': 0.0,
                    'n_blocks': 0,
                    'mode': 'nonlinear_readout'
                }

            # Linear readout path (existing behavior)
            if T_eff < max(n_blocks * min_block_size, 100):
                U_B = self._ridge_stage2_ub_matrix_with_grad(H, M_curr, self.lambda_dB)
                M_pred = H @ U_B

                prediction_loss = torch.norm(M_pred - M_curr, p='fro') ** 2 / M_curr.size(0)
                regularization_loss = self.lambda_dB * torch.norm(U_B, p='fro') ** 2
                loss_stage2 = prediction_loss + regularization_loss

                loss_stage2.backward()
                optimizer_psi.step()

                self._stage2_cache['U_B'] = U_B.detach()
                return {
                    'stage2_loss': loss_stage2.item(),
                    'stage2_pred_loss': prediction_loss.item(),
                    'stage2_reg_loss': regularization_loss.item(),
                    'n_blocks': 0,
                    'mode': 'no_crossfitting'
                }

            from .cross_fitting import CrossFittingManager
            cf_manager = CrossFittingManager(T_eff, n_blocks=n_blocks, min_block_size=min_block_size)

            k = epoch % cf_manager.n_blocks
            optimizer_psi.zero_grad()

            psi_obs_current = self.psi_omega(M_features)
            psi_curr_grad = psi_obs_current[1:T_eff+1]
            V_B_current = self._ridge_stage1_vb_with_grad(phi_prev, psi_curr_grad, self.lambda_B)
            H = (V_B_current @ phi_prev.T).T

            oof_indices = cf_manager.get_out_of_fold_indices(k)
            U_B_k = self._ridge_stage2_ub_matrix_with_grad(H[oof_indices], M_curr[oof_indices], self.lambda_dB)

            block_indices = cf_manager.get_block_indices(k)
            M_pred_k = H[block_indices] @ U_B_k

            pred_loss_k = torch.norm(M_pred_k - M_curr[block_indices], p='fro') ** 2 / M_curr[block_indices].size(0)
            reg_loss_k = self.lambda_dB * torch.norm(U_B_k, p='fro') ** 2
            loss_k = pred_loss_k + reg_loss_k

            loss_k.backward()
            optimizer_psi.step()

            # Recompute on full data to reflect psi_omega updates
            with torch.no_grad():
                X_hat_states = self._stage1_cache['X_hat']
                phi_final = self.phi_theta(X_hat_states)
                psi_final = self.psi_omega(M_features)
                phi_prev_final = phi_final[:-1]
                psi_curr_final = psi_final[1:T_eff+1]
                V_B_final = self._ridge_stage1_vb(phi_prev_final, psi_curr_final, self.lambda_B)
                H_final = (V_B_final @ phi_prev_final.T).T
                M_curr_final = M_features[1:T_eff+1]
                U_B_final = self._ridge_stage2_ub_matrix(H_final, M_curr_final, self.lambda_dB)
                self._stage2_cache['U_B'] = U_B_final

            return {
                'stage2_loss': loss_k.item(),
                'stage2_pred_loss': pred_loss_k.item(),
                'stage2_reg_loss': reg_loss_k.item(),
                'current_block': k,
                'n_blocks': cf_manager.n_blocks
            }

        finally:
            if fix_phi_theta and phi_original_states:
                self._restore_parameters(self.phi_theta, phi_original_states)

    def fit_two_stage(
        self,
        X_hat_states: torch.Tensor,
        M_features: torch.Tensor,
        use_cross_fitting: bool = True,
        verbose: bool = False
    ) -> 'DFObservationLayer':
        """
        Legacy two-stage cross-fitting training (multivariate).

        Note: For Phase-1 training, use train_stage1_with_gradients
        and train_stage2_with_gradients instead.
        """
        T_x, r = X_hat_states.shape

        if M_features.dim() != 2:
            raise ValueError(f"Multivariate features must be 2D: got shape {M_features.shape}")

        if M_features.size(1) != self.multivariate_feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.multivariate_feature_dim}, got {M_features.size(1)}")

        T_m = M_features.size(0)

        if T_x != T_m:
            raise ValueError(f"State prediction and feature time series length mismatch: {T_x} vs {T_m}")

        if T_x < 2:
            raise ValueError(f"Time series too short: T={T_x}")

        with torch.no_grad():
            Phi_instrument = self.phi_theta(X_hat_states)

        with torch.no_grad():
            Psi_obs = self.psi_omega(M_features)

        Phi_prev = Phi_instrument[:-1]
        Psi_curr = Psi_obs[1:]
        M_curr = M_features[1:]
        
        if use_cross_fitting and T_x >= 20:
            self._fit_with_cross_fitting(Phi_prev, Psi_curr, M_curr, verbose)
        else:
            self._fit_without_cross_fitting(Phi_prev, Psi_curr, M_curr, verbose)
        
        self._is_fitted = True
        return self
    
    def _fit_with_cross_fitting(
        self,
        Phi_prev: torch.Tensor,
        Psi_curr: torch.Tensor,
        M_curr: torch.Tensor,
        verbose: bool
    ):
        """Training with cross-fitting (multivariate)."""
        T_eff = Phi_prev.size(0)
        cf_manager = CrossFittingManager(T_eff, **self.cf_config)
        cf_fitter = TwoStageCrossFitter(cf_manager)

        if verbose:
            print(f"DF-B cross-fitting: T={T_eff}, n_blocks={cf_manager.n_blocks}")

        # Stage-1: Observation transfer operator estimation
        VB_list = cf_fitter.cross_fit_stage1(
            Phi_prev, Psi_curr,
            self._ridge_stage1_vb,
            reg_lambda=self.lambda_B
        )

        self.V_B = torch.stack(VB_list).mean(dim=0)
        H_cf = cf_fitter.compute_out_of_fold_features(Phi_prev, VB_list)

        # Stage-2: Observation readout estimation
        if self.readout_type == 'nonlinear':
            self._train_readout_net_on_data(H_cf, M_curr)
            if verbose:
                print(f"V_B shape: {self.V_B.shape}, readout_net: nonlinear")
        else:
            U_B_list = cf_fitter.cross_fit_stage2_matrix(
                H_cf, M_curr,
                self._ridge_stage2_ub_matrix,
                detach_features=True,
                reg_lambda=self.lambda_dB
            )
            self.U_B = torch.stack(U_B_list).mean(dim=0)
            if verbose:
                print(f"V_B shape: {self.V_B.shape}, U_B shape: {self.U_B.shape}")
    
    def _fit_without_cross_fitting(
        self,
        Phi_prev: torch.Tensor,
        Psi_curr: torch.Tensor,
        M_curr: torch.Tensor,
        verbose: bool
    ):
        """Training without cross-fitting (for small data, multivariate)."""
        if verbose:
            print("DF-B training without cross-fitting (multivariate)")

        self.V_B = self._ridge_stage1_vb(Phi_prev, Psi_curr, self.lambda_B)
        H = (self.V_B @ Phi_prev.T).T

        if self.readout_type == 'nonlinear':
            self._train_readout_net_on_data(H, M_curr)
            if verbose:
                print(f"V_B shape: {self.V_B.shape}, readout_net: nonlinear")
        else:
            self.U_B = self._ridge_stage2_ub_matrix(H, M_curr, self.lambda_dB)
            if verbose:
                print(f"V_B shape: {self.V_B.shape}, U_B shape: {self.U_B.shape}")
    
    def predict_one_step(self, x_hat_prev: torch.Tensor) -> torch.Tensor:
        """
        One-step feature prediction: M_hat_{t|t-1} = U_B^T V_B phi_theta(x_hat_{t|t-1}).

        Args:
            x_hat_prev: Previous-step state prediction (r,) or (batch, r)

        Returns:
            torch.Tensor: Predicted multivariate features (m,) or (batch, m)
        """
        expected_state_dim = self.df_state.phi_theta.net[0].in_features
        if x_hat_prev.dim() == 1:
            actual_dim = x_hat_prev.shape[0]
        else:
            actual_dim = x_hat_prev.shape[-1]

        if actual_dim != expected_state_dim:
            raise ValueError(
                f"predict_one_step expects state input with dimension {expected_state_dim}, "
                f"but got {actual_dim}. "
                f"Input shape: {x_hat_prev.shape}. "
                f"Note: This method expects x_hat_prev (state), not M_features (multivariate features)."
            )

        if not self._is_fitted:
            if 'V_B' in self._stage1_cache:
                V_B = self._stage1_cache['V_B']
            else:
                raise RuntimeError("Run fit_two_stage() or train_stage1/2_with_gradients() first")
        else:
            V_B = self.V_B

        phi_prev = self.phi_theta(x_hat_prev)
        V_B = V_B.to(phi_prev.device)

        if phi_prev.dim() == 1:
            h_pred = V_B @ phi_prev
        else:
            h_pred = (V_B @ phi_prev.T).T

        if self.readout_type == 'nonlinear':
            return self.readout_net(h_pred)

        # Linear readout path
        if not self._is_fitted:
            if 'U_B' in self._stage2_cache:
                U_B = self._stage2_cache['U_B']
            else:
                raise RuntimeError("Stage-2 not completed")
        else:
            U_B = self.U_B

        U_B = U_B.to(h_pred.device)
        if h_pred.dim() == 1:
            return U_B.T @ h_pred
        else:
            return h_pred @ U_B
    
    def predict_sequence(
        self,
        X_hat_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Sequence prediction: one-step-ahead feature prediction at each time step.

        Args:
            X_hat_states: State prediction sequence (T, r)

        Returns:
            torch.Tensor: Predicted feature sequence (T-1, m)
        """
        if not self._is_fitted and 'V_B' not in self._stage1_cache:
            raise RuntimeError("Model not fitted")

        T = X_hat_states.size(0)
        predictions = []

        for t in range(T - 1):
            M_pred = self.predict_one_step(X_hat_states[t])
            predictions.append(M_pred)

        return torch.stack(predictions)
    
    def get_observation_operator(self) -> torch.Tensor:
        """Get observation transfer operator V_B."""
        if self._is_fitted:
            return self.V_B.clone()
        elif 'V_B' in self._stage1_cache:
            return self._stage1_cache['V_B'].clone()
        else:
            raise RuntimeError("Model not fitted")
    
    def get_readout_matrix(self) -> torch.Tensor:
        """Get observation readout matrix U_B (linear mode only)."""
        if self.readout_type == 'nonlinear':
            raise RuntimeError(
                "Linear readout matrix not available in nonlinear mode. "
                "Use apply_readout() instead."
            )
        if self._is_fitted:
            return self.U_B.clone()
        elif 'U_B' in self._stage2_cache:
            return self._stage2_cache['U_B'].clone()
        else:
            raise RuntimeError("Model not fitted")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get trained parameters as a dictionary."""
        state_dict = {
            'psi_omega': self.psi_omega.state_dict(),
            'readout_type': self.readout_type,
            'config': {
                'obs_feature_dim': self.obs_feature_dim,
                'lambda_B': self.lambda_B,
                'lambda_dB': self.lambda_dB
            }
        }

        if self._is_fitted:
            if self.V_B is not None:
                state_dict['V_B'] = self.V_B
            if self.readout_type == 'nonlinear' and hasattr(self, 'readout_net'):
                state_dict['readout_net'] = self.readout_net.state_dict()
            elif self.U_B is not None:
                state_dict['U_B'] = self.U_B

        if self._stage1_cache:
            state_dict['stage1_cache'] = self._stage1_cache.copy()
        if self._stage2_cache:
            state_dict['stage2_cache'] = self._stage2_cache.copy()

        return state_dict
    
    def get_inference_state_dict(self) -> Dict[str, Any]:
        """Get state_dict for inference (includes V_B and readout for evaluation)."""
        state_dict = {
            'psi_omega': self.psi_omega.state_dict(),
            'readout_type': self.readout_type,
        }

        if hasattr(self, 'V_B') and self.V_B is not None:
            state_dict['V_B'] = self.V_B

        if self.readout_type == 'nonlinear' and hasattr(self, 'readout_net'):
            state_dict['readout_net'] = self.readout_net.state_dict()
        elif hasattr(self, 'U_B') and self.U_B is not None:
            state_dict['U_B'] = self.U_B

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Custom load_state_dict that also restores V_B and readout."""
        v_b = state_dict.pop('V_B', None)
        u_b = state_dict.pop('U_B', None)
        readout_net_state = state_dict.pop('readout_net', None)
        state_dict.pop('readout_type', None)

        super().load_state_dict(state_dict, strict=strict)

        if v_b is not None:
            self.V_B = v_b.to(self.device) if hasattr(self, 'device') else v_b

        if self.readout_type == 'nonlinear' and readout_net_state is not None:
            self.readout_net.load_state_dict(readout_net_state)
        elif u_b is not None:
            self.U_B = u_b.to(self.device) if hasattr(self, 'device') else u_b