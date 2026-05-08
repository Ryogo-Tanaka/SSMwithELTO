"""DSE_no_closed_form variant: V_B as nn.Parameter, pure SGD, no cross-fitting.

Counterpart to DFStateLayerSGD for DFObservationLayer. V_B is registered as
an nn.Parameter; cross-fitting and the closed-form ridge in train_stage1 /
train_stage2 are removed. For nonlinear readout (current baseline config),
U_B is unused.

Stage-1 mirrors the base class's `T_eff < threshold` fast path: psi_omega is
held under torch.no_grad (psi_omega is updated in Stage-2 only) and gradients
flow to phi_theta (via phi_prev) and V_B (Parameter). Stage-2 uses the
cached phi_prev (no grad to phi_theta because of fix_phi_theta=True in the
trainer) and updates psi_omega + readout_net (and V_B grad accumulates but is
not stepped through opt_psi - see plan file Section 2.3 for the analysis).

Saving: V_B is captured by the inherited get_inference_state_dict.
Loading: at evaluation the regular DFObservationLayer class is used.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from src.ssm.df_observation_layer import DFObservationLayer


class DFObservationLayerSGD(DFObservationLayer):
    """SGD-based variant: V_B is an nn.Parameter, no cross-fitting / ridge."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        d_A = self.df_state.feature_dim
        d_B = self.obs_feature_dim
        scale = 1.0 / math.sqrt(d_A)
        self.V_B = nn.Parameter(torch.randn(d_B, d_A) * scale)

        self._is_fitted = True

    def train_stage1_with_gradients(
        self,
        X_hat_states: torch.Tensor,
        m_features: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        fix_psi_omega: bool = True,
        epoch: int = 0,
    ) -> Dict[str, float]:
        T_x = X_hat_states.size(0)
        if T_x != m_features.size(0):
            raise ValueError(
                f"Sequence length mismatch: X_hat_states={T_x} vs m_features={m_features.size(0)}"
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

            # Pure SGD using Parameter V_B
            psi_pred = (self.V_B @ phi_prev.T).T

            prediction_loss = (
                torch.norm(psi_pred - psi_curr, p="fro") ** 2 / psi_curr.size(0)
            )
            regularization_loss = self.lambda_B * torch.norm(self.V_B, p="fro") ** 2
            loss_stage1 = prediction_loss + regularization_loss

            loss_stage1.backward()
            optimizer_phi.step()

            with torch.no_grad():
                self._stage1_cache = {
                    "V_B": self.V_B.detach().clone(),
                    "phi_prev": phi_prev.detach(),
                    "psi_curr": psi_curr.detach(),
                    "X_hat": X_hat_states.detach(),
                }

            return {
                "stage1_loss": loss_stage1.item(),
                "stage1_pred_loss": prediction_loss.item(),
                "stage1_reg_loss": regularization_loss.item(),
                "n_blocks": 0,
                "mode": "sgd_pure",
            }

        finally:
            if fix_psi_omega and psi_original_states:
                self._restore_parameters(self.psi_omega, psi_original_states)

    def train_stage2_with_gradients(
        self,
        M_features: torch.Tensor,
        optimizer_psi: torch.optim.Optimizer,
        fix_phi_theta: bool = True,
        epoch: int = 0,
    ) -> Dict[str, float]:
        if "V_B" not in self._stage1_cache:
            raise RuntimeError("Stage-1 must be executed first")

        phi_original_states = {}
        if fix_phi_theta:
            phi_original_states = self._freeze_parameters(self.phi_theta)

        try:
            phi_prev = self._stage1_cache["phi_prev"]
            T_eff = phi_prev.size(0)

            if M_features.size(0) < T_eff + 1:
                raise RuntimeError(
                    f"M_features too short: required {T_eff+1}, got {M_features.size(0)}"
                )
            M_curr = M_features[1:T_eff + 1]

            optimizer_psi.zero_grad()

            # Use Parameter V_B directly
            H = (self.V_B @ phi_prev.T).T

            if self.readout_type == "nonlinear":
                M_pred = self.readout_net(H)
                prediction_loss = (
                    torch.norm(M_pred - M_curr, p="fro") ** 2 / M_curr.size(0)
                )
                loss_stage2 = prediction_loss

                loss_stage2.backward()
                optimizer_psi.step()

                return {
                    "stage2_loss": loss_stage2.item(),
                    "stage2_pred_loss": prediction_loss.item(),
                    "stage2_reg_loss": 0.0,
                    "n_blocks": 0,
                    "mode": "sgd_nonlinear",
                }

            raise NotImplementedError(
                "DFObservationLayerSGD: linear readout path not implemented "
                "(current rebuttal config uses readout.type=nonlinear)."
            )

        finally:
            if fix_phi_theta and phi_original_states:
                self._restore_parameters(self.phi_theta, phi_original_states)

    def _fit_with_cross_fitting(self, *args, **kwargs):
        print("[DFObservationLayerSGD] _fit_with_cross_fitting called but disabled; V_B preserved.")

    def _fit_without_cross_fitting(self, *args, **kwargs):
        print("[DFObservationLayerSGD] _fit_without_cross_fitting called but disabled; V_B preserved.")
