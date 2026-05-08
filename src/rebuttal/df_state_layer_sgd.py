"""DSE_no_closed_form variant: V_A as nn.Parameter, pure SGD, no cross-fitting.

Replaces the closed-form ridge estimation of V_A in DFStateLayer with a
gradient-trained nn.Parameter. The cross-fitting block aggregation is also
removed (single full-batch backward per Stage-1/Stage-2 call). The final
closed-form refit at end of training (TwoStageTrainer._compute_final_operators)
must be skipped externally so the learned Parameter is preserved.

For the existing baseline config (readout type = "nonlinear"), the readout
matrix U_A is not used at training or inference time (the readout_net MLP
is used instead). U_A therefore stays as the inherited None attribute and
is not registered as a Parameter. Stage-2 of df_state for the linear readout
path is intentionally not implemented here; raise NotImplementedError if
that path is reached so the discrepancy surfaces immediately.

Saving: V_A is captured by the inherited get_inference_state_dict (which
stores self.V_A as a tensor). Loading: at evaluation the regular DFStateLayer
class is used (V_A loaded as a tensor via direct assignment); no special
load_state_dict override is required.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from src.ssm.df_state_layer import DFStateLayer


class DFStateLayerSGD(DFStateLayer):
    """SGD-based variant: V_A is an nn.Parameter, no cross-fitting / ridge."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        d_A = self.feature_dim
        # Xavier-style init: Var = 1/d_A, std = 1/sqrt(d_A).
        scale = 1.0 / math.sqrt(d_A)
        # Replace inherited None attribute with a Parameter. nn.Module.__setattr__
        # registers it for gradient tracking and state_dict inclusion.
        self.V_A = nn.Parameter(torch.randn(d_A, d_A) * scale)

        # Mark as fitted so apply_readout / predict_one_step take the V_A path
        # immediately. For nonlinear readout this is required so readout_net is
        # used and U_A (None) is never accessed.
        self._is_fitted = True

    def train_stage1_with_gradients(
        self,
        X_states: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        epoch: int = 0,
    ) -> Dict[str, float]:
        if X_states.size(0) < 2:
            raise ValueError(f"State sequence too short: T={X_states.size(0)}")

        optimizer_phi.zero_grad()

        phi_seq = self.phi_theta(X_states)
        phi_minus = phi_seq[:-1]
        phi_plus = phi_seq[1:]

        # Pure SGD: V_A is a Parameter, no closed-form ridge / cross-fitting.
        phi_pred = (self.V_A @ phi_minus.T).T

        prediction_loss = torch.norm(phi_pred - phi_plus, p="fro") ** 2 / phi_plus.size(0)
        regularization_loss = self.lambda_A * torch.norm(self.V_A, p="fro") ** 2
        loss_stage1 = prediction_loss + regularization_loss

        loss_stage1.backward()
        optimizer_phi.step()

        # Cache phi_minus / phi_plus / X_plus for Stage-2 (matches base contract).
        with torch.no_grad():
            self._stage1_cache = {
                "V_A": self.V_A.detach().clone(),
                "phi_minus": phi_minus.detach(),
                "phi_plus": phi_plus.detach(),
                "X_plus": X_states[1:].detach(),
            }

        return {
            "stage1_loss": loss_stage1.item(),
            "stage1_pred_loss": prediction_loss.item(),
            "stage1_reg_loss": regularization_loss.item(),
            "n_blocks": 0,
            "mode": "sgd_pure",
        }

    def train_stage2_with_gradients(
        self,
        X_states: torch.Tensor,
        optimizer_phi: torch.optim.Optimizer,
        epoch: int = 0,
    ) -> Dict[str, float]:
        if "X_plus" not in self._stage1_cache:
            raise RuntimeError("Stage-1 must be executed first")

        X_plus = self._stage1_cache["X_plus"]

        optimizer_phi.zero_grad()

        phi_seq = self.phi_theta(X_states)
        phi_minus = phi_seq[:-1]

        # Use Parameter V_A directly so gradients flow to V_A and phi_theta.
        H = (self.V_A @ phi_minus.T).T

        if self.readout_type == "nonlinear":
            X_pred = self.readout_net(H)
            prediction_loss = torch.norm(X_pred - X_plus, p="fro") ** 2 / X_plus.size(0)
            loss_stage2 = prediction_loss

            loss_stage2.backward()
            optimizer_phi.step()

            return {
                "stage2_loss": loss_stage2.item(),
                "stage2_pred_loss": prediction_loss.item(),
                "stage2_reg_loss": 0.0,
                "n_blocks": 0,
                "mode": "sgd_nonlinear",
            }

        raise NotImplementedError(
            "DFStateLayerSGD: linear readout path not implemented "
            "(current rebuttal config uses readout.type=nonlinear)."
        )

    # _fit_with_cross_fitting / _fit_without_cross_fitting are inherited but
    # should never be called when patch_trainer_for_sgd disables
    # _compute_final_operators. We override them defensively to no-op so that a
    # missed patch causes a clear log line rather than silently overwriting V_A.
    def _fit_with_cross_fitting(self, *args, **kwargs):
        print("[DFStateLayerSGD] _fit_with_cross_fitting called but disabled; V_A preserved.")

    def _fit_without_cross_fitting(self, *args, **kwargs):
        print("[DFStateLayerSGD] _fit_without_cross_fitting called but disabled; V_A preserved.")
