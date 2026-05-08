"""Helpers to monkey-patch a TwoStageTrainer instance for the DSE rebuttal variants.

`patch_trainer_for_no_cca`:
  Swap `trainer.realization` to a `StochasticRealizationPCA` instance built
  with the same constructor arguments as the original. Existing src is
  untouched.

`patch_trainer_for_no_closed_form`:
  After `trainer.train_integrated()` has called `_initialize_df_layers`
  (which constructs DFStateLayer / DFObservationLayer instances), this patch
  replaces them with the SGD subclasses, copies over the already-initialized
  phi_theta / psi_omega / readout_net submodules so weights are shared, then
  re-runs `_initialize_optimizers` with V_A / V_B added to the 'phi'
  optimizer's param list. It also no-ops `_compute_final_operators` so the
  closed-form refit at end of training does not overwrite the learned V_A /
  V_B.

Note on schedule: this patch keeps the staged design intact
(`update_strategy=encoder_decoder_only`). V_A and V_B are updated every
epoch via the `phi` optimizer in df_a Stage-1 / df_b Stage-1 respectively.
df_b Stage-2 also computes a V_B gradient but its `opt_psi.step()` does not
include V_B; the leftover gradient is cleared by the next epoch's
`opt_phi.zero_grad()` (see plan file `proj-rebuttal-uai-fuzzy-dream.md`
section 2.3 for the full analysis).
"""
from __future__ import annotations

import types
from typing import Optional

import torch

from src.rebuttal.df_observation_layer_sgd import DFObservationLayerSGD
from src.rebuttal.df_state_layer_sgd import DFStateLayerSGD
from src.rebuttal.realization_pca import StochasticRealizationPCA
from src.ssm.realization import StochasticRealizationWithEncoder


def patch_trainer_for_no_cca(trainer) -> None:
    """Swap CCA realization with PCA realization on past block features."""
    original = trainer.realization
    if not isinstance(original, StochasticRealizationWithEncoder):
        raise TypeError(
            "patch_trainer_for_no_cca expects trainer.realization to be a "
            f"StochasticRealizationWithEncoder, got {type(original)}"
        )

    # Reuse the encoder reference; reconstruct PCA variant with the same args.
    fm_type = original.feature_mapping_type
    fm_hidden_dims = None
    fm_activation = "relu"
    if fm_type in ("linear", "mlp") and original.component_transforms is not None:
        first = original.component_transforms[0]
        # _build_component_transform stores Linear ... activation ... Linear
        # so hidden dims = output features of all Linear except the last
        hidden_dims = []
        for module in first:
            if isinstance(module, torch.nn.Linear):
                hidden_dims.append(module.out_features)
        # Last entry is the projection to scalar (1) - drop it.
        if hidden_dims and hidden_dims[-1] == 1:
            hidden_dims = hidden_dims[:-1]
        fm_hidden_dims = hidden_dims if fm_type == "mlp" else []
        # Activation lookup: we accept the default 'relu' since saving the
        # original activation is a minor concern (visible in config).

    pca = StochasticRealizationPCA(
        encoder=original.encoder,
        encoder_output_dim=original.feature_dim,
        past_horizon=original.window_length,
        rank=original.num_components,
        ridge_param=original.ridge_param,
        jitter=original.min_eigenvalue,
        device=original.device,
        feature_mapping_type=fm_type,
        feature_mapping_hidden_dims=fm_hidden_dims,
        feature_mapping_activation=fm_activation,
    )

    # Carry over the trained component_transforms weights if they exist so the
    # variant matches the parent's architectural capacity.
    if original.component_transforms is not None:
        pca.component_transforms.load_state_dict(original.component_transforms.state_dict())

    # Move all submodules (component_transforms, encoder ref) to the trainer's device.
    pca = pca.to(trainer.device)
    pca.device = original.device  # preserve string device attr used internally
    trainer.realization = pca
    print(
        f"[patch_trainer_for_no_cca] Replaced realization with "
        f"StochasticRealizationPCA (encoder_output_dim={pca.feature_dim}, "
        f"past_horizon={pca.window_length}, rank={pca.num_components})."
    )


def _replace_df_state_with_sgd(trainer) -> DFStateLayerSGD:
    """Replace `trainer.df_state` with a DFStateLayerSGD instance, sharing weights."""
    src = trainer.df_state

    sgd = DFStateLayerSGD(
        state_dim=src.state_dim,
        feature_dim=src.feature_dim,
        lambda_A=src.lambda_A,
        lambda_B=src.lambda_B,
        feature_net_config={},
        cross_fitting_config=src.cf_config,
        readout_config=src.readout_config,
    )
    # Re-use the already-built phi_theta module (so any earlier init carries over)
    sgd.phi_theta = src.phi_theta
    if hasattr(src, "readout_net") and src.readout_type == "nonlinear":
        sgd.readout_net = src.readout_net
        sgd.readout_type = "nonlinear"
    sgd = sgd.to(trainer.device)
    trainer.df_state = sgd
    return sgd


def _replace_df_obs_with_sgd(trainer) -> DFObservationLayerSGD:
    """Replace `trainer.df_obs` with a DFObservationLayerSGD instance, sharing weights."""
    src = trainer.df_obs

    sgd = DFObservationLayerSGD(
        df_state_layer=trainer.df_state,
        obs_feature_dim=src.obs_feature_dim,
        multivariate_feature_dim=src.multivariate_feature_dim,
        lambda_B=src.lambda_B,
        lambda_dB=src.lambda_dB,
        obs_net_config={},
        cross_fitting_config=src.cf_config,
        readout_config=src.readout_config,
    )
    # Re-use existing psi_omega and (if present) readout_net
    sgd.psi_omega = src.psi_omega
    if hasattr(src, "readout_net") and src.readout_type == "nonlinear":
        sgd.readout_net = src.readout_net
        sgd.readout_type = "nonlinear"
    sgd = sgd.to(trainer.device)
    trainer.df_obs = sgd
    return sgd


def _patched_initialize_optimizers(trainer_self):
    """Re-run optimizer setup with V_A / V_B added to the phi optimizer."""
    phi_params = list(trainer_self.df_state.phi_theta.parameters())
    if hasattr(trainer_self.df_state, "readout_net") and \
       trainer_self.df_state.readout_type == "nonlinear":
        phi_params += list(trainer_self.df_state.readout_net.parameters())
    # Add V_A and V_B Parameters so opt_phi.step() updates them in df_a Stage-1
    # and df_b Stage-1 respectively.
    phi_params.append(trainer_self.df_state.V_A)
    phi_params.append(trainer_self.df_obs.V_B)
    trainer_self.optimizers["phi"] = torch.optim.Adam(
        phi_params, lr=trainer_self.config.lr_phi
    )

    psi_params = list(trainer_self.df_obs.psi_omega.parameters())
    if hasattr(trainer_self.df_obs, "readout_net") and \
       trainer_self.df_obs.readout_type == "nonlinear":
        psi_params += list(trainer_self.df_obs.readout_net.parameters())
    trainer_self.optimizers["psi"] = torch.optim.Adam(
        psi_params, lr=trainer_self.config.lr_psi
    )

    # Stage-2 optimizer: identical to base trainer for encoder_decoder_only
    param_groups = [
        {"params": list(trainer_self.encoder.parameters()), "lr": trainer_self.config.lr_encoder},
        {"params": list(trainer_self.decoder.parameters()), "lr": trainer_self.config.lr_decoder},
    ]
    if trainer_self.config.update_strategy in ("all", "joint_all"):
        param_groups.extend([
            {"params": list(trainer_self.df_state.phi_theta.parameters()), "lr": trainer_self.config.lr_phi},
            {"params": list(trainer_self.df_obs.psi_omega.parameters()), "lr": trainer_self.config.lr_psi},
        ])
    trainer_self.optimizers["e2e"] = torch.optim.Adam(param_groups)
    print(
        f"[patched_initialize_optimizers] Stage-2 optimizer: "
        f"{len(param_groups)} param groups (update_strategy={trainer_self.config.update_strategy}), "
        f"phi optimizer extended with V_A/V_B Parameters "
        f"(shapes V_A={tuple(trainer_self.df_state.V_A.shape)}, "
        f"V_B={tuple(trainer_self.df_obs.V_B.shape)})."
    )


def _patched_compute_final_operators(trainer_self, Y_train: torch.Tensor):
    """No-op: V_A / V_B are SGD-learned; do not overwrite with closed-form ridge."""
    trainer_self.encoder = trainer_self.encoder.to(trainer_self.device)
    trainer_self.decoder = trainer_self.decoder.to(trainer_self.device)
    if hasattr(trainer_self.df_state, "phi_theta"):
        trainer_self.df_state.phi_theta = trainer_self.df_state.phi_theta.to(
            trainer_self.device
        )
    if hasattr(trainer_self.df_obs, "psi_omega"):
        trainer_self.df_obs.psi_omega = trainer_self.df_obs.psi_omega.to(
            trainer_self.device
        )
    print(
        "[patched_compute_final_operators] (SGD variant) skipping closed-form "
        "refit; learned V_A and V_B Parameters preserved."
    )


def patch_trainer_for_no_closed_form(trainer) -> None:
    """Apply the SGD-variant monkey-patches.

    Must be called AFTER `_initialize_df_layers` has run (so that `df_state`
    and `df_obs` exist and have phi_theta / psi_omega / readout_net
    initialized). The standard place is to wrap `train_integrated` so the
    patch is applied between layer init and optimizer init. We achieve this
    by replacing `_initialize_df_layers` so that, on first call, it runs the
    base method, swaps the layers to SGD subclasses, and the subsequent
    `_initialize_optimizers` call (also patched) picks up V_A / V_B.
    """
    base_initialize_df_layers = trainer._initialize_df_layers

    def patched_initialize_df_layers(trainer_self, X_states):
        # Run the base layer construction
        result = base_initialize_df_layers(X_states)
        # Replace with SGD subclasses
        _replace_df_state_with_sgd(trainer_self)
        _replace_df_obs_with_sgd(trainer_self)
        print(
            "[patch_trainer_for_no_closed_form] Replaced df_state -> "
            "DFStateLayerSGD, df_obs -> DFObservationLayerSGD "
            f"(V_A {tuple(trainer_self.df_state.V_A.shape)} / "
            f"V_B {tuple(trainer_self.df_obs.V_B.shape)} are nn.Parameters)."
        )
        return result

    trainer._initialize_df_layers = types.MethodType(
        patched_initialize_df_layers, trainer
    )
    trainer._initialize_optimizers = types.MethodType(
        _patched_initialize_optimizers, trainer
    )
    trainer._compute_final_operators = types.MethodType(
        _patched_compute_final_operators, trainer
    )
    print(
        "[patch_trainer_for_no_closed_form] Monkey-patched _initialize_df_layers, "
        "_initialize_optimizers, and _compute_final_operators."
    )
