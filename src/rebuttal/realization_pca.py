"""DSE_no_cca variant: PCA on past block features instead of CCA.

Replaces the CCA-based state construction in StochasticRealizationWithEncoder
with PCA on the past block features only. The future block features and the
past-future cross-covariance are not used. State coordinates are obtained by
projecting past block features onto the top-r principal directions of the
ridge-regularized past covariance G_reg = G + lambda * I_m.

Compatibility: stores results in the same attributes as the parent class
(canonical_directions_past, canonical_directions_future, canonical_correlations,
is_fitted, B_matrix, _feature_statistics) so estimate_states() and downstream
code work unchanged. canonical_correlations is set to ones(r) since PCA has
no future-correlation concept; this neutralizes the sqrt scaling inside
estimate_states() so that x(t) = a^T phi_m(p(t)).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.ssm.realization import StochasticRealizationWithEncoder


class StochasticRealizationPCA(StochasticRealizationWithEncoder):
    """PCA-based variant of StochasticRealizationWithEncoder.

    Inherits feature construction (block features Phi_X / Phi_Y, MLP
    component_transforms, encoder reference) from the parent. Only fit() is
    overridden to perform PCA on past block features. estimate_states() and
    other inherited methods work unchanged.
    """

    def fit(
        self,
        Y: torch.Tensor,
        encoder: Optional[nn.Module] = None,
    ) -> "StochasticRealizationPCA":
        if encoder is not None:
            self.encoder = encoder

        Y = Y.to(self.device)
        self.encoder = self.encoder.to(self.device)

        # Match parent's cache-clearing behaviour to avoid stale graph references
        self._cached_feature_matrices = None
        self._last_input_shape = None

        # Build past/future block features (we only need Feat_X, but parent's
        # implementation returns both; reusing it preserves caching semantics).
        Feat_X, _Feat_Y = self._build_feature_matrices(Y)

        # Past-only covariance G = (1/N) Feat_X_centered Feat_X_centered^T
        m, N = Feat_X.shape
        Feat_X_mean = torch.mean(Feat_X, dim=1, keepdim=True)
        Feat_X_centered = Feat_X - Feat_X_mean
        G = (Feat_X_centered @ Feat_X_centered.T) / N

        # Mirror parent's _feature_statistics so downstream consumers still find it
        self._feature_statistics = {
            "past_mean": Feat_X_mean.squeeze(),
            "future_mean": Feat_X_mean.squeeze().clone(),  # placeholder for compat
            "num_samples": N,
        }

        # Ridge regularize and ensure symmetry before eigh
        I_m = torch.eye(m, device=G.device, dtype=G.dtype)
        G_reg = G + self.ridge_param * I_m
        G_reg = 0.5 * (G_reg + G_reg.T)

        eigvals, eigvecs = torch.linalg.eigh(G_reg)  # ascending

        r = min(self.num_components, m)
        # Top-r eigenvectors / eigenvalues in DESCENDING order
        top_eigvecs = eigvecs[:, -r:].flip(dims=[1])  # (m, r)
        # top_eigvals retained for diagnostics; not needed for state construction
        _top_eigvals = eigvals[-r:].flip(dims=[0])  # (r,)

        self.canonical_directions_past = top_eigvecs
        # PCA has no future direction; reuse past for attribute compatibility.
        self.canonical_directions_future = top_eigvecs.clone()
        # Set "correlations" = 1 to neutralize the sqrt scaling in estimate_states
        self.canonical_correlations = torch.ones(
            r, device=G.device, dtype=G.dtype
        )

        # Legacy B_matrix (Sigma^{1/2} a^T) for any code that consumes it
        self.B_matrix = (
            torch.diag(torch.sqrt(self.canonical_correlations))
            @ self.canonical_directions_past.T
        )

        self.is_fitted = True

        return self
