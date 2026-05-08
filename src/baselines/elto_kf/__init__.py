"""ELTO-KF baseline for quad-link image prediction (paper-based simplified re-impl).

Reference paper:
    Ke, Tanaka, Kawahara (AISTATS 2025), "Learning Stochastic Nonlinear Dynamics
    with Embedded Latent Transfer Operators".

Simplifications vs paper:
- Feature-space approximation: the paper learns ``k_theta(x, x') = k(f_theta(x),
  f_theta(x'))`` with a Gaussian RBF on encoder features (gamma = 1/#features
  in §6). Here we use the encoder feature directly (linear kernel), since the
  RBF bandwidth in the paper is small enough that encoder features dominate.
- Stage 2 decoder retraining (paper §6) is skipped; the decoder trained jointly
  with the encoder is used directly for image-space output.
"""
from .model import EltoKfModel
from .data import QuadlinkEltoKfDataset, resolve_quadlink_path, load_test_sequence

__all__ = [
    "EltoKfModel",
    "QuadlinkEltoKfDataset",
    "resolve_quadlink_path",
    "load_test_sequence",
]
