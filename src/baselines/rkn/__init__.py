"""RKN (Recurrent Kalman Network) baseline for quad-link image prediction.

Wraps the official PyTorch port (`external/rkn/rkn_cell/RKNCell.py`) inside a
custom encoder/decoder that operates directly on 48x48 images, so that 1-step
and multi-step image-space MSE can be evaluated under the DSE-aligned protocol.
"""
from .model import RknImageModel

__all__ = ["RknImageModel"]
