"""
Hankel delay embedding for time series preprocessing.

Converts low-dimensional time series to higher-dimensional delay-embedded
observations suitable for encoder input.

Usage:
    VDP: hankel_embed(y, 30)  # (T,) -> (T-29, 30)
    SL:  hankel_embed(y, 10)  # (T,2) -> (T-9, 20)
"""

import numpy as np


def hankel_embed(y: np.ndarray, window: int) -> np.ndarray:
    """
    Hankel delay embedding.

    Args:
        y: Time series array. Shape (T,) for univariate or (T, d) for multivariate.
        window: Embedding window size (number of delays).

    Returns:
        Delay-embedded array of shape (T - window + 1, window * d).
    """
    if y.ndim == 1:
        y = y[:, np.newaxis]  # (T,) -> (T, 1)

    T, d = y.shape
    T_out = T - window + 1
    if T_out <= 0:
        raise ValueError(f"Window size {window} exceeds time series length {T}")

    # Build Hankel matrix: row t contains [y_t, y_{t+1}, ..., y_{t+window-1}]
    embedded = np.empty((T_out, window * d), dtype=y.dtype)
    for i in range(window):
        embedded[:, i * d:(i + 1) * d] = y[i:i + T_out]

    return embedded
