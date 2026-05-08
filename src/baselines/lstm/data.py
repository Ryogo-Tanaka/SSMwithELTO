"""quad-link image dataset for the LSTM baseline.

Keeps the image shape (T, H, W, C) for the convolutional encoder (unlike the
NCDSSM adapter which flattens to (T, y_dim)).

* clean regime: quad{N}_n.npz, train_obs shape (1, 1500, 48, 48, 1) -> 4D
* corrupted regime: quad{N}_y.npz, train_obs shape (1500, 48, 48, 1)

Train / val split: head 1200 / tail 300 frames (matches NCDSSM val_size=300
default for ctx_len=30, pred_len=1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_quadlink_path(data_root: str, regime: str, data_seed: int) -> Path:
    suffix = "n" if regime == "clean" else "y"
    return Path(data_root) / f"quad{data_seed}_{suffix}.npz"


def _load_obs(npz_path: Path, key: str) -> np.ndarray:
    data = np.load(npz_path)
    obs = data[key]
    obs = np.asarray(obs)
    # Squeeze a leading singleton dimension if present (5D -> 4D).
    if obs.ndim == 5 and obs.shape[0] == 1:
        obs = obs[0]
    if obs.ndim != 4:
        raise ValueError(
            f"Unexpected obs shape {obs.shape} from {npz_path} (expected 4D after squeeze)."
        )
    obs = obs.astype(np.float32)
    if obs.max() > 1.5:
        obs = obs / 255.0
    return obs  # (T, H, W, C)


class QuadlinkLstmDataset(Dataset):
    """Sliding-window dataset for next-step image prediction."""

    def __init__(
        self,
        file_path: str,
        split: str = "train",
        window_len: int = 30,
        stride: int = 1,
        val_size: int = 300,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split={split}")
        self.file_path = Path(file_path)
        self.split = split
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.val_size = int(val_size)

        if split == "test":
            obs = _load_obs(self.file_path, "test_obs")
            self.full_obs = obs  # (T, H, W, C)
            self.window_starts = []  # not used for test
        else:
            obs = _load_obs(self.file_path, "train_obs")  # (T, H, W, C)
            T = obs.shape[0]
            if val_size <= 0 or val_size >= T:
                raise ValueError(f"invalid val_size={val_size} for T={T}")
            train_obs = obs[: T - val_size]
            val_obs = obs[T - val_size:]
            base = train_obs if split == "train" else val_obs
            self.full_obs = base
            self.window_starts = list(
                range(0, base.shape[0] - self.window_len + 1, self.stride)
            )

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        s = self.window_starts[idx]
        win = self.full_obs[s : s + self.window_len]
        return torch.from_numpy(np.ascontiguousarray(win))  # (T, H, W, C)


def load_test_sequence(
    data_root: str,
    regime: str,
    data_seed: int,
) -> torch.Tensor:
    """Load full test_obs for evaluation.

    Returns: float32 tensor (T, H, W, C) with pixel range [0, 1].
    """
    path = resolve_quadlink_path(data_root, regime, data_seed)
    obs = _load_obs(path, "test_obs")
    return torch.from_numpy(np.ascontiguousarray(obs))
