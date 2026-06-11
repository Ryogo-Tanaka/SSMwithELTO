"""
Unified data loader for time series and image data.

Features:
- Multi-format file loading (.npz, .npy, .csv, .json)
- Data validation and cleaning
- Normalization (standard, minmax, unit_scale)
- Chronological train/val/test splitting
- Metadata management

Expected data format: (T, d) multivariate time series as torch.FloatTensor
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@dataclass
class DataMetadata:
    """Dataset metadata."""
    original_shape: Tuple[int, int]
    feature_names: Optional[List[str]]
    time_index: Optional[List]
    sampling_rate: Optional[float]
    missing_ratio: float
    data_source: str
    normalization_method: str
    train_indices: Tuple[int, int]
    val_indices: Tuple[int, int]
    test_indices: Tuple[int, int]


class DataLoaderError(Exception):
    """Data loader exception."""
    pass


class UniversalTimeSeriesDataset(Dataset):
    """
    Unified time series dataset.

    Features:
    - Chronological splitting (preserves time order)
    - Multi-format file support
    - Automatic normalization
    - Missing value handling
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        normalization: str = "standard",
        handle_missing: str = "interpolate",
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            data_path: Path to data file
            split: "train", "val", "test"
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            normalization: "standard", "minmax", "none"
            handle_missing: "interpolate", "forward_fill", "remove"
            feature_names: List of feature names
        """
        super().__init__()

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataLoaderError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        self.data_path = Path(data_path)
        self.split = split
        self.normalization = normalization

        raw_data = self._load_raw_data()
        cleaned_data = self._validate_and_clean(raw_data, handle_missing)
        normalized_data, self.scaler = self._normalize_data(cleaned_data, normalization)
        split_data = self._split_time_series(normalized_data, train_ratio, val_ratio, test_ratio)

        self.data = split_data[split]
        self.length = self.data.shape[0]

        self._create_metadata(raw_data, split_data, feature_names)

    def _squeeze_leading_batch_dim(self, data: np.ndarray) -> np.ndarray:
        """Remove leading batch dimension if size 1: (1,T,...) -> (T,...)."""
        if data.ndim >= 2 and data.shape[0] == 1:
            return np.squeeze(data, axis=0)
        return data

    def _load_raw_data(self) -> np.ndarray:
        """Load the observation array from disk."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        ext = self.data_path.suffix.lower()

        try:
            if ext == ".npz":
                data = np.load(self.data_path)

                candidate_keys = ['Y', 'X', 'data', 'arr_0', 'train_obs', 'test_obs']
                raw_data = None

                for key in candidate_keys:
                    if key in data:
                        candidate = data[key]
                        if ((candidate.ndim == 2 and candidate.shape[0] > 1) or
                            (candidate.ndim == 4 and candidate.shape[0] > 1)):
                            raw_data = candidate
                            break

                if raw_data is None:
                    available_keys = list(data.keys())
                    for key in available_keys:
                        candidate = data[key]
                        if (hasattr(candidate, 'ndim') and
                            ((candidate.ndim == 2 and candidate.shape[0] > 1 and candidate.shape[1] > 0) or
                             (candidate.ndim == 4 and candidate.shape[0] > 1))):
                            raw_data = candidate
                            break

                if raw_data is None:
                    available_info = []
                    for key in data.keys():
                        try:
                            shape = data[key].shape if hasattr(data[key], 'shape') else 'scalar'
                            dtype = data[key].dtype if hasattr(data[key], 'dtype') else type(data[key])
                            available_info.append(f"'{key}': shape={shape}, dtype={dtype}")
                        except Exception:
                            available_info.append(f"'{key}': (unreadable)")

                        raise DataLoaderError(
                            f"No suitable data found in npz file.\n"
                            f"Available data: {', '.join(available_info)}\n"
                            f"Expected: (T, d) time series or (T, H, W, C) image data"
                        )

            elif ext == ".npy":
                raw_data = np.load(self.data_path)

                if raw_data.ndim == 1:
                    raw_data = raw_data.reshape(-1, 1)
                elif raw_data.ndim > 2:
                    raw_data = raw_data.reshape(raw_data.shape[0], -1)
                elif raw_data.ndim == 0:
                    raise DataLoaderError(f"npy data is 0-dimensional: shape={raw_data.shape}")

            elif ext == ".csv":
                df = pd.read_csv(self.data_path, index_col=0 if 'time' in pd.read_csv(self.data_path, nrows=1).columns else None)
                raw_data = df.values
                self._csv_feature_names = df.columns.tolist()

            elif ext == ".json":
                with open(self.data_path, 'r') as f:
                    json_data = json.load(f)
                if 'data' in json_data:
                    raw_data = np.array(json_data['data'])
                    self._json_metadata = {k: v for k, v in json_data.items() if k != 'data'}
                else:
                    raw_data = np.array(json_data)

            else:
                raise DataLoaderError(f"Unsupported file format: {ext}. Supported: .npz, .npy, .csv, .json")

        except Exception as e:
            raise DataLoaderError(f"Data loading error ({self.data_path}): {e}")

        return raw_data

    def _validate_and_clean(self, data: np.ndarray, handle_missing: str) -> np.ndarray:
        """Validate and clean data (supports image data)."""
        if data.ndim == 1:
            warnings.warn("Converting 1D data to 2D")
            data = data.reshape(-1, 1)
        elif data.ndim == 2:
            pass
        elif data.ndim == 4:
            pass
        else:
            raise DataLoaderError(f"Unsupported data shape: {data.shape}. Supported: (T,), (T, d), (T, H, W, C)")

        T = data.shape[0]
        if T < 10:
            warnings.warn(f"Data length too short: T={T}. Minimum 10 recommended")

        missing_mask = np.isnan(data) | np.isinf(data)
        missing_ratio = missing_mask.sum() / data.size

        if missing_ratio > 0:
            warnings.warn(f"Missing values detected: {missing_ratio:.1%}")

            if handle_missing == "interpolate":
                if data.ndim == 2:
                    d = data.shape[1]
                    for j in range(d):
                        col_data = data[:, j]
                        missing_idx = np.isnan(col_data) | np.isinf(col_data)
                        if missing_idx.any():
                            valid_idx = ~missing_idx
                            if valid_idx.sum() > 1:
                                col_data[missing_idx] = np.interp(
                                    np.where(missing_idx)[0],
                                    np.where(valid_idx)[0],
                                    col_data[valid_idx]
                                )
                            else:
                                col_data[missing_idx] = 0.0
                elif data.ndim == 4:
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)

            elif handle_missing == "forward_fill":
                if data.ndim == 2:
                    data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
                else:
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)

            elif handle_missing == "remove":
                if data.ndim == 2:
                    valid_rows = ~missing_mask.any(axis=1)
                    data = data[valid_rows]
                    warnings.warn(f"Removed missing rows: {T} -> {data.shape[0]}")
                else:
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)

            if np.isnan(data).any() or np.isinf(data).any():
                warnings.warn("NaN/Inf remain after missing value handling. Replacing with zeros.")
                if data.ndim == 4:
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)
                else:
                    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

        self.missing_ratio = missing_ratio
        return data

    def _normalize_data(self, data: np.ndarray, method: str) -> Tuple[np.ndarray, Optional[object]]:
        """Normalize data (supports image data)."""
        if method == "none":
            return data, None

        elif method == "standard":
            if data.ndim == 4:
                data_flat = data.reshape(-1, data.shape[-1])
                scaler = StandardScaler()
                normalized_flat = scaler.fit_transform(data_flat)
                normalized = normalized_flat.reshape(data.shape)
            else:
                scaler = StandardScaler()
                normalized = scaler.fit_transform(data)

        elif method == "minmax":
            if data.ndim == 4:
                data_flat = data.reshape(-1, data.shape[-1])
                scaler = MinMaxScaler()
                normalized_flat = scaler.fit_transform(data_flat)
                normalized = normalized_flat.reshape(data.shape)
            else:
                scaler = MinMaxScaler()
                normalized = scaler.fit_transform(data)

        elif method == "unit_scale":
            # [0, 255] -> [0, 1] for image data
            if data.dtype == np.uint8:
                normalized = data.astype(np.float32) / 255.0
            else:
                normalized = data.astype(np.float32)
            scaler = None

        else:
            raise DataLoaderError(f"Unsupported normalization: {method}. Available: 'standard', 'minmax', 'unit_scale', 'none'")

        return normalized, scaler

    def _split_time_series(self, data: np.ndarray, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, np.ndarray]:
        """Split data chronologically."""
        T = data.shape[0]

        train_end = int(train_ratio * T)
        val_end = int((train_ratio + val_ratio) * T)

        # Ensure val/test are non-empty
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        if val_data.shape[0] == 0:
            val_data = data[max(0, train_end-1):train_end]
        if test_data.shape[0] == 0:
            test_data = data[-1:]

        splits = {
            "train": data[:train_end],
            "val": val_data,
            "test": test_data
        }

        self.split_indices = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, T)
        }

        return splits

    def _create_metadata(self, raw_data: np.ndarray, split_data: Dict[str, np.ndarray], feature_names: Optional[List[str]]):
        """Create dataset metadata."""
        if feature_names:
            final_feature_names = feature_names
        elif hasattr(self, '_csv_feature_names'):
            final_feature_names = self._csv_feature_names
        else:
            if hasattr(raw_data, 'shape'):
                if raw_data.ndim == 2:
                    final_feature_names = [f"feature_{i}" for i in range(raw_data.shape[1])]
                elif raw_data.ndim == 4:
                    T, H, W, C = raw_data.shape
                    final_feature_names = [f"image_pixel_{H}x{W}x{C}"]
                else:
                    final_feature_names = ["feature_0"]
            else:
                final_feature_names = ["feature_0"]

        self.metadata = DataMetadata(
            original_shape=raw_data.shape,
            feature_names=final_feature_names,
            time_index=None,
            sampling_rate=None,
            missing_ratio=self.missing_ratio,
            data_source=str(self.data_path),
            normalization_method=self.normalization,
            train_indices=self.split_indices["train"],
            val_indices=self.split_indices["val"],
            test_indices=self.split_indices["test"]
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get one observation sample."""
        return torch.from_numpy(self.data[idx]).float()

    def get_full_data(self) -> torch.Tensor:
        """Get the full split data as a tensor."""
        return torch.from_numpy(self.data).float()

    def inverse_transform(self, normalized_data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Inverse normalization transform."""
        if self.scaler is None:
            return normalized_data

        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()

        return self.scaler.inverse_transform(normalized_data)


def load_experimental_data_with_architecture(
    data_path: str,
    config: Dict[str, Any],
    split: str = "train",
    return_dataloaders: bool = False
) -> Union[Dataset, Dict[str, Dataset], DataLoader, Dict[str, DataLoader]]:
    """
    Architecture-aware unified data loader.

    Args:
        data_path: Path to data file
        config: Experiment config (uses model.encoder.type for dispatch)
        split: Data split ("train" | "val" | "test" | "all")
        return_dataloaders: Return DataLoaders instead of Datasets

    Returns:
        Dataset/DataLoader or dict of Datasets/DataLoaders
    """
    encoder_type = config.get('model', {}).get('encoder', {}).get('type', 'time_invariant')
    data_config = config.get('data', {})

    if encoder_type == "cnn_image":
        dataset_params = {k: v for k, v in data_config.items()
                         if k not in ['batch_size', 'num_workers', 'pin_memory',
                                     'image_shape', 'paper_data_protocol']}

        if split == "all":
            datasets = {}
            for s in ["train", "val", "test"]:
                datasets[s] = UniversalTimeSeriesDataset(
                    data_path=data_path,
                    split=s,
                    **dataset_params
                )

            if return_dataloaders:
                loaders = {}
                batch_size = data_config.get('batch_size', 16)
                for s, dataset in datasets.items():
                    loaders[s] = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=data_config.get('num_workers', 4),
                        pin_memory=data_config.get('pin_memory', True)
                    )
                return loaders
            else:
                return datasets
        else:
            dataset = UniversalTimeSeriesDataset(
                data_path=data_path,
                split=split,
                **dataset_params
            )

            if return_dataloaders:
                batch_size = data_config.get('batch_size', 16)
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=data_config.get('num_workers', 4),
                    pin_memory=data_config.get('pin_memory', True)
                )
            else:
                return dataset

    elif encoder_type == "time_invariant":
        dataset_params = {k: v for k, v in data_config.items()
                         if k not in ['batch_size', 'num_workers', 'pin_memory',
                                     'image_shape', 'paper_data_protocol']}

        if split == "all":
            datasets = {}
            for s in ["train", "val", "test"]:
                datasets[s] = UniversalTimeSeriesDataset(
                    data_path=data_path,
                    split=s,
                    **dataset_params
                )

            if return_dataloaders:
                loaders = {}
                batch_size = data_config.get('batch_size', 32)
                for s, dataset in datasets.items():
                    loaders[s] = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=data_config.get('num_workers', 4),
                        pin_memory=data_config.get('pin_memory', True)
                    )
                return loaders
            else:
                return datasets
        else:
            dataset = UniversalTimeSeriesDataset(
                data_path=data_path,
                split=split,
                **dataset_params
            )

            if return_dataloaders:
                batch_size = data_config.get('batch_size', 32)
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=data_config.get('num_workers', 4),
                    pin_memory=data_config.get('pin_memory', True)
                )
            else:
                return dataset
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def load_experimental_data(
    data_path: str,
    config: Optional[Dict[str, Any]] = None,
    split: str = "all",
    return_dataloaders: bool = False
) -> Union[Dict[str, torch.Tensor], Dict[str, Dataset], Dict[str, DataLoader]]:
    """
    Load data with the architecture-aware loader.

    Args:
        data_path: Path to data file
        config: Experiment config (None assumes time_invariant)
        split: Data split
        return_dataloaders: Return DataLoaders

    Returns:
        Data tensors, datasets, or dataloaders depending on the flags.
    """
    if config is None:
        config = {
            'model': {'encoder': {'type': 'time_invariant'}},
            'data': {'batch_size': 32}
        }

    result = load_experimental_data_with_architecture(
        data_path=data_path,
        config=config,
        split=split,
        return_dataloaders=return_dataloaders
    )

    if not return_dataloaders and split == "all" and isinstance(result, dict):
        if all(hasattr(dataset, 'get_full_data') for dataset in result.values()):
            tensor_dict = {}
            for s, dataset in result.items():
                tensor_dict[s] = dataset.get_full_data()
            tensor_dict['metadata'] = result['train'].metadata
            return tensor_dict

    return result
