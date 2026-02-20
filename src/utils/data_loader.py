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

import os
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict

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
    has_target_data: bool = False
    target_shape: Optional[Tuple[int, int]] = None
    target_feature_names: Optional[List[str]] = None
    target_dtype: Optional[str] = None


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
        feature_names: Optional[List[str]] = None,
        experiment_mode: str = "reconstruction"
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
            experiment_mode: "reconstruction" or "target_prediction"
        """
        super().__init__()

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataLoaderError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        self.data_path = Path(data_path)
        self.split = split
        self.normalization = normalization
        self.experiment_mode = experiment_mode

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

    def _detect_target_data(self, data: dict) -> Dict[str, Any]:
        """Auto-detect target data (quad_link format, with class-level caching)."""
        cache_key = str(self.data_path)
        if not hasattr(UniversalTimeSeriesDataset, '_class_target_cache'):
            UniversalTimeSeriesDataset._class_target_cache = {}

        if cache_key in UniversalTimeSeriesDataset._class_target_cache:
            return UniversalTimeSeriesDataset._class_target_cache[cache_key]

        target_info = {
            'has_target': False,
            'input_data': None,
            'target_data': None,
            'target_test_data': None
        }

        target_keys_train = ['train_targets', 'y_train', 'target_train', 'labels_train']
        target_keys_test = ['test_targets', 'y_test', 'target_test', 'labels_test']
        input_keys_train = ['train_obs', 'X_train', 'input_train', 'obs_train']
        input_keys_test = ['test_obs', 'X_test', 'input_test', 'obs_test']

        target_train = None
        target_test = None
        input_train = None
        input_test = None

        for key in target_keys_train:
            if key in data:
                target_train = self._squeeze_leading_batch_dim(data[key])
                break

        for key in target_keys_test:
            if key in data:
                target_test = self._squeeze_leading_batch_dim(data[key])
                break

        for key in input_keys_train:
            if key in data:
                input_train = self._squeeze_leading_batch_dim(data[key])
                break

        for key in input_keys_test:
            if key in data:
                input_test = self._squeeze_leading_batch_dim(data[key])
                break

        if target_train is not None and input_train is not None:
            target_info['has_target'] = True
            target_info['input_data'] = input_train
            target_info['target_data'] = target_train
            target_info['target_test_data'] = target_test
            target_info['input_test_data'] = input_test

        UniversalTimeSeriesDataset._class_target_cache[cache_key] = target_info
        return target_info

    def _load_raw_data(self) -> np.ndarray:
        """Load raw data with automatic target detection."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        ext = self.data_path.suffix.lower()

        try:
            if ext == ".npz":
                data = np.load(self.data_path)

                if self.experiment_mode == "target_prediction":
                    target_info = self._detect_target_data(data)

                    if target_info['has_target']:
                        self.has_target = True
                        self.target_data = target_info['target_data']
                        self.target_test_data = target_info.get('target_test_data', None)
                        self.input_test_data = target_info.get('input_test_data', None)
                        raw_data = target_info['input_data']
                    else:
                        raise DataLoaderError(
                            f"Target prediction mode specified but no target data found.\n"
                            f"Available keys: {list(data.keys())}\n"
                            f"Expected keys: train_targets, y_train, target_train, etc."
                        )
                else:
                    self.has_target = False
                    self.target_data = None
                    self.target_test_data = None
                    self.input_test_data = None

                    # Search for data array by priority
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
                            except:
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

        has_target = getattr(self, 'has_target', False)
        target_shape = None
        target_feature_names = None
        target_dtype = None

        if has_target and hasattr(self, 'target_data') and self.target_data is not None:
            target_shape = self.target_data.shape
            target_dtype = str(self.target_data.dtype)
            target_feature_names = [f"target_{i}" for i in range(self.target_data.shape[1])]

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
            test_indices=self.split_indices["test"],
            has_target_data=has_target,
            target_shape=target_shape,
            target_feature_names=target_feature_names,
            target_dtype=target_dtype
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get data item (with optional target)."""
        sample = self.data[idx]

        if hasattr(self, 'has_target') and self.has_target:
            if hasattr(self, 'target_data') and self.target_data is not None:
                target_sample = self._get_target_for_split(idx)
                return (torch.from_numpy(sample).float(),
                       torch.from_numpy(target_sample).float())
            else:
                return torch.from_numpy(sample).float()
        else:
            return torch.from_numpy(sample).float()

    def _get_target_for_split(self, idx: int) -> np.ndarray:
        """Get target data for the current split."""
        if not hasattr(self, 'target_data') or self.target_data is None:
            raise ValueError("Target data not set")

        split_start, split_end = self.split_indices[self.split]
        target_split_data = self.target_data[split_start:split_end]

        return target_split_data[idx]

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


def load_experimental_data(
    data_path: str,
    config: Optional[Dict[str, Any]] = None,
    return_loaders: bool = False
) -> Union[Dict[str, torch.Tensor], Dict[str, DataLoader]]:
    """
    Load and preprocess experimental data.

    Args:
        data_path: Path to data file
        config: Configuration dict (train_ratio, val_ratio, normalization, etc.)
        return_loaders: Return DataLoaders instead of Tensors

    Returns:
        Data dict or DataLoader dict
    """
    default_config = {
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'test_ratio': 0.1,
        'normalization': 'standard',
        'handle_missing': 'interpolate',
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True
    }

    if config:
        default_config.update(config)

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = UniversalTimeSeriesDataset(
            data_path=data_path,
            split=split,
            train_ratio=default_config['train_ratio'],
            val_ratio=default_config['val_ratio'],
            test_ratio=default_config['test_ratio'],
            normalization=default_config['normalization'],
            handle_missing=default_config['handle_missing']
        )

    if return_loaders:
        loaders = {}
        for split, dataset in datasets.items():
            loaders[split] = DataLoader(
                dataset,
                batch_size=default_config['batch_size'],
                shuffle=False,  # Preserve time order
                num_workers=default_config['num_workers'],
                pin_memory=default_config['pin_memory']
            )
        return loaders
    else:
        data_dict = {split: dataset.get_full_data() for split, dataset in datasets.items()}
        data_dict['metadata'] = datasets['train'].metadata
        return data_dict


def create_data_loader_from_tensor(
    tensor: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = False,
    **dataloader_kwargs
) -> DataLoader:
    """
    Create a DataLoader directly from a tensor.

    Args:
        tensor: Data tensor (T, d)
        batch_size: Batch size
        shuffle: Whether to shuffle
        **dataloader_kwargs: Additional DataLoader arguments

    Returns:
        DataLoader
    """

    class TensorDataset(Dataset):
        def __init__(self, data: torch.Tensor):
            self.data = data

        def __len__(self) -> int:
            return self.data.shape[0]

        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.data[idx]

    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)


def build_dataloaders(
    file_path: str,
    batch_size: int,
    split: str = "train",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Legacy compatibility wrapper."""
    dataset = UniversalTimeSeriesDataset(
        data_path=file_path,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


class QuadImageDataset(Dataset):
    """
    Quadcopter image dataset for image self-reconstruction.

    Data shape:
    - Images: (T, H, W, C) = (1500, 48, 48, 1)

    Returns (image, image) pairs for autoencoder-style training.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        image_normalization: str = "unit_scale",
        **kwargs
    ):
        """
        Args:
            data_path: Path to image npz file (containing train_obs, test_obs)
            split: "train", "val", "test"
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            image_normalization: "unit_scale", "standard", "none"
        """
        super().__init__()

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataLoaderError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        self.data_path = Path(data_path)
        self.split = split
        self.image_normalization = image_normalization

        data = np.load(data_path)

        if split in ["train", "val"]:
            self.images = data['train_obs']
        else:
            self.images = data['test_obs']

        # Chronological split within train data
        T = len(self.images)
        if split == "train":
            start_idx = 0
            end_idx = int(T * train_ratio)
        elif split == "val":
            start_idx = int(T * train_ratio)
            end_idx = int(T * (train_ratio + val_ratio))
        else:
            start_idx = 0
            end_idx = T

        self.images = self.images[start_idx:end_idx]

        self._normalize_data()

        self.metadata = DataMetadata(
            original_shape=(T, 48, 48, 1),
            feature_names=['image_features'],
            time_index=None,
            sampling_rate=None,
            missing_ratio=0.0,
            data_source=str(data_path),
            normalization_method=f"image:{image_normalization}",
            train_indices=(0, int(T * train_ratio)),
            val_indices=(int(T * train_ratio), int(T * (train_ratio + val_ratio))),
            test_indices=(int(T * (train_ratio + val_ratio)), T)
        )

    def _normalize_data(self):
        """Normalize image data."""
        if self.image_normalization == "unit_scale":
            self.images = self.images.astype(np.float32) / 255.0
        elif self.image_normalization == "standard":
            mean = self.images.mean()
            std = self.images.std()
            self.images = (self.images - mean) / (std + 1e-8)
        elif self.image_normalization == "none":
            self.images = self.images.astype(np.float32)

    def __getitem__(self, idx):
        """Returns (image, image) self-reconstruction pair."""
        image = torch.FloatTensor(self.images[idx])
        return image, image

    def __len__(self):
        return len(self.images)

    def get_full_data(self):
        """Get full split data (chronological order)."""
        images = torch.FloatTensor(self.images)
        return images


def load_experimental_data_with_architecture(
    data_path: str,
    config: Dict[str, Any],
    split: str = "train",
    return_dataloaders: bool = False,
    experiment_mode: Optional[str] = None
) -> Union[Dataset, Dict[str, Dataset], DataLoader, Dict[str, DataLoader]]:
    """
    Architecture-aware unified data loader (backward compatible).

    Args:
        data_path: Path to data file
        config: Experiment config (uses model.encoder.type for dispatch)
        split: Data split ("train" | "val" | "test" | "all")
        return_dataloaders: Return DataLoaders instead of Datasets
        experiment_mode: "reconstruction" | "target_prediction" (None = auto from config)

    Returns:
        Dataset/DataLoader or dict of Datasets/DataLoaders
    """
    encoder_type = config.get('model', {}).get('encoder', {}).get('type', 'time_invariant')
    data_config = config.get('data', {})

    if experiment_mode is None:
        experiment_mode = config.get('experiment', {}).get('mode', 'reconstruction')

    if encoder_type == "cnn_image":
        # Filter out non-Dataset parameters
        dataset_params = {k: v for k, v in data_config.items()
                         if k not in ['batch_size', 'num_workers', 'pin_memory',
                                     'image_shape', 'target_shape',
                                     'paper_data_protocol']}

        if split == "all":
            datasets = {}
            for s in ["train", "val", "test"]:
                datasets[s] = UniversalTimeSeriesDataset(
                    data_path=data_path,
                    split=s,
                    experiment_mode=experiment_mode,
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
                experiment_mode=experiment_mode,
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
                                     'image_shape', 'target_shape',
                                     'paper_data_protocol']}

        if split == "all":
            datasets = {}
            for s in ["train", "val", "test"]:
                datasets[s] = UniversalTimeSeriesDataset(
                    data_path=data_path,
                    split=s,
                    experiment_mode=experiment_mode,
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
                experiment_mode=experiment_mode,
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
    Backward-compatible data loading wrapper.

    Args:
        data_path: Path to data file
        config: Experiment config (None assumes time_invariant)
        split: Data split
        return_dataloaders: Return DataLoaders

    Returns:
        Data dict maintaining backward compatibility
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


if __name__ == "__main__":
    print("Testing unified data loader")

    test_data = np.random.randn(100, 5)
    test_path = "test_data.npy"
    np.save(test_path, test_data)

    try:
        data_dict = load_experimental_data(test_path)

        print("Split results:")
        for split, data in data_dict.items():
            if isinstance(data, torch.Tensor):
                print(f"  {split}: {data.shape}")
            else:
                print(f"  {split}: {type(data)}")

        print("\nData loader test completed")

    finally:
        if os.path.exists(test_path):
            os.remove(test_path)
