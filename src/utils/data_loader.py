# src/utils/data_loader.py
"""
タスク3用統一データローダー

機能:
- 多様なファイル形式の統一読み込み (.npz, .npy, .csv, .json)
- データ品質チェック・前処理
- 正規化・標準化 
- 時系列順を保った学習/検証/テスト分割
- メタデータ管理

前提データ型: (T, d) 多変量時系列 torch.FloatTensor
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
    """データメタデータ"""
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
    """データローダー専用例外"""
    pass


class UniversalTimeSeriesDataset(Dataset):
    """
    統一時系列データセット
    
    特徴:
    - 時系列順保持分割
    - 複数ファイル形式対応
    - 自動正規化
    - 欠損値処理
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
            data_path: データファイルパス
            split: "train", "val", "test"
            train_ratio: 訓練データ比率
            val_ratio: 検証データ比率 
            test_ratio: テストデータ比率
            normalization: "standard", "minmax", "none"
            handle_missing: "interpolate", "forward_fill", "remove"
            feature_names: 特徴量名リスト
        """
        super().__init__()
        
        # パラメータ検証
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataLoaderError(f"分割比率の合計が1.0でありません: {train_ratio + val_ratio + test_ratio}")
        
        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")
            
        # データ読み込み
        self.data_path = Path(data_path)
        self.split = split
        self.normalization = normalization
        
        # 生データ読み込み
        raw_data = self._load_raw_data()
        
        # データ検証・クリーニング
        cleaned_data = self._validate_and_clean(raw_data, handle_missing)
        
        # 正規化
        normalized_data, self.scaler = self._normalize_data(cleaned_data, normalization)
        
        # 時系列順分割
        split_data = self._split_time_series(normalized_data, train_ratio, val_ratio, test_ratio)
        
        # 分割データ取得
        self.data = split_data[split]
        self.length = self.data.shape[0]
        
        # メタデータ作成
        self._create_metadata(raw_data, split_data, feature_names)
    
    def _load_raw_data(self) -> np.ndarray:
        """生データ読み込み"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"データファイルが存在しません: {self.data_path}")
        
        ext = self.data_path.suffix.lower()
        
        try:
            if ext == ".npz":
                data = np.load(self.data_path)
                # 以前のチャットの仕様に従い、'Y' または 'arr_0' キーを探す
                if 'Y' in data:
                    raw_data = data['Y']
                elif 'arr_0' in data:
                    raw_data = data['arr_0']
                elif 'X' in data:  # 既存の実装との互換性
                    raw_data = data['X']
                else:
                    available_keys = list(data.keys())
                    raise DataLoaderError(f"npzファイルに 'Y', 'arr_0', 'X' キーが見つかりません。利用可能: {available_keys}")
                    
            elif ext == ".npy":
                raw_data = np.load(self.data_path)
                
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
                raise DataLoaderError(f"対応していないファイル形式: {ext}. 対応形式: .npz, .npy, .csv, .json")
                
        except Exception as e:
            raise DataLoaderError(f"データ読み込みエラー ({self.data_path}): {e}")
        
        return raw_data
    
    def _validate_and_clean(self, data: np.ndarray, handle_missing: str) -> np.ndarray:
        """データ検証・クリーニング"""
        # 形状チェック
        if data.ndim != 2:
            if data.ndim == 1:
                warnings.warn("1次元データを2次元に変換します")
                data = data.reshape(-1, 1)
            else:
                raise DataLoaderError(f"データは(T, d)の2次元配列である必要があります。実際の形状: {data.shape}")
        
        T, d = data.shape
        if T < 10:
            warnings.warn(f"データ長が短すぎます: T={T}. 最低10以上推奨")
        
        # 欠損値チェック
        missing_mask = np.isnan(data) | np.isinf(data)
        missing_ratio = missing_mask.sum() / data.size
        
        if missing_ratio > 0:
            warnings.warn(f"欠損値を検出: {missing_ratio:.1%}")
            
            if handle_missing == "interpolate":
                # 線形補間
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
                            
            elif handle_missing == "forward_fill":
                data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
                
            elif handle_missing == "remove":
                valid_rows = ~missing_mask.any(axis=1)
                data = data[valid_rows]
                warnings.warn(f"欠損行を削除: {T} -> {data.shape[0]} 行")
            
            # 無限大・NaNの最終チェック
            if np.isnan(data).any() or np.isinf(data).any():
                warnings.warn("欠損値処理後にもNaN/Infが残存。0で置換します。")
                data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.missing_ratio = missing_ratio
        return data
    
    def _normalize_data(self, data: np.ndarray, method: str) -> Tuple[np.ndarray, Optional[object]]:
        """データ正規化"""
        if method == "none":
            return data, None
            
        elif method == "standard":
            scaler = StandardScaler()
            normalized = scaler.fit_transform(data)
            
        elif method == "minmax":
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(data)
            
        else:
            raise DataLoaderError(f"対応していない正規化方法: {method}. 使用可能: 'standard', 'minmax', 'none'")
        
        return normalized, scaler
    
    def _split_time_series(self, data: np.ndarray, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, np.ndarray]:
        """時系列順分割"""
        T = data.shape[0]
        
        # 分割点計算
        train_end = int(train_ratio * T)
        val_end = int((train_ratio + val_ratio) * T)
        
        splits = {
            "train": data[:train_end],
            "val": data[train_end:val_end],
            "test": data[val_end:]
        }
        
        # 分割インデックス記録
        self.split_indices = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, T)
        }
        
        return splits
    
    def _create_metadata(self, raw_data: np.ndarray, split_data: Dict[str, np.ndarray], feature_names: Optional[List[str]]):
        """メタデータ作成"""
        # 特徴量名の決定
        if feature_names:
            final_feature_names = feature_names
        elif hasattr(self, '_csv_feature_names'):
            final_feature_names = self._csv_feature_names
        else:
            final_feature_names = [f"feature_{i}" for i in range(raw_data.shape[1])]
        
        self.metadata = DataMetadata(
            original_shape=raw_data.shape,
            feature_names=final_feature_names,
            time_index=None,  # TODO: 必要に応じて実装
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
        """データ取得"""
        sample = self.data[idx]  # shape: (d,)
        return torch.from_numpy(sample).float()
    
    def get_full_data(self) -> torch.Tensor:
        """分割データ全体を取得"""
        return torch.from_numpy(self.data).float()
    
    def inverse_transform(self, normalized_data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """正規化逆変換"""
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
    実験用データ読み込み・前処理パイプライン
    
    Args:
        data_path: データファイルパス
        config: 設定辞書（train_ratio, val_ratio, normalization等）
        return_loaders: DataLoaderを返すかTensorを返すか
        
    Returns:
        データ辞書またはDataLoader辞書
    """
    # デフォルト設定
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
    
    # データセット作成
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
        # DataLoader作成
        loaders = {}
        for split, dataset in datasets.items():
            # 時系列データはシャッフルしない
            loaders[split] = DataLoader(
                dataset,
                batch_size=default_config['batch_size'],
                shuffle=False,  # 時系列順保持
                num_workers=default_config['num_workers'],
                pin_memory=default_config['pin_memory']
            )
        return loaders
    else:
        # Tensor辞書作成
        data_dict = {split: dataset.get_full_data() for split, dataset in datasets.items()}
        # メタデータも含める
        data_dict['metadata'] = datasets['train'].metadata
        return data_dict


def create_data_loader_from_tensor(
    tensor: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = False,
    **dataloader_kwargs
) -> DataLoader:
    """
    Tensorから直接DataLoaderを作成
    
    Args:
        tensor: データテンソル (T, d)
        batch_size: バッチサイズ
        shuffle: シャッフルするか
        **dataloader_kwargs: DataLoader追加引数
        
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


# 既存互換性のための関数
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
    """
    既存コードとの互換性のためのラッパー関数
    """
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


if __name__ == "__main__":
    # 使用例とテスト
    print("統一データローダーのテスト")
    
    # テストデータ作成
    test_data = np.random.randn(100, 5)
    test_path = "test_data.npy"
    np.save(test_path, test_data)
    
    try:
        # データ読み込みテスト
        data_dict = load_experimental_data(test_path)
        
        print("データ分割結果:")
        for split, data in data_dict.items():
            if isinstance(data, torch.Tensor):
                print(f"  {split}: {data.shape}")
            else:
                print(f"  {split}: {type(data)}")
        
        print("\n✅ 統一データローダーのテストが完了しました")
        
    finally:
        # テストファイル削除
        if os.path.exists(test_path):
            os.remove(test_path)