# src/ssm/utils/data.py

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

def load_data_npz(path: str):
    """
    .npzファイルからX, Yを読み込みtorch.Tensorに変換
    """
    d = np.load(path)
    X = torch.from_numpy(d["X"].astype(np.float32))
    Y = torch.from_numpy(d["Y"].astype(np.float32))
    return X, Y

def build_dataloaders(
    data_path: str,
    seq_length: int = None,
    train_ratio: float = 0.6,
    batch_size: int = 32,
    seed: int = 0,
):
    """
    データをロードし、seq_length指定時は先頭を切り出して訓練/テストDataLoaderを返す

    Args:
      data_path: .npzファイルパス
      seq_length: None or 正の整数
      train_ratio: 訓練データ比率
      batch_size: ミニバッチサイズ
      seed: 乱数シード
    Returns:
      train_loader, test_loader
    """
    # フルデータロード
    X, Y = load_data_npz(data_path)

    # 先頭を切り出す
    if seq_length is not None and seq_length < X.size(0):
        X = X[:seq_length]
        Y = Y[:seq_length]

    # 訓練/テスト分割
    N = X.size(0)
    idx = torch.arange(N)
    torch.manual_seed(seed)
    perm = idx[torch.randperm(N)]
    n_train = int(train_ratio * N)
    idx_train = perm[:n_train]
    idx_test = perm[n_train:]

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_test,  Y_test  = X[idx_test],  Y[idx_test]

    # DataLoader化
    train_ds = TensorDataset(X_train, Y_train)
    test_ds  = TensorDataset(X_test,  Y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
