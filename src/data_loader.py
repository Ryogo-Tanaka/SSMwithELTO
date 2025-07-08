# src/data_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DataLoaderError(Exception):
    """データ読み込み中に発生したエラーを示すカスタム例外"""
    pass

class NpTimeSeriesDataset(Dataset):
    """
    .npz または .npy の時系列データを、
    時系列順を保ったまま train/val/test に分割して返す Dataset。

    - .npz ファイル: {"X":…, "y":…} のキーで保存（X: (T,d), y: (T,) または (T,k)）
    - .npy ファイル: 単一の配列 X: (T,d)
    """

    def __init__(
        self,
        file_path: str,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """
        Args:
          file_path   : .npz or .npy ファイルへのパス
          split       : "train", "val", "test"
          train_ratio : 訓練データの割合 (0～1)
          val_ratio   : 検証データの割合 (0～1)
          test_ratio  : テストデータの割合 (0～1)
        """
        super().__init__()

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npz":
            data = np.load(file_path)
            if "X" not in data:
                raise DataLoaderError(f".npz must contain key 'X'; found keys: {list(data.keys())}")
            X_all = data["X"]              # shape: (T, d)
            y_all = data.get("y", None)
            if y_all is not None and y_all.shape[0] != X_all.shape[0]:
                raise DataLoaderError("Length of 'y' must match length of 'X'")
        elif ext == ".npy":
            X_all = np.load(file_path)    # shape: (T, d)
            y_all = None
        else:
            raise DataLoaderError(f"Unsupported file extension '{ext}'. Use .npz or .npy.")

        if X_all.ndim != 2:
            raise DataLoaderError(f"X must be a 2D array of shape (T, d); got {X_all.shape}")
        T, d = X_all.shape

        # 時系列順に分割
        n_train = int(train_ratio * T)
        n_val   = int(val_ratio   * T)
        # n_test は残り
        if train_ratio + val_ratio + test_ratio != 1.0:
            # テスト比率は自動的に残りを使う
            test_ratio = 1.0 - (train_ratio + val_ratio)
        n_test = T - (n_train + n_val)

        if split == "train":
            start, end = 0, n_train
        elif split == "val":
            start, end = n_train, n_train + n_val
        elif split == "test":
            start, end = n_train + n_val, T
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        # 連続したインデックス範囲をそのまま使う
        self.X = X_all[start:end]
        self.y = None if y_all is None else y_all[start:end]
        self.length = end - start

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_np = self.X[idx]                     # shape: (d,)
        x_tensor = torch.from_numpy(x_np).float()
        if self.y is not None:
            y_np = self.y[idx]
            if np.issubdtype(y_np.dtype, np.integer):
                y_tensor = torch.from_numpy(y_np).long()
            else:
                y_tensor = torch.from_numpy(y_np).float()
            return x_tensor, y_tensor
        else:
            return x_tensor, torch.tensor(0)

def build_dataloaders(
    file_path: str,
    batch_size: int,
    split: str = "train",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    NpTimeSeriesDataset を作成し、DataLoader を返すユーティリティ。
    時系列順を保つため、shuffle=False 固定です。

    Args:
      file_path   : .npz または .npy ファイルへのパス
      batch_size  : ミニバッチサイズ
      split       : "train", "val", "test"
      train_ratio : 訓練データ割合
      val_ratio   : 検証データ割合
      test_ratio  : テストデータ割合
      num_workers : DataLoader のワーカープロセス数
      pin_memory  : pin_memory フラグ
    Returns:
      DataLoader
    """
    dataset = NpTimeSeriesDataset(
        file_path=file_path,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,           # 時系列順を変えない
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return loader





# # src/data_loader.py

# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader

# class DataLoaderError(Exception):
#     """データ読み込み中に発生したエラーを示すカスタム例外"""
#     pass

# class NpDataset(Dataset):
#     """
#     拡張子 .npz または .npy のデータファイルを読み込み、
#     train/val/test に分割してバッチ単位で返す Dataset。

#     - .npz ファイル: 複数の配列を {"X":…, "y":…} のように保存できる形式
#     - .npy ファイル: 単一の配列のみ保存できる形式（ラベルなしデータ向け）
#     """

#     def __init__(
#         self,
#         file_path: str,
#         split: str = "train",
#         train_ratio: float = 0.7,
#         val_ratio: float = 0.15,
#         test_ratio: float = 0.15,
#         seed: int = 42
#     ):
#         """
#         Args:
#           file_path:   .npz または .npy ファイルへのパス
#           split:       "train", "val", "test"
#           train_ratio: 訓練データの割合
#           val_ratio:   検証データの割合
#           test_ratio:  テストデータの割合
#           seed:        分割時の乱数シード
#         """

#         super().__init__()

#         # 1) ファイル存在チェック
#         if not os.path.isfile(file_path):
#             raise FileNotFoundError(f"Data file not found: {file_path}")

#         # 2) 拡張子を判定しながらデータ読み込み
#         ext = os.path.splitext(file_path)[1].lower()  # e.g. ".npz" or ".npy"

#         if ext == ".npz":
#             data = np.load(file_path)
#             if "X" not in data:
#                 raise DataLoaderError(f".npz must contain key 'X'; found keys: {list(data.keys())}")
#             X_all = data["X"]  # shape: (T, d)
#             # y があれば取り出し、なければラベルなしとみなす
#             if "y" in data:
#                 y_all = data["y"]
#                 if y_all.shape[0] != X_all.shape[0]:
#                     raise DataLoaderError("Length of 'y' must match length of 'X'")
#             else:
#                 y_all = None

#         elif ext == ".npy":
#             # 単一配列のみ保存できる形式 → ラベルなしデータとして扱う
#             X_all = np.load(file_path)  # shape: (T, d)
#             y_all = None
#         else:
#             raise DataLoaderError(f"Unsupported file extension '{ext}'. Use .npz or .npy.")

#         # 3) X_all の形状チェック
#         if X_all.ndim != 2:
#             raise DataLoaderError(f"X must be a 2D array of shape (T, d); got shape {X_all.shape}")

#         T, d = X_all.shape

#         # 4) インデックスをシャッフルして train/val/test に分割
#         indices = np.arange(T)
#         rng = np.random.default_rng(seed)
#         rng.shuffle(indices)

#         n_train = int(train_ratio * T)
#         n_val   = int(val_ratio   * T)
#         n_test  = T - (n_train + n_val)

#         if split == "train":
#             selected_idx = indices[:n_train]
#         elif split == "val":
#             selected_idx = indices[n_train : n_train + n_val]
#         elif split == "test":
#             selected_idx = indices[n_train + n_val :]
#         else:
#             raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

#         # 5) 選ばれたインデックスでサブセットを作成
#         self.X = X_all[selected_idx]            # NumPy array, shape (split_size, d)
#         self.y = None if y_all is None else y_all[selected_idx]
#         self.length = self.X.shape[0]

#     def __len__(self):
#         """
#         Dataset の長さ（サンプル数）を返す。
#         """
#         return self.length

#     def __getitem__(self, idx):
#         """
#         idx 番目のサンプルを返す。
#         ラベルがある場合は (x_tensor, y_tensor) 、
#         ラベルなしの場合は (x_tensor, torch.tensor(0)) を返す。
#         """
#         # X を Tensor に変換
#         x_np = self.X[idx]  # NumPy array of shape (d,)
#         x_tensor = torch.from_numpy(x_np).float()  # FloatTensor に

#         if self.y is not None:
#             # y が整数であれば LongTensor、浮動小数点なら FloatTensor に
#             y_np = self.y[idx]
#             if np.issubdtype(y_np.dtype, np.integer):
#                 y_tensor = torch.from_numpy(y_np).long()
#             else:
#                 y_tensor = torch.from_numpy(y_np).float()
#             return x_tensor, y_tensor
#         else:
#             # ラベルなしデータはダミーラベル 0 を返す
#             return x_tensor, torch.tensor(0)

# def build_dataloaders(
#     file_path: str,
#     batch_size: int,
#     split: str = "train",
#     train_ratio: float = 0.7,
#     val_ratio: float = 0.15,
#     test_ratio: float = 0.15,
#     seed: int = 42,
#     num_workers: int = 4,
#     pin_memory: bool = True
# ):
#     """
#     NpDataset を作成し、DataLoader を返すユーティリティ。

#     Args:
#       file_path:   .npz または .npy ファイルへのパス
#       batch_size:  ミニバッチサイズ
#       split:       "train", "val" または "test"
#       train_ratio: 訓練データの割合
#       val_ratio:   検証データの割合
#       test_ratio:  テストデータの割合
#       seed:        分割用乱数シード
#       num_workers: DataLoader のワーカープロセス数
#       pin_memory:  DataLoader の pin_memory フラグ
#     Returns:
#       DataLoader オブジェクト
#     """
#     dataset = NpDataset(
#         file_path=file_path,
#         split=split,
#         train_ratio=train_ratio,
#         val_ratio=val_ratio,
#         test_ratio=test_ratio,
#         seed=seed
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(split == "train"),  # 訓練時のみデータをシャッフル→要確認
#         num_workers=num_workers,
#         pin_memory=pin_memory
#     )
#     return loader
