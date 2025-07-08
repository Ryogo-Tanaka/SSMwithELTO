# _tests/test_realization.py

import os
import sys
import torch
import numpy as np
import pytest

# プロジェクトルートを sys.path に追加して import 可能にする
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

from src.ssm.realization import (
    Realization      as ExactRealization,
    build_realization,
    RealizationError,
)

def generate_dummy_Y(T=20, p=5):
    """
    ランダムな時系列データ Y: shape (T, p) を作成して返すヘルパー。
    """
    np.random.seed(0)
    Y_np = np.random.randn(T, p).astype(np.float32)
    return torch.from_numpy(Y_np)

def test_exact_realization_shapes_and_values():
    """
    ExactRealization (alias: Realization) が正しく動作するかテスト。
    """
    Y = generate_dummy_Y(T=25, p=6)
    h = 3
    model = ExactRealization(past_horizon=h)

    # 正常に fit → filter → singular_value_sum が動くこと
    model.fit(Y)
    X_state = model.filter(Y)
    sv_sum = model.singular_value_reg()

    # 期待される出力形状: N = T - 2*h + 1 = 25 - 6 + 1 = 20
    N_expected = Y.shape[0] - 2*h + 1
    assert isinstance(X_state, torch.Tensor)
    assert X_state.shape == (N_expected, X_state.shape[1])
    assert X_state.shape[1] > 0
    assert isinstance(sv_sum, torch.Tensor)
    assert sv_sum.dim() == 0

# def test_approximate_realization_low_rank():
#     """
#     ApproximateRealization が低ランク近似で動作するかテスト。
#     """
#     Y = generate_dummy_Y(T=30, p=4)
#     h = 4
#     rank = 2
#     model = ApproximateRealization(past_horizon=h, rank=rank)

#     model.fit(Y)
#     X_state = model.filter(Y)
#     sv_sum = model.singular_value_sum()
 
#     # 行数: 30 - 2*4 + 1 = 23、列数が rank と一致
#     assert X_state.shape == (Y.shape[0] - 2*h + 1, rank)
#     assert isinstance(sv_sum, torch.Tensor)
#     assert sv_sum.dim() == 0

def test_build_realization_factory():
    """
    build_realization() が Realization を返し、
    rank オプションが正しく設定されていることをテスト。
    """
    from types import SimpleNamespace
    from src.ssm.realization import build_realization, Realization

    # デフォルト (rank=None -> 0) の場合
    cfg_default = SimpleNamespace(
        h=2, jitter=1e-6, cond_thresh=1e10,
        rank=None,
    )
    model_def = build_realization(cfg_default)
    assert isinstance(model_def, Realization)
    # デフォルト rank は 0
    assert model_def.rank == 0

    # rank=3 を渡した場合
    cfg_opt = SimpleNamespace(
        h=2, jitter=1e-6, cond_thresh=1e10,
        rank=3,
    )
    model_opt = build_realization(cfg_opt)
    assert isinstance(model_opt, Realization)
    assert model_opt.rank == 3

    # 無効な設定 (rank が負数) でエラー
    cfg_bad = SimpleNamespace(
        h=2, jitter=1e-6, cond_thresh=1e10,
        rank=-1,
    )
    with pytest.raises(ValueError):
        build_realization(cfg_bad)


def test_realization_error_on_small_sequence():
    """
    シーケンス長が小さすぎて RealizationError を投げるかテスト。
    """
    Y = generate_dummy_Y(T=5, p=4)
    h = 3  # 5 - 2*3 + 1 = 0
    model = ExactRealization(past_horizon=h)
    with pytest.raises(RealizationError):
        model.fit(Y)
