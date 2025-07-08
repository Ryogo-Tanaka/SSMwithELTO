# # _tests/test_observation.py

# import os
# import sys
# import torch
# import numpy as np
# import pytest

# # プロジェクトルートを sys.path に追加
# current_dir  = os.path.dirname(__file__)
# project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.insert(0, project_root)

# from src.ssm.observation import CMEObservation, RealizationError  # 例外名を統一

# def generate_dummy_states_and_features(N=20, r=3, p=4):
#     """
#     状態系列 X_state と特徴系列 Y を生成
#     """
#     torch.manual_seed(0)
#     X_state = torch.randn(N, r)
#     W       = torch.randn(r, p)
#     Y       = X_state @ W + 0.1 * torch.randn(N, p)
#     return X_state, Y

# def test_cme_observation_linear_kernel():
#     X_state, Y = generate_dummy_states_and_features(N=25, r=3, p=5)
#     obs = CMEObservation(kernel="linear", gamma=None, reg_lambda=1e-3, approx=False, approx_rank=None)
#     obs.fit(X_state, Y)
#     Y_pred = obs.decode(X_state)
#     assert isinstance(Y_pred, torch.Tensor)
#     assert Y_pred.shape == Y.shape
#     mse = torch.mean((Y_pred - Y) ** 2).item()
#     assert mse < 1e-2

# def test_cme_observation_rbf_kernel():
#     X_state, Y = generate_dummy_states_and_features(N=30, r=2, p=3)
#     obs = CMEObservation(kernel="rbf", gamma=0.5, reg_lambda=1e-4, approx=False, approx_rank=None)
#     obs.fit(X_state, Y)
#     Y_pred = obs.decode(X_state)
#     assert isinstance(Y_pred, torch.Tensor)
#     assert Y_pred.shape == Y.shape
#     mse = torch.mean((Y_pred - Y) ** 2).item()
#     assert mse < 1e-1

# def test_cme_observation_approximate_gram():
#     X_state, Y = generate_dummy_states_and_features(N=40, r=3, p=4)
#     obs = CMEObservation(kernel="rbf", gamma=1.0, reg_lambda=1e-3, approx=True, approx_rank=5)
#     obs.fit(X_state, Y)
#     Y_pred = obs.decode(X_state)
#     assert Y_pred.shape == Y.shape

# def test_observation_error_on_singular_kernel():
#     """
#     正則化なしで特異行列の場合に例外が投げられることをチェック
#     """
#     N, r, p = 10, 3, 4
#     X_state = torch.zeros(N, r)
#     Y       = torch.randn(N, p)
#     obs = CMEObservation(kernel="linear", gamma=None, reg_lambda=0.0, approx=False, approx_rank=None)
#     with pytest.raises(RealizationError):
#         obs.fit(X_state, Y)
