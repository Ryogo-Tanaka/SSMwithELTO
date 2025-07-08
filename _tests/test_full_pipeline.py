# _tests/test_full_pipeline.py
import os
import sys

import pytest
import torch

# プロジェクトの src ディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ssm.realization import Realization
from ssm.observation import CMEObservation
from models.architectures._mlp import _mlpEncoder, _mlpDecoder


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(0)


def test_full_pipeline_shapes():
    # --- ハイパーパラメータ ---
    T = 50         # 元データ系列長
    d = 8          # 観測次元
    p = 6          # 特徴次元
    h = 3          # Realization の past_horizon
    r = 4          # 潜在次元

    # 1) ダミー観測データを生成し、エンコーダで特徴化
    X = torch.randn(T, d)
    encoder = _mlpEncoder(input_dim=d, hidden_sizes=[16, 16], output_dim=p, activation="ReLU")
    Y = encoder(X)  # => (T, p)
    assert Y.shape == (T, p), "Encoder 出力の形状が期待と異なります"

    # 2) サブスペース同定 (Realization) で潜在状態系列を推定
    real = Realization(past_horizon=h, jitter=1e-6, cond_thresh=1e12, rank=r)
    real.fit(Y)
    X_states = real.filter(Y)  # => (T - 2h + 1, r)
    N_x = X_states.shape[0]
    assert X_states.shape == (T - 2*h + 1, r), "Realization.filter の出力形状が期待と異なります"

    # 3) CMEObservation に与えるためのダミー特徴列を用意
    #    — fit 内部で (n_feats - n_states + 1)/2 = h が成り立つように長さを調整
    Y_feats = torch.randn(N_x + 2*h, p)
    obs = CMEObservation(kernel="rbf", gamma=1.0, reg_lambda=1e-3)
    obs.fit(X_states, Y_feats)
    # decode では X_states 全体を入力とし、(N_x, p) の予測特徴を返す
    Y_pred = obs.decode(X_states)
    assert Y_pred.shape == (N_x -1, p), "CMEObservation.decode の出力形状が期待と異なります"

    # 4) 最終デコーダで観測空間に再構成
    decoder = _mlpDecoder(input_dim=p, output_dim=d, hidden_sizes=[16, 16], activation="ReLU")
    X_rec = decoder(Y_pred)
    assert X_rec.shape == (N_x, d), "Decoder 出力の形状が期待と異なります"
