# _tests/test_train_smoke.py

import os
import sys
import numpy as np
import torch
import yaml
import subprocess
import tempfile
from click.testing import CliRunner

# train.py から main 関数だけをインポート
from src.train import main as train_main

def create_dummy_npz(tmp_path, T=50, d=5):
    """
    ダミー時系列データを .npz で作成してパスを返す。
    DataLoader が 'X' キーを探すので必ず 'X' という名前で保存。
    """
    X = np.random.randn(T, d).astype(np.float32)
    fp = tmp_path / "dummy.npz"
    # 特徴量は必ず 'X' キー
    np.savez(fp, X=X)
    return str(fp)

def create_tmp_config(tmp_path, data_path):
    """demo.yaml 風の最小設定ファイルを作ってパスを返す"""
    cfg = {
        "model": {
            "encoder": {"type":"_mlp","input_dim":5,"output_dim":8,"hidden_sizes":[16],"activation":"ReLU"},
            "decoder_y2d": {"type":"_mlp","input_dim":8,"output_dim":5,"hidden_sizes":[16],"activation":"ReLU"},
        },
        "ssm": {
            "realization": {"h":2,"jitter":1e-6,"cond_thresh":1e12,"rank":3,"reg_type":"sum"},
            "observation": {"kernel":"linear","gamma":None,"reg_lambda":1e-3,"approx":False,"approx_rank":2},
        },
        "training": {
            "data_path": data_path,
            "batch_size": 16,
            "epochs": 1,
            "lr": 1e-3,
            "svd_weight": 0.1
        },
        "visualization": {"output_dir": str(tmp_path / "figs")},
    }
    fp = tmp_path / "tmp_config.yaml"
    with open(fp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return str(fp)

def test_train_runs_without_error(tmp_path, monkeypatch):
    """
    1エポックだけ回して、例外なく終了するかを確認するスモークテスト
    """
    data_fp = create_dummy_npz(tmp_path)
    cfg_fp  = create_tmp_config(tmp_path, data_fp)

    # 標準出力を抑え、例外の有無だけチェック
    # runner = CliRunner()
    # result = runner.invoke(train_main, ["--config", cfg_fp])
    # assert result.exit_code == 0, result.output
    import sys
    # sys.argv を擬似的に書き換え
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", cfg_fp])
    # main 関数を直接呼び出し。例外が出なければ OK。
    train_main()
