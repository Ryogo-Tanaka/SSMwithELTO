# src/config.py

import argparse
import yaml
from types import SimpleNamespace

def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """辞書を再帰的にSimpleNamespaceに変換"""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns

def load_cfg() -> SimpleNamespace:
    """
    YAMLファイルからSimpleNamespace形式で設定を読み込み

    使い方:
      python train.py --config configs/model.yaml
      cfg = load_cfg()
      cfg.model.encoder.type でアクセス可能
    """
    # コマンドライン引数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML設定ファイルパス")
    args = parser.parse_args()

    # YAML読み込み
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Namespace変換
    cfg = _dict_to_namespace(cfg_dict)

    # デフォルト値設定
    if not hasattr(cfg, "visualization"):
        cfg.visualization = _dict_to_namespace({})
    if not hasattr(cfg.visualization, "output_dir"):
        cfg.visualization.output_dir = "results/figs"

    return cfg
