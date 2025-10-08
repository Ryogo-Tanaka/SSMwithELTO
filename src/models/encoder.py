# src/models/encoder.py

import pkgutil
import importlib
from pathlib import Path
from . import architectures

# architecturesフォルダから<name>Encoderクラスを自動登録
_ENCODERS: dict[str, type] = {}
pkg_path = Path(architectures.__file__).parent

for module_info in pkgutil.iter_modules([str(pkg_path)]):
    name = module_info.name
    module = importlib.import_module(f"{architectures.__package__}.{name}")
    cls_name = name + "Encoder"
    if hasattr(module, cls_name):
        _ENCODERS[name] = getattr(module, cls_name)

def build_encoder(cfg):
    """
    エンコーダファクトリ関数
    Args:
      cfg: 設定オブジェクト (type: エンコーダ種類, その他: 初期化引数)
    Returns:
      エンコーダインスタンス
    """
    # dict/Namespaceから'type'取得
    if isinstance(cfg, dict):
        cfg_type = cfg.get('type')
    else:
        cfg_type = getattr(cfg, 'type', None)
    if cfg_type is None:
        raise ValueError("encoder config must include 'type' key or attribute")

    # tcnエンコーダサポート確認
    if cfg_type == "tcn" and "tcn" not in _ENCODERS:
        raise ValueError("tcnEncoder が見つかりません。tcn.py が正しくインポートされているか確認してください")

    try:
        cls = _ENCODERS[cfg_type]
    except KeyError:
        available_types = list(_ENCODERS.keys())
        raise ValueError(f"Unknown encoder type: {cfg_type}. Available types: {available_types}")

    # 初期化引数抽出
    if isinstance(cfg, dict):
        init_args = {k: v for k, v in cfg.items() if k != "type"}
    else:
        init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}

    # TCNエンコーダデフォルト設定
    if cfg_type == "tcn" and "output_dim" not in init_args:
        init_args["output_dim"] = 1
        print(f"[INFO] TCNエンコーダの output_dim を 1 に自動設定しました")

    return cls(**init_args)
