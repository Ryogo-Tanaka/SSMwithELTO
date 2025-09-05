# src/models/decoder.py (修正版)

import pkgutil
import importlib
from pathlib import Path
from . import architectures

# Scan architectures/ for any <name>Decoder classes
_DECODERS: dict[str, type] = {}
pkg_path = Path(architectures.__file__).parent

for module_info in pkgutil.iter_modules([str(pkg_path)]):
    name = module_info.name              # e.g. "_mlp", "_cnn", "tcn"
    module = importlib.import_module(f"{architectures.__package__}.{name}")
    cls_name = name + "Decoder"
    if hasattr(module, cls_name):
        _DECODERS[name] = getattr(module, cls_name)

def build_decoder(cfg):
    """
    Factory for decoders.
    Args:
      cfg: dict / Namespace with
        - type: key in _DECODERS (e.g. "tcn", "_mlp")
        - other keys: kwargs for that decoder class
    Returns:
      An instance of the chosen Decoder class.
    """
    # cfg が dict ならキーアクセス、Namespace なら属性アクセスで 'type' を取得
    if isinstance(cfg, dict):
        cfg_type = cfg.get('type')
    else:
        cfg_type = getattr(cfg, 'type', None)
    if cfg_type is None:
        raise ValueError("decoder config must include 'type' key or attribute")
    
    # **修正**: tcn デコーダのサポート確認
    if cfg_type == "tcn" and "tcn" not in _DECODERS:
        raise ValueError("tcnDecoder が見つかりません。tcn.py が正しくインポートされているか確認してください")
    
    try:
        cls = _DECODERS[cfg_type]
    except KeyError:
        available_types = list(_DECODERS.keys())
        raise ValueError(f"Unknown decoder type: {cfg_type}. Available types: {available_types}")
    
    # Dictionary or Namespace に対応する
    # 残りのキーを初期化引数に
    if isinstance(cfg, dict):
        init_args = {k: v for k, v in cfg.items() if k != "type"}
    else:
        init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}
    
    # **新機能**: TCNデコーダの自動設定
    if cfg_type == "tcn":
        if "output_dim" not in init_args:
            raise ValueError("tcnDecoder には output_dim の指定が必要です")
        print(f"[INFO] TCNデコーダを output_dim={init_args['output_dim']} で初期化します")
    
    return cls(**init_args)