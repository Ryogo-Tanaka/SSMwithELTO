# src/models/decoder.py

import pkgutil
import importlib
from pathlib import Path
from . import architectures

# architecturesから<name>Decoderと<name>_targetDecoderクラスを自動登録
_DECODERS: dict[str, type] = {}
_TARGET_DECODERS: dict[str, type] = {}
pkg_path = Path(architectures.__file__).parent

for module_info in pkgutil.iter_modules([str(pkg_path)]):
    name = module_info.name
    module = importlib.import_module(f"{architectures.__package__}.{name}")

    # 通常デコーダ: <name>Decoder
    cls_name = name + "Decoder"
    if hasattr(module, cls_name):
        _DECODERS[name] = getattr(module, cls_name)

    # ターゲット予測デコーダ: <name>_targetDecoder
    target_cls_name = name + "_targetDecoder"
    if hasattr(module, target_cls_name):
        _TARGET_DECODERS[name] = getattr(module, target_cls_name)

def build_decoder(cfg, experiment_mode=None):
    """
    デコーダファクトリ関数 (experiment_mode対応)
    Args:
      cfg: 設定オブジェクト (type: デコーダ種類, その他: 初期化引数)
      experiment_mode: "target_prediction" | "reconstruction" | None
    Returns:
      デコーダインスタンス
    """
    # dict/Namespaceから'type'取得
    if isinstance(cfg, dict):
        cfg_type = cfg.get('type')
    else:
        cfg_type = getattr(cfg, 'type', None)
    if cfg_type is None:
        raise ValueError("decoder config must include 'type' key or attribute")

    # experiment_mode: target_prediction時は<type>_targetDecoderを優先
    if experiment_mode == "target_prediction":
        if cfg_type in _TARGET_DECODERS:
            cls = _TARGET_DECODERS[cfg_type]
            print(f"ターゲット予測モード: {cfg_type}_targetDecoderを使用")
        else:
            if cfg_type in _DECODERS:
                cls = _DECODERS[cfg_type]
                print(f"警告: {cfg_type}_targetDecoderが見つかりません。通常の{cfg_type}Decoderを使用します")
            else:
                available_types = list(_DECODERS.keys())
                available_target_types = list(_TARGET_DECODERS.keys())
                raise ValueError(f"Unknown decoder type: {cfg_type}. Available types: {available_types}, Available target types: {available_target_types}")

        if isinstance(cfg, dict):
            init_args = {k: v for k, v in cfg.items() if k != "type"}
        else:
            init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}

        return cls(**init_args)

    # tcnデコーダサポート確認
    if cfg_type == "tcn" and "tcn" not in _DECODERS:
        raise ValueError("tcnDecoder が見つかりません。tcn.py が正しくインポートされているか確認してください")

    try:
        cls = _DECODERS[cfg_type]
    except KeyError:
        available_types = list(_DECODERS.keys())
        raise ValueError(f"Unknown decoder type: {cfg_type}. Available types: {available_types}")

    # 初期化引数抽出
    if isinstance(cfg, dict):
        init_args = {k: v for k, v in cfg.items() if k != "type"}
    else:
        init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}

    # TCNデコーダ設定
    if cfg_type == "tcn":
        if "output_dim" not in init_args:
            raise ValueError("tcnDecoder には output_dim の指定が必要です")
        print(f"[INFO] TCNデコーダを output_dim={init_args['output_dim']} で初期化します")

    return cls(**init_args)
