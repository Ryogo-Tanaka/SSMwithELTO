import pkgutil
import importlib
from pathlib import Path
from . import architectures

# Scan the architectures folder and register any <name>Encoder classes
_ENCODERS: dict[str, type] = {}
pkg_path = Path(architectures.__file__).parent

for module_info in pkgutil.iter_modules([str(pkg_path)]):
    name = module_info.name  # e.g. "_mlp", "_cnn",  …
    module = importlib.import_module(f"{architectures.__package__}.{name}")
    # Expect class names like "_MlpEncoder", "_CnnEncoder"
    cls_name = name + "Encoder"
    if hasattr(module, cls_name):
        _ENCODERS[name] = getattr(module, cls_name)

def build_encoder(cfg):
    """
    Factory function for encoders.
    Args:
      cfg: a configuration object or dict containing:
        - type:   the key matching one of _ENCODERS, e.g. "mlp"
        - other keys: constructor arguments for that encoder class
    Returns:
      An instance of the selected Encoder class, initialized with cfg values.
    """
    # cfg が dict ならキーアクセス、Namespace なら属性アクセスで 'type' を取得
    if isinstance(cfg, dict):
        cfg_type = cfg.get('type')
    else:
        cfg_type = getattr(cfg, 'type', None)
    if cfg_type is None:
        raise ValueError("encoder config must include 'type' key or attribute")
    try:
        cls = _ENCODERS[cfg_type]
    except KeyError:
        raise ValueError(f"Unknown encoder type: {cfg_type}")
    # try:
    #     cls = _ENCODERS[cfg.type]
    # except KeyError:
    #     raise ValueError(f"Unknown encoder type: {cfg.type}")
    
    # Dictionary or Namespace に対応する
    # 残りのキーを初期化引数に
    if isinstance(cfg, dict):
        init_args = {k: v for k, v in cfg.items() if k != "type"}
    else:
        init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}
    # if isinstance(cfg, dict):
    #     init_args = {k: v for k, v in cfg.items() if k != "type"}
    # else:
    #     # SimpleNamespace など属性オブジェクトの場合
    #     init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}
    
    return cls(**init_args)
