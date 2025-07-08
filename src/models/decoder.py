import pkgutil
import importlib
from pathlib import Path
from . import architectures

# Scan architectures/ for any <name>Decoder classes
_DECODERS: dict[str, type] = {}
pkg_path = Path(architectures.__file__).parent

for module_info in pkgutil.iter_modules([str(pkg_path)]):
    name = module_info.name              # e.g. "_mlp", "_cnn", …
    module = importlib.import_module(f"{architectures.__package__}.{name}")
    cls_name = name + "Decoder"
    if hasattr(module, cls_name):
        _DECODERS[name] = getattr(module, cls_name)

def build_decoder(cfg):
    """
    Factory for decoders.
    Args:
      cfg: dict / Namespace with
        - type: key in _DECODERS (e.g. "mlp")
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
    try:
        cls = _DECODERS[cfg_type]
    except KeyError:
        raise ValueError(f"Unknown decoder type: {cfg_type}")    
    # try:
    #     cls = _DECODERS[cfg.type]
    # except KeyError:
    #     raise ValueError(f"Unknown decoder type: {cfg.type}")
    
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
