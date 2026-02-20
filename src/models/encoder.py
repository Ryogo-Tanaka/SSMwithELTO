# src/models/encoder.py

import pkgutil
import importlib
from pathlib import Path
from . import architectures

# Scan the architectures folder and register any <name>Encoder classes
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
    Factory function for encoders.
    Args:
      cfg: a configuration object or dict containing:
        - type:   the key matching one of _ENCODERS, e.g. "tcn", "_mlp"
        - other keys: constructor arguments for that encoder class
    Returns:
      An instance of the selected Encoder class, initialized with cfg values.
    """
    if isinstance(cfg, dict):
        cfg_type = cfg.get('type')
    else:
        cfg_type = getattr(cfg, 'type', None)
    if cfg_type is None:
        raise ValueError("encoder config must include 'type' key or attribute")

    if cfg_type == "tcn" and "tcn" not in _ENCODERS:
        raise ValueError("tcnEncoder not found. Verify tcn.py is importable.")

    try:
        cls = _ENCODERS[cfg_type]
    except KeyError:
        available_types = list(_ENCODERS.keys())
        raise ValueError(f"Unknown encoder type: {cfg_type}. Available types: {available_types}")

    if isinstance(cfg, dict):
        init_args = {k: v for k, v in cfg.items() if k != "type"}
    else:
        init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}

    if cfg_type == "tcn" and "output_dim" not in init_args:
        init_args["output_dim"] = 1

    return cls(**init_args)
