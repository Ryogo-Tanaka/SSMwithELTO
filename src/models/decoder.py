# src/models/decoder.py

import pkgutil
import importlib
from pathlib import Path
from . import architectures

# Scan architectures/ for any <name>Decoder and <name>_targetDecoder classes
_DECODERS: dict[str, type] = {}
_TARGET_DECODERS: dict[str, type] = {}
pkg_path = Path(architectures.__file__).parent

for module_info in pkgutil.iter_modules([str(pkg_path)]):
    name = module_info.name
    module = importlib.import_module(f"{architectures.__package__}.{name}")

    cls_name = name + "Decoder"
    if hasattr(module, cls_name):
        _DECODERS[name] = getattr(module, cls_name)

    target_cls_name = name + "_targetDecoder"
    if hasattr(module, target_cls_name):
        _TARGET_DECODERS[name] = getattr(module, target_cls_name)

def build_decoder(cfg, experiment_mode=None):
    """
    Factory for decoders with experiment mode support.
    Args:
      cfg: dict / Namespace with
        - type: key in _DECODERS (e.g. "tcn", "cnn_image", "time_invariant")
        - other keys: kwargs for that decoder class
      experiment_mode: Optional str ("target_prediction" | "reconstruction")
                      Overrides decoder selection for target prediction
    Returns:
      An instance of the chosen Decoder class.
    """
    if isinstance(cfg, dict):
        cfg_type = cfg.get('type')
    else:
        cfg_type = getattr(cfg, 'type', None)
    if cfg_type is None:
        raise ValueError("decoder config must include 'type' key or attribute")

    # Target prediction mode: use <type>_targetDecoder if available
    if experiment_mode == "target_prediction":
        if cfg_type in _TARGET_DECODERS:
            cls = _TARGET_DECODERS[cfg_type]
            print(f"Target prediction mode: using {cfg_type}_targetDecoder")
        else:
            if cfg_type in _DECODERS:
                cls = _DECODERS[cfg_type]
                print(f"Warning: {cfg_type}_targetDecoder not found. Using standard {cfg_type}Decoder")
            else:
                available_types = list(_DECODERS.keys())
                available_target_types = list(_TARGET_DECODERS.keys())
                raise ValueError(f"Unknown decoder type: {cfg_type}. Available: {available_types}, Target: {available_target_types}")

        if isinstance(cfg, dict):
            init_args = {k: v for k, v in cfg.items() if k != "type"}
        else:
            init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}

        return cls(**init_args)

    if cfg_type == "tcn" and "tcn" not in _DECODERS:
        raise ValueError("tcnDecoder not found. Verify tcn.py is importable.")

    try:
        cls = _DECODERS[cfg_type]
    except KeyError:
        available_types = list(_DECODERS.keys())
        raise ValueError(f"Unknown decoder type: {cfg_type}. Available types: {available_types}")

    if isinstance(cfg, dict):
        init_args = {k: v for k, v in cfg.items() if k != "type"}
    else:
        init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}

    if cfg_type == "tcn":
        if "output_dim" not in init_args:
            raise ValueError("tcnDecoder requires output_dim")

    return cls(**init_args)
