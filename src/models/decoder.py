import pkgutil
import importlib
from pathlib import Path
from . import architectures

# Auto-register <name>Decoder and <name>_targetDecoder classes from architectures/
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
    """Factory: build a decoder from cfg.type.

    If experiment_mode="target_prediction", prefers <type>_targetDecoder when available.
    """
    if isinstance(cfg, dict):
        cfg_type = cfg.get('type')
    else:
        cfg_type = getattr(cfg, 'type', None)
    if cfg_type is None:
        raise ValueError("decoder config must include 'type' key or attribute")

    # Target prediction mode: prefer <type>_targetDecoder if available
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

    try:
        cls = _DECODERS[cfg_type]
    except KeyError:
        available_types = list(_DECODERS.keys())
        raise ValueError(f"Unknown decoder type: {cfg_type}. Available types: {available_types}")

    if isinstance(cfg, dict):
        init_args = {k: v for k, v in cfg.items() if k != "type"}
    else:
        init_args = {k: getattr(cfg, k) for k in vars(cfg) if k != "type"}

    return cls(**init_args)
