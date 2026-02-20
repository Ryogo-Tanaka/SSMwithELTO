# src/config.py

import argparse
import yaml
from types import SimpleNamespace

def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """
    Recursively convert a nested dict to SimpleNamespace for dot-access.

    Example:
      {"model": {"encoder": {"type":"mlp", "input_dim":10}}, "training": {"lr":0.001}}
    becomes:
      Namespace(model=Namespace(encoder=Namespace(type="mlp", input_dim=10)),
                training=Namespace(lr=0.001))
    """
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns

def load_cfg() -> SimpleNamespace:
    """
    Load configuration from a YAML file specified via --config CLI argument.

    1. Parse --config from command line
    2. Load YAML file into dict
    3. Convert to SimpleNamespace for dot-access (e.g., cfg.model.encoder.type)
    4. Set default visualization.output_dir if not specified

    Usage:
      python train.py --config configs/model.yaml
      cfg = load_cfg()
      # cfg.model.encoder.type is now accessible
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = _dict_to_namespace(cfg_dict)

    if not hasattr(cfg, "visualization"):
        cfg.visualization = _dict_to_namespace({})
    if not hasattr(cfg.visualization, "output_dir"):
        cfg.visualization.output_dir = "results/figs"

    return cfg
