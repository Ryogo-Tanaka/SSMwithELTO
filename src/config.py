import argparse
import yaml
from types import SimpleNamespace

def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict to SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns

def load_cfg() -> SimpleNamespace:
    """Load configuration from a YAML file specified via --config CLI argument.

    Returns a SimpleNamespace tree for dot-access (e.g., cfg.model.encoder.type).
    Sets default visualization.output_dir if not specified in the YAML.
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
