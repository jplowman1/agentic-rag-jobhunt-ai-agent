"""
Loads preferences config.
Prefers config/preferences.local.yaml if it exists (gitignored, personal values),
falls back to config/preferences.yaml (template, committed to GitHub).
"""
from pathlib import Path
import yaml

_CONFIG_DIR = Path(__file__).parent.parent / "config"
_LOCAL = _CONFIG_DIR / "preferences.local.yaml"
_DEFAULT = _CONFIG_DIR / "preferences.yaml"


def load_config() -> dict:
    path = _LOCAL if _LOCAL.exists() else _DEFAULT
    with open(path) as f:
        return yaml.safe_load(f)
