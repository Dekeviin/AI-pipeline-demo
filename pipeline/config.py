"""Config loader — reads config.yaml once, resolves paths against the project root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve project-relative paths to absolute so stages can run from anywhere.
    cfg["data"]["raw_dir"] = str(ROOT / cfg["data"]["raw_dir"])
    cfg["data"]["db_path"] = str(ROOT / cfg["data"]["db_path"])
    cfg["artifacts_dir"] = str(ROOT / cfg["artifacts_dir"])
    return cfg
