# -*- coding: utf-8 -*-
"""Configuration handling for the Deer Detection system.

Loads settings from a ``.env`` file (using ``python-dotenv``) and provides
defaults for all required parameters.  If a ``config.yaml`` exists it is also
merged, giving precedence to the explicit environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Project root = parent of the config/ package directory
BASE_DIR = Path(__file__).resolve().parent.parent  # wiserli/

# Load .env (if present)
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

def _env(key: str, default: Any = None, cast: Any = str) -> Any:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return cast(val)
    except Exception:
        return default

# Base configuration from .env
CONFIG: Dict[str, Any] = {
    "HOST": _env("HOST", "0.0.0.0"),
    "PORT": _env("PORT", 8000, int),
    "MODEL_PATH": _env("MODEL_PATH", "model/best.pt"),
    "YOLO_CONFIDENCE": _env("YOLO_CONFIDENCE", 0.3, float),
    "YOLO_IOU": _env("YOLO_IOU", 0.5, float),
    "YOLO_IMG_SIZE": _env("YOLO_IMG_SIZE", 640, int),
    "YOLO_DEVICE": _env("YOLO_DEVICE", "cuda"),
    "CAMERA_TYPE": _env("CAMERA_TYPE", "LOCAL_VIDEO"),
    "CAMERA_SOURCE": _env("CAMERA_SOURCE", "videos/deer_test.mp4"),
    "CAMERA_TIMEOUT": _env("CAMERA_TIMEOUT", 10, int),
    "CAMERA_RECONNECT_DELAY": _env("CAMERA_RECONNECT_DELAY", 5, int),
    "ROAD_ZONE": _env("ROAD_ZONE", ""),  # JSON list e.g. [[x1,y1],...]
    "ROAD_DISTANCE_THRESHOLD": _env("ROAD_DISTANCE_THRESHOLD", 20.0, float),
    "CAUTION_SPEED": _env("CAUTION_SPEED", 2.0, float),
    "HIGH_RISK_SPEED": _env("HIGH_RISK_SPEED", 5.0, float),
    "PREDICTION_HORIZON": _env("PREDICTION_HORIZON", 5, int),
    "MAX_HISTORY": _env("MAX_HISTORY", 30, int),
    "VLM_ENABLED": _env("VLM_ENABLED", "true").lower() == "true",
    "VLM_MODEL": _env("VLM_MODEL", "Salesforce/blip-image-captioning-base"),
    "VLM_COOLDOWN": _env("VLM_COOLDOWN", 30, int),
}

# Optional structured config.yaml – merges with env values (env wins)
yaml_path = BASE_DIR / "config.yaml"
if yaml_path.is_file():
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        # Only set keys not already set by env
        for k, v in yaml_cfg.items():
            if k not in CONFIG:
                CONFIG[k] = v
    except Exception:
        pass


def get(key: str, default: Any = None) -> Any:
    return CONFIG.get(key, default)


# Helper to parse ROAD_ZONE JSON string into list of points
def parse_road_zone() -> list:
    import json
    raw = CONFIG.get("ROAD_ZONE", "")
    if not raw:
        return []
    try:
        pts = json.loads(raw)
        if isinstance(pts, list) and all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in pts
        ):
            return [[float(coord) for coord in p] for p in pts]
    except Exception:
        pass
    return []
