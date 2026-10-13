"""Shared configuration loaded after the startup settings window."""
import json
from pathlib import Path
import cv2
from app_storage import DATA_ROOT, data_path, initialize_storage

from config_validation import validate_config
import machine_settings

ROOT = Path(__file__).resolve().parent
# config.json beside the code holds the defaults and is replaced by every update;
# what was changed on this PC is kept in Documents and laid over it.
CONFIG = machine_settings.load_config(ROOT / "config.json", validate=validate_config)

validate_config(CONFIG)
CAMERA_COUNT = CONFIG.get('camera_count', len(CONFIG['cameras']))
ACTIVE_CAMERAS = CONFIG['cameras'][:CAMERA_COUNT]
cv2.setNumThreads(CONFIG.get('performance', {}).get('opencv_threads', 1))
initialize_storage()


def camera_config(index):
    return next(c for c in CONFIG["cameras"] if c["index"] == index)


def project_path(path):
    """Compatibility alias: configured data now lives in Documents."""
    return str(data_path(path))
