"""Central application configuration.

The module exposes a typed ``settings`` object and keeps the legacy module
constants so existing scripts can continue importing ``FACE_DB_PATH`` etc.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_path(name: str, default: str) -> Path:
    raw = _env_str(name, default)
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def _camera_source() -> int | str:
    raw = _env_str("CAMERA_SOURCE", "0")
    return int(raw) if raw.isdigit() else raw


@dataclass(frozen=True)
class Settings:
    host: str = _env_str("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)

    camera_source: int | str = _camera_source()
    camera_width: int = _env_int("CAMERA_WIDTH", 1280)
    camera_height: int = _env_int("CAMERA_HEIGHT", 720)
    camera_fps: int = _env_int("CAMERA_FPS", 30)

    face_backend: str = _env_str("FACE_BACKEND", "auto")
    face_threshold: float = _env_float("FACE_THRESHOLD", 0.6)
    liveness_threshold: float = _env_float("LIVENESS_THRESHOLD", 0.8)
    detection_size: int = _env_int("DETECTION_SIZE", 640)

    cooldown_minutes: int = _env_int("CHECKIN_COOLDOWN_MINUTES", 5)
    late_threshold: str = _env_str("LATE_THRESHOLD", "08:30")

    database_url: str = _env_str("DATABASE_URL", "sqlite:///data/attendance.db")
    face_db_path: Path = _env_path("FACE_DB_PATH", "models/face_db.pkl")
    data_raw_dir: Path = _env_path("DATA_RAW_DIR", "data/raw")
    data_aug_dir: Path = _env_path("DATA_AUG_DIR", "data/augmented")
    log_dir: Path = _env_path("LOG_DIR", "logs")


settings = Settings()


# Backward-compatible constants used by existing scripts.
HOST = settings.host
PORT = settings.port
CAMERA_SOURCE = settings.camera_source
CAMERA_WIDTH = settings.camera_width
CAMERA_HEIGHT = settings.camera_height
CAMERA_FPS = settings.camera_fps
INSIGHTFACE_MODEL = _env_str("INSIGHTFACE_MODEL", "buffalo_sc")
FACE_BACKEND = settings.face_backend
FACE_THRESHOLD = settings.face_threshold
LIVENESS_THRESHOLD = settings.liveness_threshold
DETECTION_SIZE = settings.detection_size
COOLDOWN_MINUTES = settings.cooldown_minutes
LATE_THRESHOLD = settings.late_threshold
DATABASE_URL = settings.database_url
FACE_DB_PATH = str(settings.face_db_path)
DATA_RAW_DIR = str(settings.data_raw_dir)
DATA_AUG_DIR = str(settings.data_aug_dir)
LOG_DIR = str(settings.log_dir)


import json

def get_runtime_settings() -> dict:
    default_path = BASE_DIR / "data" / "settings.json"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    defaults = {
        "late_threshold": LATE_THRESHOLD,
        "cooldown_minutes": COOLDOWN_MINUTES,
        "face_threshold": FACE_THRESHOLD,
    }
    if default_path.exists():
        try:
            with open(default_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure type compatibility
                if "cooldown_minutes" in data:
                    data["cooldown_minutes"] = int(data["cooldown_minutes"])
                if "face_threshold" in data:
                    data["face_threshold"] = float(data["face_threshold"])
                return {**defaults, **data}
        except Exception:
            pass
    return defaults


def save_runtime_settings(settings_dict: dict) -> None:
    default_path = BASE_DIR / "data" / "settings.json"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    # Validate types before saving
    clean_dict = {
        "late_threshold": str(settings_dict.get("late_threshold", LATE_THRESHOLD)),
        "cooldown_minutes": int(settings_dict.get("cooldown_minutes", COOLDOWN_MINUTES)),
        "face_threshold": float(settings_dict.get("face_threshold", FACE_THRESHOLD)),
    }
    with open(default_path, "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=2)
