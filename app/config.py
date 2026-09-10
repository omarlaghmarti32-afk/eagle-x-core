"""Central configuration."""

from __future__ import annotations

import os
from pathlib import Path

VERSION = "1.3.0"
SEAL = "EAGLE-CORE-1.3"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EAGLE_DATA_DIR", BASE_DIR / "data"))
LOG_DIR = Path(os.environ.get("EAGLE_LOG_DIR", BASE_DIR / "logs"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "eagle.db"
KEY_PATH = DATA_DIR / "master.key"
MODEL_PATH = Path(os.environ.get("EAGLE_MODEL_PATH", DATA_DIR / "detector_state.joblib"))
BASELINE_PATH = Path(os.environ.get("EAGLE_BASELINE_PATH", DATA_DIR / "baselines.json"))

API_TOKEN = os.environ.get("EAGLE_API_TOKEN", "eagle-dev-token-change-me")
REQUIRE_STRONG_TOKEN = os.environ.get("EAGLE_REQUIRE_STRONG_TOKEN", "0") not in (
    "0",
    "false",
    "False",
)

LIVE_MONITOR = os.environ.get("EAGLE_LIVE_MONITOR", "1") not in ("0", "false", "False")
MONITOR_INTERVAL = float(os.environ.get("EAGLE_MONITOR_INTERVAL", "2.0"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

WEBHOOK_URL = os.environ.get("EAGLE_WEBHOOK_URL", "").strip()
WEBHOOK_FORMAT = os.environ.get("EAGLE_WEBHOOK_FORMAT", "auto").strip().lower()
WEBHOOK_MIN_VOTES = int(os.environ.get("EAGLE_WEBHOOK_MIN_VOTES", "3"))
WEBHOOK_MIN_SEVERITY = os.environ.get("EAGLE_WEBHOOK_MIN_SEVERITY", "medium").lower()
WEBHOOK_TIMEOUT = float(os.environ.get("EAGLE_WEBHOOK_TIMEOUT", "5.0"))
WEBHOOK_ALLOW_PRIVATE = os.environ.get("EAGLE_WEBHOOK_ALLOW_PRIVATE", "0") not in (
    "0",
    "false",
    "False",
)

STATE_SAVE_EVERY = int(os.environ.get("EAGLE_STATE_SAVE_EVERY", "30"))

CORS_ORIGINS = os.environ.get("EAGLE_CORS_ORIGINS", "*")
RATE_LIMIT_PER_MIN = int(os.environ.get("EAGLE_RATE_LIMIT_PER_MIN", "120"))
PROTECT_READ_APIS = os.environ.get("EAGLE_PROTECT_READ_APIS", "0") not in (
    "0",
    "false",
    "False",
)
