"""Health and readiness probes."""

from __future__ import annotations

import shutil
import time
from typing import Any

import psutil

from .config import DATA_DIR, SEAL, VERSION


def run_checks(
    *,
    store=None,
    crypto=None,
    uptime: int = 0,
    scans: int = 0,
    live: bool = False,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    try:
        usage = shutil.disk_usage(DATA_DIR)
        free_mb = usage.free // (1024 * 1024)
        checks["disk"] = {
            "ok": free_mb >= 50,
            "free_mb": free_mb,
            "used_percent": round(usage.used / usage.total * 100, 1),
        }
    except Exception as e:
        checks["disk"] = {"ok": False, "error": str(e)}

    try:
        checks["host"] = {
            "ok": True,
            "cpu_percent": psutil.cpu_percent(interval=0.05),
            "mem_percent": psutil.virtual_memory().percent,
        }
    except Exception as e:
        checks["host"] = {"ok": False, "error": str(e)}

    if store is not None:
        try:
            checks["database"] = {
                "ok": True,
                "threats": store.count_threats(),
                "path": str(store.path),
            }
        except Exception as e:
            checks["database"] = {"ok": False, "error": str(e)}

    if crypto is not None:
        try:
            tok = crypto.encrypt("ping")
            ok = crypto.decrypt(tok) == "ping" and crypto.verify("ping", crypto.sign("ping"))
            checks["crypto"] = {"ok": ok, "alg": "AES-256-GCM+Ed25519"}
        except Exception as e:
            checks["crypto"] = {"ok": False, "error": str(e)}

    failed = [n for n, c in checks.items() if not c.get("ok")]
    status = "ok" if not failed else ("degraded" if len(failed) < len(checks) else "down")

    return {
        "status": status,
        "version": VERSION,
        "seal": SEAL,
        "uptime_seconds": uptime,
        "scans": scans,
        "live_monitor": live,
        "checks": checks,
        "failed": failed,
        "timestamp": time.time(),
    }
