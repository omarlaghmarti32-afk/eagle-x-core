"""Outbound webhook notifications for high-confidence threats."""

from __future__ import annotations

import logging
import time
from typing import Any

from .config import WEBHOOK_MIN_SEVERITY, WEBHOOK_MIN_VOTES, WEBHOOK_TIMEOUT, WEBHOOK_URL

logger = logging.getLogger("eagle-core.webhook")

_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def should_notify(analysis: dict[str, Any]) -> bool:
    if not WEBHOOK_URL:
        return False
    if not analysis.get("threat_detected"):
        return False
    consensus = (analysis.get("scores") or {}).get("consensus") or {}
    votes = int(consensus.get("vote_count") or 0)
    sev = str(analysis.get("severity") or "none").lower()
    min_sev = WEBHOOK_MIN_SEVERITY
    if votes >= WEBHOOK_MIN_VOTES:
        return True
    if _SEVERITY_RANK.get(sev, 0) >= _SEVERITY_RANK.get(min_sev, 2) and votes >= 1:
        return True
    return False


def build_payload(
    analysis: dict[str, Any],
    *,
    threat_id: int | None = None,
    source: str = "live_monitor",
) -> dict[str, Any]:
    scores = analysis.get("scores") or {}
    consensus = scores.get("consensus") or {}
    return {
        "event": "eagle.threat",
        "ts": time.time(),
        "source": source,
        "threat_id": threat_id,
        "threat_type": analysis.get("threat_type"),
        "confidence": analysis.get("confidence"),
        "severity": analysis.get("severity"),
        "votes": consensus.get("votes") or [],
        "vote_count": consensus.get("vote_count") or 0,
        "hits": analysis.get("hits"),
        "features": analysis.get("features"),
    }


def send_webhook_sync(payload: dict[str, Any]) -> dict[str, Any]:
    """Blocking POST; safe to run via asyncio.to_thread."""
    if not WEBHOOK_URL:
        return {"ok": False, "reason": "disabled"}
    try:
        import urllib.error
        import urllib.request
        import json

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "EAGLE-X-Core/1.1",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT) as resp:
            body = resp.read()[:500]
            return {
                "ok": True,
                "status": getattr(resp, "status", 200),
                "body": body.decode("utf-8", errors="replace"),
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Webhook failed: %s", exc)
        return {"ok": False, "error": str(exc)}
