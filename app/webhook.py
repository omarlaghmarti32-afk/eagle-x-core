"""Outbound webhook notifications — raw JSON, Slack, or Discord."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .config import (
    WEBHOOK_FORMAT,
    WEBHOOK_MIN_SEVERITY,
    WEBHOOK_MIN_VOTES,
    WEBHOOK_TIMEOUT,
    WEBHOOK_URL,
)

logger = logging.getLogger("eagle-core.webhook")

_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_SEV_COLOR = {
    "low": 0x3B82F6,
    "medium": 0xF59E0B,
    "high": 0xEF4444,
    "critical": 0xDC2626,
}


def detect_format(url: str, explicit: str = "auto") -> str:
    if explicit and explicit != "auto":
        return explicit
    u = (url or "").lower()
    if "hooks.slack.com" in u:
        return "slack"
    if "discord.com/api/webhooks" in u or "discordapp.com/api/webhooks" in u:
        return "discord"
    return "raw"


def should_notify(analysis: dict[str, Any]) -> bool:
    if not WEBHOOK_URL:
        return False
    if not analysis.get("threat_detected"):
        return False
    consensus = (analysis.get("scores") or {}).get("consensus") or {}
    votes = int(consensus.get("vote_count") or 0)
    sev = str(analysis.get("severity") or "none").lower()
    if votes >= WEBHOOK_MIN_VOTES:
        return True
    if _SEVERITY_RANK.get(sev, 0) >= _SEVERITY_RANK.get(WEBHOOK_MIN_SEVERITY, 2) and votes >= 1:
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


def _to_slack(payload: dict[str, Any]) -> dict[str, Any]:
    votes = ", ".join(payload.get("votes") or []) or "—"
    sev = str(payload.get("severity") or "medium")
    text = (
        f":warning: *EAGLE-X Threat* `{payload.get('threat_type')}`\n"
        f"*Severity:* {sev}  |  *Confidence:* {payload.get('confidence')}\n"
        f"*Votes ({payload.get('vote_count')}):* {votes}\n"
        f"*ID:* {payload.get('threat_id')}  |  *Source:* {payload.get('source')}"
    )
    return {
        "text": text,
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"EAGLE-X: {payload.get('threat_type')}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Severity*\n{sev}"},
                    {"type": "mrkdwn", "text": f"*Confidence*\n{payload.get('confidence')}"},
                    {"type": "mrkdwn", "text": f"*Votes*\n{votes}"},
                    {"type": "mrkdwn", "text": f"*Threat ID*\n{payload.get('threat_id')}"},
                ],
            },
        ],
    }


def _to_discord(payload: dict[str, Any]) -> dict[str, Any]:
    votes = ", ".join(payload.get("votes") or []) or "—"
    sev = str(payload.get("severity") or "medium").lower()
    color = _SEV_COLOR.get(sev, 0xF59E0B)
    return {
        "username": "EAGLE-X Core",
        "embeds": [
            {
                "title": f"Threat: {payload.get('threat_type')}",
                "color": color,
                "fields": [
                    {"name": "Severity", "value": str(sev), "inline": True},
                    {
                        "name": "Confidence",
                        "value": str(payload.get("confidence")),
                        "inline": True,
                    },
                    {
                        "name": "Votes",
                        "value": f"{payload.get('vote_count')}: {votes}",
                        "inline": False,
                    },
                    {
                        "name": "ID / Source",
                        "value": f"#{payload.get('threat_id')} · {payload.get('source')}",
                        "inline": False,
                    },
                ],
                "footer": {"text": "EAGLE-X Core security monitor"},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(payload.get("ts") or time.time())),
            }
        ],
    }


def format_body(payload: dict[str, Any], fmt: str | None = None) -> dict[str, Any]:
    fmt = detect_format(WEBHOOK_URL, fmt or WEBHOOK_FORMAT)
    if fmt == "slack":
        return _to_slack(payload)
    if fmt == "discord":
        return _to_discord(payload)
    return payload


def send_webhook_sync(payload: dict[str, Any]) -> dict[str, Any]:
    if not WEBHOOK_URL:
        return {"ok": False, "reason": "disabled"}
    try:
        import urllib.request

        fmt = detect_format(WEBHOOK_URL, WEBHOOK_FORMAT)
        body = format_body(payload, fmt)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "EAGLE-X-Core/1.2",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT) as resp:
            raw = resp.read()[:500]
            return {
                "ok": True,
                "status": getattr(resp, "status", 200),
                "format": fmt,
                "body": raw.decode("utf-8", errors="replace"),
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Webhook failed: %s", exc)
        return {"ok": False, "error": str(exc)}
