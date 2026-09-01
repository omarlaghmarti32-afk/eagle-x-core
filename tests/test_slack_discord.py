"""Slack / Discord webhook formatting tests."""

from app.webhook import build_payload, detect_format, format_body

ANALYSIS = {
    "threat_detected": True,
    "threat_type": "ENSEMBLE_ANOMALY",
    "confidence": 0.82,
    "severity": "high",
    "hits": {"patterns": ["isolation_forest", "one_class_svm"]},
    "features": {"cpu_percent": 99},
    "scores": {
        "consensus": {
            "votes": ["iforest", "lof", "ocsvm"],
            "vote_count": 3,
        }
    },
}


def test_detect_slack():
    assert detect_format("https://hooks.slack.com/services/T/B/X") == "slack"


def test_detect_discord():
    assert (
        detect_format("https://discord.com/api/webhooks/123/abc") == "discord"
    )


def test_detect_raw():
    assert detect_format("https://example.com/hook") == "raw"


def test_slack_body():
    p = build_payload(ANALYSIS, threat_id=7)
    body = format_body(p, "slack")
    assert "text" in body
    assert "blocks" in body
    assert "ENSEMBLE" in body["text"] or "EAGLE" in body["text"]


def test_discord_body():
    p = build_payload(ANALYSIS, threat_id=7)
    body = format_body(p, "discord")
    assert body["username"] == "EAGLE-X Core"
    assert body["embeds"][0]["title"].startswith("Threat:")
