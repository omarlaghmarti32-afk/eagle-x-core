"""Persistence and webhook helper tests."""

from pathlib import Path

from app.detector import ThreatDetector
from app.persist import load_baselines, load_models, save_baselines, save_models
from app.webhook import build_payload, should_notify

NORMAL = {
    "cpu_percent": 12,
    "mem_percent": 35,
    "net_sent_rate": 50_000,
    "net_recv_rate": 80_000,
    "process_count": 120,
    "connection_count": 40,
    "disk_percent": 45,
}


def test_save_load_baselines(tmp_path):
    d = ThreatDetector()
    for _ in range(5):
        d.analyze(NORMAL)
    path = tmp_path / "b.json"
    save_baselines(d, path)
    d2 = ThreatDetector()
    assert load_baselines(d2, path) is True
    assert d2.baselines["cpu_percent"].n == d.baselines["cpu_percent"].n


def test_save_load_models(tmp_path):
    d = ThreatDetector(ml_min_train=20)
    for i in range(25):
        s = dict(NORMAL)
        s["cpu_percent"] = 10 + (i % 4)
        d.analyze(s)
    if not d._sklearn_available:
        return
    path = tmp_path / "m.joblib"
    assert save_models(d, path) is not None
    d2 = ThreatDetector()
    assert load_models(d2, path) is True
    assert d2._iforest_trained is True


def test_build_payload():
    analysis = {
        "threat_detected": True,
        "threat_type": "HOST_ANOMALY",
        "confidence": 0.7,
        "severity": "high",
        "hits": {"patterns": ["isolation_forest"]},
        "features": NORMAL,
        "scores": {"consensus": {"votes": ["iforest", "lof", "ocsvm"], "vote_count": 3}},
    }
    p = build_payload(analysis, threat_id=42)
    assert p["threat_id"] == 42
    assert p["vote_count"] == 3


def test_should_notify_respects_disabled(monkeypatch):
    monkeypatch.setattr("app.webhook.WEBHOOK_URL", "")
    analysis = {
        "threat_detected": True,
        "severity": "critical",
        "scores": {"consensus": {"vote_count": 5}},
    }
    assert should_notify(analysis) is False


def test_should_notify_on_votes(monkeypatch):
    monkeypatch.setattr("app.webhook.WEBHOOK_URL", "https://example.com/hook")
    monkeypatch.setattr("app.webhook.WEBHOOK_MIN_VOTES", 3)
    analysis = {
        "threat_detected": True,
        "severity": "medium",
        "scores": {"consensus": {"vote_count": 3}},
    }
    assert should_notify(analysis) is True
