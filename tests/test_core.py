"""Core unit tests for EAGLE-X Core."""

from __future__ import annotations

import os

os.environ["EAGLE_LIVE_MONITOR"] = "0"
os.environ["EAGLE_API_TOKEN"] = "test-token"

from fastapi.testclient import TestClient

from app.crypto import CryptoEngine
from app.db import Store
from app.detector import ThreatDetector
from app.main import app


def test_crypto_roundtrip(tmp_path):
    c = CryptoEngine(key_path=tmp_path / "k.key")
    tok = c.encrypt("hello")
    assert c.decrypt(tok) == "hello"
    sig = c.sign("hello")
    assert c.verify("hello", sig)


def test_detector_clean():
    d = ThreatDetector()
    r = d.analyze(
        {
            "cpu_percent": 10,
            "mem_percent": 20,
            "net_sent_rate": 1000,
            "net_recv_rate": 2000,
            "process_count": 80,
            "connection_count": 20,
            "disk_percent": 40,
        }
    )
    assert r["threat_detected"] is False


def test_detector_hot():
    d = ThreatDetector()
    r = d.analyze(
        {
            "cpu_percent": 97,
            "mem_percent": 95,
            "net_sent_rate": 5e6,
            "net_recv_rate": 5e6,
            "process_count": 500,
            "connection_count": 600,
            "disk_percent": 95,
        }
    )
    assert r["threat_detected"] is True
    assert r["confidence"] > 0.3


def test_store(tmp_path):
    s = Store(path=tmp_path / "t.db")
    tid = s.add_threat(threat_type="TEST", confidence=0.9, severity="high")
    assert tid >= 1
    assert s.count_threats() == 1


def test_api_health():
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_ready():
    client = TestClient(app)
    r = client.get("/api/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_api_detect_auth():
    client = TestClient(app)
    assert client.post("/api/detect", json={"features": {"cpu_percent": 99}}).status_code == 401
    r = client.post(
        "/api/detect",
        json={
            "features": {
                "cpu_percent": 99,
                "mem_percent": 95,
                "net_sent_rate": 5e6,
                "net_recv_rate": 5e6,
                "process_count": 500,
                "connection_count": 500,
                "disk_percent": 95,
            }
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert r.status_code == 200
    assert "analysis" in r.json()
