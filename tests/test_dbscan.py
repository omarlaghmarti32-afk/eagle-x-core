"""DBSCAN clustering anomaly tests."""

import pytest

from app.detector import ThreatDetector, _HAS_SKLEARN

NORMAL = {
    "cpu_percent": 12,
    "mem_percent": 35,
    "net_sent_rate": 50_000,
    "net_recv_rate": 80_000,
    "process_count": 120,
    "connection_count": 40,
    "disk_percent": 45,
}


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
def test_dbscan_trains():
    d = ThreatDetector(ml_min_train=32, ml_refit_every=8, dbscan_eps=2.0, dbscan_min_samples=4)
    for i in range(40):
        s = dict(NORMAL)
        s["cpu_percent"] = 10 + (i % 5)
        s["mem_percent"] = 30 + (i % 3)
        d.analyze(s)
    assert d._dbscan_trained is True
    snap = d.baseline_snapshot()
    assert snap["dbscan"]["trained"] is True


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
def test_dbscan_flags_far_point():
    d = ThreatDetector(ml_min_train=32, sensitivity=0.4, dbscan_eps=1.0, dbscan_min_samples=4)
    for _ in range(40):
        d.analyze(NORMAL)
    hot = {
        "cpu_percent": 99,
        "mem_percent": 97,
        "net_sent_rate": 9e6,
        "net_recv_rate": 8e6,
        "process_count": 600,
        "connection_count": 900,
        "disk_percent": 98,
    }
    r = d.analyze(hot)
    assert "dbscan" in r["scores"]
    assert r["scores"]["dbscan"]["trained"] is True
    assert r["threat_detected"] is True


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
def test_train_dbscan_now():
    d = ThreatDetector(ml_min_train=20)
    for _ in range(25):
        d.analyze(NORMAL)
    result = d.train_dbscan_now()
    assert result["ok"] is True
    assert result["trained"] is True


def test_dbscan_meta_present():
    d = ThreatDetector()
    r = d.analyze(NORMAL)
    assert "dbscan" in r["scores"]
    assert "available" in r["scores"]["dbscan"]
