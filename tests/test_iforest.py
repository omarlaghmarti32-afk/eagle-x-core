"""Isolation Forest integration tests."""

import pytest

from app.detector import ThreatDetector, _HAS_SKLEARN, FEATURE_KEYS

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
def test_iforest_trains_and_flags_outlier():
    d = ThreatDetector(sensitivity=0.45, iforest_min_train=32, iforest_refit_every=8)
    # feed mostly normal traffic
    for i in range(40):
        sample = dict(NORMAL)
        sample["cpu_percent"] = 10 + (i % 5)
        sample["mem_percent"] = 30 + (i % 4)
        d.analyze(sample)

    assert d._iforest_trained is True
    snap = d.baseline_snapshot()
    assert snap["isolation_forest"]["trained"] is True

    # strong outlier
    hot = {
        "cpu_percent": 99,
        "mem_percent": 97,
        "net_sent_rate": 9_000_000,
        "net_recv_rate": 8_000_000,
        "process_count": 600,
        "connection_count": 900,
        "disk_percent": 98,
    }
    r = d.analyze(hot)
    assert r["threat_detected"] is True
    if_meta = r["scores"]["isolation_forest"]
    assert if_meta["available"] is True
    assert if_meta["trained"] is True
    # hard ceilings alone should fire; IF may also
    assert r["confidence"] > 0.4


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
def test_force_train():
    d = ThreatDetector(iforest_min_train=20)
    for _ in range(25):
        d.analyze(NORMAL)
    result = d.train_iforest_now()
    assert result["ok"] is True
    assert result["trained"] is True


def test_iforest_meta_always_present():
    d = ThreatDetector()
    r = d.analyze(NORMAL)
    assert "isolation_forest" in r["scores"]
    assert "available" in r["scores"]["isolation_forest"]
