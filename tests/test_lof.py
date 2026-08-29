"""Local Outlier Factor tests."""

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
def test_lof_trains():
    d = ThreatDetector(ml_min_train=32, ml_refit_every=8, lof_n_neighbors=15)
    for i in range(40):
        s = dict(NORMAL)
        s["cpu_percent"] = 10 + (i % 5)
        d.analyze(s)
    assert d._lof_trained is True
    snap = d.baseline_snapshot()
    assert snap["lof"]["trained"] is True


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
def test_lof_scores_present_on_outlier():
    d = ThreatDetector(ml_min_train=32, sensitivity=0.4)
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
    assert "lof" in r["scores"]
    assert r["scores"]["lof"]["trained"] is True
    assert r["threat_detected"] is True


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
def test_train_lof_now():
    d = ThreatDetector(ml_min_train=20)
    for _ in range(25):
        d.analyze(NORMAL)
    result = d.train_lof_now()
    assert result["ok"] is True
    assert result["trained"] is True


def test_lof_meta_without_train():
    d = ThreatDetector()
    r = d.analyze(NORMAL)
    assert "lof" in r["scores"]
    assert r["scores"]["lof"]["available"] is _HAS_SKLEARN
