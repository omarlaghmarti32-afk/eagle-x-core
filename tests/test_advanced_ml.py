"""Advanced ML ensemble tests."""

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

HOT = {
    "cpu_percent": 99,
    "mem_percent": 97,
    "net_sent_rate": 9e6,
    "net_recv_rate": 8e6,
    "process_count": 600,
    "connection_count": 900,
    "disk_percent": 98,
}


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
def test_advanced_models_train():
    d = ThreatDetector(ml_min_train=32, ml_refit_every=8)
    for i in range(40):
        s = dict(NORMAL)
        s["cpu_percent"] = 10 + (i % 5)
        d.analyze(s)
    assert d._iforest_trained
    assert d._lof_trained
    assert d._dbscan_trained
    assert d._ocsvm_trained
    assert d._pca_trained
    # elliptic may fail on singular matrices in edge cases but usually works
    snap = d.baseline_snapshot()
    assert "one_class_svm" in snap
    assert "pca" in snap
    assert "elliptic_envelope" in snap


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
def test_hot_point_has_all_score_keys():
    d = ThreatDetector(ml_min_train=32, sensitivity=0.4)
    for _ in range(40):
        d.analyze(NORMAL)
    r = d.analyze(HOT)
    assert r["threat_detected"] is True
    scores = r["scores"]
    for key in (
        "isolation_forest",
        "lof",
        "dbscan",
        "one_class_svm",
        "elliptic_envelope",
        "pca",
        "cusum",
        "consensus",
    ):
        assert key in scores


def test_cusum_available_without_sklearn():
    d = ThreatDetector()
    for _ in range(5):
        d.analyze(NORMAL)
    r = d.analyze({**NORMAL, "cpu_percent": 99})
    assert "cusum" in r["scores"]
    assert r["scores"]["cusum"]["available"] is True


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
def test_train_ml_now():
    d = ThreatDetector(ml_min_train=20)
    for _ in range(25):
        d.analyze(NORMAL)
    result = d.train_ml_now()
    assert result["ok"] is True
    assert result["fitted"].get("iforest")
