"""Deep anomaly detector tests."""

from app.detector import ThreatDetector

NORMAL = {
    "cpu_percent": 12,
    "mem_percent": 35,
    "net_sent_rate": 50_000,
    "net_recv_rate": 80_000,
    "process_count": 120,
    "connection_count": 40,
    "disk_percent": 45,
}


def test_warmup_then_spike_detected():
    d = ThreatDetector(sensitivity=0.4)
    # establish baseline
    for _ in range(12):
        r = d.analyze(NORMAL)
        assert r["threat_detected"] is False

    assert r["baseline"]["warmup_complete"] is True

    hot = dict(NORMAL)
    hot["cpu_percent"] = 98
    hot["mem_percent"] = 96
    hot["net_sent_rate"] = 9_000_000
    r2 = d.analyze(hot)
    assert r2["threat_detected"] is True
    assert r2["confidence"] >= 0.4
    assert r2["threat_type"] != "NONE"
    assert r2["hits"]["hard"] or r2["hits"]["zscore"] or r2["hits"]["patterns"]


def test_egress_pattern():
    d = ThreatDetector(sensitivity=0.35)
    for _ in range(10):
        d.analyze(NORMAL)
    feat = dict(NORMAL)
    feat["net_sent_rate"] = 3_000_000
    feat["net_recv_rate"] = 20_000
    feat["cpu_percent"] = 60
    r = d.analyze(feat)
    assert "egress_dominant" in r["hits"]["patterns"] or r["threat_detected"]


def test_baseline_snapshot():
    d = ThreatDetector()
    d.analyze(NORMAL)
    snap = d.baseline_snapshot()
    assert "cpu_percent" in snap["features"]
    assert snap["features"]["cpu_percent"]["n"] == 1


def test_reset_baseline():
    d = ThreatDetector()
    for _ in range(5):
        d.analyze(NORMAL)
    d.reset_baseline()
    assert d.baseline_snapshot()["features"]["cpu_percent"]["n"] == 0


def test_no_update_baseline_flag():
    d = ThreatDetector()
    for _ in range(5):
        d.analyze(NORMAL)
    n_before = d.baselines["cpu_percent"].n
    d.analyze({**NORMAL, "cpu_percent": 99}, update_baseline=False)
    assert d.baselines["cpu_percent"].n == n_before
