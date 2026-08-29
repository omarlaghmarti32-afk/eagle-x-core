"""Lightweight anomaly scoring (rules + z-like thresholds)."""

from __future__ import annotations

from typing import Any


class ThreatDetector:
    """Score host feature dicts without sklearn dependency."""

    THRESHOLDS = {
        "cpu_percent": 90.0,
        "mem_percent": 90.0,
        "net_sent_rate": 2_000_000.0,
        "net_recv_rate": 2_000_000.0,
        "process_count": 350.0,
        "connection_count": 400.0,
        "disk_percent": 92.0,
    }

    WEIGHTS = {
        "cpu_percent": 0.2,
        "mem_percent": 0.2,
        "net_sent_rate": 0.15,
        "net_recv_rate": 0.15,
        "process_count": 0.1,
        "connection_count": 0.1,
        "disk_percent": 0.1,
    }

    def analyze(self, features: dict[str, float] | list[float]) -> dict[str, Any]:
        if isinstance(features, list):
            keys = list(self.THRESHOLDS.keys())
            features = {keys[i]: float(features[i]) for i in range(min(len(keys), len(features)))}

        hits: list[str] = []
        score = 0.0
        for k, thr in self.THRESHOLDS.items():
            val = float(features.get(k, 0))
            if val >= thr:
                hits.append(k)
                # how far above threshold (capped)
                over = min((val - thr) / max(thr, 1.0), 2.0)
                score += self.WEIGHTS.get(k, 0.1) * (0.5 + over)

        confidence = min(round(score, 3), 1.0)
        threat = confidence >= 0.35 or len(hits) >= 3

        if not threat:
            threat_type = "NONE"
            severity = "none"
        elif "net_sent_rate" in hits or "net_recv_rate" in hits:
            threat_type = "TRAFFIC_ANOMALY"
            severity = "high" if confidence > 0.7 else "medium"
        elif "cpu_percent" in hits or "mem_percent" in hits:
            threat_type = "RESOURCE_ABUSE"
            severity = "high" if confidence > 0.7 else "medium"
        elif "connection_count" in hits or "process_count" in hits:
            threat_type = "PROCESS_FLOOD"
            severity = "medium"
        else:
            threat_type = "HOST_ANOMALY"
            severity = "low"

        return {
            "threat_detected": threat,
            "threat_type": threat_type,
            "confidence": confidence,
            "severity": severity,
            "hits": hits,
            "features": features,
        }
