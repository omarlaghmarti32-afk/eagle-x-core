"""Deep anomaly detection: adaptive baselines + rules + multi-signal patterns.

Layers (combined):
  1. Hard ceilings — absolute unsafe host levels
  2. EWMA baseline — online mean/variance, robust z-scores
  3. Spike / rate-of-change — sudden jumps vs previous sample
  4. Multi-signal patterns — correlated abuse signatures

No sklearn required. Works from the first sample; confidence improves
as the baseline warms up (see `warmup_left`).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

FEATURE_KEYS = [
    "cpu_percent",
    "mem_percent",
    "net_sent_rate",
    "net_recv_rate",
    "process_count",
    "connection_count",
    "disk_percent",
]

# Absolute ceilings (always score when exceeded)
HARD_CEILINGS = {
    "cpu_percent": 95.0,
    "mem_percent": 95.0,
    "net_sent_rate": 8_000_000.0,
    "net_recv_rate": 8_000_000.0,
    "process_count": 500.0,
    "connection_count": 800.0,
    "disk_percent": 97.0,
}

FEATURE_WEIGHTS = {
    "cpu_percent": 0.18,
    "mem_percent": 0.18,
    "net_sent_rate": 0.14,
    "net_recv_rate": 0.14,
    "process_count": 0.12,
    "connection_count": 0.14,
    "disk_percent": 0.10,
}

# Relative jump that counts as a spike (fraction of previous value)
SPIKE_RATIO = {
    "cpu_percent": 2.5,
    "mem_percent": 1.8,
    "net_sent_rate": 4.0,
    "net_recv_rate": 4.0,
    "process_count": 1.6,
    "connection_count": 2.0,
    "disk_percent": 1.15,
}


@dataclass
class FeatureBaseline:
    mean: float = 0.0
    var: float = 1.0
    n: int = 0
    last: float = 0.0


@dataclass
class ThreatDetector:
    """Adaptive host anomaly detector."""

    sensitivity: float = field(
        default_factory=lambda: float(os.environ.get("EAGLE_AI_SENSITIVITY", "0.55"))
    )
    alpha: float = 0.12  # EWMA smoothing (higher = adapt faster)
    z_threshold: float = 3.0
    min_samples: int = 8  # samples before z-scores fully trusted
    baselines: dict[str, FeatureBaseline] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sensitivity = float(min(max(self.sensitivity, 0.1), 0.95))
        for k in FEATURE_KEYS:
            if k not in self.baselines:
                self.baselines[k] = FeatureBaseline()

    # ------------------------------------------------------------------ utils
    def _normalize(self, features: dict[str, float] | list[float]) -> dict[str, float]:
        if isinstance(features, list):
            return {
                FEATURE_KEYS[i]: float(features[i])
                for i in range(min(len(FEATURE_KEYS), len(features)))
            }
        out: dict[str, float] = {}
        for k in FEATURE_KEYS:
            if k in features:
                out[k] = float(features[k])
        # allow partial feature sets from API
        for k, v in features.items():
            if k not in out:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    pass
        return out

    def _update_baseline(self, key: str, value: float) -> tuple[float, float]:
        """Return (z_score, spike_score in 0..1) and update EWMA."""
        b = self.baselines.setdefault(key, FeatureBaseline())
        spike = 0.0
        if b.n > 0 and b.last > 0:
            ratio = value / max(b.last, 1e-9)
            need = SPIKE_RATIO.get(key, 2.0)
            if ratio >= need:
                spike = min((ratio - need) / need, 2.0) / 2.0

        # z vs current baseline (before update)
        std = math.sqrt(max(b.var, 1e-6))
        z = (value - b.mean) / std if b.n >= 2 else 0.0

        # EWMA update
        if b.n == 0:
            b.mean = value
            b.var = max(value * 0.05, 1.0) ** 2
        else:
            diff = value - b.mean
            b.mean = (1 - self.alpha) * b.mean + self.alpha * value
            b.var = (1 - self.alpha) * b.var + self.alpha * (diff**2)
        b.n += 1
        b.last = value
        return z, spike

    def _hard_hits(self, features: dict[str, float]) -> list[str]:
        hits = []
        for k, ceiling in HARD_CEILINGS.items():
            if float(features.get(k, 0)) >= ceiling:
                hits.append(k)
        return hits

    def _pattern_bonus(self, features: dict[str, float], z_map: dict[str, float]) -> tuple[float, list[str]]:
        """Multi-signal attack-like combinations."""
        patterns: list[str] = []
        bonus = 0.0
        cpu = features.get("cpu_percent", 0)
        mem = features.get("mem_percent", 0)
        sent = features.get("net_sent_rate", 0)
        recv = features.get("net_recv_rate", 0)
        procs = features.get("process_count", 0)
        conns = features.get("connection_count", 0)

        # Crypto-miner / resource abuse: high CPU + high mem
        if cpu >= 85 and mem >= 80:
            patterns.append("cpu_mem_pressure")
            bonus += 0.22

        # Exfiltration-ish: outbound >> inbound while CPU moderate-high
        if sent > 1_500_000 and sent > max(recv * 3, 1):
            patterns.append("egress_dominant")
            bonus += 0.2

        # C2 / scan flood: many connections + process growth
        if conns >= 300 and procs >= 250:
            patterns.append("conn_process_flood")
            bonus += 0.2

        # Correlated z-anomalies on network both directions
        if z_map.get("net_sent_rate", 0) >= self.z_threshold and z_map.get(
            "net_recv_rate", 0
        ) >= self.z_threshold:
            patterns.append("bidirectional_traffic_shift")
            bonus += 0.15

        # Disk filling fast is often ransomware precursor when + high IO proxy (net)
        if features.get("disk_percent", 0) >= 90 and (sent + recv) > 1_000_000:
            patterns.append("disk_and_io_pressure")
            bonus += 0.18

        return min(bonus, 0.55), patterns

    def _classify(
        self,
        confidence: float,
        hard: list[str],
        z_hits: list[str],
        spikes: list[str],
        patterns: list[str],
    ) -> tuple[str, str]:
        if confidence < self.sensitivity and not hard:
            return "NONE", "none"

        net_keys = {"net_sent_rate", "net_recv_rate"}
        res_keys = {"cpu_percent", "mem_percent"}
        flood_keys = {"process_count", "connection_count"}

        signals = set(hard) | set(z_hits) | set(spikes)

        if "egress_dominant" in patterns or signals & net_keys:
            t = "TRAFFIC_ANOMALY"
        elif "cpu_mem_pressure" in patterns or signals & res_keys:
            t = "RESOURCE_ABUSE"
        elif "conn_process_flood" in patterns or signals & flood_keys:
            t = "PROCESS_FLOOD"
        elif "disk_and_io_pressure" in patterns or "disk_percent" in signals:
            t = "STORAGE_ANOMALY"
        else:
            t = "HOST_ANOMALY"

        if confidence >= 0.8 or len(hard) >= 2:
            sev = "critical"
        elif confidence >= 0.65 or hard:
            sev = "high"
        elif confidence >= 0.45:
            sev = "medium"
        else:
            sev = "low"
        return t, sev

    # ----------------------------------------------------------------- public
    def analyze(
        self,
        features: dict[str, float] | list[float],
        *,
        update_baseline: bool = True,
    ) -> dict[str, Any]:
        feats = self._normalize(features)
        z_map: dict[str, float] = {}
        spike_map: dict[str, float] = {}
        z_hits: list[str] = []
        spike_hits: list[str] = []
        score = 0.0

        for key in FEATURE_KEYS:
            if key not in feats:
                continue
            val = feats[key]
            if update_baseline:
                z, spike = self._update_baseline(key, val)
            else:
                b = self.baselines.get(key, FeatureBaseline())
                std = math.sqrt(max(b.var, 1e-6))
                z = (val - b.mean) / std if b.n >= 2 else 0.0
                spike = 0.0
                if b.n > 0 and b.last > 0:
                    ratio = val / max(b.last, 1e-9)
                    need = SPIKE_RATIO.get(key, 2.0)
                    if ratio >= need:
                        spike = min((ratio - need) / need, 2.0) / 2.0

            z_map[key] = round(z, 3)
            spike_map[key] = round(spike, 3)
            w = FEATURE_WEIGHTS.get(key, 0.1)

            # Warm-up: damp z contribution until enough samples
            b = self.baselines.get(key, FeatureBaseline())
            warm = min(b.n / max(self.min_samples, 1), 1.0)

            if abs(z) >= self.z_threshold and warm >= 0.5:
                z_hits.append(key)
                score += w * min(abs(z) / self.z_threshold, 3.0) * 0.35 * warm

            if spike >= 0.25:
                spike_hits.append(key)
                score += w * spike * 0.4

        hard = self._hard_hits(feats)
        for key in hard:
            w = FEATURE_WEIGHTS.get(key, 0.1)
            ceiling = HARD_CEILINGS[key]
            over = min((feats[key] - ceiling) / max(ceiling, 1.0), 2.0)
            score += w * (0.55 + over * 0.35)

        pattern_bonus, patterns = self._pattern_bonus(feats, z_map)
        score += pattern_bonus

        confidence = min(round(score, 3), 1.0)
        threat = confidence >= self.sensitivity or len(hard) >= 1 or (
            len(z_hits) + len(spike_hits) >= 3 and confidence >= self.sensitivity * 0.75
        )

        threat_type, severity = self._classify(
            confidence if threat else 0.0, hard, z_hits, spike_hits, patterns
        )
        if not threat:
            threat_type, severity = "NONE", "none"

        samples = min((self.baselines[k].n for k in FEATURE_KEYS if k in self.baselines), default=0)

        return {
            "threat_detected": bool(threat),
            "threat_type": threat_type,
            "confidence": confidence,
            "severity": severity,
            "hits": {
                "hard": hard,
                "zscore": z_hits,
                "spike": spike_hits,
                "patterns": patterns,
            },
            "scores": {
                "z": z_map,
                "spike": {k: v for k, v in spike_map.items() if v > 0},
                "pattern_bonus": round(pattern_bonus, 3),
            },
            "baseline": {
                "samples": samples,
                "warmup_complete": samples >= self.min_samples,
                "warmup_left": max(self.min_samples - samples, 0),
                "sensitivity": self.sensitivity,
                "z_threshold": self.z_threshold,
            },
            "features": feats,
        }

    def baseline_snapshot(self) -> dict[str, Any]:
        out = {}
        for k, b in self.baselines.items():
            out[k] = {
                "mean": round(b.mean, 3),
                "std": round(math.sqrt(max(b.var, 0)), 3),
                "n": b.n,
                "last": round(b.last, 3),
            }
        return {
            "features": out,
            "sensitivity": self.sensitivity,
            "z_threshold": self.z_threshold,
            "min_samples": self.min_samples,
        }

    def reset_baseline(self) -> None:
        self.baselines = {k: FeatureBaseline() for k in FEATURE_KEYS}
