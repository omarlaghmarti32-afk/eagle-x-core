"""Deep anomaly detection ensemble.

Layers:
  1. Hard ceilings
  2. EWMA z-scores
  3. Spike / rate-of-change
  4. Multi-signal patterns
  5. Isolation Forest
  6. Local Outlier Factor (LOF)
  7. DBSCAN density clustering — points far from clusters = anomaly

Shared rolling buffer. Without sklearn, layers 1–4 still run.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

FEATURE_KEYS = [
    "cpu_percent",
    "mem_percent",
    "net_sent_rate",
    "net_recv_rate",
    "process_count",
    "connection_count",
    "disk_percent",
]

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

SPIKE_RATIO = {
    "cpu_percent": 2.5,
    "mem_percent": 1.8,
    "net_sent_rate": 4.0,
    "net_recv_rate": 4.0,
    "process_count": 1.6,
    "connection_count": 2.0,
    "disk_percent": 1.15,
}

try:
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    DBSCAN = None  # type: ignore
    IsolationForest = None  # type: ignore
    LocalOutlierFactor = None  # type: ignore
    StandardScaler = None  # type: ignore
    _HAS_SKLEARN = False


@dataclass
class FeatureBaseline:
    mean: float = 0.0
    var: float = 1.0
    n: int = 0
    last: float = 0.0


@dataclass
class ThreatDetector:
    sensitivity: float = field(
        default_factory=lambda: float(os.environ.get("EAGLE_AI_SENSITIVITY", "0.55"))
    )
    alpha: float = 0.12
    z_threshold: float = 3.0
    min_samples: int = 8

    ml_min_train: int = 32
    ml_refit_every: int = 16
    ml_buffer_size: int = 256

    iforest_contamination: float = field(
        default_factory=lambda: float(os.environ.get("EAGLE_IFOREST_CONTAMINATION", "0.08"))
    )
    iforest_weight: float = 0.22

    lof_contamination: float = field(
        default_factory=lambda: float(os.environ.get("EAGLE_LOF_CONTAMINATION", "0.08"))
    )
    lof_n_neighbors: int = field(
        default_factory=lambda: int(os.environ.get("EAGLE_LOF_NEIGHBORS", "20"))
    )
    lof_weight: float = 0.22

    dbscan_eps: float = field(
        default_factory=lambda: float(os.environ.get("EAGLE_DBSCAN_EPS", "1.2"))
    )
    dbscan_min_samples: int = field(
        default_factory=lambda: int(os.environ.get("EAGLE_DBSCAN_MIN_SAMPLES", "5"))
    )
    dbscan_weight: float = 0.22

    baselines: dict[str, FeatureBaseline] = field(default_factory=dict)
    _buffer: Deque[list[float]] = field(default_factory=lambda: deque(maxlen=256))
    _iforest: Any = field(default=None, repr=False)
    _lof: Any = field(default=None, repr=False)
    _dbscan_scaler: Any = field(default=None, repr=False)
    _dbscan_core: Any = field(default=None, repr=False)  # scaled core points
    _dbscan_labels: Any = field(default=None, repr=False)
    _dbscan_n_clusters: int = 0
    _dbscan_n_noise: int = 0
    _iforest_trained: bool = False
    _lof_trained: bool = False
    _dbscan_trained: bool = False
    _ml_seen: int = 0
    _sklearn_available: bool = field(default_factory=lambda: _HAS_SKLEARN)

    def __post_init__(self) -> None:
        self.sensitivity = float(min(max(self.sensitivity, 0.1), 0.95))
        self.iforest_contamination = float(min(max(self.iforest_contamination, 0.01), 0.3))
        self.lof_contamination = float(min(max(self.lof_contamination, 0.01), 0.3))
        self.lof_n_neighbors = max(5, int(self.lof_n_neighbors))
        self.dbscan_eps = float(max(self.dbscan_eps, 0.1))
        self.dbscan_min_samples = max(2, int(self.dbscan_min_samples))
        self._buffer = deque(maxlen=self.ml_buffer_size)
        for k in FEATURE_KEYS:
            if k not in self.baselines:
                self.baselines[k] = FeatureBaseline()

    @property
    def _iforest_available(self) -> bool:
        return self._sklearn_available

    def _vector(self, feats: dict[str, float]) -> list[float]:
        return [float(feats.get(k, 0.0)) for k in FEATURE_KEYS]

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
        for k, v in features.items():
            if k not in out:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    pass
        return out

    def _update_baseline(self, key: str, value: float) -> tuple[float, float]:
        b = self.baselines.setdefault(key, FeatureBaseline())
        spike = 0.0
        if b.n > 0 and b.last > 0:
            ratio = value / max(b.last, 1e-9)
            need = SPIKE_RATIO.get(key, 2.0)
            if ratio >= need:
                spike = min((ratio - need) / need, 2.0) / 2.0

        std = math.sqrt(max(b.var, 1e-6))
        z = (value - b.mean) / std if b.n >= 2 else 0.0

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
        return [k for k, ceiling in HARD_CEILINGS.items() if float(features.get(k, 0)) >= ceiling]

    def _pattern_bonus(
        self, features: dict[str, float], z_map: dict[str, float]
    ) -> tuple[float, list[str]]:
        patterns: list[str] = []
        bonus = 0.0
        cpu = features.get("cpu_percent", 0)
        mem = features.get("mem_percent", 0)
        sent = features.get("net_sent_rate", 0)
        recv = features.get("net_recv_rate", 0)
        procs = features.get("process_count", 0)
        conns = features.get("connection_count", 0)

        if cpu >= 85 and mem >= 80:
            patterns.append("cpu_mem_pressure")
            bonus += 0.22
        if sent > 1_500_000 and sent > max(recv * 3, 1):
            patterns.append("egress_dominant")
            bonus += 0.2
        if conns >= 300 and procs >= 250:
            patterns.append("conn_process_flood")
            bonus += 0.2
        if (
            z_map.get("net_sent_rate", 0) >= self.z_threshold
            and z_map.get("net_recv_rate", 0) >= self.z_threshold
        ):
            patterns.append("bidirectional_traffic_shift")
            bonus += 0.15
        if features.get("disk_percent", 0) >= 90 and (sent + recv) > 1_000_000:
            patterns.append("disk_and_io_pressure")
            bonus += 0.18
        return min(bonus, 0.55), patterns

    def _fit_ml(self) -> dict[str, bool]:
        result = {"iforest": False, "lof": False, "dbscan": False}
        if not self._sklearn_available or len(self._buffer) < self.ml_min_train:
            return result
        X = np.array(list(self._buffer), dtype=float)

        iforest = IsolationForest(
            n_estimators=100,
            contamination=self.iforest_contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=1,
        )
        iforest.fit(X)
        self._iforest = iforest
        self._iforest_trained = True
        result["iforest"] = True

        n_neighbors = min(self.lof_n_neighbors, max(5, len(self._buffer) - 1))
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.lof_contamination,
            novelty=True,
            n_jobs=1,
        )
        lof.fit(X)
        self._lof = lof
        self._lof_trained = True
        result["lof"] = True

        # DBSCAN on standardized features (rates vs percents differ in scale)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=min(self.dbscan_min_samples, max(2, len(self._buffer) // 8)),
            n_jobs=1,
        )
        labels = clustering.fit_predict(Xs)
        core_mask = labels != -1
        self._dbscan_scaler = scaler
        self._dbscan_labels = labels
        self._dbscan_n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        self._dbscan_n_noise = int(np.sum(labels == -1))
        if np.any(core_mask):
            self._dbscan_core = Xs[core_mask]
            self._dbscan_trained = True
            result["dbscan"] = True
        else:
            # all noise — still mark trained but core empty → everything anomalous-ish
            self._dbscan_core = Xs
            self._dbscan_trained = True
            result["dbscan"] = True

        return result

    def _score_iforest(self, vec: list[float]) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "available": self._sklearn_available,
            "trained": self._iforest_trained,
            "buffer_size": len(self._buffer),
            "score": None,
            "anomaly": False,
            "contribution": 0.0,
        }
        if not self._iforest_trained or self._iforest is None:
            return meta
        X = np.array([vec], dtype=float)
        raw = float(self._iforest.decision_function(X)[0])
        pred = int(self._iforest.predict(X)[0])
        strength = max(0.0, min(1.0, (0.15 - raw) / 0.5))
        if pred == -1:
            strength = max(strength, 0.55)
        meta.update(
            {
                "trained": True,
                "score": round(raw, 4),
                "anomaly": pred == -1,
                "contribution": round(strength * self.iforest_weight, 3),
            }
        )
        return meta

    def _score_lof(self, vec: list[float]) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "available": self._sklearn_available,
            "trained": self._lof_trained,
            "buffer_size": len(self._buffer),
            "n_neighbors": self.lof_n_neighbors,
            "score": None,
            "anomaly": False,
            "contribution": 0.0,
        }
        if not self._lof_trained or self._lof is None:
            return meta
        X = np.array([vec], dtype=float)
        raw = float(self._lof.decision_function(X)[0])
        pred = int(self._lof.predict(X)[0])
        strength = max(0.0, min(1.0, (0.1 - raw) / 0.6))
        if pred == -1:
            strength = max(strength, 0.55)
        meta.update(
            {
                "trained": True,
                "score": round(raw, 4),
                "anomaly": pred == -1,
                "contribution": round(strength * self.lof_weight, 3),
            }
        )
        return meta

    def _score_dbscan(self, vec: list[float]) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "available": self._sklearn_available,
            "trained": self._dbscan_trained,
            "buffer_size": len(self._buffer),
            "eps": self.dbscan_eps,
            "min_samples": self.dbscan_min_samples,
            "n_clusters": self._dbscan_n_clusters,
            "n_noise_in_fit": self._dbscan_n_noise,
            "distance": None,
            "anomaly": False,
            "contribution": 0.0,
        }
        if (
            not self._dbscan_trained
            or self._dbscan_scaler is None
            or self._dbscan_core is None
            or len(self._dbscan_core) == 0
        ):
            return meta

        X = np.array([vec], dtype=float)
        Xs = self._dbscan_scaler.transform(X)
        # min Euclidean distance to any in-cluster (core) point
        diffs = self._dbscan_core - Xs[0]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        min_dist = float(np.min(dists))
        # beyond eps → noise / anomaly; strength grows with distance
        anomaly = min_dist > self.dbscan_eps
        strength = max(0.0, min(1.0, (min_dist - self.dbscan_eps * 0.5) / max(self.dbscan_eps, 1e-6)))
        if anomaly:
            strength = max(strength, 0.5)

        meta.update(
            {
                "trained": True,
                "distance": round(min_dist, 4),
                "anomaly": anomaly,
                "contribution": round(strength * self.dbscan_weight, 3),
            }
        )
        return meta

    def _maybe_update_ml(self, vec: list[float], update: bool) -> None:
        if not self._sklearn_available or not update:
            return
        self._buffer.append(list(vec))
        self._ml_seen += 1
        ready = len(self._buffer) >= self.ml_min_train
        need_first = (
            not (self._iforest_trained and self._lof_trained and self._dbscan_trained) and ready
        )
        need_refit = self._iforest_trained and self._ml_seen % self.ml_refit_every == 0 and ready
        if need_first or need_refit:
            self._fit_ml()

    def _classify(
        self,
        confidence: float,
        hard: list[str],
        z_hits: list[str],
        spikes: list[str],
        patterns: list[str],
        ml_anomaly: bool,
    ) -> tuple[str, str]:
        if confidence < self.sensitivity and not hard and not ml_anomaly:
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
        elif "dbscan_noise" in patterns and "isolation_forest" not in patterns and "local_outlier_factor" not in patterns:
            t = "DBSCAN_ANOMALY"
        elif "local_outlier_factor" in patterns and "isolation_forest" not in patterns:
            t = "LOF_ANOMALY"
        elif "isolation_forest" in patterns:
            t = "IFOREST_ANOMALY"
        else:
            t = "HOST_ANOMALY"

        if confidence >= 0.8 or len(hard) >= 2:
            sev = "critical"
        elif confidence >= 0.65 or hard:
            sev = "high"
        elif confidence >= 0.45 or ml_anomaly:
            sev = "medium"
        else:
            sev = "low"
        return t, sev

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

        vec = self._vector(feats)
        self._maybe_update_ml(vec, update=update_baseline)
        if_meta = self._score_iforest(vec)
        lof_meta = self._score_lof(vec)
        db_meta = self._score_dbscan(vec)
        score += float(if_meta.get("contribution") or 0.0)
        score += float(lof_meta.get("contribution") or 0.0)
        score += float(db_meta.get("contribution") or 0.0)

        if if_meta.get("anomaly"):
            patterns = list(patterns) + ["isolation_forest"]
        if lof_meta.get("anomaly"):
            patterns = list(patterns) + ["local_outlier_factor"]
        if db_meta.get("anomaly"):
            patterns = list(patterns) + ["dbscan_noise"]

        confidence = min(round(score, 3), 1.0)
        ml_anomaly = bool(
            if_meta.get("anomaly") or lof_meta.get("anomaly") or db_meta.get("anomaly")
        )
        threat = (
            confidence >= self.sensitivity
            or len(hard) >= 1
            or (
                len(z_hits) + len(spike_hits) >= 3
                and confidence >= self.sensitivity * 0.75
            )
            or (
                if_meta.get("anomaly")
                and float(if_meta.get("contribution") or 0) >= self.iforest_weight * 0.5
            )
            or (
                lof_meta.get("anomaly")
                and float(lof_meta.get("contribution") or 0) >= self.lof_weight * 0.5
            )
            or (
                db_meta.get("anomaly")
                and float(db_meta.get("contribution") or 0) >= self.dbscan_weight * 0.5
            )
        )

        threat_type, severity = self._classify(
            confidence if threat else 0.0,
            hard,
            z_hits,
            spike_hits,
            patterns,
            ml_anomaly,
        )
        if not threat:
            threat_type, severity = "NONE", "none"

        samples = min(
            (self.baselines[k].n for k in FEATURE_KEYS if k in self.baselines),
            default=0,
        )

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
                "isolation_forest": if_meta,
                "lof": lof_meta,
                "dbscan": db_meta,
            },
            "baseline": {
                "samples": samples,
                "warmup_complete": samples >= self.min_samples,
                "warmup_left": max(self.min_samples - samples, 0),
                "sensitivity": self.sensitivity,
                "z_threshold": self.z_threshold,
                "iforest_trained": self._iforest_trained,
                "lof_trained": self._lof_trained,
                "dbscan_trained": self._dbscan_trained,
                "sklearn_available": self._sklearn_available,
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
            "isolation_forest": {
                "available": self._sklearn_available,
                "trained": self._iforest_trained,
                "buffer_size": len(self._buffer),
                "min_train": self.ml_min_train,
                "contamination": self.iforest_contamination,
                "weight": self.iforest_weight,
            },
            "lof": {
                "available": self._sklearn_available,
                "trained": self._lof_trained,
                "buffer_size": len(self._buffer),
                "min_train": self.ml_min_train,
                "n_neighbors": self.lof_n_neighbors,
                "contamination": self.lof_contamination,
                "weight": self.lof_weight,
            },
            "dbscan": {
                "available": self._sklearn_available,
                "trained": self._dbscan_trained,
                "buffer_size": len(self._buffer),
                "min_train": self.ml_min_train,
                "eps": self.dbscan_eps,
                "min_samples": self.dbscan_min_samples,
                "n_clusters": self._dbscan_n_clusters,
                "n_noise_in_fit": self._dbscan_n_noise,
                "weight": self.dbscan_weight,
            },
        }

    def reset_baseline(self) -> None:
        self.baselines = {k: FeatureBaseline() for k in FEATURE_KEYS}
        self._buffer.clear()
        self._iforest = None
        self._lof = None
        self._dbscan_scaler = None
        self._dbscan_core = None
        self._dbscan_labels = None
        self._dbscan_n_clusters = 0
        self._dbscan_n_noise = 0
        self._iforest_trained = False
        self._lof_trained = False
        self._dbscan_trained = False
        self._ml_seen = 0

    def train_iforest_now(self) -> dict[str, Any]:
        fitted = self._fit_ml()
        return {
            "ok": bool(fitted.get("iforest")),
            "trained": self._iforest_trained,
            "buffer_size": len(self._buffer),
            "available": self._sklearn_available,
        }

    def train_lof_now(self) -> dict[str, Any]:
        fitted = self._fit_ml()
        return {
            "ok": bool(fitted.get("lof")),
            "trained": self._lof_trained,
            "buffer_size": len(self._buffer),
            "available": self._sklearn_available,
            "n_neighbors": self.lof_n_neighbors,
        }

    def train_dbscan_now(self) -> dict[str, Any]:
        fitted = self._fit_ml()
        return {
            "ok": bool(fitted.get("dbscan")),
            "trained": self._dbscan_trained,
            "buffer_size": len(self._buffer),
            "available": self._sklearn_available,
            "n_clusters": self._dbscan_n_clusters,
            "n_noise": self._dbscan_n_noise,
            "eps": self.dbscan_eps,
        }

    def train_ml_now(self) -> dict[str, Any]:
        fitted = self._fit_ml()
        return {
            "ok": any(fitted.values()),
            "iforest": self._iforest_trained,
            "lof": self._lof_trained,
            "dbscan": self._dbscan_trained,
            "n_clusters": self._dbscan_n_clusters,
            "buffer_size": len(self._buffer),
            "available": self._sklearn_available,
        }
