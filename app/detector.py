"""Advanced host anomaly detection ensemble (production core).

Layers:
  1. Hard ceilings
  2. EWMA z-scores
  3. Spike / rate-of-change
  4. Multi-signal patterns
  5. Isolation Forest
  6. Local Outlier Factor (LOF)
  7. DBSCAN density clustering
  8. One-Class SVM (novelty)
  9. Elliptic Envelope (robust Mahalanobis)
 10. PCA reconstruction error (lightweight autoencoder-style)
 11. CUSUM sequential change detection
 12. Ensemble consensus vote

Shared rolling buffer trains sklearn models together.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

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
    from sklearn.covariance import EllipticEnvelope
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    DBSCAN = IsolationForest = LocalOutlierFactor = None  # type: ignore
    EllipticEnvelope = PCA = OneClassSVM = StandardScaler = None  # type: ignore
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
    iforest_weight: float = 0.12

    lof_contamination: float = field(
        default_factory=lambda: float(os.environ.get("EAGLE_LOF_CONTAMINATION", "0.08"))
    )
    lof_n_neighbors: int = field(
        default_factory=lambda: int(os.environ.get("EAGLE_LOF_NEIGHBORS", "20"))
    )
    lof_weight: float = 0.12

    dbscan_eps: float = field(
        default_factory=lambda: float(os.environ.get("EAGLE_DBSCAN_EPS", "1.2"))
    )
    dbscan_min_samples: int = field(
        default_factory=lambda: int(os.environ.get("EAGLE_DBSCAN_MIN_SAMPLES", "5"))
    )
    dbscan_weight: float = 0.12

    ocsvm_nu: float = field(
        default_factory=lambda: float(os.environ.get("EAGLE_OCSVM_NU", "0.08"))
    )
    ocsvm_weight: float = 0.12

    elliptic_contamination: float = field(
        default_factory=lambda: float(os.environ.get("EAGLE_ELLIPTIC_CONTAMINATION", "0.08"))
    )
    elliptic_weight: float = 0.12

    pca_components: int = field(
        default_factory=lambda: int(os.environ.get("EAGLE_PCA_COMPONENTS", "3"))
    )
    pca_weight: float = 0.12

    cusum_threshold: float = field(
        default_factory=lambda: float(os.environ.get("EAGLE_CUSUM_THRESHOLD", "5.0"))
    )
    cusum_drift: float = 0.5
    cusum_weight: float = 0.10

    consensus_min_votes: int = 2
    consensus_weight: float = 0.15

    baselines: dict[str, FeatureBaseline] = field(default_factory=dict)
    _buffer: Deque[list[float]] = field(default_factory=lambda: deque(maxlen=256))

    _iforest: Any = field(default=None, repr=False)
    _lof: Any = field(default=None, repr=False)
    _dbscan_scaler: Any = field(default=None, repr=False)
    _dbscan_core: Any = field(default=None, repr=False)
    _dbscan_labels: Any = field(default=None, repr=False)
    _dbscan_n_clusters: int = 0
    _dbscan_n_noise: int = 0

    _ocsvm: Any = field(default=None, repr=False)
    _ocsvm_scaler: Any = field(default=None, repr=False)
    _elliptic: Any = field(default=None, repr=False)
    _elliptic_scaler: Any = field(default=None, repr=False)
    _pca: Any = field(default=None, repr=False)
    _pca_scaler: Any = field(default=None, repr=False)
    _pca_err_mean: float = 0.0
    _pca_err_std: float = 1.0

    _cusum_pos: dict[str, float] = field(default_factory=dict)
    _cusum_neg: dict[str, float] = field(default_factory=dict)

    _iforest_trained: bool = False
    _lof_trained: bool = False
    _dbscan_trained: bool = False
    _ocsvm_trained: bool = False
    _elliptic_trained: bool = False
    _pca_trained: bool = False
    _ml_seen: int = 0
    _sklearn_available: bool = field(default_factory=lambda: _HAS_SKLEARN)

    def __post_init__(self) -> None:
        self.sensitivity = float(min(max(self.sensitivity, 0.1), 0.95))
        self.iforest_contamination = float(min(max(self.iforest_contamination, 0.01), 0.3))
        self.lof_contamination = float(min(max(self.lof_contamination, 0.01), 0.3))
        self.elliptic_contamination = float(min(max(self.elliptic_contamination, 0.01), 0.3))
        self.ocsvm_nu = float(min(max(self.ocsvm_nu, 0.01), 0.5))
        self.lof_n_neighbors = max(5, int(self.lof_n_neighbors))
        self.dbscan_eps = float(max(self.dbscan_eps, 0.1))
        self.dbscan_min_samples = max(2, int(self.dbscan_min_samples))
        self.pca_components = max(1, min(int(self.pca_components), len(FEATURE_KEYS) - 1))
        self._buffer = deque(maxlen=self.ml_buffer_size)
        for k in FEATURE_KEYS:
            if k not in self.baselines:
                self.baselines[k] = FeatureBaseline()
            self._cusum_pos.setdefault(k, 0.0)
            self._cusum_neg.setdefault(k, 0.0)

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

    def _update_cusum(self, key: str, value: float) -> float:
        """Two-sided CUSUM; returns peak |s| after update."""
        b = self.baselines.get(key, FeatureBaseline())
        mean = b.mean if b.n else value
        std = math.sqrt(max(b.var, 1e-6)) if b.n >= 2 else max(abs(value) * 0.1, 1.0)
        z = (value - mean) / std
        self._cusum_pos[key] = max(0.0, self._cusum_pos.get(key, 0.0) + z - self.cusum_drift)
        self._cusum_neg[key] = min(0.0, self._cusum_neg.get(key, 0.0) + z + self.cusum_drift)
        return max(self._cusum_pos[key], abs(self._cusum_neg[key]))

    def _score_cusum(self, feats: dict[str, float], update: bool) -> dict[str, Any]:
        peaks: dict[str, float] = {}
        hits: list[str] = []
        for k in FEATURE_KEYS:
            if k not in feats:
                continue
            if update:
                peak = self._update_cusum(k, feats[k])
            else:
                peak = max(self._cusum_pos.get(k, 0.0), abs(self._cusum_neg.get(k, 0.0)))
            peaks[k] = round(peak, 3)
            if peak >= self.cusum_threshold:
                hits.append(k)
        strength = 0.0
        if hits:
            strength = min(1.0, max(peaks[h] for h in hits) / max(self.cusum_threshold * 2, 1e-6))
            strength = max(strength, 0.45)
        return {
            "available": True,
            "threshold": self.cusum_threshold,
            "peaks": {k: v for k, v in peaks.items() if v > 0},
            "hits": hits,
            "anomaly": bool(hits),
            "contribution": round(strength * self.cusum_weight, 3),
        }

    def _fit_ml(self) -> dict[str, bool]:
        result = {
            "iforest": False,
            "lof": False,
            "dbscan": False,
            "ocsvm": False,
            "elliptic": False,
            "pca": False,
        }
        if not self._sklearn_available or len(self._buffer) < self.ml_min_train:
            return result
        X = np.array(list(self._buffer), dtype=float)

        # Isolation Forest (raw space)
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

        # LOF novelty
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

        # Shared scaler for density / boundary models
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # DBSCAN
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
        self._dbscan_core = Xs[core_mask] if np.any(core_mask) else Xs
        self._dbscan_trained = True
        result["dbscan"] = True

        # One-Class SVM
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=self.ocsvm_nu)
        ocsvm.fit(Xs)
        self._ocsvm = ocsvm
        self._ocsvm_scaler = scaler
        self._ocsvm_trained = True
        result["ocsvm"] = True

        # Elliptic Envelope (robust covariance)
        try:
            elliptic = EllipticEnvelope(
                contamination=self.elliptic_contamination,
                random_state=42,
                support_fraction=None,
            )
            elliptic.fit(Xs)
            self._elliptic = elliptic
            self._elliptic_scaler = scaler
            self._elliptic_trained = True
            result["elliptic"] = True
        except Exception:
            self._elliptic_trained = False

        # PCA reconstruction
        n_comp = min(self.pca_components, Xs.shape[1], max(1, Xs.shape[0] - 1))
        pca = PCA(n_components=n_comp, random_state=42)
        Z = pca.fit_transform(Xs)
        X_hat = pca.inverse_transform(Z)
        errs = np.sqrt(np.sum((Xs - X_hat) ** 2, axis=1))
        self._pca = pca
        self._pca_scaler = scaler
        self._pca_err_mean = float(np.mean(errs))
        self._pca_err_std = float(max(np.std(errs), 1e-6))
        self._pca_trained = True
        result["pca"] = True

        return result

    def _score_iforest(self, vec: list[float]) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "available": self._sklearn_available,
            "trained": self._iforest_trained,
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
            "eps": self.dbscan_eps,
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
        Xs = self._dbscan_scaler.transform(np.array([vec], dtype=float))
        diffs = self._dbscan_core - Xs[0]
        min_dist = float(np.sqrt(np.sum(diffs * diffs, axis=1)).min())
        anomaly = min_dist > self.dbscan_eps
        strength = max(
            0.0,
            min(1.0, (min_dist - self.dbscan_eps * 0.5) / max(self.dbscan_eps, 1e-6)),
        )
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

    def _score_ocsvm(self, vec: list[float]) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "available": self._sklearn_available,
            "trained": self._ocsvm_trained,
            "nu": self.ocsvm_nu,
            "score": None,
            "anomaly": False,
            "contribution": 0.0,
        }
        if not self._ocsvm_trained or self._ocsvm is None or self._ocsvm_scaler is None:
            return meta
        Xs = self._ocsvm_scaler.transform(np.array([vec], dtype=float))
        raw = float(self._ocsvm.decision_function(Xs)[0])
        pred = int(self._ocsvm.predict(Xs)[0])
        strength = max(0.0, min(1.0, (0.0 - raw) / 1.0))
        if pred == -1:
            strength = max(strength, 0.55)
        meta.update(
            {
                "trained": True,
                "score": round(raw, 4),
                "anomaly": pred == -1,
                "contribution": round(strength * self.ocsvm_weight, 3),
            }
        )
        return meta

    def _score_elliptic(self, vec: list[float]) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "available": self._sklearn_available,
            "trained": self._elliptic_trained,
            "score": None,
            "anomaly": False,
            "contribution": 0.0,
        }
        if not self._elliptic_trained or self._elliptic is None or self._elliptic_scaler is None:
            return meta
        Xs = self._elliptic_scaler.transform(np.array([vec], dtype=float))
        raw = float(self._elliptic.decision_function(Xs)[0])
        pred = int(self._elliptic.predict(Xs)[0])
        strength = max(0.0, min(1.0, (0.0 - raw) / 20.0))
        if pred == -1:
            strength = max(strength, 0.55)
        meta.update(
            {
                "trained": True,
                "score": round(raw, 4),
                "anomaly": pred == -1,
                "contribution": round(strength * self.elliptic_weight, 3),
            }
        )
        return meta

    def _score_pca(self, vec: list[float]) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "available": self._sklearn_available,
            "trained": self._pca_trained,
            "components": self.pca_components,
            "error": None,
            "z_error": None,
            "anomaly": False,
            "contribution": 0.0,
        }
        if not self._pca_trained or self._pca is None or self._pca_scaler is None:
            return meta
        Xs = self._pca_scaler.transform(np.array([vec], dtype=float))
        Z = self._pca.transform(Xs)
        X_hat = self._pca.inverse_transform(Z)
        err = float(np.sqrt(np.sum((Xs - X_hat) ** 2)))
        z_err = (err - self._pca_err_mean) / self._pca_err_std
        anomaly = z_err >= 2.5
        strength = max(0.0, min(1.0, z_err / 4.0))
        if anomaly:
            strength = max(strength, 0.5)
        meta.update(
            {
                "trained": True,
                "error": round(err, 4),
                "z_error": round(z_err, 3),
                "anomaly": anomaly,
                "contribution": round(strength * self.pca_weight, 3),
            }
        )
        return meta

    def _maybe_update_ml(self, vec: list[float], update: bool) -> None:
        if not self._sklearn_available or not update:
            return
        self._buffer.append(list(vec))
        self._ml_seen += 1
        ready = len(self._buffer) >= self.ml_min_train
        all_ready = (
            self._iforest_trained
            and self._lof_trained
            and self._dbscan_trained
            and self._ocsvm_trained
            and self._pca_trained
        )
        need_first = (not all_ready) and ready
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
        elif "ensemble_consensus" in patterns:
            t = "ENSEMBLE_ANOMALY"
        elif "cusum_shift" in patterns:
            t = "SEQUENTIAL_SHIFT"
        elif "pca_reconstruction" in patterns:
            t = "PCA_ANOMALY"
        elif "elliptic_envelope" in patterns:
            t = "MAHALANOBIS_ANOMALY"
        elif "one_class_svm" in patterns:
            t = "OCSVM_ANOMALY"
        elif "dbscan_noise" in patterns:
            t = "DBSCAN_ANOMALY"
        elif "local_outlier_factor" in patterns:
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
        ocsvm_meta = self._score_ocsvm(vec)
        ell_meta = self._score_elliptic(vec)
        pca_meta = self._score_pca(vec)
        cusum_meta = self._score_cusum(feats, update=update_baseline)

        ml_metas = [if_meta, lof_meta, db_meta, ocsvm_meta, ell_meta, pca_meta, cusum_meta]
        for m in ml_metas:
            score += float(m.get("contribution") or 0.0)

        votes = []
        if if_meta.get("anomaly"):
            patterns = list(patterns) + ["isolation_forest"]
            votes.append("iforest")
        if lof_meta.get("anomaly"):
            patterns = list(patterns) + ["local_outlier_factor"]
            votes.append("lof")
        if db_meta.get("anomaly"):
            patterns = list(patterns) + ["dbscan_noise"]
            votes.append("dbscan")
        if ocsvm_meta.get("anomaly"):
            patterns = list(patterns) + ["one_class_svm"]
            votes.append("ocsvm")
        if ell_meta.get("anomaly"):
            patterns = list(patterns) + ["elliptic_envelope"]
            votes.append("elliptic")
        if pca_meta.get("anomaly"):
            patterns = list(patterns) + ["pca_reconstruction"]
            votes.append("pca")
        if cusum_meta.get("anomaly"):
            patterns = list(patterns) + ["cusum_shift"]
            votes.append("cusum")

        consensus = {
            "votes": votes,
            "vote_count": len(votes),
            "min_required": self.consensus_min_votes,
            "anomaly": len(votes) >= self.consensus_min_votes,
            "contribution": 0.0,
        }
        if consensus["anomaly"]:
            patterns = list(patterns) + ["ensemble_consensus"]
            c_strength = min(1.0, len(votes) / max(self.consensus_min_votes + 2, 1))
            consensus["contribution"] = round(c_strength * self.consensus_weight, 3)
            score += consensus["contribution"]

        confidence = min(round(score, 3), 1.0)
        ml_anomaly = bool(votes)

        threat = (
            confidence >= self.sensitivity
            or len(hard) >= 1
            or (len(z_hits) + len(spike_hits) >= 3 and confidence >= self.sensitivity * 0.75)
            or consensus["anomaly"]
            or any(
                m.get("anomaly") and float(m.get("contribution") or 0) >= 0.05 for m in ml_metas
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
                "one_class_svm": ocsvm_meta,
                "elliptic_envelope": ell_meta,
                "pca": pca_meta,
                "cusum": cusum_meta,
                "consensus": consensus,
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
                "ocsvm_trained": self._ocsvm_trained,
                "elliptic_trained": self._elliptic_trained,
                "pca_trained": self._pca_trained,
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
                "weight": self.iforest_weight,
            },
            "lof": {
                "available": self._sklearn_available,
                "trained": self._lof_trained,
                "n_neighbors": self.lof_n_neighbors,
                "weight": self.lof_weight,
            },
            "dbscan": {
                "available": self._sklearn_available,
                "trained": self._dbscan_trained,
                "eps": self.dbscan_eps,
                "n_clusters": self._dbscan_n_clusters,
                "n_noise_in_fit": self._dbscan_n_noise,
                "weight": self.dbscan_weight,
            },
            "one_class_svm": {
                "available": self._sklearn_available,
                "trained": self._ocsvm_trained,
                "nu": self.ocsvm_nu,
                "weight": self.ocsvm_weight,
            },
            "elliptic_envelope": {
                "available": self._sklearn_available,
                "trained": self._elliptic_trained,
                "contamination": self.elliptic_contamination,
                "weight": self.elliptic_weight,
            },
            "pca": {
                "available": self._sklearn_available,
                "trained": self._pca_trained,
                "components": self.pca_components,
                "err_mean": round(self._pca_err_mean, 4),
                "err_std": round(self._pca_err_std, 4),
                "weight": self.pca_weight,
            },
            "cusum": {
                "available": True,
                "threshold": self.cusum_threshold,
                "weight": self.cusum_weight,
            },
            "buffer_size": len(self._buffer),
            "ml_min_train": self.ml_min_train,
        }

    def reset_baseline(self) -> None:
        self.baselines = {k: FeatureBaseline() for k in FEATURE_KEYS}
        self._buffer.clear()
        self._iforest = self._lof = None
        self._dbscan_scaler = self._dbscan_core = self._dbscan_labels = None
        self._ocsvm = self._ocsvm_scaler = None
        self._elliptic = self._elliptic_scaler = None
        self._pca = self._pca_scaler = None
        self._pca_err_mean, self._pca_err_std = 0.0, 1.0
        self._dbscan_n_clusters = self._dbscan_n_noise = 0
        self._iforest_trained = self._lof_trained = self._dbscan_trained = False
        self._ocsvm_trained = self._elliptic_trained = self._pca_trained = False
        self._cusum_pos = {k: 0.0 for k in FEATURE_KEYS}
        self._cusum_neg = {k: 0.0 for k in FEATURE_KEYS}
        self._ml_seen = 0

    def train_ml_now(self) -> dict[str, Any]:
        fitted = self._fit_ml()
        return {
            "ok": any(fitted.values()),
            "fitted": fitted,
            "buffer_size": len(self._buffer),
            "available": self._sklearn_available,
            "n_clusters": self._dbscan_n_clusters,
        }

    def train_iforest_now(self) -> dict[str, Any]:
        f = self._fit_ml()
        return {"ok": f.get("iforest", False), "trained": self._iforest_trained}

    def train_lof_now(self) -> dict[str, Any]:
        f = self._fit_ml()
        return {"ok": f.get("lof", False), "trained": self._lof_trained}

    def train_dbscan_now(self) -> dict[str, Any]:
        f = self._fit_ml()
        return {
            "ok": f.get("dbscan", False),
            "trained": self._dbscan_trained,
            "n_clusters": self._dbscan_n_clusters,
        }
