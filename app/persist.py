"""Persist / restore ThreatDetector baselines and sklearn models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import BASELINE_PATH, MODEL_PATH
from .detector import FEATURE_KEYS, FeatureBaseline, ThreatDetector

logger = logging.getLogger("eagle-core.persist")


def save_baselines(detector: ThreatDetector, path: Path | None = None) -> Path:
    path = Path(path or BASELINE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sensitivity": detector.sensitivity,
        "baselines": {
            k: {
                "mean": b.mean,
                "var": b.var,
                "n": b.n,
                "last": b.last,
            }
            for k, b in detector.baselines.items()
        },
        "cusum_pos": dict(detector._cusum_pos),
        "cusum_neg": dict(detector._cusum_neg),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_baselines(detector: ThreatDetector, path: Path | None = None) -> bool:
    path = Path(path or BASELINE_PATH)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "sensitivity" in data:
            detector.sensitivity = float(data["sensitivity"])
        for k, raw in (data.get("baselines") or {}).items():
            detector.baselines[k] = FeatureBaseline(
                mean=float(raw.get("mean", 0)),
                var=float(raw.get("var", 1)),
                n=int(raw.get("n", 0)),
                last=float(raw.get("last", 0)),
            )
        for k in FEATURE_KEYS:
            detector.baselines.setdefault(k, FeatureBaseline())
        detector._cusum_pos = {k: float(v) for k, v in (data.get("cusum_pos") or {}).items()}
        detector._cusum_neg = {k: float(v) for k, v in (data.get("cusum_neg") or {}).items()}
        for k in FEATURE_KEYS:
            detector._cusum_pos.setdefault(k, 0.0)
            detector._cusum_neg.setdefault(k, 0.0)
        logger.info("Baselines loaded from %s", path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load baselines: %s", exc)
        return False


def save_models(detector: ThreatDetector, path: Path | None = None) -> Path | None:
    if not detector._sklearn_available:
        return None
    path = Path(path or MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        blob: dict[str, Any] = {
            "iforest": detector._iforest,
            "lof": detector._lof,
            "dbscan_scaler": detector._dbscan_scaler,
            "dbscan_core": detector._dbscan_core,
            "dbscan_n_clusters": detector._dbscan_n_clusters,
            "dbscan_n_noise": detector._dbscan_n_noise,
            "ocsvm": detector._ocsvm,
            "ocsvm_scaler": detector._ocsvm_scaler,
            "elliptic": detector._elliptic,
            "elliptic_scaler": detector._elliptic_scaler,
            "pca": detector._pca,
            "pca_scaler": detector._pca_scaler,
            "pca_err_mean": detector._pca_err_mean,
            "pca_err_std": detector._pca_err_std,
            "flags": {
                "iforest": detector._iforest_trained,
                "lof": detector._lof_trained,
                "dbscan": detector._dbscan_trained,
                "ocsvm": detector._ocsvm_trained,
                "elliptic": detector._elliptic_trained,
                "pca": detector._pca_trained,
            },
            "buffer": list(detector._buffer),
        }
        joblib.dump(blob, path)
        logger.info("Models saved to %s", path)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save models: %s", exc)
        return None


def load_models(detector: ThreatDetector, path: Path | None = None) -> bool:
    if not detector._sklearn_available:
        return False
    path = Path(path or MODEL_PATH)
    if not path.exists():
        return False
    try:
        import joblib

        blob = joblib.load(path)
        detector._iforest = blob.get("iforest")
        detector._lof = blob.get("lof")
        detector._dbscan_scaler = blob.get("dbscan_scaler")
        detector._dbscan_core = blob.get("dbscan_core")
        detector._dbscan_n_clusters = int(blob.get("dbscan_n_clusters") or 0)
        detector._dbscan_n_noise = int(blob.get("dbscan_n_noise") or 0)
        detector._ocsvm = blob.get("ocsvm")
        detector._ocsvm_scaler = blob.get("ocsvm_scaler")
        detector._elliptic = blob.get("elliptic")
        detector._elliptic_scaler = blob.get("elliptic_scaler")
        detector._pca = blob.get("pca")
        detector._pca_scaler = blob.get("pca_scaler")
        detector._pca_err_mean = float(blob.get("pca_err_mean") or 0.0)
        detector._pca_err_std = float(blob.get("pca_err_std") or 1.0)
        flags = blob.get("flags") or {}
        detector._iforest_trained = bool(flags.get("iforest") and detector._iforest is not None)
        detector._lof_trained = bool(flags.get("lof") and detector._lof is not None)
        detector._dbscan_trained = bool(flags.get("dbscan") and detector._dbscan_core is not None)
        detector._ocsvm_trained = bool(flags.get("ocsvm") and detector._ocsvm is not None)
        detector._elliptic_trained = bool(flags.get("elliptic") and detector._elliptic is not None)
        detector._pca_trained = bool(flags.get("pca") and detector._pca is not None)
        buf = blob.get("buffer") or []
        detector._buffer.clear()
        for row in buf[-detector.ml_buffer_size :]:
            detector._buffer.append(list(row))
        logger.info("Models loaded from %s", path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load models: %s", exc)
        return False


def save_all(detector: ThreatDetector) -> dict[str, Any]:
    b = save_baselines(detector)
    m = save_models(detector)
    return {"baselines": str(b), "models": str(m) if m else None}


def load_all(detector: ThreatDetector) -> dict[str, bool]:
    return {
        "baselines": load_baselines(detector),
        "models": load_models(detector),
    }
