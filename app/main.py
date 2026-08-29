"""EAGLE-X Core — FastAPI application."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import (
    API_TOKEN,
    LIVE_MONITOR,
    LOG_DIR,
    LOG_LEVEL,
    SEAL,
    VERSION,
)
from .crypto import CryptoEngine
from .db import Store
from .detector import ThreatDetector
from .health import run_checks
from .monitor import HostMonitor

LOG_FILE = LOG_DIR / "eagle-core.log"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | EAGLE-CORE | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eagle-core")

crypto = CryptoEngine()
store = Store()
detector = ThreatDetector()
monitor = HostMonitor()

started = time.time()
scans = 0
_running = True
_task: Optional[asyncio.Task] = None


async def live_loop() -> None:
    global scans
    logger.info(
        "Live monitor started (EWMA + IsolationForest available=%s)",
        detector._iforest_available,
    )
    try:
        async for snap in monitor.stream():
            if not _running:
                break
            scans += 1
            analysis = detector.analyze(snap)
            if scans % 15 == 0:
                store.record_metrics(
                    scans,
                    store.count_threats(),
                    snap.get("cpu_percent", 0),
                    snap.get("mem_percent", 0),
                )
            if analysis["threat_detected"]:
                sealed = crypto.seal(analysis)
                tid = store.add_threat(
                    threat_type=analysis["threat_type"],
                    confidence=analysis["confidence"],
                    severity=analysis["severity"],
                    source="live_monitor",
                    features=analysis["features"],
                    action_taken="logged",
                    status="detected",
                    sealed=sealed["ciphertext"],
                )
                store.add_audit(
                    "threat_detected",
                    {
                        "id": tid,
                        "type": analysis["threat_type"],
                        "confidence": analysis["confidence"],
                        "hits": analysis.get("hits"),
                        "iforest": analysis.get("scores", {}).get("isolation_forest"),
                    },
                )
                logger.warning(
                    "Threat #%s %s conf=%.2f iforest=%s",
                    tid,
                    analysis["threat_type"],
                    analysis["confidence"],
                    analysis.get("scores", {}).get("isolation_forest", {}).get("anomaly"),
                )
    except asyncio.CancelledError:
        logger.info("Live monitor cancelled")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _task, _running
    _running = True
    store.add_audit(
        "startup",
        {
            "version": VERSION,
            "seal": SEAL,
            "iforest": detector._iforest_available,
        },
    )
    if LIVE_MONITOR:
        _task = asyncio.create_task(live_loop())
    yield
    _running = False
    if _task:
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
    store.add_audit("shutdown", {})


app = FastAPI(
    title="EAGLE-X Core",
    description="Host security monitor — EWMA + Isolation Forest ensemble",
    version=VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_token(authorization: Optional[str] = Header(default=None)):
    if not API_TOKEN:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.split(" ", 1)[1].strip() != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    return True


class DetectBody(BaseModel):
    features: dict[str, float] | list[float] = Field(...)
    indicator: Optional[str] = None
    update_baseline: bool = True


class HealBody(BaseModel):
    threat_type: str = "MANUAL"
    indicator: Optional[str] = None


class SensitivityBody(BaseModel):
    sensitivity: float = Field(..., ge=0.1, le=0.95)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html = Path(__file__).parent.parent / "static" / "dashboard.html"
    if html.exists():
        return html.read_text(encoding="utf-8")
    return f"<h1>EAGLE-X Core {VERSION}</h1><p><a href='/api/health'>/api/health</a></p>"


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": VERSION,
        "seal": SEAL,
        "uptime_seconds": int(time.time() - started),
        "scans": scans,
        "iforest_available": detector._iforest_available,
        "iforest_trained": detector._iforest_trained,
    }


@app.get("/api/ready")
async def ready():
    report = run_checks(store=store, crypto=crypto, uptime=int(time.time() - started), scans=scans)
    critical = {"database", "crypto"}
    failed = [c for c in report["failed"] if c in critical]
    body = {
        "ready": not failed,
        "status": "ready" if not failed else "not_ready",
        "failed_critical": failed,
    }
    return JSONResponse(body, status_code=200 if not failed else 503)


@app.get("/api/health/deep")
async def health_deep():
    report = run_checks(
        store=store,
        crypto=crypto,
        uptime=int(time.time() - started),
        scans=scans,
        live=LIVE_MONITOR,
    )
    report["isolation_forest"] = detector.baseline_snapshot().get("isolation_forest")
    code = 200 if report["status"] in ("ok", "degraded") else 503
    return JSONResponse(report, status_code=code)


@app.get("/api/status")
async def status():
    return {
        "version": VERSION,
        "seal": SEAL,
        "uptime_seconds": int(time.time() - started),
        "scans": scans,
        "threats_total": store.count_threats(),
        "live_monitor": LIVE_MONITOR,
        "detector": detector.baseline_snapshot(),
        "crypto_pub": crypto.public_key_b64()[:16] + "…",
    }


@app.get("/api/stats")
async def stats():
    snap = await asyncio.to_thread(monitor.snapshot)
    return {"scans": scans, "threats": store.count_threats(), "host": snap}


@app.get("/api/threats")
async def threats():
    rows = store.list_threats(50)
    return {
        "threats": [
            {
                "id": r["id"],
                "timestamp": r["ts"],
                "type": r["threat_type"],
                "confidence": r["confidence"],
                "severity": r["severity"],
                "status": r["status"],
            }
            for r in rows
        ]
    }


@app.get("/api/detector/baseline")
async def detector_baseline():
    return detector.baseline_snapshot()


@app.post("/api/detector/sensitivity")
async def set_sensitivity(body: SensitivityBody, _: bool = Depends(require_token)):
    detector.sensitivity = float(body.sensitivity)
    store.add_audit("sensitivity", {"value": detector.sensitivity})
    return {"ok": True, "sensitivity": detector.sensitivity}


@app.post("/api/detector/reset")
async def reset_baseline(_: bool = Depends(require_token)):
    detector.reset_baseline()
    store.add_audit("baseline_reset", {})
    return {"ok": True, "baseline": detector.baseline_snapshot()}


@app.post("/api/detector/iforest/train")
async def train_iforest(_: bool = Depends(require_token)):
    result = detector.train_iforest_now()
    store.add_audit("iforest_train", result)
    return result


@app.post("/api/detect")
async def detect(body: DetectBody, _: bool = Depends(require_token)):
    analysis = detector.analyze(body.features, update_baseline=body.update_baseline)
    sealed = crypto.seal(analysis)
    if analysis["threat_detected"]:
        store.add_threat(
            threat_type=analysis["threat_type"],
            confidence=analysis["confidence"],
            severity=analysis["severity"],
            source="api",
            features=analysis["features"],
            action_taken="api_detect",
            status="detected",
            sealed=sealed["ciphertext"],
        )
        if body.indicator:
            store.add_block(body.indicator, analysis["threat_type"])
    return {"analysis": analysis, "seal": sealed}


@app.post("/api/heal")
async def heal(body: HealBody, _: bool = Depends(require_token)):
    if body.indicator:
        store.add_block(body.indicator, body.threat_type)
    store.add_audit("heal", {"threat_type": body.threat_type, "indicator": body.indicator})
    return {"ok": True, "threat_type": body.threat_type, "indicator": body.indicator}


@app.get("/api/blocklist")
async def blocklist(_: bool = Depends(require_token)):
    return {"blocks": store.list_blocks()}
