"""SQLite persistence for threats, metrics, audit."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .config import DB_PATH


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS threats (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts REAL NOT NULL,
                  threat_type TEXT NOT NULL,
                  confidence REAL,
                  severity TEXT,
                  source TEXT,
                  features TEXT,
                  action_taken TEXT,
                  status TEXT,
                  sealed TEXT
                );
                CREATE TABLE IF NOT EXISTS metrics (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts REAL NOT NULL,
                  scans INTEGER,
                  threats INTEGER,
                  cpu REAL,
                  mem REAL
                );
                CREATE TABLE IF NOT EXISTS audit (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts REAL NOT NULL,
                  event TEXT NOT NULL,
                  detail TEXT
                );
                CREATE TABLE IF NOT EXISTS blocks (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts REAL NOT NULL,
                  indicator TEXT NOT NULL,
                  reason TEXT
                );
                """
            )

    def add_threat(self, **kw: Any) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO threats
                   (ts, threat_type, confidence, severity, source, features,
                    action_taken, status, sealed)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    time.time(),
                    kw.get("threat_type", "UNKNOWN"),
                    float(kw.get("confidence", 0)),
                    kw.get("severity", "medium"),
                    kw.get("source", "system"),
                    json.dumps(kw.get("features") or {}),
                    kw.get("action_taken", "logged"),
                    kw.get("status", "detected"),
                    kw.get("sealed"),
                ),
            )
            return int(cur.lastrowid)

    def list_threats(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM threats ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_threats(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM threats").fetchone()[0])

    def record_metrics(self, scans: int, threats: int, cpu: float, mem: float) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO metrics (ts, scans, threats, cpu, mem) VALUES (?,?,?,?,?)",
                (time.time(), scans, threats, cpu, mem),
            )

    def add_audit(self, event: str, detail: dict | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO audit (ts, event, detail) VALUES (?,?,?)",
                (time.time(), event, json.dumps(detail or {})),
            )

    def add_block(self, indicator: str, reason: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO blocks (ts, indicator, reason) VALUES (?,?,?)",
                (time.time(), indicator, reason),
            )

    def list_blocks(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM blocks ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
