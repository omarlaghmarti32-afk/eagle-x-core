# EAGLE-X Core

نظام مراقبة أمنية تشغيلية خفيف ونظيف للجهاز المضيف.

**EAGLE-X Core** is a from-scratch rebuild: real host metrics, AES-256-GCM + Ed25519, SQLite audit, anomaly scoring, and a FastAPI control plane with health/readiness probes.

## Features

| Module | Role |
|--------|------|
| `app/monitor.py` | CPU, memory, disk, network rates, process/connection counts (psutil) |
| `app/detector.py` | Rule + statistical anomaly scoring (no heavy ML runtime required) |
| `app/crypto.py` | AES-256-GCM encryption + Ed25519 signatures |
| `app/db.py` | SQLite threats, metrics, audit trail |
| `app/health.py` | Deep health + readiness checks |
| `app/main.py` | FastAPI API + optional live monitor loop |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

export EAGLE_API_TOKEN=change-me
export EAGLE_LIVE_MONITOR=1
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Open http://127.0.0.1:8080  ·  Health: http://127.0.0.1:8080/api/health

## Docker

```bash
docker compose up --build -d
curl -s http://127.0.0.1:8080/api/ready
```

## API (summary)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | no | Liveness |
| GET | `/api/ready` | no | Readiness |
| GET | `/api/health/deep` | no | Component checks |
| GET | `/api/status` | no | Runtime status |
| GET | `/api/stats` | no | Host snapshot + counters |
| GET | `/api/threats` | no | Recent threats |
| POST | `/api/detect` | Bearer | Score a feature vector |
| POST | `/api/heal` | Bearer | Record a healing action |

```bash
curl -X POST http://127.0.0.1:8080/api/detect \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"features":{"cpu_percent":95,"mem_percent":92,"net_sent_rate":5e6,"net_recv_rate":5e6,"process_count":400,"connection_count":500,"disk_percent":88}}'
```

## Tests

```bash
EAGLE_LIVE_MONITOR=0 pytest -q
```

## Relation to eagle-x-v3.3

`eagle-x-v3.3` remains the experimental line (PQC/pcap/Caddy experiments).  
**This repository is the clean production core.**

## License

MIT — see [LICENSE](LICENSE).
