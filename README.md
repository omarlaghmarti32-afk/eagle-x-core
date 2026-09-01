# EAGLE-X Core

نظام مراقبة أمنية تشغيلية للجهاز المضيف — تجميع كشف شذوذ متقدم + تنبيهات Webhook + حفظ الحالة.

**Public:** https://github.com/omarlaghmarti32-afk/eagle-x-core  
**Version:** 1.1.0

## Capabilities

- **12-layer ensemble:** hard ceilings, EWMA, spikes, patterns, Isolation Forest, LOF, DBSCAN, One-Class SVM, Elliptic Envelope, PCA reconstruction, CUSUM, consensus vote
- **Webhooks:** POST JSON when `vote_count >= 3` (or high severity)
- **Persistence:** baselines (JSON) + sklearn models (joblib) survive restarts
- **Crypto seal** of threat records (AES-GCM + Ed25519)
- FastAPI + Docker + health endpoints

## Quick start

```bash
git clone https://github.com/omarlaghmarti32-afk/eagle-x-core.git
cd eagle-x-core
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export EAGLE_API_TOKEN=change-me
export EAGLE_LIVE_MONITOR=1
export EAGLE_WEBHOOK_URL=https://hooks.example.com/eagle   # optional
export EAGLE_WEBHOOK_MIN_VOTES=3

uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Docker:

```bash
docker compose up --build -d
```

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `EAGLE_API_TOKEN` | dev token | Bearer auth |
| `EAGLE_LIVE_MONITOR` | `1` | background host scan |
| `EAGLE_WEBHOOK_URL` | empty | disable if empty |
| `EAGLE_WEBHOOK_MIN_VOTES` | `3` | consensus threshold |
| `EAGLE_WEBHOOK_MIN_SEVERITY` | `medium` | also notify on severity |
| `EAGLE_DATA_DIR` | `./data` | db + keys + models |
| `EAGLE_STATE_SAVE_EVERY` | `30` | autosave every N scans |

## API

| Method | Path | Auth |
|--------|------|------|
| GET | `/api/health` | no |
| GET | `/api/detector/baseline` | no |
| POST | `/api/detect` | Bearer (`notify=true` optional) |
| POST | `/api/detector/ml/train` | Bearer |
| POST | `/api/detector/state/save` | Bearer |
| POST | `/api/detector/state/load` | Bearer |

Webhook body example:

```json
{
  "event": "eagle.threat",
  "threat_type": "ENSEMBLE_ANOMALY",
  "confidence": 0.81,
  "severity": "high",
  "vote_count": 4,
  "votes": ["iforest", "lof", "ocsvm", "pca"]
}
```

## Tests

```bash
EAGLE_LIVE_MONITOR=0 pytest -q
```

## License

MIT
