# EAGLE-X Core

نظام مراقبة أمنية تشغيلية للجهاز المضيف — نواة نظيفة مع **تجميع كشف شذوذ متقدم**.

**Public repo:** https://github.com/omarlaghmarti32-afk/eagle-x-core

## Anomaly ensemble (12 layers)

| # | Technique | Role |
|---|-----------|------|
| 1 | Hard ceilings | Absolute unsafe levels |
| 2 | EWMA z-scores | Adaptive statistical baseline |
| 3 | Spike detection | Sudden rate-of-change |
| 4 | Multi-signal patterns | Correlated abuse signatures |
| 5 | Isolation Forest | Path-length isolation |
| 6 | LOF | Local density outliers |
| 7 | DBSCAN | Density clustering / noise |
| 8 | One-Class SVM | Novelty boundary |
| 9 | Elliptic Envelope | Robust Mahalanobis |
| 10 | PCA reconstruction | Lightweight autoencoder-style error |
| 11 | CUSUM | Sequential change detection |
| 12 | Consensus vote | Multi-model agreement |

## Quick start

```bash
git clone https://github.com/omarlaghmarti32-afk/eagle-x-core.git
cd eagle-x-core
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export EAGLE_API_TOKEN=change-me
export EAGLE_LIVE_MONITOR=1
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

- Dashboard: http://127.0.0.1:8080  
- Health: http://127.0.0.1:8080/api/health  
- Baseline / models: http://127.0.0.1:8080/api/detector/baseline  

```bash
docker compose up --build -d
```

## API highlights

| Method | Path | Auth |
|--------|------|------|
| GET | `/api/health` | no |
| GET | `/api/ready` | no |
| GET | `/api/detector/baseline` | no |
| POST | `/api/detect` | Bearer |
| POST | `/api/detector/ml/train` | Bearer |
| POST | `/api/detector/reset` | Bearer |

## Tests

```bash
EAGLE_LIVE_MONITOR=0 pytest -q
```

## License

MIT
