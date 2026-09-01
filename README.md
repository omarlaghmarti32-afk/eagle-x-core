# EAGLE-X Core

نظام مراقبة أمنية تشغيلية — تجميع 12 طبقة + Webhooks (Slack/Discord) + حفظ الحالة + لوحة تحكم حيّة.

**https://github.com/omarlaghmarti32-afk/eagle-x-core** · **v1.2.0**

## Features

- Ensemble: EWMA, spikes, patterns, Isolation Forest, LOF, DBSCAN, One-Class SVM, Elliptic Envelope, PCA, CUSUM, consensus
- **Live dashboard** at `/` — models, host metrics, threats, baselines
- **Webhooks**: auto-detect Slack / Discord / raw JSON
- **Persistence**: baselines + models across restarts

## Quick start

```bash
git clone https://github.com/omarlaghmarti32-afk/eagle-x-core.git
cd eagle-x-core
pip install -r requirements.txt

export EAGLE_API_TOKEN=change-me
export EAGLE_LIVE_MONITOR=1

# Optional alerts
export EAGLE_WEBHOOK_URL=https://hooks.slack.com/services/...   # or Discord webhook
export EAGLE_WEBHOOK_FORMAT=auto   # auto | slack | discord | raw
export EAGLE_WEBHOOK_MIN_VOTES=3

uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Open http://127.0.0.1:8080

## Slack / Discord

| Provider | URL pattern | Format |
|----------|-------------|--------|
| Slack | `hooks.slack.com/services/...` | auto → Slack blocks |
| Discord | `discord.com/api/webhooks/...` | auto → embeds |
| Custom | any HTTPS URL | raw EAGLE JSON |

Force format: `EAGLE_WEBHOOK_FORMAT=slack`

## API

| Method | Path |
|--------|------|
| GET | `/` dashboard |
| GET | `/api/health` |
| GET | `/api/detector/baseline` |
| POST | `/api/detect` Bearer, optional `notify` |
| POST | `/api/detector/ml/train` |
| POST | `/api/detector/state/save` |

## Tests

```bash
EAGLE_LIVE_MONITOR=0 pytest -q
```

## License

MIT
