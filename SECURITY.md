# Security Policy — EAGLE-X Core

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.3.x   | Yes       |
| < 1.3   | No        |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report privately via:

1. GitHub Security Advisories (preferred): use the **Security** tab → Report a vulnerability
2. Or contact the maintainer through the account associated with this repository

Include:
- Description and impact
- Steps to reproduce / PoC
- Affected version and configuration

You can expect an initial response within a reasonable timeframe. Confirmed issues will be fixed and credited (if desired).

## Hardening Checklist (operators)

1. Set a strong `EAGLE_API_TOKEN` (≥16 random characters):
   ```bash
   export EAGLE_API_TOKEN=$(openssl rand -hex 24)
   ```
2. Keep `EAGLE_REQUIRE_STRONG_TOKEN=1` in production (default in Docker).
3. Optionally lock read APIs: `EAGLE_PROTECT_READ_APIS=1`.
4. Restrict CORS: `EAGLE_CORS_ORIGINS=https://your-domain.com`.
5. Webhook URLs are checked against private/metadata IPs (SSRF guard). Set `EAGLE_WEBHOOK_ALLOW_PRIVATE=1` only if you intentionally need private targets.
6. Do not expose the service directly without TLS.

## Built-in Controls (v1.3+)

- Constant-time Bearer comparison (`secrets.compare_digest`)
- Strong-token startup policy
- Rate limiting on mutating `/api/*` routes
- Security headers (CSP, X-Frame-Options, nosniff, HSTS on HTTPS)
- Configurable CORS
- Optional protection of read endpoints
- Webhook SSRF guard
- Path hardening for model/baseline files under `DATA_DIR`
- Docker: non-root user, no baked-in token, `no-new-privileges`
