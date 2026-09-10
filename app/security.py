"""Security helpers: auth, rate limit, SSRF guards, headers."""

from __future__ import annotations

import ipaddress
import secrets
import time
from collections import defaultdict, deque
from typing import Deque
from urllib.parse import urlparse

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .config import API_TOKEN, CORS_ORIGINS, RATE_LIMIT_PER_MIN, REQUIRE_STRONG_TOKEN

_DEFAULT_TOKENS = {
    "",
    "eagle-dev-token-change-me",
    "change-me",
    "change-me-to-a-long-random-secret",
    "test-token",
}


def token_is_weak(token: str | None) -> bool:
    t = (token or "").strip()
    if t in _DEFAULT_TOKENS:
        return True
    if len(t) < 16:
        return True
    return False


def assert_runtime_token_policy() -> None:
    if REQUIRE_STRONG_TOKEN and token_is_weak(API_TOKEN):
        raise RuntimeError(
            "EAGLE_API_TOKEN is missing or weak. "
            "Set a long random secret (>=16 chars) or set EAGLE_REQUIRE_STRONG_TOKEN=0 for local dev."
        )


def require_token(authorization: str | None = Header(default=None)) -> bool:
    if not API_TOKEN:
        raise HTTPException(status_code=503, detail="API token not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    provided = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(provided, API_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid token")
    return True


class RateLimiter:
    def __init__(self, limit: int = 60, window: float = 60.0) -> None:
        self.limit = max(1, limit)
        self.window = window
        self._hits: dict[str, Deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.time()
        q = self._hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)


rate_limiter = RateLimiter(limit=RATE_LIMIT_PER_MIN, window=60.0)


def client_ip(request: Request) -> str:
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def is_safe_webhook_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in {"localhost", "metadata.google.internal", "metadata"}:
        return False
    if host.endswith(".local") or host.endswith(".internal"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
        if str(ip) in {"169.254.169.254", "169.254.170.2"}:
            return False
    except ValueError:
        pass
    return True


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'",
        )
        if request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


def cors_origin_list() -> list[str]:
    if not CORS_ORIGINS or CORS_ORIGINS.strip() == "*":
        return ["*"]
    return [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
