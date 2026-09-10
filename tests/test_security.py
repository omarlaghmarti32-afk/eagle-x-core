"""Security hardening tests."""

import os

os.environ["EAGLE_LIVE_MONITOR"] = "0"
os.environ["EAGLE_API_TOKEN"] = "test-token-secure-enough"
os.environ["EAGLE_REQUIRE_STRONG_TOKEN"] = "0"

from fastapi import HTTPException

from app.security import is_safe_webhook_url, require_token, token_is_weak


def test_token_weak_defaults():
    assert token_is_weak("eagle-dev-token-change-me")
    assert token_is_weak("short")
    assert not token_is_weak("a-sufficiently-long-secret-key")


def test_ssrf_guard_blocks_localhost():
    assert is_safe_webhook_url("http://127.0.0.1/hook") is False
    assert is_safe_webhook_url("http://localhost/hook") is False
    assert is_safe_webhook_url("http://169.254.169.254/latest") is False
    assert is_safe_webhook_url("http://192.168.1.1/hook") is False
    assert is_safe_webhook_url("https://hooks.slack.com/services/T/B/X") is True
    assert is_safe_webhook_url("https://discord.com/api/webhooks/1/2") is True


def test_require_token_rejects_bad():
    try:
        require_token("Bearer wrong-token-value-here")
        assert False
    except HTTPException as e:
        assert e.status_code in (401, 403)
