"""
Root conftest — sets the minimal env vars required for
scripts.prod_generation_api and friends to import cleanly, before pytest
collects any test module (tests/test_worker_endpoints.py reads
WORKER_SECRET at module level; prod_generation_api raises at import time
if CORS_ORIGINS resolves to a wildcard).

setdefault() so a real environment (CI secrets, a developer's .env) is
never clobbered — these are fallbacks for a bare `pytest` run only.
"""
import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("CORS_ORIGINS", "https://testserver")
os.environ.setdefault("AUTH_SECRET", "test-auth-secret-do-not-use-in-production")
os.environ.setdefault("WORKER_SECRET", "test-worker-secret-do-not-use-in-production")
os.environ.setdefault("TELEGRAM_WEBHOOK_SECRET", "test-telegram-webhook-secret-do-not-use-in-production")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: paid test against a real external API; skipped unless explicitly enabled")
