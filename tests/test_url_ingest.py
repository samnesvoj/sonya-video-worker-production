"""
Pure unit tests for scripts/url_ingest.py: platform detection and the
SSRF guard. No network access — DNS resolution for the "public" cases is
real (loopback/private cases resolve locally without a network call, and
the one public-host case is skipped if resolution is unavailable in the
sandbox).
"""
from __future__ import annotations

import socket

import pytest

from scripts import url_ingest


# ── detect_platform ─────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
    ("https://youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
    ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
    ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
    ("https://vk.com/video-12345_67890", "vk"),
    ("https://vkvideo.ru/video-12345_67890", "vk"),
    ("https://www.vk.com/video-12345_67890", "vk"),
    ("https://www.twitch.tv/videos/123456", "twitch"),
    ("https://clips.twitch.tv/SomeClipSlug", "twitch"),
    ("https://example.com/path/movie.mp4", "direct"),
    ("https://example.com/path/movie.mkv", "direct"),
    ("https://example.com/stream.m3u8", "direct"),
    ("https://example.com/page.html", "unsupported"),
    ("https://example.com/", "unsupported"),
    ("ftp://example.com/video.mp4", "unsupported"),
    ("javascript:alert(1)", "unsupported"),
    ("", "unsupported"),
    ("not a url at all", "unsupported"),
])
def test_detect_platform(url, expected):
    assert url_ingest.detect_platform(url) == expected


def test_detect_platform_rejects_lookalike_host():
    # "youtube.com" must be the actual host, not a subdomain-spoofing trick
    # like youtube.com.evil.example — a classic SSRF/phishing bypass shape.
    assert url_ingest.detect_platform("https://youtube.com.evil.example/watch?v=x") == "unsupported"
    assert url_ingest.detect_platform("https://notvk.com/video-1_1") == "unsupported"


# ── assert_safe_url (SSRF guard) ────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/video.mp4",
    "http://127.0.0.1:8080/video.mp4",
    "http://localhost/video.mp4",
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
    "http://10.0.0.5/video.mp4",
    "http://172.16.0.5/video.mp4",
    "http://192.168.1.1/video.mp4",
    "http://[::1]/video.mp4",
    "file:///etc/passwd",
    "gopher://127.0.0.1/video.mp4",
])
def test_assert_safe_url_blocks_private_targets(url):
    with pytest.raises(url_ingest.UnsafeUrlError):
        url_ingest.assert_safe_url(url)


def test_assert_safe_url_allows_public_host():
    try:
        socket.getaddrinfo("www.youtube.com", None)
    except socket.gaierror:
        pytest.skip("no DNS resolution available in this sandbox")
    url_ingest.assert_safe_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_assert_safe_url_rejects_unresolvable_host():
    with pytest.raises(url_ingest.UnsafeUrlError):
        url_ingest.assert_safe_url("https://this-host-should-not-exist.invalid/video.mp4")


# ── size / duration limits (probe) honored via env, checked in isolation ──

def test_max_bytes_reads_env(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "10")
    assert url_ingest._max_bytes() == 10 * 1024 * 1024


def test_max_duration_reads_env(monkeypatch):
    monkeypatch.setenv("URL_DOWNLOAD_MAX_DURATION_SEC", "120")
    assert url_ingest._max_duration_sec() == 120


# ── D: mode-conditional limits (PHASE A) ────────────────────────────────
# streamer gets its own long-form ceiling; every other mode (None included,
# the default for any pre-PHASE-A caller) keeps the original env vars.

def test_streamer_duration_limit_uses_dedicated_env(monkeypatch):
    monkeypatch.setenv("STREAMER_URL_MAX_DURATION_SEC", "25200")  # 7h
    assert url_ingest._max_duration_sec(mode="streamer") == 25200


def test_streamer_duration_limit_default_is_7_hours(monkeypatch):
    monkeypatch.delenv("STREAMER_URL_MAX_DURATION_SEC", raising=False)
    assert url_ingest._max_duration_sec(mode="streamer") == 7 * 60 * 60


def test_streamer_upload_size_limit_uses_dedicated_env(monkeypatch):
    monkeypatch.setenv("STREAMER_MAX_UPLOAD_SIZE_MB", "30000")
    assert url_ingest._max_bytes(mode="streamer") == 30000 * 1024 * 1024


def test_other_modes_keep_original_duration_limit_even_with_streamer_env_set(monkeypatch):
    """The streamer-only env var must never leak into any other mode's
    limit, even when both are set at once."""
    monkeypatch.setenv("URL_DOWNLOAD_MAX_DURATION_SEC", "3600")
    monkeypatch.setenv("STREAMER_URL_MAX_DURATION_SEC", "25200")

    assert url_ingest._max_duration_sec(mode="virality") == 3600
    assert url_ingest._max_duration_sec(mode=None) == 3600
    assert url_ingest._max_duration_sec(mode="streamer") == 25200


def test_other_modes_keep_original_upload_limit_even_with_streamer_env_set(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "2048")
    monkeypatch.setenv("STREAMER_MAX_UPLOAD_SIZE_MB", "20480")

    assert url_ingest._max_bytes(mode="educational") == 2048 * 1024 * 1024
    assert url_ingest._max_bytes(mode=None) == 2048 * 1024 * 1024
    assert url_ingest._max_bytes(mode="streamer") == 20480 * 1024 * 1024


# ── safe filename helper ────────────────────────────────────────────────

def test_safe_filename_strips_forbidden_chars():
    assert url_ingest._safe_filename('a<b>c:d"e/f\\g|h?i*j.mp4') == "a_b_c_d_e_f_g_h_i_j.mp4"


def test_safe_filename_empty_falls_back():
    assert url_ingest._safe_filename("") == "video.mp4"
