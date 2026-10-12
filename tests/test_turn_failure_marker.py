"""The failed-turn marker is one string shared by the live stream and history.

Regression: a Snark turn died on a proxy stall after streaming partial text.
History got the "⚠️ Turn failed" marker, but the live bubble only got a
``service.warning`` sidecar, which the UI ignores when partial text exists — so
the reply just stopped mid-thought until a reload.
"""
from __future__ import annotations

from llm_bawt.service.turn_stream_finalize import turn_failure_suffix

ERR = "claude-code error: upstream API: API Error: Proxy stream failed: ChatGPTEventTimeout"


def test_partial_reply_gets_separated_marker():
    suffix = turn_failure_suffix("Echo now uses Node 24.", ERR)
    assert suffix.startswith("\n\n⚠️ **Turn failed** — ")
    assert ERR in suffix


def test_empty_reply_is_marker_only():
    for partial in ("", "   \n", None):
        suffix = turn_failure_suffix(partial, ERR)
        assert suffix.startswith("⚠️ **Turn failed** — ")


def test_missing_error_text_has_fallback_note():
    assert "upstream backend error" in turn_failure_suffix("x", None)
    assert "upstream backend error" in turn_failure_suffix("x", "  ")


def test_live_stream_text_matches_persisted_text():
    """Live = streamed partial + suffix; persisted = partial.rstrip() + suffix."""
    for partial in ("Working on it.", "Working on it.\n\n", ""):
        live = partial + turn_failure_suffix(partial, ERR)
        persisted = partial.rstrip() + turn_failure_suffix(partial, ERR)
        live_prose, live_marker = live.split("⚠️", 1)
        saved_prose, saved_marker = persisted.split("⚠️", 1)
        assert live_marker == saved_marker
        assert live_prose.strip() == saved_prose.strip()
