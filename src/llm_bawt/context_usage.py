"""Distinguish reset-time context estimates from provider measurements."""

from __future__ import annotations

from typing import Any, Mapping


def is_estimated_context(usage: Mapping[str, Any]) -> bool:
    """Reset acknowledgements have no provider request behind their counts.

    Their partial estimates may omit system prompt sections and tool schemas.
    Recognize provenance too, so older persisted rows without ``usage_status``
    cannot become measured health or size-based reset evidence.
    """
    source = usage.get("resident_source")
    return usage.get("usage_status") in ("estimated", "reset") or (
        isinstance(source, str)
        and (source.endswith("_estimate") or source == "clean_session_reset")
    )
