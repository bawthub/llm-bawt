"""Per-run native compaction settings and the matching usage denominator."""
from __future__ import annotations


COMPACT_LIMIT_KEY = "model_auto_compact_token_limit"


def compact_config_overrides(value: object) -> tuple[str, ...]:
    """Accept a positive CLI i64 from command fields, never arbitrary TOML."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return ()
    try:
        limit = int(value)
    except ValueError:
        return ()
    if not 0 < limit <= 2**63 - 1:
        return ()
    return (f"{COMPACT_LIMIT_KEY}={limit}",)


def effective_context_window(window: int | None, overrides: tuple[str, ...]) -> int | None:
    """Cap a known provider window by the exact setting sent to this CLI run."""
    if window is None or window <= 0:
        return None
    prefix = f"{COMPACT_LIMIT_KEY}="
    for override in overrides:
        if override.startswith(prefix):
            setting = compact_config_overrides(override[len(prefix):])
            if setting:
                return min(window, int(setting[0][len(prefix):]))
    return window
