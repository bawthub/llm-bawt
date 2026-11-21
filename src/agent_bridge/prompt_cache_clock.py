"""A clock domain shared only by processes on the same boot and time namespace."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path


def sdk_session_ref(session_id: str) -> str:
    return hashlib.sha256(f"prompt-cache-sdk-session:{session_id}".encode()).hexdigest()


def prompt_cache_clock() -> tuple[str, int] | None:
    """Linux suspend-aware elapsed milliseconds; no wall-clock fallback."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        namespace = Path("/proc/self/ns/time").readlink()
        if not boot:
            return None
        domain = hashlib.sha256(f"prompt-cache-clock:{boot}:{namespace}".encode()).hexdigest()
        return domain, time.clock_gettime_ns(time.CLOCK_BOOTTIME) // 1_000_000
    except (AttributeError, OSError, ValueError):
        return None
