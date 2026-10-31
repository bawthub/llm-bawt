"""Opt-in, bounded memory embedding repair (TASK-1843).

The global DB setting ``memory_embedding_repair_mode`` defaults to report.
No app startup writes memory or creates a progress table until set to repair.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from ..memory.embedding_repair import EmbeddingRepair
from ..runtime_setting_resolution import resolve_global_runtime_setting

log = logging.getLogger(__name__)
_INTERVAL_SECONDS = 900
_MAX_BACKOFF_SECONDS = 3600


def run_pass(config, engine, stop: threading.Event) -> bool:
    """Run one off-loop report/repair; True means transient embed outage."""
    mode = resolve_global_runtime_setting(config, "memory_embedding_repair_mode")
    if mode not in ("report", "repair"):
        log.warning("embedding repair: unsupported mode %r; reporting only", mode)
        mode = "report"
    repair = EmbeddingRepair(config, engine)
    counts = repair.report()
    log.info("embedding repair (%s): eligible bots with missing vectors=%d; content=%d meaning=%d",
             mode, len(counts), sum(c["content"] for c in counts.values()),
             sum(c["meaning"] for c in counts.values()))
    if mode != "repair":
        return False
    for bot_id in sorted(counts):
        if stop.is_set():
            break
        result = repair.repair_bot(bot_id, stop=stop)
        if result.get("excluded") or result["busy"]:
            continue
        log.info("embedding repair bot=%s: examined=%d content=%d meaning=%d "
                 "permanent_failures=%d transient_failures=%d stopped=%s",
                 bot_id, result["examined"], result["repaired_content"],
                 result["repaired_meaning"], result["permanent_failures"],
                 result["transient_failures"], result["stopped"])
        if (result["stopped"] and result["transient_failures"] >= 3
                or result["dimension_mismatch"]):
            return True
    return False


async def run_embedding_repair_loop(config, engine, stop: threading.Event) -> None:
    """Only blocking worker code touches DB/HTTP; cancellation checks stop."""
    backoff = _INTERVAL_SECONDS
    while not stop.is_set():
        try:
            outage = await asyncio.to_thread(run_pass, config, engine, stop)
            backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS) if outage else _INTERVAL_SECONDS
        except asyncio.CancelledError:
            stop.set()
            raise
        except Exception:
            log.exception("embedding repair pass failed; retrying after backoff")
            backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
        try:
            # Shutdown cancels this task after setting stop; no default-executor
            # worker needs to stay parked for the entire backoff interval.
            await asyncio.sleep(backoff)
        except asyncio.CancelledError:
            stop.set()
            raise
