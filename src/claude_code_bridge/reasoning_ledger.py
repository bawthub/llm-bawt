"""Resident reasoning for a native Claude session (TASK-2343).

Claude keeps every earlier turn's thinking in context: the CLI resends the
signed thinking blocks and the API counts them in full as input. The blocks'
text is redacted in the transcript, so neither the transcript nor the CLI's
``/context`` mix can size them. The API reports each response's
``output_tokens_details.thinking_tokens``; this module keeps those counts in a
sidecar next to the session transcript, keyed by Anthropic message id, and sums
the ones still resident (after the last compaction boundary).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from ._bridge_helpers import _session_transcript_path

logger = logging.getLogger("claude_code_bridge.bridge")

SIDECAR_SUFFIX = ".reasoning.json"
# A compacted session drops its old ids from the transcript tail; this cap only
# bounds a pathological never-compacted session's sidecar file.
MAX_ENTRIES = 20_000


def _sidecar(transcript: Path) -> Path:
    return transcript.with_name(transcript.stem + SIDECAR_SUFFIX)


def _load(path: Path) -> dict[str, int]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        k: int(v) for k, v in raw.items()
        if isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 0
    }


def record_thinking(session_id: str | None, counts: dict[str, int]) -> None:
    """Merge this turn's ``{message_id: thinking_tokens}`` into the sidecar."""
    if not session_id or not counts:
        return
    try:
        transcript = _session_transcript_path(session_id)
        if transcript is None:
            return
        path = _sidecar(transcript)
        ledger = _load(path)
        ledger.update(counts)
        if len(ledger) > MAX_ENTRIES:
            ledger = dict(list(ledger.items())[-MAX_ENTRIES:])
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:
        logger.debug("reasoning ledger write failed for %s", session_id, exc_info=True)


def _thinking_message_ids(transcript: Path) -> list[str]:
    """Main-thread assistant message ids with a thinking block, after the last compaction."""
    ids: dict[str, None] = {}
    with open(transcript, "r", encoding="utf-8") as fh:
        for line in fh:
            if "compact_boundary" in line:
                try:
                    if json.loads(line).get("subtype") == "compact_boundary":
                        ids.clear()
                        continue
                except (ValueError, TypeError, AttributeError):
                    pass
            if '"thinking' not in line and '"redacted_thinking' not in line:
                continue
            try:
                entry = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(entry, dict) or entry.get("type") != "assistant" or entry.get("isSidechain"):
                continue
            message = entry.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("id"), str):
                continue
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") for b in content
            ):
                ids[message["id"]] = None
    return list(ids)


def resident_reasoning(session_id: str | None) -> tuple[int, bool] | None:
    """``(tokens, complete)`` of thinking still in context, or None with none resident.

    ``complete`` is False when some resident thinking predates the sidecar
    (earlier bridge versions), so ``tokens`` is a lower bound.
    """
    try:
        transcript = _session_transcript_path(session_id)
        if transcript is None:
            return None
        ids = _thinking_message_ids(transcript)
        if not ids:
            return None
        ledger = _load(_sidecar(transcript))
        known = [ledger[i] for i in ids if i in ledger]
        return sum(known), len(known) == len(ids)
    except Exception:
        logger.debug("resident reasoning read failed for %s", session_id, exc_info=True)
        return None
