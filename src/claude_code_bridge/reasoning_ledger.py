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


def _thinking_id(entry: dict) -> str | None:
    """Anthropic message id of a main-thread assistant row carrying thinking."""
    if entry.get("type") != "assistant" or entry.get("isSidechain"):
        return None
    message = entry.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("id"), str):
        return None
    content = message.get("content")
    if isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") for b in content
    ):
        return message["id"]
    return None


def _preserved_uuids(metadata: object, parents: dict[str, str | None]) -> set[str] | None:
    """Transcript row uuids a compaction kept in context; None when unknowable.

    The CLI keeps ``compactMetadata.preservedMessages.uuids`` (or the legacy
    ``preservedSegment`` head..tail, walked back through ``parentUuid``) and
    drops every other row before the boundary.
    """
    if not isinstance(metadata, dict):
        return set()
    preserved = metadata.get("preservedMessages")
    if preserved is not None:
        uuids = preserved.get("uuids") if isinstance(preserved, dict) else None
        if not isinstance(uuids, list) or not all(isinstance(u, str) and u in parents for u in uuids):
            return None  # the CLI aborts its resume on a missing preserved uuid
        return set(uuids)
    segment = metadata.get("preservedSegment")
    if segment is None:
        return set()
    if not isinstance(segment, dict):
        return None
    head, tail = segment.get("headUuid"), segment.get("tailUuid")
    if not isinstance(head, str) or not isinstance(tail, str):
        return None
    kept: set[str] = set()
    current: str | None = tail
    while current is not None and current not in kept:
        if current not in parents:
            return None
        kept.add(current)
        if current == head:
            return kept
        current = parents[current]
    return None  # head not reached from tail


def _thinking_message_ids(transcript: Path) -> tuple[list[str], bool]:
    """Resident main-thread message ids with thinking, and whether that set is certain.

    Rows before the last compaction count only if that compaction preserved
    them. An unreadable preservation record makes the set uncertain.
    """
    parents: dict[str, str | None] = {}
    row_message: dict[str, str] = {}  # row uuid -> assistant message id
    resident: dict[str, None] = {}  # thinking message ids in context
    dropped: dict[str, str] = {}  # row uuid -> thinking message id cut by the last boundary
    certain = True
    with open(transcript, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(entry, dict):
                continue
            uuid = entry.get("uuid") if isinstance(entry.get("uuid"), str) else None
            parent = entry.get("parentUuid") if isinstance(entry.get("parentUuid"), str) else None
            if uuid is not None:
                parents[uuid] = parent
            if entry.get("subtype") == "compact_boundary":
                kept = _preserved_uuids(entry.get("compactMetadata"), parents)
                if kept is None:
                    certain = False
                    kept = set()
                # A kept row keeps its whole message (the CLI writes one row per block).
                kept_messages = {row_message[u] for u in kept if u in row_message}
                dropped = {
                    u: m for u, m in row_message.items()
                    if m in resident and m not in kept_messages
                }
                resident = {m: None for m in resident if m in kept_messages}
                continue
            attachment = entry.get("attachment") if entry.get("type") == "attachment" else None
            if isinstance(attachment, dict) and attachment.get("type") == "thinking_stripped":
                # Earlier thinking is replaced on the wire: all of it, or from one
                # message (and block) onward.
                start = attachment.get("from") if attachment.get("scope") == "partial" else None
                order = list(resident)
                if isinstance(start, dict) and start.get("messageId") in resident:
                    cut = order.index(start["messageId"])
                    if start.get("thinkingIndex") not in (0, None):
                        certain = False  # that message keeps part of its thinking
                    resident = {m: None for m in order[:cut]}
                else:
                    if attachment.get("scope") != "all":
                        certain = False
                    resident = {}
                dropped = {}
                continue
            # The CLI also keeps a pre-boundary row that a later row hangs off.
            if parent is not None and parent in dropped:
                revived = dropped.pop(parent)
                resident[revived] = None
                dropped = {u: m for u, m in dropped.items() if m != revived}
            msg_id = None
            if entry.get("type") == "assistant" and not entry.get("isSidechain"):
                message = entry.get("message")
                if isinstance(message, dict) and isinstance(message.get("id"), str):
                    msg_id = message["id"]
                    if uuid is not None:
                        row_message[uuid] = msg_id
            if msg_id is not None and _thinking_id(entry) is not None:
                resident[msg_id] = None
    return list(resident), certain


def resident_reasoning(session_id: str | None) -> tuple[int, bool] | None:
    """``(tokens, complete)`` of thinking still in context, or None with none resident.

    ``complete`` is False when some resident thinking predates the sidecar
    (earlier bridge versions), so ``tokens`` is a lower bound.
    """
    try:
        transcript = _session_transcript_path(session_id)
        if transcript is None:
            return None
        ids, certain = _thinking_message_ids(transcript)
        if not ids:
            return None
        ledger = _load(_sidecar(transcript))
        known = [ledger[i] for i in ids if i in ledger]
        return sum(known), certain and len(known) == len(ids)
    except Exception:
        logger.debug("resident reasoning read failed for %s", session_id, exc_info=True)
        return None
