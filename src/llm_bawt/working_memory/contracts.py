"""Working-memory item contract: caps, validation and the create fingerprint.

Design: docs/WORKING_MEMORY.md sections 5, 6 and 12 (TASK-1281). Everything a
caller supplies is validated and normalized here, before any lock is taken, so
the store only ever sees canonical values.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from collections.abc import Mapping
from typing import Any

OPEN_STATUSES = ("planned", "active", "blocked")
ARCHIVED_STATUSES = ("done", "cancelled")
STATUSES = OPEN_STATUSES + ARCHIVED_STATUSES
KINDS = ("personal", "initiative_link")

MAX_OPEN_ITEMS = 20
MAX_OBJECTIVE = 160
MAX_NEXT_ACTION = 400
MAX_BLOCKERS = 5
MAX_BLOCKER = 200
MAX_CHECKPOINT = 600
MAX_REFS = 20
MAX_REFS_BYTES = 4096
MAX_REF_EXCERPT = 200
REF_KEYS = ("kind", "id", "label", "excerpt")

# Same shape BawtHub accepts for a task reference (taskConversationValidation.ts).
_TASK_REF = re.compile(r"^(?:TASK-\d+|[A-Za-z0-9][A-Za-z0-9_-]{0,63})$", re.IGNORECASE)
_SHORT_TASK_REF = re.compile(r"^TASK-\d+$", re.IGNORECASE)
_CANONICAL_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Fields a caller may change after create. ``kind`` and ``initiative_task_id``
# are fixed at create.
UPDATABLE_FIELDS = ("objective", "status", "next_action", "blockers", "checkpoint", "priority", "refs")
NULLABLE_FIELDS = ("next_action", "checkpoint", "priority")
LINK_FORBIDDEN_FIELDS = ("next_action", "blockers")


class WorkingMemoryError(Exception):
    """A refused request. ``code`` is stable and model-visible."""

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, **self.detail}


def _invalid(message: str) -> WorkingMemoryError:
    return WorkingMemoryError("invalid_request", message)


def _text(value: Any, field: str, limit: int, *, required: bool = False) -> str | None:
    """NFC-normalize and strip; blank means absent. Length is in code points."""
    if value is None:
        if required:
            raise _invalid(f"{field} is required")
        return None
    if not isinstance(value, str):
        raise _invalid(f"{field} must be a string")
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized:
        if required:
            raise _invalid(f"{field} is required")
        return None
    if len(normalized) > limit:
        raise WorkingMemoryError("cap_exceeded", f"{field} exceeds {limit} characters", cap=field, limit=limit)
    return normalized


def normalize_identity(bot_id: Any, user_id: Any) -> tuple[str, str]:
    bot = str(bot_id or "").strip().lower()
    user = str(user_id or "").strip()
    if not bot or not user or len(bot) > 128 or len(user) > 128:
        raise _invalid("bot and user are required (at most 128 characters)")
    return bot, user


def normalize_idempotency_key(value: Any) -> str:
    """Accept only a canonical 8-4-4-4-12 hex UUID; store it lower-case (5.2)."""
    raw = value.strip() if isinstance(value, str) else ""
    lowered = raw.lower()
    if not _CANONICAL_UUID.fullmatch(lowered):
        raise WorkingMemoryError(
            "invalid_idempotency_key",
            "idempotency_key must be a UUID (8-4-4-4-12 hex); generate a fresh one per create",
        )
    return str(uuid.UUID(lowered))


def normalize_item_id(value: Any) -> str:
    try:
        return str(uuid.UUID(str(value).strip()))
    except (ValueError, AttributeError, TypeError):
        raise WorkingMemoryError("not_found", "no such working-memory item") from None


def normalize_status(value: Any, *, allowed: tuple[str, ...] = STATUSES) -> str:
    status = str(value or "").strip().lower()
    if status not in allowed:
        raise _invalid(f"status must be one of {', '.join(allowed)}")
    return status


def normalize_priority(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid("priority must be an integer")
    if not -1000 <= value <= 1000:
        raise _invalid("priority must be between -1000 and 1000")
    return value


def normalize_blockers(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(b, str) for b in value):
        raise _invalid("blockers must be a list of strings")
    blockers = [b for b in (_text(b, "blocker", MAX_BLOCKER) for b in value) if b]
    if len(blockers) > MAX_BLOCKERS:
        raise WorkingMemoryError("cap_exceeded", f"at most {MAX_BLOCKERS} blockers", cap="blockers", limit=MAX_BLOCKERS)
    return blockers


def normalize_refs(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise _invalid("refs must be a list")
    if len(value) > MAX_REFS:
        raise WorkingMemoryError("cap_exceeded", f"at most {MAX_REFS} refs", cap="refs", limit=MAX_REFS)
    refs: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) - set(REF_KEYS):
            raise _invalid(f"each ref is an object with only {', '.join(REF_KEYS)}")
        ref = {
            "kind": _text(raw.get("kind"), "ref kind", MAX_OBJECTIVE, required=True),
            "id": _text(raw.get("id"), "ref id", MAX_OBJECTIVE, required=True),
        }
        label = _text(raw.get("label"), "ref label", MAX_OBJECTIVE)
        excerpt = _text(raw.get("excerpt"), "ref excerpt", MAX_REF_EXCERPT)
        if label:
            ref["label"] = label
        if excerpt:
            ref["excerpt"] = excerpt
        refs.append(ref)
    if len(canonical_json(refs).encode("utf-8")) > MAX_REFS_BYTES:
        raise WorkingMemoryError("cap_exceeded", f"refs exceed {MAX_REFS_BYTES} bytes", cap="refs", limit=MAX_REFS_BYTES)
    return refs


def normalize_task_ref(value: Any) -> str:
    """Format check only; the task's existence is not looked up (slice 1)."""
    ref = str(value or "").strip()
    if not _TASK_REF.fullmatch(ref):
        raise _invalid("initiative_task_id must be a task key (TASK-N or a task id)")
    return ref.upper() if _SHORT_TASK_REF.fullmatch(ref) else ref


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_create(fields: Mapping[str, Any]) -> dict[str, Any]:
    """The validated create payload, defaults filled (6.1)."""
    kind = str(fields.get("kind") or "personal").strip().lower()
    if kind not in KINDS:
        raise _invalid(f"kind must be one of {', '.join(KINDS)}")
    payload = {
        "kind": kind,
        "objective": _text(fields.get("objective"), "objective", MAX_OBJECTIVE, required=True),
        "next_action": _text(fields.get("next_action"), "next_action", MAX_NEXT_ACTION),
        "status": normalize_status(fields.get("status") or "planned", allowed=OPEN_STATUSES),
        "priority": normalize_priority(fields.get("priority")),
        "refs": normalize_refs(fields.get("refs")),
        "initiative_task_id": None,
    }
    if kind == "initiative_link":
        if fields.get("initiative_task_id") is None:
            raise _invalid("initiative_link items require initiative_task_id")
        if payload["next_action"] is not None:
            raise _invalid("initiative_link items cannot set next_action; use initiative_state")
        payload["initiative_task_id"] = normalize_task_ref(fields["initiative_task_id"])
    elif fields.get("initiative_task_id") is not None:
        raise _invalid("initiative_task_id is only for kind=initiative_link")
    return payload


def create_fingerprint(payload: Mapping[str, Any]) -> str:
    """sha256 of the canonical create payload; compared on replay, never stored on the item."""
    return hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def normalize_changes(kind: str, changes: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an update. A present key with None clears a nullable field."""
    unknown = set(changes) - set(UPDATABLE_FIELDS)
    if unknown:
        raise _invalid(f"cannot update {', '.join(sorted(unknown))}")
    if kind == "initiative_link":
        forbidden = [f for f in LINK_FORBIDDEN_FIELDS if f in changes]
        if forbidden:
            raise _invalid(f"initiative_link items cannot set {', '.join(forbidden)}; use initiative_state")
    out: dict[str, Any] = {}
    for field, value in changes.items():
        if field == "objective":
            out[field] = _text(value, "objective", MAX_OBJECTIVE, required=True)
        elif field == "status":
            out[field] = normalize_status(value)
        elif field == "next_action":
            out[field] = _text(value, "next_action", MAX_NEXT_ACTION)
        elif field == "checkpoint":
            out[field] = _text(value, "checkpoint", MAX_CHECKPOINT)
        elif field == "priority":
            out[field] = normalize_priority(value)
        elif field == "blockers":
            out[field] = normalize_blockers(value)
        elif field == "refs":
            out[field] = normalize_refs(value)
    return out


__all__ = [
    "ARCHIVED_STATUSES", "KINDS", "MAX_OPEN_ITEMS", "NULLABLE_FIELDS", "OPEN_STATUSES", "STATUSES",
    "UPDATABLE_FIELDS", "WorkingMemoryError", "canonical_json", "create_fingerprint", "normalize_changes",
    "normalize_create", "normalize_idempotency_key", "normalize_identity", "normalize_item_id",
    "normalize_priority", "normalize_status",
]
