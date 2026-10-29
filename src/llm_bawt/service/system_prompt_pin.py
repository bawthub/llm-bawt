"""Session-pinned agent system prompt (TASK-1318).

An agent harness keeps the system prompt it saw at the start of a session:
the Claude Code CLI records it on the session's first request and ignores a
changed prompt on resume until a compaction; the Codex bridge pastes it into
the thread once and the thread history carries it. A fresh render on every
turn would therefore log prompts the model never saw.

llm-bawt pins the prompt per session instead. ``sessions.session_metadata``
holds a pointer (:data:`PIN_KEY`) to the turn whose composed prompt is in
effect; that turn's ``turn_logs.request_json[PIN_KEY].text`` holds the bytes.
Every turn sends the pinned bytes and records the pin it used. The pin moves
only at a session boundary:

* cold SDK session — no resume id, a structured reset, or a non-explicit
  ``/new`` (``/new``, reset policies, model switch, first run). A bare reset
  (``reset_only``) never reaches a model, so it leaves a
  :data:`BOUNDARY_KEY` marker instead: the Claude bridge answers it by
  binding a seed transcript that carries no prompt snapshot, so the next
  turn resumes a session whose first snapshot is that turn's bytes. The
  marker counts only once the resume id differs from the one the reset
  turn saw, so a reset that never ran cannot pass as a boundary;
* the first turn after a successful *manual* ``/compact`` llm-bawt ran.
  Auto-compaction inside a turn is deliberately not a boundary: the CLI
  re-records that turn's bytes, which are already the pinned bytes;
* fallback — the pin turn row was deleted (delivery recovery, bot purge) or
  its text is missing or corrupt, or a resumed legacy session has no pin yet.
  These re-pin the current render with ``snapshot_unverified`` because the
  harness may still hold older bytes until its next boundary.

A move is two-phase. ``resolve`` stages it under :data:`PENDING_KEY` and
marks the turn's audit ``pending``; :func:`confirm_session_pin` promotes it
to :data:`PIN_KEY` once the bridge's first real output proves the harness
received the bytes. A boundary turn that is cancelled or fails before that
leaves the pointer where it was, so the next turn decides the boundary again
(and the Codex bridge pastes the prompt then) instead of resuming on a pin
the harness never saw.

Lock order matches ``record_turn_context``: turn row, then session row.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import text

from agent_bridge.maintenance import is_compact_command, is_new_command

from .turn_termination import dump_payload, turn_payload

logger = logging.getLogger(__name__)

#: Key on ``sessions.session_metadata`` (the pointer) and on every turn's
#: ``turn_logs.request_json`` (the per-turn audit; the pin turn adds ``text``).
PIN_KEY = "system_prompt_pin"
#: Key on ``sessions.session_metadata`` for a staged move awaiting confirmation.
PENDING_KEY = "system_prompt_pin_pending"
#: Key on ``sessions.session_metadata`` left by a bare reset turn: the next
#: turn that resumes a new SDK session is a cold start.
BOUNDARY_KEY = "system_prompt_pin_boundary"
PIN_VERSION = 1
#: Key on the pin turn's audit record holding the skill manifest pinned with
#: its bytes (TASK-1309), and the backend kwarg that carries it.
SKILL_MANIFEST_KEY = "skill_manifest"
SKILL_SELECTION_KWARG = "skill_selection"
#: Key on the pin turn's audit record naming the skills whose bodies its bytes
#: carry (TASK-1691), and the backend kwarg that carries them.
INJECTED_SKILLS_KEY = "injected_skills"
INJECTED_SKILLS_KWARG = "injected_skills"
#: Key on ``sessions.session_metadata`` holding ``{"refs", "tokens", "turn_id"}``
#: for the skill bodies a reset turn's own render carried (TASK-1750); the
#: app-authored /new receipt cites it. Absent means unknown, never zero.
RESET_SKILLS_KEY = "reset_injected_skills"

#: Pin-move reasons. Boundaries the harness itself observes are verified;
#: fallbacks can disagree with the harness until its next boundary.
COLD_START = "cold_start"
COMPACTION = "compaction"
MISSING_ROW = "missing_row"
MISSING_TEXT = "missing_text"
SHA_MISMATCH = "sha_mismatch"
LEGACY_ADOPT = "legacy_adopt"
UNVERIFIED_REASONS = frozenset({MISSING_ROW, MISSING_TEXT, SHA_MISMATCH, LEGACY_ADOPT})


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PinDecision:
    """The system prompt this turn sends, and why.

    ``is_boundary`` is True when the pin moved on this turn: these bytes take
    effect now. ``drift`` is True when a fresh render differs from the pinned
    bytes — the hook for "lands at the next boundary" notices (TASK-1281).
    ``pinned_sections`` is the content-free section metadata of the prompt
    the harness holds (see :meth:`SystemPromptPinStore.resolve`).
    ``candidate_sections`` is set only while the pin is unverified: the
    sections of the fallback render the harness adopts at its next
    (possibly automatic) compaction. Readers take the conservative view.
    """

    text: str
    pin_turn_id: str
    sha256: str
    action: str  # "pinned" | "kept"
    reason: str | None
    is_boundary: bool
    rendered_sha256: str
    snapshot_unverified: bool = False
    pinned_sections: dict[str, dict] | None = field(default=None, compare=False)
    candidate_sections: dict[str, dict] | None = field(default=None, compare=False)
    #: TASK-1309: the skill manifest pinned with these bytes (None = unmanaged).
    skill_manifest: dict | None = field(default=None, compare=False)
    #: TASK-1691: refs whose bodies these pinned bytes carry.
    injected_skills: tuple[str, ...] = field(default=(), compare=False)

    @property
    def drift(self) -> bool:
        return self.rendered_sha256 != self.sha256

    def audit(self) -> dict[str, Any]:
        """Per-turn ``request_json[PIN_KEY]`` record (without the pin text)."""
        record: dict[str, Any] = {
            "v": PIN_VERSION,
            "pin_turn_id": self.pin_turn_id,
            "sha256": self.sha256,
            "chars": len(self.text),
            "action": self.action,
        }
        if self.reason:
            record["reason"] = self.reason
        if self.snapshot_unverified:
            record["snapshot_unverified"] = True
        if self.drift:
            record["rendered_sha256"] = self.rendered_sha256
        return record


def pin_move_reason(
    *,
    cold_start: bool,
    pin: dict | None,
    pin_row_found: bool,
    pinned_text: str | None,
    compacted_since_pin: bool,
) -> str | None:
    """Why the pin must move on this turn, or None to keep it.

    A proven manual compaction beats every fallback: the CLI adopts this
    turn's bytes either way, so the move is verified even when the old pin
    text is gone or corrupt.
    """
    if cold_start:
        return COLD_START
    if pin is None:
        return LEGACY_ADOPT
    if compacted_since_pin:
        return COMPACTION
    if not pin_row_found:
        return MISSING_ROW
    if pinned_text is None:
        return MISSING_TEXT
    if prompt_sha256(pinned_text) != pin.get("sha256"):
        return SHA_MISMATCH
    return None


def is_cold_start(binding: dict | None, *, bridge_reset_action: str | None, user_prompt: str) -> bool:
    """Whether the bridge will open a fresh SDK session for this turn.

    Mirrors the bridges' own preflight: no resume id is a cold start, and a
    reset (structured, or a non-explicit ``/new``) drops the resume id.
    """
    binding = binding or {}
    if not str(binding.get("thread_resume_id") or "").strip():
        return True
    if bridge_reset_action:
        return True
    return not binding.get("explicit_thread") and is_new_command(user_prompt)


def _valid_pin(value: Any) -> dict | None:
    if (
        isinstance(value, dict)
        and isinstance(value.get("turn_id"), str) and value["turn_id"]
        and isinstance(value.get("sha256"), str) and value["sha256"]
    ):
        return value
    return None


def _sections_value(value: Any) -> dict[str, dict] | None:
    """A JSON-safe copy of section metadata, or None when absent or malformed."""
    if not isinstance(value, Mapping):
        return None
    return json.loads(json.dumps({str(key): dict(item) for key, item in value.items()
                                  if isinstance(item, Mapping)}))


def _manifest_value(value: Any) -> dict | None:
    """A JSON-safe copy of a skill manifest, or None when absent or malformed."""
    return json.loads(json.dumps(dict(value))) if isinstance(value, Mapping) else None


def _pinned_manifest(record: Mapping) -> dict | None:
    """The manifest pinned on a pin turn's record; a corrupt one fails the turn.

    Absent means the session was pinned unmanaged (or before skills were
    pinned) and stays so until its next boundary, never the current render's.
    """
    from agent_bridge.skill_selection import ManagedSelection

    value = record.get(SKILL_MANIFEST_KEY)
    if value is None:
        return None
    ManagedSelection.from_manifest(value)  # raises SkillSelectionError
    return _manifest_value(value)


def _refs_value(value: Any) -> tuple[str, ...]:
    """Valid skill refs from a stored list; anything malformed is dropped."""
    from agent_bridge.skill_sources import _REF

    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(r for r in value if isinstance(r, str) and _REF.fullmatch(r)))


def fresh_injected_skills(ctx) -> tuple[str, ...]:
    """Refs whose bodies this turn's render carries (empty if none were added)."""
    return _refs_value(list(getattr(getattr(ctx, "llm_bawt", None), "_injected_skill_refs", ())))


def held_injected_skills(decision: PinDecision | None, ctx) -> tuple[str, ...]:
    """Skills whose bodies the prompt the harness holds carries.

    A pinned turn names its pin's; an unverified pin names none, because the
    harness may still hold older bytes. An unpinned turn names its render's.
    """
    if decision is None:
        return fresh_injected_skills(ctx)
    return () if decision.snapshot_unverified else decision.injected_skills


def fresh_injected_skill_facts(ctx) -> dict | None:
    """``{"refs", "tokens"}`` for the skill bodies this turn's render carried.

    None when the render never resolved skills (the section failed closed or
    the backend renders none), so a receipt says nothing rather than guess.
    Tokens are the resolver's admitted counts for exactly those refs.
    """
    resolution = getattr(getattr(ctx, "llm_bawt", None), "_skill_resolution", None)
    if resolution is None:
        return None
    refs = fresh_injected_skills(ctx)
    tokens = {e.ref: e.tokens or 0 for e in resolution.injected}
    if any(ref not in tokens for ref in refs):
        return None
    return {"refs": list(refs), "tokens": sum(tokens[ref] for ref in refs)}


def fresh_skill_manifest(ctx) -> dict | None:
    """This turn's resolved skill manifest (rendered with its injected section)."""
    resolution = getattr(getattr(ctx, "llm_bawt", None), "_skill_resolution", None)
    return resolution.manifest() if resolution is not None else None


def pin_backend_kwargs(ctx, decision: PinDecision | None, *, agent: bool) -> dict[str, Any]:
    """Backend kwargs for this turn's pinned prompt and its skill selection.

    A pinned turn sends the selection pinned with its bytes, so skills move
    only at the session's boundaries; an unpinned turn sends this render's.
    """
    if not agent:
        return {}
    kwargs: dict[str, Any] = {}
    if decision is not None:
        kwargs[PIN_KEY] = decision
    manifest = decision.skill_manifest if decision is not None else fresh_skill_manifest(ctx)
    if manifest is not None:
        kwargs[SKILL_SELECTION_KWARG] = manifest
    injected = held_injected_skills(decision, ctx)
    if injected:
        kwargs[INJECTED_SKILLS_KWARG] = list(injected)
    return kwargs


def _held_sections(reason: str, sections: Mapping[str, dict] | None, pin: dict | None) -> dict | None:
    """Section metadata of the prompt the harness holds after this pin move."""
    if reason in (COLD_START, COMPACTION):
        return _sections_value(sections)
    if reason == LEGACY_ADOPT or pin is None:
        return None
    return _sections_value(pin.get("sections"))


def _candidate_sections(reason: str, sections: Mapping[str, dict] | None) -> dict | None:
    """Sections of an unverified re-pin's render; verified boundaries clear them."""
    return _sections_value(sections) if reason in UNVERIFIED_REASONS else None


def _manual_compaction_succeeded(user_prompt: str | None, token_usage_json: str | None) -> bool:
    if not is_compact_command(user_prompt):
        return False
    try:
        usage = json.loads(token_usage_json) if token_usage_json else None
    except (TypeError, ValueError):
        return False
    return (
        isinstance(usage, dict)
        and usage.get("context_action") == "compact"
        and usage.get("context_action_outcome") == "success"
    )


def _resume_id(binding: Mapping[str, Any] | None) -> str | None:
    return str((binding or {}).get("thread_resume_id") or "").strip() or None


def _reset_boundary_applies(marker: Any, resume_id: str | None) -> bool:
    """Whether a bare reset's marker proves this turn opens a new snapshot.

    Only a resume id the reset turn did not see proves the reset installed a
    new SDK session. With no resume id the turn is cold anyway; with the same
    one the reset never ran and the harness still holds its old snapshot.
    """
    if not isinstance(marker, Mapping) or not marker.get("turn_id"):
        return False
    return resume_id is not None and resume_id != (marker.get("resume_id") or None)


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else None


def _record_reset_skills(ctx, *, session_id: str, turn_id: str) -> None:
    """Record the reset turn's injected skills for the /new receipt (TASK-1750)."""
    facts = fresh_injected_skill_facts(ctx)
    if facts is None:
        return
    try:
        SystemPromptPinStore(ctx.svc._turn_log_store.engine).record_reset_skills(
            session_id=session_id, facts={**facts, "turn_id": turn_id},
        )
    except Exception:
        logger.warning("reset skill facts not recorded for session %s turn %s",
                       session_id, turn_id, exc_info=True)


def apply_session_pin(
    ctx, client, messages: list, *, turn_id: str, sections: Mapping[str, dict] | None = None,
) -> tuple[list, PinDecision | None]:
    """Swap the turn's system messages for the session's pinned prompt.

    Returns ``(messages, decision)``. ``decision`` is None when the backend
    does not pin, the turn never reaches a model (bare reset), there is no
    session to pin against, or the pin store failed; the turn then sends its
    fresh render exactly as before. The caller forwards a decision to the
    client as ``system_prompt_pin`` so the backend sends the bytes verbatim.
    ``sections`` is this render's section metadata, passed to ``resolve``.
    """
    from agent_bridge.skill_selection import SkillSelectionError

    from ..models.message import Message

    backend = getattr(client, "_backend", None)
    if not getattr(backend, "pins_system_prompt", False):
        return messages, None
    binding = ctx.thread_binding or {}
    session_id = str(binding.get("thread_session_id") or "").strip()
    resume_id = _resume_id(binding)
    if ctx.bridge_reset_action and session_id:
        _record_reset_skills(ctx, session_id=session_id, turn_id=turn_id)
    if ctx.bridge_reset_action == "reset_only":
        if session_id:
            try:
                SystemPromptPinStore(ctx.svc._turn_log_store.engine).mark_reset_boundary(
                    session_id=session_id, turn_id=turn_id, resume_id=resume_id,
                )
            except Exception:
                logger.warning(
                    "system prompt pin reset marker failed for session %s turn %s",
                    session_id, turn_id, exc_info=True,
                )
        return messages, None
    if not session_id:
        return messages, None
    from ..memory.postgresql import _sanitize_table_name

    base = "\n\n".join(
        m.content for m in messages
        if m.role == "system" and isinstance(m.content, str) and m.content
    )
    try:
        decision = SystemPromptPinStore(ctx.svc._turn_log_store.engine).resolve(
            session_id=session_id,
            turn_id=turn_id,
            storage_bot=_sanitize_table_name(ctx.bot_id),
            rendered=client.compose_system_prompt(base),
            cold_start=is_cold_start(
                binding, bridge_reset_action=ctx.bridge_reset_action,
                user_prompt=ctx.user_prompt,
            ),
            sections=sections,
            resume_id=resume_id,
            skill_manifest=fresh_skill_manifest(ctx),
            injected_skills=fresh_injected_skills(ctx),
        )
    except SkillSelectionError:
        raise  # a corrupt pinned skill selection is never swapped for the current one
    except Exception:
        logger.warning(
            "system prompt pin failed for session %s turn %s; sending the fresh render",
            session_id, turn_id, exc_info=True,
        )
        return messages, None
    if decision is None:
        return messages, None
    pinned = Message(role="system", content=decision.text)
    out: list = []
    placed = False
    for message in messages:
        if message.role != "system":
            out.append(message)
        elif not placed:
            out.append(pinned)
            placed = True
    if not placed:
        out.insert(0, pinned)
    return out, decision


#: Agent-stream queue events that only a model call produces, so the harness
#: has consumed this turn's prompt. Bridge bookkeeping is excluded:
#: ``turn_cancelled`` (a queued Claude cancel publishes ASSISTANT_DONE before
#: any SDK call), ``upstream_status``, ``metadata``, ``token_usage`` and
#: ``attachments`` (all can ride an aborted ASSISTANT_DONE).
MODEL_OUTPUT_EVENTS = frozenset({
    "reasoning", "live_usage", "tool_call", "tool_result", "tool_executing",
    "tool_preapproved", "approval_required", "await_tool_result", "file_changed",
    "subagent_started", "subagent_progress", "subagent_done",
})


def proves_prompt_consumed(item: Any) -> bool:
    """Whether a stream item is positive evidence the model read the prompt."""
    if isinstance(item, str):
        return bool(item)
    return isinstance(item, dict) and item.get("event") in MODEL_OUTPUT_EVENTS


class SessionPinConfirmer:
    """Promotes a turn's staged pin move on the first model output.

    ``observe`` every stream item. Kept turns have nothing to confirm. A
    turn that ends without model output (cancelled, failed before the SDK
    ran) leaves the move staged: the next turn decides the boundary again
    and re-sends its own render. A confirm failure is logged and likewise
    left staged.
    """

    def __init__(self, ctx, decision: PinDecision | None, *, turn_id: str) -> None:
        self._ctx = ctx
        self._turn_id = turn_id
        self._pending = decision is not None and decision.is_boundary

    def observe(self, item: Any) -> None:
        if not self._pending or not proves_prompt_consumed(item):
            return
        self._pending = False
        ctx, turn_id = self._ctx, self._turn_id
        session_id = str((ctx.thread_binding or {}).get("thread_session_id") or "").strip()
        try:
            SystemPromptPinStore(ctx.svc._turn_log_store.engine).confirm(
                session_id=session_id, turn_id=turn_id,
            )
        except Exception:
            logger.warning(
                "system prompt pin confirm failed for session %s turn %s", session_id, turn_id,
                exc_info=True,
            )


class SystemPromptPinStore:
    """Reads and moves a session's system-prompt pin."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def get(self, session_id: str) -> dict | None:
        """The session's pin pointer, or None."""
        if self.engine is None or not session_id:
            return None
        with self.engine.connect() as conn:
            metadata = conn.execute(
                text("SELECT session_metadata FROM sessions WHERE id=:sid"),
                {"sid": session_id},
            ).scalar_one_or_none()
        return _valid_pin((metadata or {}).get(PIN_KEY)) if isinstance(metadata, dict) else None

    def pinned_text(self, session_id: str) -> str | None:
        """The pinned prompt bytes, verified against the pointer's sha."""
        pin = self.get(session_id)
        if pin is None:
            return None
        with self.engine.connect() as conn:
            prior = conn.execute(
                text("SELECT request_json FROM turn_logs WHERE id=:id"),
                {"id": pin["turn_id"]},
            ).scalar_one_or_none()
        record = turn_payload(prior).get(PIN_KEY)
        value = record.get("text") if isinstance(record, dict) else None
        if isinstance(value, str) and prompt_sha256(value) == pin["sha256"]:
            return value
        return None

    def resolve(
        self,
        *,
        session_id: str,
        turn_id: str,
        storage_bot: str,
        rendered: str,
        cold_start: bool,
        sections: Mapping[str, dict] | None = None,
        resume_id: str | None = None,
        skill_manifest: Mapping | None = None,
        injected_skills: Sequence[str] = (),
    ) -> PinDecision | None:
        """Decide this turn's prompt, stage a pin move if needed, record the audit.

        One transaction. Returns None (send the render, record nothing) when
        the turn or session row is missing — there is nothing to pin against.
        Decisions read only the confirmed pointer; a move is staged under
        :data:`PENDING_KEY` until :meth:`confirm`, and any older staged move
        (a boundary turn that never ran) is dropped.

        ``sections`` is content-free metadata about this render's sections
        (TASK-1281: working-memory version/state). The pointer stores what
        the harness holds: this render's sections at a verified boundary,
        the previous pointer's at an unverified re-pin (the harness keeps
        its old snapshot until its next boundary), none on a legacy adopt.
        Unverified re-pins and legacy adopts also store this render's
        sections as ``candidate_sections``: the harness adopts them at its
        next compaction, which may be automatic and invisible here. Kept
        turns carry both until a verified boundary clears the candidate.

        ``resume_id`` is the SDK session this turn resumes. A
        :data:`BOUNDARY_KEY` marker makes the turn a cold start once the
        resume id differs from the reset turn's; it stays until
        :meth:`confirm`, so an unconfirmed boundary is decided again. A
        marker whose reset never installed a new session is dropped.

        ``skill_manifest`` (TASK-1309) is this render's resolved skills. It is
        pinned with the bytes on the pin turn's audit record, so a kept turn
        replays the pinned one; a pin from before skills were pinned replays
        None (unmanaged) until its next boundary.

        ``injected_skills`` (TASK-1691) names the skills whose bodies this
        render carries; it is pinned the same way, and an older pin replays
        none.
        """
        if self.engine is None or not session_id or not turn_id:
            return None
        resume_id = (resume_id or "").strip() or None
        rendered_sha = prompt_sha256(rendered)
        with self.engine.begin() as conn:
            turn = conn.execute(
                text("SELECT request_json, created_at FROM turn_logs WHERE id=:id FOR UPDATE"),
                {"id": turn_id},
            ).first()
            if turn is None:
                return None
            metadata = conn.execute(
                text("SELECT session_metadata FROM sessions WHERE id=:sid FOR UPDATE"),
                {"sid": session_id},
            ).first()
            if metadata is None:
                return None
            session_metadata = metadata[0] if isinstance(metadata[0], dict) else {}
            turn_request = turn_payload(turn[0])
            pin = _valid_pin(session_metadata.get(PIN_KEY))
            if BOUNDARY_KEY in session_metadata:
                if _reset_boundary_applies(session_metadata[BOUNDARY_KEY], resume_id):
                    cold_start = True
                elif resume_id is not None:  # same SDK session: the reset never ran
                    self._drop_keys(conn, session_id, BOUNDARY_KEY)

            pin_row_found = False
            pinned_text: str | None = None
            record: Any = None
            compacted = False
            if pin is not None and not cold_start:
                if pin["turn_id"] == turn_id:
                    pin_row_found, pin_request = True, turn_request
                else:
                    prior = conn.execute(
                        text("SELECT request_json FROM turn_logs WHERE id=:id"),
                        {"id": pin["turn_id"]},
                    ).first()
                    pin_row_found = prior is not None
                    pin_request = turn_payload(prior[0]) if prior is not None else {}
                record = pin_request.get(PIN_KEY)
                value = record.get("text") if isinstance(record, dict) else None
                pinned_text = value if isinstance(value, str) else None
                if pin.get("pinned_at"):  # needs only the pointer, not the old row or text
                    compacted = self._compacted_since(
                        conn, session_id=session_id, storage_bot=storage_bot,
                        pinned_at=pin["pinned_at"], turn_id=turn_id,
                    )

            reason = pin_move_reason(
                cold_start=cold_start, pin=pin, pin_row_found=pin_row_found,
                pinned_text=pinned_text, compacted_since_pin=compacted,
            )
            if reason is not None:
                held_sections = _held_sections(reason, sections, pin)
                candidate_sections = _candidate_sections(reason, sections)
                decision = PinDecision(
                    text=rendered, pin_turn_id=turn_id, sha256=rendered_sha,
                    action="pinned", reason=reason, is_boundary=True,
                    rendered_sha256=rendered_sha,
                    snapshot_unverified=reason in UNVERIFIED_REASONS,
                    pinned_sections=held_sections,
                    candidate_sections=candidate_sections,
                    skill_manifest=_manifest_value(skill_manifest),
                    injected_skills=_refs_value(list(injected_skills)),
                )
                new_pin = {
                    "v": PIN_VERSION,
                    "turn_id": turn_id,
                    "sha256": rendered_sha,
                    "chars": len(rendered),
                    "pinned_at": _iso(turn[1]),
                    "reason": reason,
                    "snapshot_unverified": decision.snapshot_unverified,
                }
                if held_sections is not None:
                    new_pin["sections"] = held_sections
                if candidate_sections is not None:
                    new_pin["candidate_sections"] = candidate_sections
                conn.execute(
                    text("""
                        UPDATE sessions SET session_metadata=jsonb_set(
                            COALESCE(session_metadata, '{}'::jsonb),
                            '{system_prompt_pin_pending}', CAST(:pin AS jsonb)
                        ) WHERE id=:sid
                    """),
                    {"pin": json.dumps(new_pin), "sid": session_id},
                )
                if reason in UNVERIFIED_REASONS and reason != LEGACY_ADOPT:
                    logger.warning(
                        "system prompt pin for session %s re-pinned to turn %s (%s); "
                        "previous pin turn %s", session_id, turn_id, reason,
                        pin["turn_id"] if pin else None,
                    )
            elif pin is not None and pinned_text is not None:
                decision = PinDecision(
                    text=pinned_text, pin_turn_id=pin["turn_id"], sha256=pin["sha256"],
                    action="kept", reason=None, is_boundary=False,
                    rendered_sha256=rendered_sha,
                    snapshot_unverified=bool(pin.get("snapshot_unverified")),
                    pinned_sections=_sections_value(pin.get("sections")),
                    candidate_sections=(
                        _sections_value(pin.get("candidate_sections"))
                        if pin.get("snapshot_unverified") else None
                    ),
                    skill_manifest=_pinned_manifest(record),
                    injected_skills=_refs_value(
                        record.get(INJECTED_SKILLS_KEY) if isinstance(record, dict) else None),
                )
            else:  # pin_move_reason keeps a pin only when both exist
                raise RuntimeError("system prompt pin kept without pinned text")
            if not decision.is_boundary and PENDING_KEY in session_metadata:
                self._drop_pending(conn, session_id)

            audit = decision.audit()
            if decision.is_boundary:
                audit["pending"] = True
            if decision.pin_turn_id == turn_id:
                audit["text"] = decision.text
                if decision.skill_manifest is not None:
                    audit[SKILL_MANIFEST_KEY] = decision.skill_manifest
                if decision.injected_skills:
                    audit[INJECTED_SKILLS_KEY] = list(decision.injected_skills)
            turn_request[PIN_KEY] = audit
            conn.execute(
                text("UPDATE turn_logs SET request_json=:rj WHERE id=:id"),
                {"rj": dump_payload(turn_request), "id": turn_id},
            )
        return decision

    def confirm(self, *, session_id: str, turn_id: str) -> bool:
        """Promote the move staged by ``turn_id`` to the session's pin.

        Returns False when nothing matching is staged: a later turn already
        replaced or dropped it, or this turn kept the pin.
        """
        if self.engine is None or not session_id or not turn_id:
            return False
        with self.engine.begin() as conn:
            turn = conn.execute(
                text("SELECT request_json FROM turn_logs WHERE id=:id FOR UPDATE"),
                {"id": turn_id},
            ).first()
            metadata = conn.execute(
                text("SELECT session_metadata FROM sessions WHERE id=:sid FOR UPDATE"),
                {"sid": session_id},
            ).scalar_one_or_none()
            staged = _valid_pin(metadata.get(PENDING_KEY)) if isinstance(metadata, dict) else None
            if staged is None or staged["turn_id"] != turn_id:
                return False
            # A confirmed move consumes any bare-reset marker: turns serialize
            # per session, so a staged move after the marker is its boundary.
            conn.execute(
                text("""
                    UPDATE sessions SET session_metadata=jsonb_set(
                        session_metadata - 'system_prompt_pin_pending' - 'system_prompt_pin_boundary',
                        '{system_prompt_pin}', CAST(:pin AS jsonb)
                    ) WHERE id=:sid
                """),
                {"pin": json.dumps(staged), "sid": session_id},
            )
            if turn is not None:
                turn_request = turn_payload(turn[0])
                audit = turn_request.get(PIN_KEY)
                if isinstance(audit, dict) and audit.pop("pending", None) is not None:
                    conn.execute(
                        text("UPDATE turn_logs SET request_json=:rj WHERE id=:id"),
                        {"rj": dump_payload(turn_request), "id": turn_id},
                    )
        return True

    def record_reset_skills(self, *, session_id: str, facts: Mapping[str, Any]) -> bool:
        """Store :data:`RESET_SKILLS_KEY` on the session; False if the row is missing."""
        if self.engine is None or not session_id:
            return False
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                    UPDATE sessions SET session_metadata=jsonb_set(
                        COALESCE(session_metadata, '{}'::jsonb),
                        '{reset_injected_skills}', CAST(:facts AS jsonb)
                    ) WHERE id=:sid
                """),
                {"facts": json.dumps(dict(facts)), "sid": session_id},
            )
            return bool(result.rowcount)

    def mark_reset_boundary(self, *, session_id: str, turn_id: str, resume_id: str | None) -> bool:
        """Record that a bare reset turn handed this session to a new SDK session.

        The turn never reaches a model, so nothing is pinned or audited.
        ``resume_id`` is the SDK session the reset turn saw; the next turn
        counts as a cold start only once its resume id differs from it.
        Returns False when the session row is missing.
        """
        if self.engine is None or not session_id or not turn_id:
            return False
        marker = {"v": PIN_VERSION, "turn_id": turn_id, "resume_id": (resume_id or "").strip() or None}
        with self.engine.begin() as conn:
            conn.execute(  # lock order: turn row, then session row
                text("SELECT 1 FROM turn_logs WHERE id=:id FOR UPDATE"), {"id": turn_id},
            )
            found = conn.execute(
                text("SELECT 1 FROM sessions WHERE id=:sid FOR UPDATE"), {"sid": session_id},
            ).first()
            if found is None:
                return False
            conn.execute(
                text("""
                    UPDATE sessions SET session_metadata=jsonb_set(
                        COALESCE(session_metadata, '{}'::jsonb),
                        '{system_prompt_pin_boundary}', CAST(:marker AS jsonb)
                    ) WHERE id=:sid
                """),
                {"marker": json.dumps(marker), "sid": session_id},
            )
        return True

    @staticmethod
    def _drop_pending(conn, session_id: str) -> None:
        SystemPromptPinStore._drop_keys(conn, session_id, PENDING_KEY)

    @staticmethod
    def _drop_keys(conn, session_id: str, *keys: str) -> None:
        for key in keys:
            conn.execute(
                text("UPDATE sessions SET session_metadata=session_metadata - CAST(:key AS text) WHERE id=:sid"),
                {"key": key, "sid": session_id},
            )

    @staticmethod
    def _compacted_since(conn, *, session_id: str, storage_bot: str, pinned_at: str, turn_id: str) -> bool:
        rows = conn.execute(
            text("""
                SELECT t.user_prompt, t.token_usage_json
                FROM turn_logs t
                JOIN messages m ON m.bot_id=:storage_bot AND m.id=t.trigger_message_id
                WHERE m.session_id=:sid
                  AND t.created_at > CAST(:pinned_at AS timestamptz)
                  AND t.id <> :turn_id
                  AND ltrim(t.user_prompt) LIKE '/compact%'
            """),
            {"storage_bot": storage_bot, "sid": session_id,
             "pinned_at": pinned_at, "turn_id": turn_id},
        ).all()
        return any(_manual_compaction_succeeded(row[0], row[1]) for row in rows)
