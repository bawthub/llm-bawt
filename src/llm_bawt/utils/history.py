import os
from urllib.parse import quote
import json
import time
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Any

from ..clients.base import LLMClient
from ..models.message import Message
from rich.rule import Rule
from .config import Config
from ..setting_definitions import HISTORY_SCOPES, setting_default

if TYPE_CHECKING:
    from ..memory.postgresql import PostgreSQLShortTermManager as ShortTermMemoryManager

logger = logging.getLogger(__name__)

# A role='system' row that carries tool-execution evidence rather than the
# persona/system prompt. Detected by content prefix so the two are separable in
# the context payload (chat keeps these; the plain system row is discarded by
# both delivery modes). Keep in sync with core/base.py's inline check.
_TOOL_RESULT_PREFIXES = ("[Tool Results]", "[Tools used:")


def _file_safe(value: str) -> str:
    """Filename-safe, injective form of a user id (TASK-1774 per-user files).

    Percent-encoding with no safe characters: distinct ids never share a
    file ('a@b' -> 'a%40b', 'a_b' -> 'a_b') and '/' can't leave the dir.
    """
    return quote(value, safe="")


def _is_tool_result_system(msg) -> bool:
    if getattr(msg, "role", "") != "system":
        return False
    content = getattr(msg, "content", "") or ""
    return content.startswith(_TOOL_RESULT_PREFIXES)


@dataclass
class ContextPayload:
    """The ONE context-assembly result (TASK-493).

    Single source of truth for *what* prior context a conversation gets — for
    BOTH chat turns and agent session-seeds. The two orthogonal user controls
    resolve here identically for every bot_type:

    - ``continuity`` (bool): carry prior context at all.
    - ``history_scope`` ("inline+summaries" | "inline"): whether summaries are
      part of it.

    Delivery (chat inline vs agent seed) is NOT decided here — it is expressed by
    which composed view the caller reads (``inline_history`` vs ``seed_messages``).
    The buckets are kept separate so each delivery composes its own shape without
    re-deriving the summary gate. ``summary_messages`` is already gated: it is
    empty when summaries are excluded.
    """

    system_messages: list = field(default_factory=list)       # plain persona/system rows
    summary_messages: list = field(default_factory=list)       # already gated (empty if excluded)
    tool_result_messages: list = field(default_factory=list)   # role=system tool-evidence rows
    regular_messages: list = field(default_factory=list)       # user / assistant

    @property
    def inline_history(self) -> list:
        """Rows a CHAT turn appends after its own system-prompt builder.

        Order preserves what the pre-refactor loop produced: tool-result system
        rows (hoisted to the front by ``get_context_messages``), then summaries,
        then the regular conversation. The plain persona system row is omitted —
        the chat path builds its own.
        """
        return self.tool_result_messages + self.summary_messages + self.regular_messages

    @property
    def seed_messages(self) -> list:
        """Rows an AGENT cold-start seed carries: summaries + conversation, no
        system rows at all (the SDK injects its own byte-stable system prompt —
        TASK-288). Matches the pre-refactor ``role != 'system'`` strip.
        """
        return self.summary_messages + self.regular_messages


def scope_flags(scope: str | None) -> tuple[bool, bool]:
    """Decode a ``history_scope`` enum into two independent booleans (TASK-518).

    Returns ``(include_history, include_summaries)``. The scope is a set of bits
    expressed as a string; membership is substring-tested so the four canonical
    values map cleanly:

        "inline+summaries" -> (True,  True)   recent messages + rolling summaries
        "inline"           -> (True,  False)  recent messages only
        "summaries"        -> (False, True)   dense summary-only (no raw messages)
        "none"             -> (False, False)  carry nothing (cold turn)

    An empty/unknown scope falls back to the registry default (both on).
    """
    s = scope or "inline+summaries"
    return ("inline" in s, "summaries" in s)


def effective_history_scope(config_resolver, override: str | None = None) -> str:
    """The scope a /new summary and seed use: an explicit override, else the bot's.

    ``override`` is a server-owned per-operation choice (a scheduled reset's
    continuity parameter); it never writes the bot's setting. Resolver errors
    propagate so each caller keeps its established fail-closed behavior.
    """
    if override is not None:
        if override not in HISTORY_SCOPES:
            raise ValueError(f"unsupported history scope override: {override!r}")
        return override
    return str(config_resolver.resolve_config_setting("history_scope").value or "inline+summaries")


def estimate_tokens(text: str) -> int:
    """Rough token estimate for a text string.

    Uses ~4 chars per token as a fast heuristic, within 10-15%
    of tiktoken for English text. No external dependencies.
    """
    return len(text) // 4


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens for a list of Message objects.

    Adds a small per-message overhead for role/framing.
    """
    return sum(len(m.content) // 4 + 4 for m in messages)


class ContextBudgetError(RuntimeError):
    """The system prompt alone exceeds the physical prompt budget.

    TASK-612: the ONE hard-fail in the allocation ladder. Every other budget
    shortfall degrades and logs (shed summaries, then trim raw, but never below
    the newest complete turn). If the system prompt cannot fit the model's
    prompt budget there is nothing sane to send, so we fail loudly rather than
    ship a silently-truncated system prompt.
    """


class HistoryManager:
    messages: list[Message] = []
    client: LLMClient
    config: Config
    _db_backend: "ShortTermMemoryManager | None"
    bot_id: str

    def __init__(
        self,
        client: LLMClient,
        config: Config,
        db_backend: "ShortTermMemoryManager | None" = None,
        bot_id: str = "nova",
        settings_getter: Callable[[str, Any], Any] | None = None,
        user_id: str | None = None,
    ):
        self.client = client
        self.config = config
        self.bot_id = bot_id
        # TASK-1774: a bot's message partition holds every user's history;
        # DB reads are limited to this user's sessions (see load_history).
        self.user_id = user_id
        self.messages = []
        self._db_backend = db_backend
        self._settings_getter = settings_getter
        
        # Use per-bot history files for local mode
        if config.HISTORY_FILE:
            # Get the base path and insert bot_id
            base_path = config.HISTORY_FILE
            dir_name = os.path.dirname(base_path)
            file_name = os.path.basename(base_path)
            name, ext = os.path.splitext(file_name)
            # Create bot-specific filename: history.json -> history_spark.json.
            # TASK-1774: with a user, one file per (bot, user):
            # history_spark__alice.json, so users never share a file.
            owner = f"{bot_id}__{_file_safe(user_id)}" if user_id else bot_id
            self.history_file = os.path.join(dir_name, f"{name}_{owner}{ext}")
        else:
            self.history_file = None
            
        if db_backend:
            logger.debug(f"HistoryManager using PostgreSQL short-term backend for bot: {bot_id}")
        else:
            logger.debug(f"HistoryManager using text file backend for bot: {bot_id} ({self.history_file})")

    def _setting(self, key: str, fallback: Any) -> Any:
        """Resolve setting through optional runtime resolver."""
        if self._settings_getter is None:
            return fallback
        try:
            return self._settings_getter(key, fallback)
        except Exception:
            return fallback

    def load_history(
        self, since_minutes: int | None = None, session_id: str | None = None
    ):
        self.messages = []

        # TASK-1774: without a user there is no safe slice of the bot's shared
        # partition. Refuse visibly rather than serve every user's history.
        if self._db_backend and not self.user_id:
            self._warn_userless_history_load()
            return

        # TASK-284 (reverted design): session threads are PROVENANCE METADATA
        # ONLY — a queryable session_id for restoring a conversation, never a
        # context boundary. Context is built from the CONTINUOUS message
        # history (all unsummarized raw + rolling summaries, budget-bounded
        # downstream) — that is the DEFAULT and the primary mode.
        #
        # TASK-251: ``session_id`` is the ONE sanctioned exception — an
        # EXPLICITLY selected thread (user picked a conversation in the UI).
        # The pool becomes that thread's raw bubbles + the bot's rolling
        # summaries; the downstream assembler (build_context_payload) budgets
        # it identically. Never set implicitly — a request without session_id
        # must never take this branch.
        if session_id and self._db_backend:
            scoped_loader = getattr(self._db_backend, "load_session_scoped", None)
            if callable(scoped_loader):
                try:
                    scoped = scoped_loader(
                        since_minutes=since_minutes, session_id=session_id,
                        user_id=self.user_id,
                    )
                    if scoped is not None:
                        self.messages = scoped
                        logger.debug(
                            "Loaded %d messages scoped to thread %s (TASK-251)",
                            len(self.messages), session_id,
                        )
                        return
                except Exception as e:
                    logger.warning(
                        "Scoped history load failed for thread %s, "
                        "falling back to continuous: %s", session_id, e,
                    )
            else:
                logger.warning(
                    "Backend lacks load_session_scoped; ignoring session_id=%s "
                    "and loading continuous history", session_id,
                )

        # Use PostgreSQL backend if available
        if self._db_backend:
            try:
                self.messages = self._db_backend.get_messages(
                    since_minutes=since_minutes, user_id=self.user_id,
                )
                logger.debug(f"Loaded {len(self.messages)} messages from PostgreSQL short-term memory")
                return
            except Exception as e:
                # TASK-1774: no file fallback. A DB-backed bot's history lives
                # only in its user-scoped partition reads; a file is not.
                logger.warning(
                    "History load failed for bot=%s user=%s; context has no "
                    "history: %s", self.bot_id, self.user_id, e,
                )
                self.messages = []
                return

        # File-based fallback
        if not self.history_file or not os.path.exists(self.history_file):
            print("No history file found. Skipping history load.")
            logger.debug("No history file found. Skipping history load.")
            return
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                message_dicts = json.load(f)
                self.messages = [Message.from_dict(msg) for msg in message_dicts]
                logger.debug(f"Loaded {len(self.messages)} messages from history file.")
        except UnicodeDecodeError as e:
            self.client.console.print(f"[bold red]Error loading history:[/bold red] Unable to decode file. Ensure it is saved in UTF-8 format. ({e})")
        except json.JSONDecodeError as e:
            self.client.console.print(f"[bold red]Error loading history:[/bold red] Invalid JSON format in {self.history_file}. ({e})")
        except Exception as e:
            self.client.console.print(f"[bold red]Error loading history:[/bold red] {e}")
        if since_minutes is not None and self.messages:
            # Note: since_minutes param is actually in seconds (legacy naming)
            logger.debug(f"Loading history from {since_minutes} seconds ago ({len(self.messages)} messages)")
            cutoff = time.time() - since_minutes
            self.messages = [msg for msg in self.messages if msg.timestamp >= cutoff]

    def _warn_userless_history_load(self) -> None:
        request_id = None
        try:
            from ..service.logging import get_request_id

            request_id = get_request_id()
        except Exception:  # CLI/standalone: no service request context
            pass
        logger.warning(
            "History load refused for bot=%s request=%s: no user_id, so the "
            "bot's shared history cannot be scoped (TASK-1774); context has "
            "no history",
            self.bot_id, request_id,
        )

    def save_history(self):
        """Persist message history to the history file."""
        # PostgreSQL backend saves on add_message, so nothing to do here
        if self._db_backend:
            return
            
        if not self.history_file:
            return
        try:
            message_dicts = [msg.to_dict() for msg in self.messages]
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(message_dicts, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.client.console.print(f"[bold red]Error saving history:[/bold red] {e}")

    def _compact_summary_content(self, content: str) -> str:
        """Reduce summary verbosity for prompt context while preserving key continuity."""
        text = (content or "").strip()
        if not text:
            return text

        # TASK-611: Tier-3 canonical key (was summarization_compact_context).
        # Registry default only — no retired config env attr.
        compact_enabled = bool(
            self._setting("compact_context", setting_default("compact_context", True))
        )
        if not compact_enabled:
            return text

        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text

        from ..memory.summarization import compress_structured_summary_text, extract_summary_sections

        prefix_lines: list[str] = []
        sections = extract_summary_sections(compress_structured_summary_text(text))
        section_summary = (sections.get("summary") or "").strip()
        section_key_details = (sections.get("key_details") or "").strip()
        section_intent = (sections.get("intent") or "").strip()
        section_open_loops = (sections.get("open_loops") or "").strip()

        for line in lines:
            lowered = line.lower()
            if lowered.startswith("[historical summary]"):
                continue
            if lowered.startswith("summary:") or lowered.startswith("key details:") or lowered.startswith("intent:") or lowered.startswith("tone:") or lowered.startswith("open loops:"):
                continue
            # Keep non-section metadata prefix lines (legacy lines like "On YYYY-MM-DD...").
            # Exclude bullet lines — they belong to a section (e.g. Key Details) and would
            # otherwise be hoisted to the top AND duplicated inside their section body.
            if ":" not in line and not line.startswith("[") and not line.lstrip().startswith(("-", "*", "•")):
                prefix_lines.append(line)

        compact_lines: list[str] = []
        compact_lines.extend(prefix_lines[:2])
        if section_summary:
            compact_lines.append(section_summary)
        if section_key_details:
            compact_lines.append(f"Details: {section_key_details}")
        if section_intent:
            compact_lines.append(f"Intent: {section_intent}")
        if section_open_loops:
            compact_lines.append(f"Open: {section_open_loops}")

        # Legacy fallback: keep as-is if we couldn't identify structured sections.
        if not compact_lines:
            return text

        return "\n".join(compact_lines).strip()

    @staticmethod
    def _unanswered_trailing_turn(msgs: list) -> list:
        """The in-flight turn (TASK-650): the trailing run from the newest
        user row — user message plus any interleaved tool-evidence rows —
        provided NO assistant reply follows it. Empty when the newest turn is
        already answered (that's prior history, not the current turn)."""
        turn_start = -1
        for i in range(len(msgs) - 1, -1, -1):
            role = getattr(msgs[i], "role", "")
            if role == "assistant":
                break
            if role == "user":
                turn_start = i
                break
        return list(msgs[turn_start:]) if turn_start >= 0 else []

    def get_context_messages(
        self,
        max_tokens: int = 0,
        charge_system: bool = True,
        *,
        want_history: bool = True,
        want_summaries: bool = True,
        floor_current_turn: bool = False,
    ):
        """Assemble LLM context under Nick's strict allocation ladder (TASK-612).

        Fill order, highest priority first:
          1. **Plain system** prompt (persona) — always kept. If it alone exceeds
             the physical budget this raises ``ContextBudgetError`` (the ONLY
             hard-fail). Tool-execution evidence rows (role='system' with a
             ``[Tool Results]`` / ``[Tools used:]`` prefix) are NOT persona — they
             are raw historical context and ride the raw bucket below, so they can
             never trigger this hard-fail.
          2. **Raw** unsummarized history — newest-first, bounded by the Tier-3
             ``history_tokens`` policy knob (default 12000) *and* by physical
             headroom. Includes user/assistant rows plus (inline delivery only)
             interleaved tool-evidence rows. The newest complete turn is floored
             in even when ``history_tokens`` is set absurdly small (degrade-and-log).
          3. **Summaries** — fill the *remaining physical budget* after system +
             raw, newest-first, capped by ``summary_count`` (default 5).
             ``summary_count=0`` carries none (raw-only bot).

        This deliberately reverses the old flow (summaries filled before raw, a
        ``memory_protected_recent_turns`` reservation carved off the top). Under
        the ladder raw already outranks summaries and newest already outranks
        oldest, so the protected-turn reservation is redundant and gone; the
        recent-turn guarantee now falls out of raw-first + newest-first ordering
        plus the newest-complete-turn floor. ``memory_protected_recent_turns`` is
        a Tier-1 summarization-job parameter now (TASK-610), not an assembly knob.

        The raw vs summary partition is the DB ``summarized`` flag (summarized
        sessions become ``role='summary'`` rows; unsummarized stay raw), so the
        two buckets are disjoint by construction — no dedup here.

        Output order is chronological: ``system + summaries + raw``.

        Args:
            max_tokens: physical prompt budget (Tier-2 ``resolve_context_budget``).
                        0 = no budget (return everything, back-compat).
            charge_system: whether system rows consume the budget. True for
                        inline (chat) delivery — system is sent in the request
                        window. False for seed delivery (TASK-613): the seed
                        strips system rows (the SDK carries its own byte-stable
                        system prompt), so they must NOT reserve seed tokens and
                        cannot trigger the system-over-budget hard-fail. System
                        rows are still returned/separated either way; only the
                        budgeting differs.
            want_history: whether the caller will actually deliver raw history.
                        When False (summary-only scope), raw is not allocated at
                        all — otherwise it would consume budget here and then be
                        discarded downstream, starving summaries that would have
                        fit (TASK-612 finding 1). Must mirror the caller's
                        ``include_history`` so allocation matches delivery.
            want_summaries: whether the caller will deliver summaries. When False,
                        summaries are not allocated. Mirrors ``include_summaries``.
            floor_current_turn: (TASK-650) when True, an UNANSWERED trailing
                        turn (the in-flight user message + its tool-evidence)
                        is reserved as a mandatory raw floor even when
                        ``want_history`` is False — scope governs PRIOR
                        context, never the message being answered. Summaries
                        then fill the remainder. Inline delivery only; seeds
                        are built between turns and pass False.
        """
        # Only the PLAIN persona/system prompt is the non-negotiable system
        # bucket. A role='system' row that carries tool-execution EVIDENCE
        # (``[Tool Results]`` / ``[Tools used:]``) is historical raw context, not
        # the persona — so it is budgeted with raw history (bounded by
        # history_tokens, part of the newest-turn floor, obeys want_history) and
        # can never trigger the system-over-budget hard-fail (TASK-612 follow-up:
        # tool-result rows previously rode the non-negotiable system budget).
        # Seed delivery strips ALL system rows (seed_messages), so tool-evidence
        # is folded into the budgeted raw stream only for inline delivery
        # (``charge_system``) — otherwise it would consume seed budget for rows
        # that get stripped, the same starvation bug as finding 1.
        plain_system_messages = []
        summary_messages = []
        raw_stream = []  # chronological user/assistant (+ inline tool-evidence)

        for msg in self.messages:
            if _is_tool_result_system(msg):
                if charge_system:
                    raw_stream.append(msg)
            elif msg.role == "system":
                plain_system_messages.append(msg)
            elif msg.role == "summary":
                # Add time context to summaries
                time_ago = self._format_time_ago(msg.timestamp)
                compact_summary = self._compact_summary_content(msg.content)
                enhanced_content = f"[Previous conversation {time_ago}]\n{compact_summary}"
                summary_messages.append(Message(
                    role="summary",
                    content=enhanced_content,
                    timestamp=msg.timestamp,
                    db_id=msg.db_id,
                ))
            else:
                raw_stream.append(msg)

        if not plain_system_messages:
            plain_system_messages.append(Message(role="system", content=self.config.SYSTEM_MESSAGE))

        # (TASK-615) The legacy message-count cap (max_context_messages) is
        # retired: history_tokens (the token budget below) is the sole control
        # over how much raw history is carried. No count-based truncation here.

        # (TASK-647) Age bound on the raw bucket: raw messages older than
        # history_max_age_hours are excluded up front, regardless of token
        # budget (applies to the no-budget path too). 0 = unlimited (default).
        # Raw ONLY — summaries keep their own temporal framing and are bounded
        # by summary_count. Rows without a timestamp are kept defensively.
        # Deliberate consequence: if the entire newest turn is older than the
        # cutoff, the newest-complete-turn floor below has nothing to force in —
        # an operator setting an age cap is explicitly asking stale raw context
        # NOT to be carried.
        max_age_hours = int(
            self._setting(
                "history_max_age_hours",
                setting_default("history_max_age_hours", 0),
            )
        )
        if max_age_hours > 0 and raw_stream:
            cutoff = time.time() - max_age_hours * 3600
            kept = [m for m in raw_stream if not m.timestamp or m.timestamp >= cutoff]
            if len(kept) != len(raw_stream):
                logger.debug(
                    "history_max_age_hours=%s: dropped %s/%s raw messages older "
                    "than the cutoff.",
                    max_age_hours, len(raw_stream) - len(kept), len(raw_stream),
                )
                raw_stream = kept

        # Without a token budget, return everything (chronological: system,
        # summaries, then raw recent messages).
        if max_tokens <= 0:
            return plain_system_messages + summary_messages + raw_stream

        # ── Allocation ladder ─────────────────────────────────────
        budget = max_tokens
        # System rows only consume the budget when they will actually be
        # delivered (inline chat). Seed delivery strips them, so a seed spends
        # its whole budget on the delivered summary+raw content (TASK-613).
        system_cost = estimate_messages_tokens(plain_system_messages) if charge_system else 0

        # (1) System is non-negotiable when delivered. System-alone-over-budget
        #     is the one unrecoverable state — fail loudly instead of truncating
        #     it. Not applicable to seeds (system isn't delivered).
        if charge_system and system_cost > budget:
            raise ContextBudgetError(
                f"System prompt ({system_cost} tokens) exceeds the prompt budget "
                f"({budget} tokens) for bot {self.bot_id!r}. Raise the model "
                f"context window or shrink the system prompt."
            )

        # (2) Raw history: newest-first, bounded by BOTH history_tokens (Tier-3
        #     policy) and the remaining physical budget. Skipped entirely when the
        #     caller won't deliver raw history (summary-only scope) — otherwise raw
        #     would consume budget here and then be discarded downstream, starving
        #     summaries that would have fit (TASK-612 finding 1).
        physical_headroom = budget - system_cost
        history_tokens = int(
            self._setting("history_tokens", setting_default("history_tokens", 12000))
        )
        # history_tokens should always be a positive bound (TASK-602 killed the
        # 0=whole-window footgun); defensively treat <=0 as "use all headroom".
        raw_cap = physical_headroom if history_tokens <= 0 else min(history_tokens, physical_headroom)

        included_raw: list[Message] = []
        raw_used = 0
        forced = ""
        if want_history:
            for msg in reversed(raw_stream):  # newest-first
                cost = estimate_messages_tokens([msg])
                if raw_used + cost <= raw_cap:
                    included_raw.insert(0, msg)  # keep chronological order
                    raw_used += cost
                else:
                    break  # older messages only get larger-or-equal in aggregate

        # Newest-COMPLETE-turn floor: history_tokens must never chop the most
        # recent turn in half. The newest complete turn is the trailing run
        # back through the most recent user message (captures user + assistant
        # + any interleaved tool-evidence rows). Guarantee it whenever it fits
        # the PHYSICAL budget, even if history_tokens was too small to have fit
        # it in the bounded fill above (TASK-612 finding 2 — a single-message
        # floor returned the assistant without its user). If the whole turn
        # can't fit the physical budget, keep the largest newest-first suffix
        # that does (degrade); if not even the newest message fits, drop it.
        #
        # (TASK-650) The floor ALSO applies when the caller carries no raw
        # history (history_scope none/summaries on inline delivery): an
        # unanswered trailing turn is the in-flight message being answered,
        # not prior context, so it is reserved HERE — before summaries fill —
        # so allocation stays within the physical budget instead of a
        # post-hoc append blowing past it.
        turn_msgs: list[Message] = []
        if raw_stream:
            if want_history:
                turn_start = 0
                for i in range(len(raw_stream) - 1, -1, -1):
                    if raw_stream[i].role == "user":
                        turn_start = i
                        break
                turn_msgs = raw_stream[turn_start:]
            elif floor_current_turn:
                turn_msgs = self._unanswered_trailing_turn(raw_stream)
        if turn_msgs and len(turn_msgs) > len(included_raw):  # fill didn't cover the turn
            turn_cost = estimate_messages_tokens(turn_msgs)
            if system_cost + turn_cost <= budget:
                included_raw = list(turn_msgs)
                raw_used = turn_cost
                forced = "turn"
                if want_history:
                    logger.warning(
                        "history_tokens=%s too small for the newest complete "
                        "turn (%s msgs); forcing it in under the physical "
                        "budget (%s).",
                        history_tokens, len(turn_msgs), budget,
                    )
                else:
                    logger.debug(
                        "Current-turn floor (scope carries no raw history): "
                        "reserved %s in-flight message(s), %s tokens.",
                        len(turn_msgs), turn_cost,
                    )
            else:
                fit: list[Message] = []
                used = 0
                for msg in reversed(turn_msgs):  # newest-first
                    cost = estimate_messages_tokens([msg])
                    if system_cost + used + cost <= budget:
                        fit.insert(0, msg)
                        used += cost
                    else:
                        break
                included_raw = fit
                raw_used = used
                forced = "partial-turn" if fit else "dropped"
                logger.warning(
                    "Newest turn (%s tokens) exceeds physical headroom "
                    "(%s); kept %s of %s msgs (degrade).",
                    turn_cost, physical_headroom, len(fit), len(turn_msgs),
                )

        # (3) Summaries: fill the remaining physical budget after system + raw,
        #     newest-first, capped by summary_count. Skipped when the caller won't
        #     deliver summaries. Guard the slice because summary_messages[-0:]
        #     would select the WHOLE list, not none.
        remaining = budget - system_cost - raw_used
        max_summaries = int(
            self._setting("summary_count", setting_default("summary_count", 5))
        )
        candidate_summaries = (
            summary_messages[-max_summaries:]
            if (want_summaries and max_summaries > 0)
            else []
        )
        included_summaries: list[Message] = []
        summary_used = 0
        for s in reversed(candidate_summaries):  # newest summary first under pressure
            cost = estimate_messages_tokens([s])
            if summary_used + cost <= remaining:
                included_summaries.insert(0, s)  # keep chronological order
                summary_used += cost
            else:
                break

        raw_dropped = len(raw_stream) - len(included_raw)
        used = system_cost + raw_used + summary_used

        def pct(v: int) -> str:
            return f"{v * 100 // budget}%" if budget > 0 else "n/a"

        logger.debug(
            f"Ladder budget: {budget} total | used={used} ({pct(used)}) | "
            f"system={system_cost} ({pct(system_cost)}), "
            f"raw={raw_used} ({pct(raw_used)}, {len(included_raw)} msgs, "
            f"{raw_dropped} dropped, cap={raw_cap}{f', forced-{forced}' if forced else ''}), "
            f"summaries={summary_used} ({pct(summary_used)}, "
            f"{len(included_summaries)}/{len(summary_messages)})"
        )

        return plain_system_messages + included_summaries + included_raw

    def build_context_payload(
        self,
        *,
        include_history: bool = True,
        include_summaries: bool = True,
        delivery: str = "inline",
        max_tokens: int = 0,
    ) -> ContextPayload:
        """The ONE context-assembly handler (TASK-493/518).

        Decides *what* prior context a conversation gets, identically for chat
        turns and agent seeds. Wraps the shared assembler ``get_context_messages``
        and categorises its output into buckets. The two carried buckets are
        gated by two INDEPENDENT flags — there is no coupling between them:

            regular (inline) messages  included iff  include_history
            summary rows               included iff  include_summaries

        Both may be on (recent + summaries), either alone (recent-only, or the
        dense summary-only seed), or both off (carry nothing — a cold turn). The
        two flags come from the ``history_scope`` enum at the call site:
        ``"inline" in scope`` and ``"summaries" in scope`` respectively.

        System-row handling is NOT decided here — the caller reads the view that
        matches its delivery (``inline_history`` for chat, ``seed_messages`` for a
        seed). ``delivery`` is accepted for call-site clarity and future use; it
        does not change the buckets (both plain-system and tool-result-system rows
        are always separated so either delivery can compose correctly).

        Args:
            include_history: carry raw recent messages (regular_messages bucket).
            include_summaries: carry rolling summaries (summary_messages bucket).
            delivery: "inline" (chat) or "seed" (agent) — documentary only.
            max_tokens: token budget passed through to ``get_context_messages``.
        """
        # TASK-613: seed delivery strips system rows (the SDK carries its own
        # system prompt), so they must not consume the seed's token budget.
        # Inline (chat) delivers system in the request window, so it does.
        # (TASK-650) Current-turn floor for INLINE delivery: history_scope
        # governs PRIOR context only — it must never swallow the message the
        # user just sent. On the chat path the in-flight user message is
        # already persisted into self.messages, so with include_history=False
        # ("none"/"summaries" scope) it would be dropped with the rest of raw
        # history and the model would receive a system-only prompt (observed:
        # nova replying "tool format locked in" to "Hi"). The assembler
        # reserves the unanswered trailing turn as a mandatory raw floor
        # (summaries then fill the remainder — allocation stays within
        # budget); the id-set below exempts exactly those rows from the
        # include_history gate. Seed delivery is untouched: a seed is built
        # between turns and has no current message.
        floor_current = delivery == "inline" and not include_history
        current_turn_ids: set[int] = (
            set(map(id, self._unanswered_trailing_turn(self.messages)))
            if floor_current
            else set()
        )

        payload = ContextPayload()
        for msg in self.get_context_messages(
            max_tokens=max_tokens,
            charge_system=(delivery != "seed"),
            want_history=include_history,
            want_summaries=include_summaries,
            floor_current_turn=floor_current,
        ):
            role = getattr(msg, "role", "")
            is_current_turn = id(msg) in current_turn_ids
            if role == "summary":
                if include_summaries:
                    payload.summary_messages.append(msg)
                continue
            if role == "system":
                if _is_tool_result_system(msg):
                    # Tool-evidence is raw historical context — gate it on
                    # include_history like any other raw row (TASK-612 follow-up),
                    # except the in-flight turn's own evidence (TASK-650).
                    # get_context_messages already excludes it from the budgeted
                    # stream when want_history is False; this keeps direct callers
                    # honest too.
                    if include_history or is_current_turn:
                        payload.tool_result_messages.append(msg)
                else:
                    payload.system_messages.append(msg)
                continue
            if include_history or is_current_turn:
                payload.regular_messages.append(msg)
        return payload

    def _format_time_ago(self, timestamp: float) -> str:
        """Format a timestamp as a human-readable relative time."""
        import time
        now = time.time()
        diff = now - timestamp
        
        if diff < 60:
            return "just now"
        elif diff < 3600:
            minutes = int(diff / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif diff < 7200:
            return "about 1 hour ago"
        elif diff < 86400:
            hours = int(diff / 3600)
            return f"{hours} hours ago"
        elif diff < 172800:
            return "yesterday"
        elif diff < 604800:
            days = int(diff / 86400)
            return f"{days} days ago"
        elif diff < 1209600:
            return "last week"
        elif diff < 2592000:
            weeks = int(diff / 604800)
            return f"{weeks} weeks ago"
        elif diff < 5184000:
            return "last month"
        else:
            months = int(diff / 2592000)
            return f"{months} months ago"
    
    def get_context_messages_excluding_last(self):
        """Get messages to be used as context, excluding the most recent one."""
        context_messages = self.get_context_messages()
        if len(context_messages) > 1:
            return context_messages[:-1]
        elif context_messages and context_messages[0].role == "system":
            return context_messages
        else:
            if any(msg.role == "system" for msg in context_messages):
                return [msg for msg in context_messages if msg.role == "system"]
            else:
                return [Message(role="system", content=self.config.SYSTEM_MESSAGE)]

    def add_message(self, role, content, message_id=None, attachments=None,
                    reasoning=None, session_id=None, author=None, extract_memory=True):
        """Append a message to history and save.

        If ``message_id`` is provided, it is used as the persistent ID so
        the frontend's user-message UUID matches the server-side history
        row — enabling tool-call events keyed by trigger_message_id to
        join cleanly with chat history without a separate id-mapping step.

        ``attachments`` (TASK-225): optional tiny JSONB payload written
        to the ``{bot}_messages.attachments`` column — a list of
        ``{"asset_id": "ma_...", "kind": "image"}`` refs.  Only
        meaningful on user-role rows in practice; the file-fallback path
        ignores it (the on-disk format has no concept of attachments).
        """
        provided_id = str(message_id).strip() if message_id else None
        message = Message(role, content, db_id=provided_id or str(uuid.uuid4()))
        self.messages.append(message)

        # Save to PostgreSQL if available, otherwise file
        if self._db_backend:
            # EPIC TASK-217 / TASK-357: the DB backend now RAISES on a failed
            # INSERT instead of swallowing it and handing back a valid-looking
            # id. When the DB is configured, a failed persist must NOT be
            # silently masked by a file fallback — that is exactly what left a
            # turn_logs row pointing at an uncommitted (orphan) {bot}_messages
            # row. Propagate the failure so the turn-orchestration layer marks
            # the turn as errored (background_service catches this and sets the
            # turn_log status="error", then re-raises) rather than recording a
            # ghost reference. Every caller wraps add_message in its own
            # try/except, so the raise aborts the turn loudly and safely.
            # TASK-251: ``session_id`` is the explicit thread override for
            # continue-old-thread turns. Passed ONLY when set so backends
            # (and test doubles) without the parameter keep working on the
            # continuous default path; a scoped write against such a backend
            # fails loudly instead of silently landing on the wrong thread.
            _extra = {"session_id": session_id} if session_id else {}
            if author is not None:
                _extra["author"] = author
            if not extract_memory:
                _extra["extract_memory"] = False
            persisted_id = self._db_backend.add_message(
                role, content, message.timestamp,
                message_id=provided_id,
                attachments=attachments,
                reasoning=reasoning,
                **_extra,
            )
            if persisted_id:
                message.db_id = str(persisted_id)
        else:
            self.save_history()


    def print_history(self, pairs_limit=None):
        """Print the conversation history in a formatted way.

        Args:
            pairs_limit: Number of recent conversation pairs to show (-1 for all).
        """
        if not self.messages:
            self.client.console.print("[italic]No conversation history found.[/italic]")
            return

        non_system_messages = [msg for msg in self.messages if msg.role != "system"]

        if not non_system_messages:
            self.client.console.print("[italic]No conversation messages found.[/italic]")
            return

        if pairs_limit is not None and pairs_limit != -1:
            messages_to_show = min(pairs_limit * 2, len(non_system_messages))
            non_system_messages = non_system_messages[-messages_to_show:]

        self.client.console.print()
        self.client.console.print("[bold]Conversation History:[/bold]")
        self.client.console.print(Rule(style="#555555"))

        for msg in non_system_messages:
            if msg.role == "user":
                self.client._print_user_message(msg.content)
            elif msg.role == "assistant":
                panel_title, panel_border_style = self.client.get_styling()
                parts = msg.content.split("\n\n", 1)
                first_part = parts[0]
                second_part = parts[1] if len(parts) > 1 else None
                self.client._print_assistant_message(
                    first_part,
                    second_part=second_part,
                    panel_title=panel_title,
                    panel_border_style=panel_border_style
                )
            self.client.console.print(Rule(style="#333333"))
    
    def clear_history(self):
        """Clear the conversation history from memory and disk."""
        self.messages = []
        
        # Clear PostgreSQL if available
        if self._db_backend:
            try:
                self._db_backend.clear()
                self.client.console.print("[bold red]History cleared (PostgreSQL).[/bold red]")
                return
            except Exception as e:
                logger.warning(f"Failed to clear PostgreSQL history: {e}")
                # Fall through to file-based clearing
        
        if self.history_file and os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
                self.client.console.print("[bold red]History cleared.[/bold red]")
            except Exception as e:
                self.client.console.print(f"[bold red]Error clearing history file:[/bold red] {e}")
        else:
            self.client.console.print("[dim]No history file found to clear or history file path not set.[/dim]")
            
    def get_last_assistant_message(self) -> str | None:
        """Get the content of the last assistant message, or None."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def remove_last_message_if_partial(self, role: str):
        """Remove the last message if it matches the specified role (used for cleanup on error/interrupt)."""
        if self.messages and self.messages[-1].role == role:
            self.messages.pop()
            # Remove from PostgreSQL as well if available
            if self._db_backend:
                try:
                    self._db_backend.remove_last_message_if_partial(role)
                except Exception as e:
                    logger.warning(f"Failed to remove from PostgreSQL: {e}")
            else:
                self.save_history()
