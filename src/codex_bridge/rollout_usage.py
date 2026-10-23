"""Per-turn token usage for Codex threads, read from the rollout (TASK-1319).

The Codex SDK's ``turn.completed`` usage is the *thread's* running total, not
the turn's: on a resumed thread it grows every turn, and a tool-using turn adds
every internal model call. Stored as per-turn usage it inflated context gauges
and cost (a six-turn thread showed 7.7M input tokens).

The codex CLI writes the truth to its rollout file
(``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<thread_id>.jsonl``): every
model call appends an ``event_msg``/``token_count`` line carrying both
``total_token_usage`` (thread running total) and ``last_token_usage`` (that one
call), and every turn opens with an ``event_msg``/``task_started`` line.

From the rollout this module derives the bridge-wide usage convention the
Claude bridge already reports (``send_usage.py``):

- ``input_tokens`` / ``cache_read_tokens`` / ``cache_creation_tokens``: the
  turn's last model call, as disjoint buckets (OpenAI's ``input_tokens``
  includes the cached subset);
- ``output_tokens``: the whole turn;
- ``resident_tokens``: the last call's inclusive input, i.e. what is actually
  resident in the context window.

The rollout is only trusted when its running total matches the SDK's
``turn.completed`` usage (or, should the SDK switch to per-turn usage, the
turn's delta). Anything else returns ``None``: no usage beats wrong usage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

RESIDENT_SOURCE = "codex_last_call_input"

_THREAD_ID_RE = re.compile(r"^[0-9A-Za-z-]{8,64}$")


@dataclass(frozen=True)
class TokenCounts:
    """One Codex usage record. ``input`` includes ``cached`` and ``cache_write``."""

    input: int = 0
    cached: int = 0
    cache_write: int = 0
    output: int = 0

    @classmethod
    def from_mapping(cls, raw: Any) -> TokenCounts | None:
        if not isinstance(raw, Mapping):
            return None
        try:
            return cls(
                input=int(raw.get("input_tokens") or 0),
                cached=int(raw.get("cached_input_tokens") or 0),
                cache_write=int(raw.get("cache_write_input_tokens") or 0),
                output=int(raw.get("output_tokens") or 0),
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_sdk_usage(cls, usage_or_carrier: Any) -> TokenCounts | None:
        """Read the SDK ``Usage`` (object or dict, bare or under ``.usage``)."""
        usage = usage_or_carrier
        if isinstance(usage, Mapping) and "usage" in usage:
            usage = usage["usage"]
        elif usage is not None and not isinstance(usage, Mapping) \
                and hasattr(usage, "usage") and not hasattr(usage, "input_tokens"):
            usage = usage.usage
        if usage is None:
            return None
        if isinstance(usage, Mapping):
            return cls.from_mapping(usage)
        return cls.from_mapping({
            key: getattr(usage, key, 0)
            for key in ("input_tokens", "cached_input_tokens",
                        "cache_write_input_tokens", "output_tokens")
        })

    def __sub__(self, other: TokenCounts) -> TokenCounts:
        return TokenCounts(
            input=self.input - other.input,
            cached=self.cached - other.cached,
            cache_write=self.cache_write - other.cache_write,
            output=self.output - other.output,
        )

    def is_call(self) -> bool:
        """Plausible as one model call: no negatives, cached buckets within input."""
        return (
            min(self.input, self.cached, self.cache_write, self.output) >= 0
            and self.cached + self.cache_write <= self.input
        )

    def matches_sdk(self, sdk: TokenCounts) -> bool:
        # The SDK Usage has no cache-write field; compare what it reports.
        return (self.input, self.cached, self.output) == (sdk.input, sdk.cached, sdk.output)


@dataclass(frozen=True)
class RolloutTurnUsage:
    """Usage of the rollout's latest turn."""

    thread_total: TokenCounts
    turn_total: TokenCounts
    last_call: TokenCounts
    model_calls: int
    context_window: int | None = None

    def matches_sdk(self, sdk: TokenCounts) -> bool:
        return self.thread_total.matches_sdk(sdk) or self.turn_total.matches_sdk(sdk)

    def to_token_usage(self) -> dict[str, Any]:
        last = self.last_call
        return {
            "input_tokens": max(last.input - last.cached - last.cache_write, 0),
            "cache_read_tokens": last.cached,
            "cache_creation_tokens": last.cache_write,
            "output_tokens": self.turn_total.output,
            "context_window": None,  # the caller fills these from the catalog
            "max_output_tokens": None,
            "total_cost_usd": None,
            "resident_tokens": last.input,
            "resident_source": RESIDENT_SOURCE,
            # Audit only; never summed into the context gauge.
            "turn_input_tokens": self.turn_total.input,
            "turn_cached_input_tokens": self.turn_total.cached,
            "turn_model_calls": self.model_calls,
        }


def parse_latest_turn(lines: Iterable[str]) -> RolloutTurnUsage | None:
    """Usage of the last turn in a rollout, or ``None`` if it made no model
    call or its counts don't add up.

    The turn's usage is the running total at its last ``token_count`` minus the
    running total before its ``task_started``. Within the turn every new count
    must be one plausible call that the running total grew by exactly; a total
    that shrinks or jumps (seen around compaction and resumes) or an
    unparseable count makes the turn unreadable. Lines that are not JSON (a
    partly flushed tail) are skipped.
    """
    before_turn = TokenCounts()
    running: TokenCounts | None = None
    turn_end: TokenCounts | None = None
    last_call: TokenCounts | None = None
    calls = 0
    consistent = True
    context_window: int | None = None
    for line in lines:
        if '"token_count"' not in line and '"task_started"' not in line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict) or obj.get("type") != "event_msg":
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = payload.get("type")
        if kind == "task_started":
            before_turn = running or TokenCounts()
            turn_end = last_call = None
            calls = 0
            consistent = True
            context_window = None
        elif kind == "token_count":
            info = payload.get("info")
            if not isinstance(info, dict):
                continue  # a rate-limit-only count
            total = TokenCounts.from_mapping(info.get("total_token_usage"))
            last = TokenCounts.from_mapping(info.get("last_token_usage"))
            if total is None or last is None:
                consistent = False
                continue
            if total != running:  # the CLI can repeat a count unchanged
                calls += 1
                if not last.is_call() or total - (running or TokenCounts()) != last:
                    consistent = False
            running = turn_end = total
            last_call = last
            window = info.get("model_context_window")
            context_window = window if isinstance(window, int) and window > 0 else None
    if turn_end is None or last_call is None:
        return None
    if not consistent:
        logger.warning("Codex rollout counts are inconsistent for the latest turn")
        return None
    return RolloutTurnUsage(
        thread_total=turn_end,
        turn_total=turn_end - before_turn,
        last_call=last_call,
        model_calls=calls,
        context_window=context_window,
    )


class RolloutUsageReader:
    """Find a thread's rollout and read its latest turn's usage."""

    def __init__(
        self,
        codex_home: Path | None = None,
        *,
        attempts: int = 5,
        retry_delay: float = 0.1,
    ) -> None:
        self._codex_home = codex_home
        self._attempts = max(1, attempts)
        self._retry_delay = retry_delay
        self._paths: dict[str, Path] = {}

    @property
    def codex_home(self) -> Path:
        if self._codex_home is not None:
            return self._codex_home
        return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")

    def locate(self, thread_id: str) -> Path | None:
        if not _THREAD_ID_RE.match(thread_id or ""):
            return None
        cached = self._paths.get(thread_id)
        if cached is not None and cached.is_file():
            return cached
        matches = sorted(
            (self.codex_home / "sessions").glob(f"*/*/*/rollout-*-{thread_id}.jsonl")
        )
        if not matches:
            return None
        self._paths[thread_id] = matches[-1]
        return matches[-1]

    def read_latest_turn(self, thread_id: str) -> RolloutTurnUsage | None:
        path = self.locate(thread_id)
        if path is None:
            return None
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                return parse_latest_turn(handle)
        except OSError:
            logger.warning("Codex rollout unreadable: %s", path, exc_info=True)
            return None

    async def resolve(self, thread_id: str | None, sdk: TokenCounts) -> RolloutTurnUsage | None:
        """The latest turn's usage, once the rollout agrees with the SDK.

        The CLI may still be flushing its last ``token_count`` when the SDK
        reports ``turn.completed``, so a mismatch is retried briefly.
        """
        if not thread_id:
            logger.warning("Codex usage dropped: no thread id for the rollout lookup")
            return None
        turn: RolloutTurnUsage | None = None
        for attempt in range(self._attempts):
            turn = await asyncio.to_thread(self.read_latest_turn, thread_id)
            if turn is not None and turn.matches_sdk(sdk):
                return turn
            if attempt + 1 < self._attempts:
                await asyncio.sleep(self._retry_delay)
        logger.warning(
            "Codex usage dropped: rollout for thread %s %s the SDK usage "
            "(sdk input=%s cached=%s output=%s)",
            thread_id,
            "does not match" if turn is not None else "has no usage for",
            sdk.input, sdk.cached, sdk.output,
        )
        return None
