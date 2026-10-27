"""Recurring context-waste report (TASK-1585, DESIGN v2).

Names, counts, token sizes, hashes and redacted argument *key* shapes only:
no result bodies, argument values, paths or commands ever leave this module.

- Trusted rows were measured by an accounting-v2 producer. Older rows sit
  in an unranked "unverified" bucket (they may count media as text).
- Repeated identical output is grouped by (bot, tool, sha). It is a
  candidate redundancy, not proven waste: a repeat may be a freshness check.
  The same sha under different tools is reported separately and never added
  to per-tool repeat totals.
- Results rank by one-off size. Carried cost is not measurable from turn
  logs, so it is not claimed.
"""

from __future__ import annotations

import json
import shlex
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from . import queries
from .queries import Scope

SETTING_KEY = "context_waste_thresholds"
REPEAT_WINDOW = timedelta(hours=24)
TOP_GROUPS = 25
TOP_TOOLS = 50
TOP_PER_TOOL = 5
EXAMPLE_TURNS = 3
SHA_CHARS = 12

#: Generic seed. Operators override per install in the runtime setting.
DEFAULT_THRESHOLDS: dict[str, Any] = {
    "default": 8_000,
    "repeat_min_tokens": 500,
    "per_tool": {
        "Read": 10_000, "Bash": 6_000, "shell": 6_000,
        "tasks_get": 4_000, "tasks_reviews_list": 4_000, "messages_search_all": 4_000,
        "tasks_list": 4_000, "browser_snapshot": 8_000,
    },
}

#: First-party tool → the cheaper call that usually answers the same question.
CHEAPER_ALTERNATIVES: dict[str, str] = {
    "Read": "Read with offset/limit (a line range)",
    "tasks_get": "tasks_get_context or tasks_board",
    "tasks_list": "tasks_board with q and limit",
    "tasks_reviews_list": "tasks_get_context (carries the latest verdicts)",
    "messages_search_all": "messages_find, then messages_context",
    "messages_get": "messages_find, then messages_context",
    "memory_search_all": "memory_search_source for one known bot",
    "ops_list_operations": "reuse the catalog already fetched this session",
    "browser_snapshot": "browser_take_screenshot or a scoped browser_evaluate",
}

# Explicit first-party schema-key allowlist (binding DESIGN clarification 3).
# Only these key *names* are shown; values never are. Anything else counts as
# unknown_keys.
BUILTIN_ARG_KEYS: dict[str, frozenset[str]] = {
    "Read": frozenset({"file_path", "offset", "limit", "pages"}),
    "Bash": frozenset({"command", "timeout", "description", "run_in_background"}),
    "shell": frozenset({"command", "timeout", "workdir"}),
    "Grep": frozenset({"pattern", "path", "glob", "output_mode", "type", "head_limit"}),
    "Glob": frozenset({"pattern", "path"}),
    "Edit": frozenset({"file_path", "old_string", "new_string", "replace_all"}),
    "Write": frozenset({"file_path", "content"}),
    "WebFetch": frozenset({"url", "prompt"}),
    "WebSearch": frozenset({"query"}),
    "ToolSearch": frozenset({"query", "max_results"}),
    "Skill": frozenset({"skill", "args"}),
}
FIRST_PARTY_MCP_KEYS = frozenset({
    "bot_id", "task_id", "task_ids", "q", "query", "limit", "offset", "status", "focused",
    "bot", "speaker", "since", "until", "handle", "id", "card_ref", "before", "after",
    "max_chars", "focus", "sources", "include", "sort_by", "entity_type", "entity_id",
    "target_bot_id", "message", "wait_for_reply", "delivery", "idempotency_key",
    "operation", "job_id", "project_id", "memory_id", "min_importance", "n",
})
FIRST_PARTY_MCP_SERVER = "bawthub"
#: Command heads that are safe to name; any other first word is "other".
CMD_HEADS = frozenset({
    "git", "grep", "rg", "sed", "awk", "cat", "head", "tail", "ls", "find", "wc", "make",
    "ssh", "curl", "jq", "python", "python3", "uv", "pytest", "docker", "cd", "echo",
    "pgq", "pnpm", "npm", "node", "diff", "sort", "stat", "test", "gh", "commit-slice",
})
_SHELLS = frozenset({"bash", "sh", "zsh"})

LABELS = {
    "unverified": "unverified accounting, may include media counted as text",
    "repeats": "repeat may be an intentional freshness check",
    "composition": "share of the session transcript, Claude SDK estimate, not resident context",
    "weighting": "ranked by one-off size; per-result residence is not measurable yet",
}
_BREAKDOWN_FIELDS = ("toolCallTokens", "toolResultTokens", "attachmentTokens",
                     "assistantMessageTokens", "userMessageTokens")


# ----- tool names and thresholds ------------------------------------------


def split_tool_name(tool_name: str) -> tuple[str | None, str]:
    """``mcp__<server>__<tool>`` → (server, tool); built-ins → (None, name)."""
    if tool_name.startswith("mcp__"):
        server, sep, base = tool_name[5:].partition("__")
        if sep and base:
            return server, base
    return None, tool_name


def _first_party_mcp(server: str | None, base: str) -> bool:
    if server == FIRST_PARTY_MCP_SERVER:
        return True
    if server is None and base not in BUILTIN_ARG_KEYS:
        # Codex records first-party MCP tools by their bare names.
        from llm_bawt.mcp_server.catalog_contracts import TOOL_SUMMARIES
        return base in TOOL_SUMMARIES
    return False


def _positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def resolve_thresholds(raw: Any) -> dict[str, Any]:
    """Validated overrides on top of the generic seed; bad entries are ignored."""
    resolved = {
        "default": DEFAULT_THRESHOLDS["default"],
        "repeat_min_tokens": DEFAULT_THRESHOLDS["repeat_min_tokens"],
        "per_tool": dict(DEFAULT_THRESHOLDS["per_tool"]),
    }
    if not isinstance(raw, Mapping):
        return resolved
    for key in ("default", "repeat_min_tokens"):
        if (value := _positive_int(raw.get(key))) is not None:
            resolved[key] = value
    per_tool = raw.get("per_tool")
    if isinstance(per_tool, Mapping):
        for tool, value in per_tool.items():
            if isinstance(tool, str) and tool and (value := _positive_int(value)) is not None:
                resolved["per_tool"][tool[:128]] = value
    return resolved


def threshold_for(thresholds: Mapping[str, Any], tool_name: str) -> int:
    per_tool = thresholds["per_tool"]
    _, base = split_tool_name(tool_name)
    return per_tool.get(tool_name) or per_tool.get(base) or thresholds["default"]


def alternative_for(tool_name: str) -> str | None:
    return CHEAPER_ALTERNATIVES.get(split_tool_name(tool_name)[1])


# ----- argument shape (redacted) ------------------------------------------


def _command_head(command: Any) -> str:
    if not isinstance(command, str):
        return "other"
    try:
        words = shlex.split(command)
    except ValueError:
        return "other"
    # Codex wraps commands as ``/bin/bash -lc "<script>"``; look inside once.
    if (len(words) >= 3 and words[0].rsplit("/", 1)[-1] in _SHELLS
            and words[1].startswith("-") and "c" in words[1]):
        return _command_head(words[2])
    if not words:
        return "other"
    head = words[0].rsplit("/", 1)[-1]
    return head if head in CMD_HEADS else "other"


def arg_shape(tool_name: str, arguments_json: str | None) -> dict[str, Any]:
    """Allowlisted key names plus value-free derived facts; never values."""
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
    except ValueError:
        return {"keys": [], "unknown_keys": None, "unparsed": True}
    if not isinstance(arguments, dict):
        return {"keys": [], "unknown_keys": None, "unparsed": True}
    server, base = split_tool_name(tool_name)
    if server is None and base in BUILTIN_ARG_KEYS:
        allowed = BUILTIN_ARG_KEYS[base]
    elif _first_party_mcp(server, base):
        allowed = FIRST_PARTY_MCP_KEYS
    else:
        allowed = frozenset()
    keys = sorted(key for key in arguments if key in allowed)
    shape: dict[str, Any] = {"keys": keys, "unknown_keys": len(arguments) - len(keys)}
    if server is None and base == "Read":
        shape["ranged"] = "offset" in arguments or "limit" in arguments
    if server is None and base in {"Bash", "shell"}:
        shape["cmd_head"] = _command_head(arguments.get("command"))
    return shape


# ----- metrics -------------------------------------------------------------


def _aware(value: datetime | str) -> datetime:
    """UTC-aware datetime; raw SQL on some drivers returns ISO text."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return _aware(value).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def summarize_totals(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"trusted": [], "unverified": []}
    for row in rows:
        out["trusted" if row["trusted"] else "unverified"].append({
            "bot": row["bot_id"], "tool": row["tool_name"], "calls": int(row["calls"]),
            "tokens": int(row["tokens"]), "max_tokens": int(row["max_tokens"]),
        })
    out["trusted"].sort(key=lambda item: (-item["tokens"], item["bot"], item["tool"]))
    out["trusted"] = out["trusted"][:TOP_TOOLS]
    # Unverified rows are listed, never ranked: stable name order only.
    out["unverified"].sort(key=lambda item: (item["bot"], item["tool"]))
    return out


def find_repeats(rows: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """In-session, cross-session (≤24h) and cross-tool identical output.

    Each later call counts once: in-session if the same session already saw
    the (bot, tool, sha), else cross-session if another session saw it
    within :data:`REPEAT_WINDOW`, else it is a fresh occurrence.
    """
    by_tool: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    tools_by_sha: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for row in rows:
        by_tool[(row["bot_id"], row["tool_name"], row["sha"])].append(row)
        tools_by_sha[(row["bot_id"], row["sha"])][row["tool_name"]].append(row)

    groups: dict[str, list[dict[str, Any]]] = {"in_session": [], "cross_session": []}
    for (bot, tool, sha), group in by_tool.items():
        if len(group) < 2:
            continue
        seen_sessions: set[str] = set()
        prior: list[Mapping[str, Any]] = []
        repeats = {"in_session": [], "cross_session": []}
        for row in group:
            created = _aware(row["created_at"])
            if row["session_key"] in seen_sessions:
                repeats["in_session"].append(row)
            elif any(created - _aware(p["created_at"]) <= REPEAT_WINDOW for p in prior):
                repeats["cross_session"].append(row)
            seen_sessions.add(row["session_key"])
            prior.append(row)
        for scope, hits in repeats.items():
            if hits:
                groups[scope].append({
                    "bot": bot, "tool": tool, "sha": sha[:SHA_CHARS],
                    "calls": len(group), "repeats": len(hits),
                    "repeat_tokens": sum(int(hit["tokens"]) for hit in hits),
                    "first_seen": _iso(group[0]["created_at"]),
                    "last_seen": _iso(group[-1]["created_at"]),
                    "example_turns": list(dict.fromkeys(hit["turn_id"] for hit in hits))[:EXAMPLE_TURNS],
                    "_example_id": hits[0]["id"],
                    "alternative": alternative_for(tool),
                })
    for scope in groups:
        groups[scope].sort(key=lambda item: (-item["repeat_tokens"], item["bot"], item["tool"]))
        groups[scope] = groups[scope][:TOP_GROUPS]

    cross_tool = []
    for (bot, sha), tools in tools_by_sha.items():
        if len(tools) < 2:
            continue
        members = [row for rows_ in tools.values() for row in rows_]
        cross_tool.append({
            "bot": bot, "sha": sha[:SHA_CHARS], "tools": sorted(tools),
            "calls": len(members), "tokens": sum(int(row["tokens"]) for row in members),
        })
    cross_tool.sort(key=lambda item: (-item["tokens"], item["bot"], item["sha"]))
    groups["cross_tool"] = cross_tool[:TOP_GROUPS]
    return groups


def find_oversized(rows: list[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> list[dict[str, Any]]:
    per_tool: dict[str, dict[str, Any]] = {}
    for row in rows:
        tool = row["tool_name"]
        limit = threshold_for(thresholds, tool)
        tokens = int(row["tokens"])
        if tokens <= limit:
            continue
        entry = per_tool.setdefault(tool, {
            "tool": tool, "threshold": limit, "alternative": alternative_for(tool),
            "count": 0, "tokens": 0, "top": [],
        })
        entry["count"] += 1
        entry["tokens"] += tokens
        if len(entry["top"]) < TOP_PER_TOOL:  # rows arrive largest first
            entry["top"].append({
                "bot": row["bot_id"], "turn_id": row["turn_id"], "tokens": tokens,
                "created_at": _iso(row["created_at"]),
                "arg_shape": arg_shape(tool, row.get("arguments_json")),
            })
    ranked = sorted(per_tool.values(), key=lambda item: (-item["tokens"], item["tool"]))
    return ranked[:TOP_TOOLS]


def _breakdown(raw: Any) -> tuple[int, int] | None:
    """(toolResult, transcript) when all five fields are ints ≥ 0 summing > 0.

    ``None`` means malformed: excluded from numerator and denominator.
    """
    try:
        usage = json.loads(raw)
    except (TypeError, ValueError):
        return None
    breakdown = usage.get("context_breakdown") if isinstance(usage, dict) else None
    messages = breakdown.get("message_breakdown") if isinstance(breakdown, dict) else None
    if not isinstance(messages, dict):
        return None
    values = [messages.get(field) for field in _BREAKDOWN_FIELDS]
    if not all(type(value) is int and value >= 0 for value in values):
        return None
    total = sum(values)
    return (values[1], total) if total > 0 else None


def transcript_composition(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Daily per-bot Σ toolResult / Σ transcript, with coverage and malformed."""
    days: dict[tuple[str, date], dict[str, int]] = defaultdict(
        lambda: {"turns": 0, "with_breakdown": 0, "eligible": 0, "num": 0, "den": 0})
    for row in rows:
        day = _aware(row["created_at"]).astimezone(timezone.utc).date()
        cell = days[(row["bot_id"], day)]
        cell["turns"] += 1
        raw = row["token_usage_json"]
        if not isinstance(raw, str) or "message_breakdown" not in raw:
            continue
        cell["with_breakdown"] += 1
        parsed = _breakdown(raw)
        if parsed is None:
            continue
        cell["eligible"] += 1
        cell["num"] += parsed[0]
        cell["den"] += parsed[1]
    series = []
    for (bot, day), cell in sorted(days.items()):
        series.append({
            "bot": bot, "day": day.isoformat(), "turns": cell["turns"],
            "eligible_turns": cell["eligible"],
            "malformed_turns": cell["with_breakdown"] - cell["eligible"],
            "coverage": round(cell["eligible"] / cell["turns"], 4),
            "tool_result_tokens": cell["num"], "transcript_tokens": cell["den"],
            "share": round(cell["num"] / cell["den"], 4) if cell["den"] else None,
        })
    return series


DIGEST_TOP = 10


def digest(report: Mapping[str, Any]) -> dict[str, Any]:
    """A compact model-facing view: top offenders, per-bot composition.

    Tool results are context too; the digest stays a few kilobytes so the
    report does not become the waste it measures.
    """
    unverified: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "tokens": 0})
    for item in report["totals"]["unverified"]:
        unverified[item["bot"]]["calls"] += item["calls"]
        unverified[item["bot"]]["tokens"] += item["tokens"]
    composition: dict[str, dict[str, int]] = defaultdict(
        lambda: {"turns": 0, "eligible_turns": 0, "malformed_turns": 0, "num": 0, "den": 0})
    for day in report["composition"]:
        cell = composition[day["bot"]]
        cell["turns"] += day["turns"]
        cell["eligible_turns"] += day["eligible_turns"]
        cell["malformed_turns"] += day["malformed_turns"]
        cell["num"] += day["tool_result_tokens"]
        cell["den"] += day["transcript_tokens"]

    def strip(item: Mapping[str, Any], drop: tuple[str, ...]) -> dict[str, Any]:
        return {key: value for key, value in item.items() if key not in drop}

    return {
        "status": report["status"], "window": report["window"], "bots": report["bots"],
        "accounting_cutover": report["accounting_cutover"], "labels": report["labels"],
        "top_tools": report["totals"]["trusted"][:DIGEST_TOP],
        "unverified_by_bot": [{"bot": bot, **cell} for bot, cell in sorted(unverified.items())],
        "repeats": {
            scope: [strip(item, ("first_seen", "last_seen")) for item in report["repeats"][scope][:DIGEST_TOP]]
            for scope in ("in_session", "cross_session", "cross_tool")
        },
        "oversized": [{**strip(entry, ("top",)), "top": entry["top"][:2]}
                      for entry in report["oversized"][:DIGEST_TOP]],
        "composition_by_bot": [{
            "bot": bot, "turns": cell["turns"], "eligible_turns": cell["eligible_turns"],
            "malformed_turns": cell["malformed_turns"],
            "share": round(cell["num"] / cell["den"], 4) if cell["den"] else None,
        } for bot, cell in sorted(composition.items())],
        "truncated": report["truncated"],
    }


# ----- assembly ------------------------------------------------------------


def build_report(engine, scope: Scope, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """The whole report for one user, bot set and window."""
    totals = summarize_totals(queries.load_tool_totals(engine, scope))
    repeat_rows = queries.load_repeat_candidates(
        engine, scope, min_tokens=thresholds["repeat_min_tokens"])
    repeats_truncated = len(repeat_rows) > queries.MAX_REPEAT_ROWS
    repeats = find_repeats(repeat_rows[:queries.MAX_REPEAT_ROWS])
    examples = [item for key in ("in_session", "cross_session") for item in repeats[key]]
    shapes = queries.load_arguments(engine, scope, sorted({item["_example_id"] for item in examples}))
    for item in examples:
        item["arg_shape"] = arg_shape(item["tool"], shapes.get(item.pop("_example_id")))

    smallest = min([thresholds["default"], *thresholds["per_tool"].values()])
    oversized_rows = queries.load_oversized_candidates(engine, scope, min_tokens=smallest)
    oversized_truncated = len(oversized_rows) > queries.MAX_OVERSIZED_ROWS
    turn_rows = queries.load_turn_usage(engine, scope)
    turns_truncated = len(turn_rows) > queries.MAX_TURN_ROWS
    return {
        "status": "ok",
        "window": {"since": _iso(scope.since), "until": _iso(scope.until)},
        "bots": list(scope.bots),
        "accounting_cutover": _iso(queries.load_accounting_cutover(engine, scope)),
        "thresholds": thresholds,
        "labels": LABELS,
        "totals": totals,
        "repeats": repeats,
        "oversized": find_oversized(oversized_rows[:queries.MAX_OVERSIZED_ROWS], thresholds),
        "composition": transcript_composition(turn_rows[:queries.MAX_TURN_ROWS]),
        "truncated": {"repeats": repeats_truncated, "oversized": oversized_truncated,
                      "turns": turns_truncated},
    }
