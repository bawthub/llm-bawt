"""Per-bot tool-result waste report (TASK-1614).

Ranks tool x bot rows by estimated token-request exposure so the next fixes
(cache/trim a static payload, prompt guidance against refetches, trim an
oversized payload) can be routed to an owner. Read-only over rows that are
already stored; computed on request, no rollup table.

Only aggregate numbers leave this module: never arguments, results, commands
or hashes.

Definitions (see the TASK-1614 design):

* Epoch: a run of turns sharing one non-null ``turn_logs.agent_session_key``.
  A turn whose ``token_usage_json`` records a successful compaction starts a
  new epoch. A turn with a null key is its own epoch. The window start also
  starts an epoch (earlier turns are not fetched).
* Scope: a call's producing response (``tool_call_records.response_id`` ->
  ``llm_response_records``). ``host_parent`` null = main stream; anything else
  is a worker result, whose exposure/refetch status is unknown and unranked.
* Verified tokens: ``result_accounting`` equals the current producer version.
* Refetch candidate: a later main-stream call in the same epoch with the same
  tool and the same non-null hash.
* Exposure: verified tokens x the number of later main-stream responses in the
  same epoch. Order = (turn created_at, turn id, first_seen_seq, response_id);
  ``first_seen_seq`` is only meaningful within a turn. Epochs with a turn that
  has no response rows, or with an unlinked call, have unknown exposure.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import bindparam, text

from agent_bridge.tool_results import TOOL_RESULT_ACCOUNTING_VERSION

MAX_DAYS = 30
TOP_N = 20
LOW_SAMPLE_CALLS = 20
STATIC_SHARE = 0.8
STATIC_MIN_CALLS = 50
STATIC_MIN_SESSIONS = 5
REFETCH_SHARE = 0.3
OVERSIZED_P95_TOKENS = 8000

# First chip wins the owner, in this order.
CLASS_ORDER = ("oversized", "static", "refetch")
OWNERS = {
    "oversized": "tool coder (payload trim)",
    "static": "tool coder (cache or trim)",
    "refetch": "prompt registry and bot coordinator",
}


def _compacted(token_usage_json: str | None) -> bool:
    try:
        usage = json.loads(token_usage_json) if token_usage_json else None
    except (TypeError, ValueError):
        return False
    return (isinstance(usage, dict) and usage.get("context_action") == "compact"
            and usage.get("context_action_outcome") == "success")


def fetch(conn, *, days: int, protected: Sequence[str], now: datetime | None = None
          ) -> tuple[list[dict], list[dict], list[dict]]:
    """Windowed rows for :func:`compute`, protected and bot-less rows excluded in SQL."""
    days = max(1, min(int(days), MAX_DAYS))
    since = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    prot = sorted({p.lower() for p in protected}) or ["\x00"]
    params = {"since": since, "prot": prot}
    calls = conn.execute(text("""
        SELECT c.bot_id, c.turn_id, c.tool_name, c.response_id, c.result_total_tokens,
               c.result_accounting, c.result_sha256
        FROM tool_call_records c
        WHERE c.created_at >= :since AND c.bot_id IS NOT NULL
          AND lower(c.bot_id) NOT IN :prot
    """).bindparams(bindparam("prot", expanding=True)), params).mappings().all()
    turns = conn.execute(text("""
        SELECT t.id, t.bot_id, t.created_at, t.agent_session_key, t.token_usage_json
        FROM turn_logs t
        WHERE t.created_at >= :since AND t.bot_id IS NOT NULL
          AND lower(t.bot_id) NOT IN :prot
    """).bindparams(bindparam("prot", expanding=True)), params).mappings().all()
    responses = conn.execute(text("""
        SELECT r.turn_id, r.response_id, r.host_parent, r.first_seen_seq
        FROM llm_response_records r JOIN turn_logs t ON t.id = r.turn_id
        WHERE t.created_at >= :since AND t.bot_id IS NOT NULL
          AND lower(t.bot_id) NOT IN :prot
    """).bindparams(bindparam("prot", expanding=True)), params).mappings().all()
    return [dict(r) for r in calls], [dict(r) for r in turns], [dict(r) for r in responses]


def _p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def compute(calls: Iterable[Mapping[str, Any]], turns: Iterable[Mapping[str, Any]],
            responses: Iterable[Mapping[str, Any]], *, protected: Sequence[str] = ()) -> dict:
    """Per-bot totals and ranked tool x bot rows. Pure; inputs are fetched rows."""
    prot = {p.lower() for p in protected}
    calls = [c for c in calls if c.get("bot_id") and c["bot_id"].lower() not in prot]
    turns = [t for t in turns if t.get("bot_id") and t["bot_id"].lower() not in prot]
    turn_by_id = {t["id"]: t for t in turns}

    # Epochs: per session key, turns in (created_at, id) order; compaction splits.
    epoch_of: dict[str, tuple] = {}
    turn_rank: dict[str, int] = {}
    session_of: dict[str, str] = {}
    by_session: dict[str, list] = defaultdict(list)
    for t in turns:
        key = t.get("agent_session_key")
        session = f"s:{key}" if key else f"t:{t['id']}"
        session_of[t["id"]] = session
        by_session[session].append(t)
    for session, ts in by_session.items():
        ts.sort(key=lambda t: (t["created_at"], t["id"]))
        n = 0
        for i, t in enumerate(ts):
            if i and _compacted(t.get("token_usage_json")):
                n += 1
            epoch_of[t["id"]] = (session, n)
            turn_rank[t["id"]] = i

    resp_by_key: dict[tuple, Mapping] = {}
    turns_with_resp: set[str] = set()
    main_orders: dict[tuple, list] = defaultdict(list)
    for r in responses:
        if r["turn_id"] not in turn_by_id:
            continue
        resp_by_key[(r["turn_id"], r["response_id"])] = r
        turns_with_resp.add(r["turn_id"])
        if r.get("host_parent") is None:
            order = (turn_rank[r["turn_id"]], r["first_seen_seq"], r["response_id"])
            main_orders[epoch_of[r["turn_id"]]].append(order)
    for orders in main_orders.values():
        orders.sort()

    # An epoch is incomplete if any of its turns lacks response rows or a call is unlinked.
    incomplete: set[tuple] = {epoch_of[t] for t in turn_by_id if t not in turns_with_resp}

    rows: list[dict] = []
    for c in calls:
        turn = turn_by_id.get(c.get("turn_id"))
        resp = resp_by_key.get((c.get("turn_id"), c.get("response_id"))) if turn else None
        epoch = epoch_of.get(c.get("turn_id")) if turn else None
        if epoch is not None and resp is None:
            incomplete.add(epoch)
        tokens = c.get("result_total_tokens")
        verified = tokens is not None and c.get("result_accounting") == TOOL_RESULT_ACCOUNTING_VERSION
        scope = None if resp is None else ("main" if resp.get("host_parent") is None else "worker")
        rows.append({"bot": c["bot_id"], "tool": c.get("tool_name") or "unknown",
                     "tokens": tokens, "verified": verified, "hash": c.get("result_sha256"),
                     "epoch": epoch, "scope": scope, "resp": resp,
                     "session": session_of.get(c.get("turn_id"))})

    # Refetch candidates: main stream, same epoch + tool + non-null hash, after the first.
    seen: set[tuple] = set()
    main_sorted = sorted(
        (r for r in rows if r["scope"] == "main"),
        key=lambda r: (turn_rank[r["resp"]["turn_id"]], r["resp"]["first_seen_seq"],
                       r["resp"]["response_id"]))
    for r in main_sorted:
        if r["hash"] is None:
            continue
        k = (r["epoch"], r["tool"], r["hash"])
        r["refetch"] = k in seen
        seen.add(k)

    for r in rows:
        r["exposure"] = None
        if r["scope"] == "main" and r["verified"] and r["epoch"] not in incomplete:
            resp = r["resp"]
            orders = main_orders[r["epoch"]]
            later = len(orders) - bisect_right(
                orders, (turn_rank[resp["turn_id"]], resp["first_seen_seq"], resp["response_id"]))
            r["exposure"] = r["tokens"] * later

    bots: dict[str, dict] = {}
    for r in rows:
        b = bots.setdefault(r["bot"], {
            "bot_id": r["bot"], "calls": 0, "verified_tokens": 0, "unverified_tokens": 0,
            "calls_without_tokens": 0, "calls_without_hash": 0, "worker_calls": 0,
            "worker_verified_tokens": 0, "refetch_candidates": 0, "refetch_candidate_tokens": 0,
            "exposure_tokens": 0, "exposure_unknown_calls": 0})
        b["calls"] += 1
        if r["tokens"] is None:
            b["calls_without_tokens"] += 1
        elif r["verified"]:
            b["verified_tokens"] += r["tokens"]
        else:
            b["unverified_tokens"] += r["tokens"]
        if r["hash"] is None:
            b["calls_without_hash"] += 1
        if r["scope"] == "worker":
            b["worker_calls"] += 1
            if r["verified"]:
                b["worker_verified_tokens"] += r["tokens"]
        if r.get("refetch"):
            b["refetch_candidates"] += 1
            if r["verified"]:
                b["refetch_candidate_tokens"] += r["tokens"]
        if r["exposure"] is None:
            b["exposure_unknown_calls"] += 1
        else:
            b["exposure_tokens"] += r["exposure"]

    # Static is a property of the tool across all (visible) bots.
    by_tool: dict[str, list] = defaultdict(list)
    for r in rows:
        by_tool[r["tool"]].append(r)
    static_tools = set()
    for tool, rs in by_tool.items():
        hashes = Counter(r["hash"] for r in rs if r["hash"] is not None)
        # Calls without a fetched turn add no session diversity.
        sessions = {r["session"] for r in rs if r["session"] is not None}
        if (len(rs) >= STATIC_MIN_CALLS and len(sessions) >= STATIC_MIN_SESSIONS and hashes
                and hashes.most_common(1)[0][1] / len(rs) >= STATIC_SHARE):
            static_tools.add(tool)

    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["tool"], r["bot"])].append(r)
    ranked, low_sample = [], []
    for (tool, bot), rs in groups.items():
        known = [r for r in rs if r["exposure"] is not None]
        if not known:
            continue  # unverified, worker or unlinked only: never ranked
        main = [r for r in rs if r["scope"] == "main"]
        refetches = sum(1 for r in main if r.get("refetch"))
        p95 = _p95([r["tokens"] for r in rs if r["verified"]])
        row = {"tool_name": tool, "bot_id": bot, "calls": len(rs), "ranked_calls": len(known),
               "hash_coverage": round(sum(1 for r in rs if r["hash"] is not None) / len(rs), 4),
               "verified_tokens": sum(r["tokens"] for r in rs if r["verified"]),
               "exposure_tokens": sum(r["exposure"] for r in known),
               "refetch_candidates": refetches, "p95_verified_tokens": p95,
               "classes": [], "owner": None, "low_sample": len(rs) < LOW_SAMPLE_CALLS}
        if row["low_sample"]:
            low_sample.append(row)
            continue
        met = {"oversized": p95 is not None and p95 >= OVERSIZED_P95_TOKENS,
               "static": tool in static_tools,
               "refetch": bool(main) and refetches / len(main) >= REFETCH_SHARE}
        row["classes"] = [c for c in CLASS_ORDER if met[c]]
        row["owner"] = OWNERS[row["classes"][0]] if row["classes"] else None
        ranked.append(row)
    order = lambda r: (-r["exposure_tokens"], r["bot_id"], r["tool_name"])  # noqa: E731
    ranked.sort(key=order)
    low_sample.sort(key=order)
    return {"bots": sorted(bots.values(), key=lambda b: (-b["verified_tokens"], b["bot_id"])),
            "ranked": ranked[:TOP_N], "low_sample": low_sample[:TOP_N]}


def report(conn, *, days: int, protected: Sequence[str], now: datetime | None = None) -> dict:
    calls, turns, responses = fetch(conn, days=days, protected=protected, now=now)
    return compute(calls, turns, responses, protected=protected)
