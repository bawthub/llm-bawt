"""Bounded read-only tool records under the messages_find provenance boundary."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text

from .find import FindScope, _ReadSnapshot, _SCOPE_FROM, serialize
from .tool_call_scrub import arguments_view, clip, scrub_text

LIST_BYTES = 4000
DETAIL_BYTES = 16000
ARG_READ_CHARS = 65536
_EMPTY = {"status": "no_match", "turns": [], "next": None,
          "scrubbing": "best_effort", "partial_turns": True}
_FROM = _SCOPE_FROM + """
    JOIN turn_logs turns ON turns.trigger_message_id = m.id
      AND turns.bot_id = p.bot_id AND turns.user_id = :authority_user
    JOIN tool_call_records t ON t.turn_id = turns.id
      AND t.bot_id = turns.bot_id AND t.user_id = turns.user_id
"""
_COLUMNS = """
    t.id, t.created_at, t.turn_id, t.bot_id, left(t.tool_name, 128) AS tool_name,
    t.is_error, t.duration_ms, left(t.approval_status, 32) AS approval_status,
    t.result_total_chars, t.result_total_bytes, left(t.result_sha256, 64) AS result_sha256,
    left(t.result_content_type, 128) AS result_content_type, t.result_complete
"""


@dataclass(frozen=True)
class CallRequest:
    bot: str | None = None
    turn_id: str | None = None
    id: int | None = None
    since: float | None = None
    until: float | None = None
    tool_name: str | None = None
    errors_only: bool = False
    limit: int = 10
    before: str | None = None
    include_result: bool = False
    max_chars: int = 4000


def _wire_size(value) -> int:
    return len(serialize(value).encode("utf-8"))


def _cursor(row) -> str:
    return f"{row['created_at'].isoformat()}|{row['id']}"


def _row(row, args) -> dict:
    return {
        "id": row["id"], "at": row["created_at"].isoformat(),
        "tool": scrub_text(row["tool_name"]), "args": args,
        "is_error": row["is_error"], "duration_ms": row["duration_ms"],
        "approval": row["approval_status"], "result_chars": row["result_total_chars"],
        "result_bytes": row["result_total_bytes"],
    }


class ToolCallLookup:
    def __init__(self, engine, *, clock=time.time):
        self.engine = engine
        self.clock = clock

    def lookup(self, access, request: CallRequest, *, trigger_id=None) -> str:
        if not 1 <= request.limit <= 25 or not 1000 <= request.max_chars <= 8000:
            raise ValueError("invalid bounds")
        if request.id is not None and (request.id <= 0 or request.turn_id
                or request.tool_name or request.before or request.since is not None
                or request.until is not None or request.errors_only):
            raise ValueError("id cannot be combined with list filters")
        if request.include_result and request.id is None:
            raise ValueError("results require id")
        if request.bot is not None and len(request.bot) > 128:
            raise ValueError("bot too long")
        if request.turn_id is not None and len(request.turn_id) > 256:
            raise ValueError("turn too long")
        if request.tool_name is not None and len(request.tool_name) > 128:
            raise ValueError("tool too long")
        if request.before is not None and len(request.before) > 128:
            raise ValueError("cursor too long")
        scope = FindScope.build(access, request.bot, trigger_id)
        if not scope.partitions:
            return serialize({"status": "not_found"} if request.id else
                             _EMPTY)
        params = scope.params()
        where = ["c.speaker IS NOT NULL"]
        if trigger_id:
            where.append("m.id <> :trigger")
            params["trigger"] = trigger_id
        if request.id is not None:
            where.append("t.id = :record")
            params["record"] = request.id
        else:
            now = self.clock()
            since = request.since if request.since is not None else (
                None if request.turn_id else now - 86400)
            until = request.until if request.until is not None else now
            if since is not None and since > until:
                raise ValueError("invalid time window")
            for name, op, value in (("since", ">=", since), ("until", "<=", until)):
                if value is not None:
                    where.append(f"t.created_at {op} :{name}")
                    params[name] = datetime.fromtimestamp(value, timezone.utc)
            if request.turn_id is not None:
                where.append("t.turn_id = :turn")
                params["turn"] = request.turn_id
            if request.tool_name is not None:
                where.append("t.tool_name = :tool")
                params["tool"] = request.tool_name
            if request.errors_only:
                where.append("t.is_error IS TRUE")
            if request.before:
                at, record = request.before.rsplit("|", 1)
                parsed = datetime.fromisoformat(at)
                if parsed.tzinfo is None or int(record) <= 0:
                    raise ValueError("invalid cursor")
                where.append("(t.created_at, t.id) < (:before_at, :before_id)")
                params.update(before_at=parsed, before_id=int(record))
        params["limit"] = 1 if request.id else request.limit + 1
        with _ReadSnapshot(self.engine) as conn:
            # Authorization precedes reading args, previews, payloads or blobs.
            rows = conn.execute(text(
                "SELECT " + _COLUMNS + _FROM + " WHERE " + " AND ".join(where)
                + " ORDER BY t.created_at DESC, t.id DESC LIMIT :limit"
            ), params).mappings().all()
            if request.id is not None:
                if not rows:
                    return serialize({"status": "not_found"})
                row = rows[0]
                result = {"status": "ok", "bot": row["bot_id"], "turn_id": row["turn_id"],
                          "call": _row(row, self._args(conn, row, request, detail=True)),
                          "scrubbing": "best_effort"}
                if request.include_result:
                    result["result"] = self._result(conn, row, request.max_chars)
                return self._fit_detail(result)
            groups, accepted = [], []
            for row in rows[:request.limit]:
                call = _row(row, self._args(conn, row, request, detail=False))
                if not groups or (groups[-1]["bot"], groups[-1]["turn_id"]) != (row["bot_id"], row["turn_id"]):
                    groups.append({"bot": row["bot_id"], "turn_id": row["turn_id"], "calls": []})
                groups[-1]["calls"].append(call)
                candidate = {"status": "ok", "turns": groups, "next": {
                    "before": _cursor(row), "since": since, "until": until},
                    "scrubbing": "best_effort", "partial_turns": True}
                if _wire_size(candidate) > LIST_BYTES:
                    groups[-1]["calls"].pop()
                    if not groups[-1]["calls"]:
                        groups.pop()
                    break
                accepted.append(row)
            has_more = len(rows) > len(accepted)
            next_page = ({"before": _cursor(accepted[-1]), "since": since, "until": until}
                         if accepted and has_more else None)
            return serialize({"status": "ok" if groups else "no_match", "turns": groups,
                              "next": next_page, "scrubbing": "best_effort",
                              "partial_turns": True})

    @staticmethod
    def _args(conn, row, request, *, detail):
        raw = conn.execute(text("SELECT left(arguments_json, :cap) AS args, "
                                "length(arguments_json) AS total FROM tool_call_records WHERE id=:id"),
                           {"cap": ARG_READ_CHARS, "id": row["id"]}).mappings().one()
        schema = None
        # Reuse the operation schema's x-sensitive masks, in addition to the
        # generic hygiene. Unknown/non-ops tools still get best-effort scrubbing.
        if row["tool_name"].rsplit("__", 1)[-1] == "ops_run":
            try:
                args = json.loads(raw["args"] or "{}")
            except (ValueError, RecursionError):
                args = {}
            if isinstance(args, dict) and isinstance(args.get("operation"), str):
                schema = conn.execute(text("SELECT args_schema_json FROM ops_operations WHERE slug=:slug"),
                                      {"slug": args["operation"]}).scalar()
                if schema is not None:
                    schema = serialize({"properties": {"args": json.loads(schema)}})
            elif (raw["total"] or 0) > ARG_READ_CHARS:
                # A cut JSON document cannot apply the operation's schema masks.
                return {"truncated": True, "source_truncated": True,
                        "total_chars": raw["total"], "reason": "oversized_ops_args"}
        view = arguments_view(raw["args"], field_chars=400 if detail else 48,
                              max_chars=min(request.max_chars, 2400) if detail else 160,
                              schema_json=schema)
        view["total_chars"] = raw["total"]
        view["source_truncated"] = (raw["total"] or 0) > ARG_READ_CHARS
        return view

    @staticmethod
    def _result(conn, row, max_chars):
        cap = max_chars * 4 + 4
        source = conn.execute(text(
            "SELECT left(result_text, :cap) AS preview, result_blob_key, result_payload_available "
            "FROM tool_call_records WHERE id=:id"), {"cap": cap, "id": row["id"]}).mappings().one()
        content, integrity, whole = None, "unverified", False
        if source["result_blob_key"]:
            from llm_bawt.service.tool_call_store import _get_tool_blob_backend
            try:
                backend = _get_tool_blob_backend()
                if backend is None:
                    return {"status": "unavailable", "reason": "result_source_unavailable"}
                blob = backend.get_range(source["result_blob_key"], f"bytes=0-{cap - 1}")
                content = blob.data.decode("utf-8", errors="replace")
                whole = blob.total_size == len(blob.data)
                if whole and row["result_sha256"]:
                    integrity = ("verified" if hashlib.sha256(blob.data).hexdigest() == row["result_sha256"]
                                 else "mismatch")
            except Exception:
                return {"status": "unavailable", "reason": "result_source_unavailable"}
        elif source["result_payload_available"]:
            payload = conn.execute(text(
                "SELECT left(content_text, :cap) AS content, length(content_text) AS total "
                "FROM tool_call_result_payloads WHERE tool_call_record_id=:id"),
                {"cap": cap, "id": row["id"]}).mappings().first()
            if payload:
                content, whole = payload["content"], payload["total"] <= cap
        else:
            content = source["preview"]
            whole = content is not None and row["result_total_chars"] == len(content)
        if whole and content is not None and row["result_sha256"] and integrity == "unverified":
            integrity = ("verified" if hashlib.sha256(content.encode("utf-8")).hexdigest()
                         == row["result_sha256"] else "mismatch")
        if integrity == "mismatch":
            return {"status": "unavailable", "reason": "integrity_mismatch"}
        if content is None:
            return {"status": "unavailable", "reason": "result_not_stored"}
        # Apply the same structured-key/string hygiene to JSON results.
        cleaned = arguments_view(content, field_chars=max_chars, max_chars=max_chars)
        rendered = (serialize(cleaned["value"]) if "value" in cleaned and not isinstance(cleaned["value"], str)
                    else cleaned.get("value", cleaned.get("text", "")))
        return {"text": clip(rendered, max_chars), "total_chars": row["result_total_chars"],
                "sha256": row["result_sha256"], "content_type": row["result_content_type"],
                "complete": row["result_complete"], "integrity": integrity,
                "truncated": cleaned["truncated"] or not whole
                             or (row["result_total_chars"] or len(content)) > max_chars
                             or len(rendered) > max_chars,
                "scrubbing": "best_effort"}

    @staticmethod
    def _fit_detail(result):
        # Unicode/control-character JSON escaping can expand a bounded field.
        # Fit the whole reply, not just the unescaped strings.
        while _wire_size(result) > DETAIL_BYTES:
            body = result.get("result", {}).get("text", "")
            if len(body) > 1:
                result["result"]["text"] = clip(body, max(1, len(body) // 2))
                result["result"]["truncated"] = True
            else:
                result["call"]["args"] = {"truncated": True}
                break
        return serialize(result)
