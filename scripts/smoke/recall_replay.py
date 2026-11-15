"""Replay recorded ``context_recall`` calls against the current recall code.

Opt-in measurement tool (TASK-1557), not a test. It re-runs real recall
requests read from ``tool_call_records`` under each original caller's identity
and writes one JSON line per call: status, coverage, finding quotes and, when an
expectation file is given, whether the answer reached the admitted evidence and
the returned findings.

It runs the real retrieval and the real utility model (``maintenance_model``),
so it needs the app's database and service environment. It is read-only by the
recall contract: no memory, message, task or session writes. The caller's scope
comes from the current bot-tag policy exactly as a live call would; a protected
or non-searchable caller is refused, never replayed. Follow-up handles are not
replayable (they are process-local), so the recorded question runs alone.

Usage, inside the app container so the environment matches the MCP server::

    docker exec -i llm-bawt-app /app/.venv/bin/python - --ids 1,2,3 \
        --expect-json '{"1": ["marker"], "3": null}' < scripts/smoke/recall_replay.py

``--system-file`` sends a candidate ``recall.synthesis.system`` body instead of
the registry's, so a prompt change can be measured before it is saved.
Relative dates in a question ("yesterday") resolve at the original call time.

Expectations map a tool-call id to answer markers (case-insensitive literals,
any one suffices) or ``null`` for "no answer exists". A call with markers hits
when the status is ``ok`` and a finding quote contains a marker; a ``null`` call
hits when the status is ``no_match`` or ``insufficient_evidence``.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import time

from sqlalchemy import text


@dataclass(frozen=True)
class ReplayContext:
    bot_id: str
    user_id: str
    session_id: str
    issued_at: float


CALL_SQL = """
    SELECT t.id, t.bot_id, t.user_id, t.arguments_json, m.session_id::text AS session_id,
           extract(epoch FROM t.created_at) AS called_at
    FROM tool_call_records t
    LEFT JOIN turn_logs turns ON turns.id = t.turn_id
    LEFT JOIN messages m ON m.id = turns.trigger_message_id AND m.bot_id = t.bot_id
    WHERE t.id = ANY(:ids) AND t.tool_name LIKE '%context_recall'
    ORDER BY t.id
"""


def _recorder(synthesizer, system: str | None = None):
    from llm_bawt.recall.contracts import Synthesizer

    class Recording(Synthesizer):
        """Delegates to the real synthesizer and keeps the admitted evidence."""

        def __init__(self):
            self.evidence: list = []

        def system_prompt(self, bot_id: str) -> str:
            return system if system is not None else synthesizer.system_prompt(bot_id)

        async def synthesize(self, system, query, evidence, output_tokens):
            self.evidence = list(evidence)
            return await synthesizer.synthesize(system, query, evidence, output_tokens)

    return Recording()


def grade(result: dict, evidence: list, markers) -> dict:
    status = result.get("status")
    if markers is None:
        return {"expected": "absent", "hit": status in {"no_match", "insufficient_evidence"}}
    # "@bot" markers answer "who" questions: they match a finding's source bot,
    # not its quote text.
    bots = {marker[1:].lower() for marker in markers if marker.startswith("@")}
    folded = [marker.lower() for marker in markers if not marker.startswith("@")]
    findings = result.get("findings", [])
    quotes = " ".join(item.get("quote", "") for item in findings).lower()
    cited_bots = {str((item.get("provenance") or {}).get("source_bot") or "").lower()
                  for item in findings}
    in_evidence = (any(marker in row.text.lower() for row in evidence for marker in folded)
                   or any(str(row.source_bot or "").lower() in bots for row in evidence))
    found = any(marker in quotes for marker in folded) or bool(bots & cited_bots)
    return {"expected": "present", "in_evidence": in_evidence,
            "hit": status == "ok" and found}


def with_sources(raw: dict, extra: tuple[str, ...]) -> dict:
    """The recorded request, searching ``extra`` sources as well (deep depth)."""
    if not extra:
        return raw
    from llm_bawt.recall.contracts import DEFAULT_SOURCES
    sources = tuple(raw.get("sources") or DEFAULT_SOURCES)
    return {**raw, "sources": list(dict.fromkeys((*sources, *extra)))}


async def replay(ids: list[int], expectations: dict, extra: tuple[str, ...] = (),
                 system: str | None = None) -> None:
    from llm_bawt.mcp_server.recall_tools import _runtime
    from llm_bawt.recall.contracts import RecallError, RecallRequest
    from llm_bawt.recall.service import RecallService

    database, live = _runtime()
    with database.engine.connect() as connection:
        calls = [dict(row) for row in connection.execute(text(CALL_SQL), {"ids": ids}).mappings()]
    recorder = _recorder(live.synthesizer, system)
    messages = live.providers.get("messages")
    service = RecallService(list(live.providers.values()), recorder)
    hits = graded = 0
    for call in calls:
        recorder.evidence = []
        if messages is not None and hasattr(messages, "clock"):
            messages.clock = (lambda at=float(call["called_at"]): at)
        raw = json.loads(call["arguments_json"] or "{}").get("request", {})
        raw = {key: value for key, value in raw.items() if key != "handle" and value is not None}
        raw = with_sources(raw, extra)
        line = {"id": call["id"], "bot": call["bot_id"], "query": raw.get("query"),
                "sources": raw.get("sources")}
        try:
            request = RecallRequest.model_validate(raw)
            context = ReplayContext(call["bot_id"], call["user_id"],
                                    call["session_id"] or "replay", time.time())
            scope = await database.scope(context)
            result = await service.recall(request, scope)
        except RecallError as error:
            result = {"status": "error", "error_code": error.code, "findings": []}
        line.update(status=result.get("status"), error_code=result.get("error_code"),
                    coverage=result.get("coverage"), truncated=result.get("truncated"),
                    uncertainty=result.get("uncertainty"),
                    evidence=[f"{row.source}:{row.id}" for row in recorder.evidence],
                    quotes=[f"{item['source']}:{item['id']}: {item['quote']}"
                            for item in result.get("findings", [])])
        key = str(call["id"])
        if key in expectations:
            line["grade"] = grade(result, recorder.evidence, expectations[key])
            graded += 1
            hits += line["grade"]["hit"]
        print(json.dumps(line, ensure_ascii=False), flush=True)
    if graded:
        print(json.dumps({"summary": {"graded": graded, "hits": hits}}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", required=True, help="comma-separated tool_call_records ids")
    parser.add_argument("--expect-json", default="{}", help="id -> markers list or null")
    parser.add_argument("--extra-sources", default="",
                        help="comma-separated sources added to every request (deep depth)")
    parser.add_argument("--system-file", default=None,
                        help="candidate synthesis system prompt (default: the registry's)")
    args = parser.parse_args()
    system = None
    if args.system_file:
        with open(args.system_file, encoding="utf-8") as handle:
            system = handle.read()
    ids = [int(value) for value in args.ids.split(",") if value.strip()]
    extra = tuple(value.strip() for value in args.extra_sources.split(",") if value.strip())
    asyncio.run(replay(ids, json.loads(args.expect_json), extra, system))


if __name__ == "__main__":
    main()
