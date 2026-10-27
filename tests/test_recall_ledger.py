"""Recall evidence ledger and UI drill-down, hermetic (TASK-1519).

The ledger records every row a recall accepted, as references only, so the
BawtHub card can list what was read without growing the bot payload. These
tests pin: what is written and when, that a sink can never change or delay
the packet, and that the drill-down re-applies the current policy.
"""
import asyncio
import threading
from contextlib import asynccontextmanager

import pytest

from llm_bawt.recall.budgets import compact_json
from llm_bawt.recall.contracts import Evidence, RecallError, RecallRequest, RecallScope
from llm_bawt.recall.drilldown import EvidenceDrilldown, visible
from llm_bawt.recall.ledger import ENTRY_FIELDS, LedgerRecord, LedgerSink
from llm_bawt.recall.service import RecallService
from llm_bawt.recall.task_source import TaskSource
from tests.test_isolated_recall import SCOPE, Model, Source

OWNER = "reviewer"
USER = "operator"


class Sink(LedgerSink):
    def __init__(self, error=None, held=False):
        self.writes = []
        self.error = error
        # Held sinks park the worker thread until the test releases it.
        self.held = held
        self.entered = threading.Event()
        self.release = threading.Event()
        self.released = []

    def write(self, handle, scope, coverage, entries, omitted):
        self.entered.set()
        if self.held:
            # Records whether the write really waited for the release; the
            # 30 s bound is only a hang guard.
            self.released.append(self.release.wait(30))
        if self.error:
            raise self.error
        self.writes.append((handle, scope, coverage, entries, omitted))


def named(source, rows, max_records=None):
    provider = Source(rows)
    provider.source = source
    if max_records:
        provider.max_records = max_records
    return provider


def message(i, follows=None, text="The relay background is cool glass."):
    return Evidence("messages", f"m{i}", OWNER, USER, text, follows=follows,
                    source_bot="peer", session_id="s1", at=f"2026-10-2{i % 9}T10:00:00.000000Z",
                    speaker="assistant")


async def run(service, request, scope=SCOPE):
    packet = await service.recall(request, scope)
    await service.drain_ledger()
    return packet


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,status", [
    (None, "ok"),
    ('{"findings":[]}', "insufficient_evidence"),
])
async def test_answered_and_insufficient_recalls_record_every_accepted_row(raw, status):
    sink = Sink()
    rows = [message(1), message(2, text="Second relay background note.")]
    service = RecallService([named("messages", rows)], Model(raw), ledger=sink)
    packet = await run(service, RecallRequest(query="relay background", sources=("messages",)))
    assert packet["status"] == status
    [(handle, scope, coverage, entries, omitted)] = sink.writes
    assert handle == packet["handle"] and scope == SCOPE and omitted == 0
    assert coverage == packet["coverage"] and coverage is not packet["coverage"]
    assert len(entries) == packet["coverage"]["messages"]["records"] == 2
    assert all(set(entry) == {*ENTRY_FIELDS, "admitted", "cited"} for entry in entries)
    assert all("text" not in entry for entry in entries), "references only"
    assert [entry["admitted"] for entry in entries] == [True, True]
    assert [entry["cited"] for entry in entries] == ([True, False] if status == "ok" else [False, False])
    assert entries[0]["source_bot"] == "peer" and entries[0]["speaker"] == "assistant"


@pytest.mark.asyncio
async def test_no_match_records_the_empty_read_and_errors_record_nothing():
    sink = Sink()
    service = RecallService([named("messages", [])], Model(), ledger=sink)
    packet = await run(service, RecallRequest(query="relay background", sources=("messages",)))
    assert packet["status"] == "no_match"
    assert sink.writes[0][3] == [] and sink.writes[0][2] == {"messages": packet["coverage"]["messages"]}

    failed = Sink()
    for provider, model in ((named("messages", [message(1)]), Model("not json")),
                            (Source(error=RecallError("source_unavailable")), Model())):
        service = RecallService([provider], model, ledger=failed)
        packet = await run(service, RecallRequest(query="relay background", sources=("messages",)))
        assert packet["status"] == "error" and packet["handle"] is None
    assert failed.writes == []


def without_elapsed(packet):
    clone = {**packet, "metrics": {**packet["metrics"], "elapsed_ms": 0, "parent_tokens": 0}, "handle": "h"}
    return compact_json(clone)


@pytest.mark.asyncio
async def test_failing_or_hanging_sink_never_changes_or_delays_the_packet(monkeypatch):
    # The write's own 2 s budget must not end the hold before the release.
    monkeypatch.setattr("llm_bawt.recall.service.LEDGER_WRITE_SECONDS", 60)
    request = RecallRequest(query="relay background", sources=("messages",))
    plain = await run(RecallService([named("messages", [message(1)])], Model()), request)

    broken = RecallService([named("messages", [message(1)])], Model(), ledger=Sink(error=RuntimeError("db down")))
    assert without_elapsed(await run(broken, request)) == without_elapsed(plain)

    hanging = Sink(held=True)
    service = RecallService([named("messages", [message(1)])], Model(), ledger=hanging)
    packet = await service.recall(request, SCOPE)
    # The packet is final, and recall has returned, while the write is still
    # parked in its worker thread.
    assert without_elapsed(packet) == without_elapsed(plain)
    assert await asyncio.to_thread(hanging.entered.wait, 30), "the write never started"
    assert hanging.writes == [] and hanging.released == []
    pending = list(service._ledger_writes)
    assert len(pending) == 1 and not pending[0].done()
    hanging.release.set()
    await service.drain_ledger()
    # The write completed only because it was released, not by timing out.
    assert hanging.released == [True]
    assert len(hanging.writes) == 1


@pytest.mark.asyncio
async def test_a_full_four_source_recall_records_all_33_rows():
    hits = [message(i) for i in range(6)]
    follow = [message(10 + 2 * i + j, follows=f"m{i}") for i in range(6) for j in range(2)]
    memories = [Evidence("memories", f"mem{i}", OWNER, USER, f"relay memory {i}") for i in range(6)]
    tools = [Evidence("tool_evidence", str(i), OWNER, USER, '{"tool_name":"relay"}') for i in range(6)]
    tasks = [Evidence("tasks", f"TASK-{i + 1}", OWNER, USER, f"TASK-{i + 1}: relay") for i in range(3)]
    sink = Sink()
    service = RecallService([named("messages", hits + follow, 18), named("memories", memories),
                             named("tool_evidence", tools), named("tasks", tasks)], Model(), ledger=sink)
    packet = await run(service, RecallRequest(
        query="relay background", sources=("messages", "memories", "tool_evidence", "tasks")))
    [(_, _, coverage, entries, omitted)] = sink.writes
    assert sum(item["records"] for item in coverage.values()) == len(entries) == 33
    assert omitted == 0
    # The model saw at most 24 records; the ledger says which.
    assert sum(entry["admitted"] for entry in entries) <= 24
    assert sum(entry["cited"] for entry in entries) == len({(f["source"], f["id"]) for f in packet["findings"]})


# --- drill-down -----------------------------------------------------------------

POLICY = frozenset({OWNER, "peer", "other"})


class Ledger:
    def __init__(self, record):
        self.record = record

    def read(self, handle, bot_id, user_id):
        record = self.record
        if record is None or handle != "h" * 20 or (bot_id, user_id) != (record.bot_id, record.user_id):
            return None
        return record


class Database:
    def __init__(self, allowed=POLICY, gone=()):
        self.allowed = allowed
        self.gone = set(gone)
        self.statements = []

    async def owner_scope(self, bot_id, user_id, session_id, expires_at=None):
        if bot_id not in self.allowed:
            raise RecallError("unauthorized")
        return RecallScope(bot_id, user_id, session_id, frozenset({"protected"}) | (POLICY - self.allowed),
                           self.allowed)

    def read(self, statement, params):
        self.statements.append(params)
        return [{"id": ref, "snippet": f"live {ref}", "more": False}
                for ref in params["ids"] if ref not in self.gone]


class Tasks(TaskSource):
    """Real authorized_task over a fake BawtHub; owners decide authority."""

    def __init__(self, owners):
        self.owners = owners

    @asynccontextmanager
    async def reader(self):
        owners = self.owners

        class Reader:
            async def json(self, method, url, **kwargs):
                if url.endswith("/associations"):
                    return {"sessions": []}
                if url.endswith("/activity"):
                    return {"activities": [], "total": 0}
                ref = url.rsplit("/", 1)[1]
                if owners[ref] == "unreachable":
                    raise RecallError("source_unavailable")
                return {"id": "c" + "a" * 24, "shortId": ref, "title": f"Title {ref}", "status": "REVIEW",
                        "agentBotId": owners[ref], "response": "PRIVATE"}
        yield Reader()


def entry(source, ref, source_bot=None, speaker=None, cited=False):
    return {"source": source, "id": ref, "source_bot": source_bot, "session_id": "s1", "at": None,
            "speaker": speaker, "follows": None, "window_kind": "matched", "start_char": 0,
            "admitted": True, "cited": cited}


def drill(entries, *, database=None, owners=None, omitted=0):
    record = LedgerRecord(OWNER, USER, "s1", {"messages": {"status": "ok", "records": len(entries)}},
                          entries, omitted)
    return EvidenceDrilldown(database or Database(), Ledger(record), Tasks(owners or {}))


def test_visibility_follows_the_current_policy_for_source_and_delivering_bot():
    scope = RecallScope(OWNER, USER, "s1", frozenset({"protected"}), frozenset({OWNER, "peer", "protected"}))
    assert visible(entry("messages", "a", "peer"), scope)
    assert visible(entry("memories", "b"), scope)
    assert not visible(entry("messages", "c", "protected"), scope)
    assert not visible(entry("messages", "d", "peer", speaker="bot:protected"), scope)
    assert not visible(entry("messages", "e", "gone-bot"), scope)
    assert visible(entry("messages", "f", "peer", speaker="bot:peer"), scope)


@pytest.mark.asyncio
async def test_pages_are_disjoint_and_totals_count_every_page():
    entries = [entry("messages", f"m{i}", "peer" if i % 5 else "other") for i in range(33)]
    database = Database(allowed=POLICY - {"other"}, gone={"m3"})
    drilldown = drill(entries, database=database)
    seen = []
    for offset in (0, 20):
        page = await drilldown.page("h" * 20, OWNER, USER, "messages", offset, 20)
        # m0, m5, ... m30 came from a bot that is no longer searchable.
        assert (page["recorded"], page["withheld_now"], page["total"]) == (33, 7, 26)
        seen += [item["id"] for item in page["entries"]]
    assert len(seen) == len(set(seen)) == 26
    assert {"m0", "m5", "m30"}.isdisjoint(seen)
    # Only the requested page is resolved live; nothing for withheld rows.
    assert [len(params["ids"]) for params in database.statements] == [20, 6]
    # A row that no longer passes its source's authority stays listed, empty.
    page = await drilldown.page("h" * 20, OWNER, USER, "messages", 0, 50)
    gone = next(item for item in page["entries"] if item["id"] == "m3")
    assert gone["available"] is False and "snippet" not in gone
    assert next(item for item in page["entries"] if item["id"] == "m1")["snippet"] == "live m1"


@pytest.mark.asyncio
async def test_tasks_are_reauthorized_and_withheld_when_now_protected():
    entries = [entry("tasks", "TASK-1"), entry("tasks", "TASK-2"), entry("tasks", "TASK-3")]
    owners = {"TASK-1": OWNER, "TASK-2": "protected", "TASK-3": "unreachable"}
    page = await drill(entries, owners=owners).page("h" * 20, OWNER, USER, "tasks", 0, 1)
    assert (page["recorded"], page["withheld_now"], page["total"]) == (3, 2, 1)
    [item] = page["entries"]
    assert (item["id"], item["title"], item["status"], item["available"]) == ("TASK-1", "Title TASK-1", "REVIEW", True)
    assert "PRIVATE" not in compact_json(page)


@pytest.mark.asyncio
async def test_task_part_ids_are_authorized_once_as_their_task():
    entries = [entry("tasks", "TASK-1#review:c" + "b" * 24), entry("tasks", "TASK-1#step:4"),
               entry("tasks", "TASK-2#response")]
    owners = {"TASK-1": OWNER, "TASK-2": "protected"}
    page = await drill(entries, owners=owners).page("h" * 20, OWNER, USER, "tasks", 0, 20)
    assert [item["id"] for item in page["entries"]] == ["TASK-1#review:c" + "b" * 24, "TASK-1#step:4"]
    assert page["withheld_now"] == 1


@pytest.mark.asyncio
async def test_owner_mismatch_expiry_and_a_now_private_owner_are_all_unavailable():
    entries = [entry("messages", "m1", "peer")]
    ok = drill(entries)
    assert await ok.page("h" * 20, OWNER, USER, "messages", 0, 20)
    assert await ok.page("h" * 20, "peer", USER, "messages", 0, 20) is None
    assert await ok.page("h" * 20, OWNER, "someone", "messages", 0, 20) is None
    assert await ok.summary("x" * 20, OWNER, USER) is None
    expired = EvidenceDrilldown(Database(), Ledger(None), Tasks({}))
    assert await expired.summary("h" * 20, OWNER, USER) is None
    private_owner = drill(entries, database=Database(allowed=POLICY - {OWNER}))
    assert await private_owner.page("h" * 20, OWNER, USER, "messages", 0, 20) is None


@pytest.mark.asyncio
async def test_summary_gives_cited_speakers_and_counts_under_current_policy():
    entries = [entry("messages", "m1", "peer", speaker="assistant", cited=True),
               entry("messages", "m2", "other", speaker="user", cited=True),
               entry("messages", "m3", "peer", speaker="bot:other"),
               entry("memories", "mem1")]
    summary = await drill(entries, database=Database(allowed=POLICY - {"other"}), omitted=2).summary(
        "h" * 20, OWNER, USER)
    assert summary["sources"] == {"messages": {"recorded": 3, "withheld_now": 2},
                                  "memories": {"recorded": 1, "withheld_now": 0}}
    assert [(item["id"], item["speaker"]) for item in summary["cited"]] == [("m1", "assistant")]
    assert summary["omitted"] == 2
    assert "start_char" not in summary["cited"][0] and "window_kind" not in summary["cited"][0]


@pytest.mark.asyncio
async def test_summary_reauthorizes_cited_tasks_like_a_page():
    # TASK-2 was cited at recall time; its owner has since become protected.
    entries = [entry("tasks", "TASK-1", cited=True), entry("tasks", "TASK-2", cited=True),
               entry("tasks", "TASK-2#review:c" + "b" * 24, cited=True),
               entry("tasks", "TASK-3#step:1", cited=True), entry("messages", "m1", "peer", cited=True)]
    owners = {"TASK-1": OWNER, "TASK-2": "protected", "TASK-3": "unreachable"}
    drilldown = drill(entries, owners=owners)
    summary = await drilldown.summary("h" * 20, OWNER, USER)
    assert summary["sources"]["tasks"] == {"recorded": 4, "withheld_now": 3}
    assert [item["id"] for item in summary["cited"]] == ["TASK-1", "m1"]
    assert "TASK-2" not in compact_json(summary) and "TASK-3" not in compact_json(summary)
    page = await drilldown.page("h" * 20, OWNER, USER, "tasks", 0, 20)
    assert page["withheld_now"] == summary["sources"]["tasks"]["withheld_now"]


def test_sources_never_run_an_unbounded_page():
    from llm_bawt.recall.drilldown import MAX_PAGE
    page = asyncio.run(drill([entry("memories", f"x{i}") for i in range(80)]).page(
        "h" * 20, OWNER, USER, "memories", 0, 500))
    assert page["limit"] == MAX_PAGE and len(page["entries"]) == MAX_PAGE
