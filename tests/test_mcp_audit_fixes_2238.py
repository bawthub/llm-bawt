"""TASK-2238: regressions for the mcp_server audit fixes, one per defect."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from llm_bawt.mcp_server import media_tools, self_tools, task_tools
from llm_bawt.mcp_server.task_step_types import normalize_step_inputs


def _created(**extra: Any) -> dict[str, Any]:
    return {"id": "c1", "shortId": "TASK-9", "status": "QUEUED", "title": "t", **extra}


def _no_images(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_turn_images(*_args: Any) -> tuple[list[str], None]:
        return [], None

    async def no_refs(refs: list[str]) -> list[str]:
        return list(refs)

    monkeypatch.setattr(task_tools, "linked_turn_images", no_turn_images)
    monkeypatch.setattr(task_tools, "resolve_image_refs", no_refs)


# Item 2: an explicit zero relevance threshold is not replaced by the default.
@pytest.mark.parametrize(("given", "sent"), [(0.0, 0.0), (None, 0.4)])
def test_memory_search_keeps_explicit_zero_threshold(given: float | None, sent: float) -> None:
    from llm_bawt.mcp_server.client import MemoryClient

    config = SimpleNamespace(MEMORY_N_RESULTS=5, MEMORY_MIN_RELEVANCE=0.4)
    client = MemoryClient(config, bot_id="b", server_url="http://mcp.invalid")  # type: ignore[arg-type]
    calls: list[dict[str, Any]] = []
    client._ensure_initialized = lambda: None  # type: ignore[method-assign]
    client._call_server = lambda tool, args: calls.append(args) or []  # type: ignore[method-assign,assignment,func-returns-value]

    client.search("q", min_relevance=given)

    assert calls[0]["min_relevance"] == sent


# Item 1: server-mode history search goes through the server, never a local DB.
def test_server_mode_message_search_uses_server(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_bawt.memory.postgresql as pg
    from llm_bawt.mcp_server.client import MemoryClient

    def no_local_backend(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("server mode constructed a local database backend")

    monkeypatch.setattr(pg, "PostgreSQLMemoryBackend", no_local_backend)
    client = MemoryClient(SimpleNamespace(), bot_id="b", server_url="http://mcp.invalid")  # type: ignore[arg-type]
    calls: list[tuple[str, dict[str, Any]]] = []
    client._ensure_initialized = lambda: None  # type: ignore[method-assign]
    row = {"id": "m1", "role": "user", "content": "hi", "timestamp": 12.5, "session_id": "s", "source": "b"}
    client._call_server = lambda tool, args: calls.append((tool, args)) or [row]  # type: ignore[method-assign,assignment,func-returns-value]

    results = client.search_messages("hi", n_results=3, role_filter="user", since=1.0, until=2.0)

    assert calls == [("messages_search_all", {
        "query": "hi", "n_results": 3, "role_filter": "user", "since": 1.0, "until": 2.0, "bot_id": "b"})]
    assert [(r.id, r.role, r.content, r.bot_id, r.session_id, r.created_at) for r in results] == [
        ("m1", "user", "hi", "b", "s", "12.5")]


# Item 3: an extracted name candidate never changes the live display name.
def test_extracted_name_candidate_leaves_display_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_bawt.mcp_server import extraction

    class Manager:
        display_names: list[str] = []

        def add_candidate(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(created_at=1, updated_at=1)

        def set_display_name(self, *_args: Any) -> None:
            self.display_names.append(_args[-1])

    manager = Manager()
    monkeypatch.setattr("llm_bawt.mcp_server.server._get_profile_manager", lambda: manager)
    fact = {"content": "User's name is Alex", "importance": 0.9, "tags": [],
            "profile_attribute": {"category": "fact", "key": "name"}}
    config = SimpleNamespace(MEMORY_PROFILE_ATTRIBUTE_ENABLED=True)

    assert extraction.extract_profile_attributes_from_fact(fact, "nick", config) is True  # type: ignore[arg-type]
    assert manager.display_names == []


# Item 4: the default "auto" pipeline lookup reads combined history.
@pytest.mark.parametrize(("kind", "history_kind"), [("auto", "all"), ("movie", "movie"), ("series", "series")])
def test_pipeline_status_history_kind(monkeypatch: pytest.MonkeyPatch, kind: str, history_kind: str) -> None:
    seen: list[str] = []

    async def lookup(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"results": []}

    async def history(kind: str, limit: int) -> dict[str, Any]:
        seen.append(kind)
        return {"sonarr": [{"title": "Show Name"}], "radarr": []}

    def offline() -> Any:
        raise RuntimeError("offline")

    monkeypatch.setattr(media_tools, "media_lookup", lookup)
    monkeypatch.setattr(media_tools, "media_history", history)
    for client in ("sonarr_client", "radarr_client", "sabnzbd_client"):
        monkeypatch.setattr(media_tools, client, offline)

    out = asyncio.run(media_tools.media_pipeline_status("show name", kind=kind))

    assert seen == [history_kind]
    assert [h["title"] for h in out["recent_history"]] == ["Show Name"]


# Item 6: self_tail's structured messages obey the same truncation as its transcript.
def test_self_tail_messages_match_truncated_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"role": "user", "content": "x" * 50, "timestamp": None},
            {"role": "assistant", "content": "new", "timestamp": None}]

    async def tail(_bot_id: str, _count: int) -> tuple[list[dict], int]:
        return rows, 2

    monkeypatch.setattr(self_tools, "_tail_bubbles", tail)
    monkeypatch.setattr(self_tools, "_MAX_TAIL_CHARS", 30)
    tool = getattr(self_tools.self_tail, "fn", self_tools.self_tail)

    result = asyncio.run(tool(bot_id="nova", count=2))

    assert result["truncated"] is True
    assert result["count_returned"] == 1
    assert [m["content"] for m in result["messages"]] == ["new"]
    assert len(result["transcript"]) <= 30


def test_tail_cap_counts_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(self_tools, "_MAX_TAIL_CHARS", 20)
    # Two 9-char entries are 18 chars of content but 20 once joined; a third
    # would push the joined text past the cap.
    transcript, count, truncated = self_tools._format_bubbles(
        [{"role": "a", "content": "1234"}, {"role": "a", "content": "5678"}, {"role": "a", "content": "9"}]
    )
    assert len(transcript) <= 20
    assert (count, truncated) == (2, True)


# Item 8: relevance (similarity) is not forwarded as an importance floor.
def test_relevance_threshold_is_not_an_importance_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_bawt.mcp_server import storage as storage_mod

    seen: dict[str, Any] = {}

    class Backend:
        def search_memories_by_embedding(self, **kwargs: Any) -> list[dict[str, Any]]:
            seen.update(kwargs)
            return []

    store = storage_mod.MemoryStorage.__new__(storage_mod.MemoryStorage)
    store.config = SimpleNamespace(MEMORY_EMBEDDING_MODEL="m", VERBOSE=False)
    store._get_backend = lambda _bot: Backend()  # type: ignore[method-assign]
    monkeypatch.setattr(storage_mod, "generate_embedding", lambda *_a, **_k: [0.1])

    asyncio.run(store.search_memories("q", bot_id="b", min_relevance=0.8))

    assert seen["min_importance"] == 0.0


# Item 10: a committed task write keeps its receipt when association transport fails.
def test_create_returns_task_when_association_transport_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_images(monkeypatch)
    posts: list[dict[str, Any]] = []

    async def post(path: str, json: dict[str, Any], headers: Any = None) -> dict[str, Any]:
        posts.append(json)
        return _created()

    async def associate(_ref: str) -> dict[str, Any]:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(task_tools, "_api_post", post)
    monkeypatch.setattr(task_tools, "associate_current_task", associate)
    tool = getattr(task_tools.create_task, "fn", task_tools.create_task)

    result = asyncio.run(tool(title="t", associate_current_turn=True))

    assert len(posts) == 1
    assert result["task"]["shortId"] == "TASK-9"
    assert result["currentTurnAssociation"] == {"error": "down"}


def test_update_returns_receipt_when_association_transport_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_images(monkeypatch)

    async def patch(path: str, json: dict[str, Any], headers: Any = None) -> dict[str, Any]:
        return _created(status="IN_PROGRESS")

    async def associate(_ref: str) -> dict[str, Any]:
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(task_tools, "_api_patch", patch)
    monkeypatch.setattr(task_tools, "associate_current_task", associate)
    tool = getattr(task_tools.update_task, "fn", task_tools.update_task)

    result = asyncio.run(tool(task_id="TASK-9", status="IN_PROGRESS", associate_current_turn=True))

    assert result["ok"] is True
    assert result["currentTurnAssociation"] == {"error": "slow"}


# Item 11: the file_path alias reaches BawtHub as filePath.
def test_step_file_path_alias_is_normalized() -> None:
    assert normalize_step_inputs([
        {"title": "a", "file_path": "src/a.py"},
        {"title": "b", "filePath": "src/b.py"},
        {"title": "c", "file_path": "src/c.py", "filePath": "other"},
        {"title": "d"},
    ]) == [
        {"title": "a", "filePath": "src/a.py"},
        {"title": "b", "filePath": "src/b.py"},
        {"title": "c", "filePath": "src/c.py"},
        {"title": "d"},
    ]


def test_create_task_forwards_step_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_images(monkeypatch)
    posts: list[dict[str, Any]] = []

    async def post(path: str, json: dict[str, Any], headers: Any = None) -> dict[str, Any]:
        posts.append(json)
        return _created()

    monkeypatch.setattr(task_tools, "_api_post", post)
    tool = getattr(task_tools.create_task, "fn", task_tools.create_task)

    asyncio.run(tool(title="t", steps=[{"title": "Inspect", "file_path": "src/example.py"}]))

    assert posts[0]["steps"] == [{"title": "Inspect", "filePath": "src/example.py"}]


# Item 12: lowercase review gets the same owner fallback as REVIEW.
def test_lowercase_review_sends_owner_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_images(monkeypatch)
    patches: list[dict[str, Any]] = []

    async def patch(path: str, json: dict[str, Any], headers: Any = None) -> dict[str, Any]:
        patches.append(json)
        return _created(status="REVIEW")

    monkeypatch.setattr(task_tools, "_api_patch", patch)
    tool = getattr(task_tools.update_task, "fn", task_tools.update_task)
    links = [{"label": "Board", "path": "/tasks", "steps": ["Open it"]}]

    asyncio.run(tool(task_id="TASK-9", status="review", bot_id="nova", check_links=links))

    assert patches[0]["status"] == "REVIEW"
    assert patches[0]["fallbackAgentBotId"] == "nova"


# Item 13: tasks_create can send the check links a REVIEW create requires.
def test_create_review_task_forwards_check_links(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_images(monkeypatch)
    posts: list[dict[str, Any]] = []

    async def post(path: str, json: dict[str, Any], headers: Any = None) -> dict[str, Any]:
        posts.append(json)
        return _created(status="REVIEW")

    monkeypatch.setattr(task_tools, "_api_post", post)
    tool = getattr(task_tools.create_task, "fn", task_tools.create_task)
    links = [{"label": "Board", "path": "/tasks", "steps": ["Open it"]}]

    asyncio.run(tool(title="t", status="REVIEW", agent_bot_id="nova", check_links=links))

    assert posts[0]["checkLinks"] == links


# Item 5: a send whose response is lost after the request went out is
# "uncertain", never "not dispatched"; a connect failure stays a refusal.
def _failing_client(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    class Client:
        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def post(self, *_args: Any, **_kwargs: Any) -> Any:
            raise exc

    monkeypatch.setattr(httpx, "AsyncClient", Client)


_READ_TIMEOUT = httpx.ReadTimeout("read timed out")
_CONNECT_ERROR = httpx.ConnectError("refused")


@pytest.mark.parametrize(("exc", "dispatched"), [(_READ_TIMEOUT, None), (_CONNECT_ERROR, False)])
def test_batch_send_lost_response_is_uncertain(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, dispatched: bool | None,
) -> None:
    from llm_bawt.mcp_server.inter_bot_fanout import send_batch
    from llm_bawt.mcp_server.inter_bot_send_mode import SendMode

    _failing_client(monkeypatch, exc)
    receipt = asyncio.run(send_batch("http://app.invalid", targets=["a", "b"],
                                     mode=SendMode("steer_or_idle", False), exclude=[],
                                     fields={"idempotency_key": "k1", "message": "m"}))

    assert receipt["dispatched"] is dispatched
    if dispatched is None:
        assert receipt["outcome"] == "uncertain"
        assert receipt["idempotency_key"] == "k1"
        assert "SAME idempotency_key" in receipt["note"]


def test_batch_send_unreadable_success_is_uncertain(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_bawt.mcp_server.inter_bot_fanout import send_batch
    from llm_bawt.mcp_server.inter_bot_send_mode import SendMode

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, content=b"not json")

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    receipt = asyncio.run(send_batch("http://app.invalid", targets=["a", "b"],
                                     mode=SendMode("steer_or_idle", False), exclude=[],
                                     fields={"idempotency_key": "k1"}))

    assert receipt["dispatched"] is None and receipt["status_code"] == 202


@pytest.mark.parametrize(("exc", "uncertain"), [(_READ_TIMEOUT, True), (_CONNECT_ERROR, False)])
def test_single_send_lost_response_is_uncertain(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, uncertain: bool,
) -> None:
    from llm_bawt.mcp_server import inter_bot_tools

    _failing_client(monkeypatch, exc)
    receipt = asyncio.run(inter_bot_tools._enqueue_durable(
        prefer_steer=True, target_bot_id="b", message="m", sender_bot_id="a", max_tokens=None,
        temperature=0.7, timeout_seconds=300, idempotency_key="k2", project_id=None, task_id=None,
        message_kind=None, metadata=None, session_policy=None, reset_session_before_delivery=None,
        retain_history=None, reset_reason=None,
    ))

    if uncertain:
        assert receipt["outcome"] == "uncertain" and receipt["queued"] is None
        assert receipt["idempotency_key"] == "k2"
    else:
        assert receipt["queued"] is False and "outcome" not in receipt
