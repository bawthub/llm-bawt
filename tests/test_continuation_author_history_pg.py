"""TASK-1807 phase 2 on disposable PostgreSQL: the real embedded history path.

The chat writers hand ``user_author`` to ``HistoryManager.add_message``; in
embedded mode that is MemoryClient -> MemoryStorage -> PostgreSQLShortTermManager
-> message store. This proves a continuation author survives that whole path
and reads back as system (TOOL), while ordinary turns stay user.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.mcp_server.client import MemoryClient
from llm_bawt.mcp_server.storage import MemoryStorage
from llm_bawt.memory.postgresql import MESSAGES_PARENT, partition_name
from llm_bawt.memory.speaker_attribution import TOOL, USER, extraction_input, speaker_of
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.inter_bot_claims import continuation_author
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage, McpToolResultContinuation
from llm_bawt.utils.history import HistoryManager

pytestmark = pytest.mark.integration

USER_ID = "user_a"


def _request(*, claim: bool, mcp: bool) -> ChatCompletionRequest:
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="x")])
    if mcp:
        request.continuation_payload = McpToolResultContinuation(
            kind="mcp_tool_result", approval_request_id="r", original_tool_use_id="t",
            tool_name="ops_run", result={}, is_error=False)
    if claim:
        request._internal_approval_claim = ("req", "token")
    return request


@pytest.fixture
def history(private_app_pg):
    """A HistoryManager wired exactly as service/core.py wires it in embedded mode."""
    config = private_app_pg
    bot = f"cont{uuid.uuid4().hex[:10]}"
    client = MemoryClient(config, bot_id=bot, user_id=USER_ID)  # no server_url: embedded
    storage = MemoryStorage(config)
    client._storage, client._initialized = storage, True
    manager = client.get_short_term_manager()
    manager.ensure_schema()
    hm = HistoryManager(client=SimpleNamespace(), config=config, db_backend=manager,
                        bot_id=bot, user_id=USER_ID)
    return SimpleNamespace(hm=hm, storage=storage, bot=bot)


def _write(history, request, content):
    author = continuation_author(request) or AuthorReference.user(USER_ID)
    history.hm.add_message("user", content, author=author)
    return history.hm.messages[-1].db_id


def _stored(history, message_id):
    table = partition_name(MESSAGES_PARENT, history.bot)
    with history.storage.engine.connect() as conn:
        return conn.execute(text(
            f'SELECT author_entity_type, author_entity_id FROM "{table}" WHERE id = :id'
        ), {"id": message_id}).one()


def test_continuations_persist_and_read_back_as_system(history):
    mcp_id = _write(history, _request(claim=True, mcp=True), "result without envelope")
    harness_id = _write(history, _request(claim=True, mcp=False), "[The user APPROVED ...]")
    human_id = _write(history, _request(claim=False, mcp=False), "my dog is called Rex")
    forged_id = _write(history, _request(claim=False, mcp=True), "payload, no claim")

    assert tuple(_stored(history, mcp_id)) == ("system", "approval-continuation:mcp")
    assert tuple(_stored(history, harness_id)) == ("system", "approval-continuation:harness")
    assert tuple(_stored(history, human_id)) == ("user", USER_ID)
    assert tuple(_stored(history, forged_id)) == ("user", USER_ID)

    rows = asyncio.run(history.storage.get_unprocessed_messages(history.bot))
    by_id = {m.id: m for m in rows}
    assert by_id[mcp_id].author == AuthorReference("system", "approval-continuation:mcp")
    assert by_id[harness_id].author == AuthorReference("system", "approval-continuation:harness")
    kinds = {mid: speaker_of(by_id[mid]).kind for mid in (mcp_id, harness_id, human_id)}
    assert kinds == {mcp_id: TOOL, harness_id: TOOL, human_id: USER}
    kept = {m.id for m in extraction_input([by_id[i] for i in (mcp_id, harness_id, human_id)])}
    assert kept == {human_id}
