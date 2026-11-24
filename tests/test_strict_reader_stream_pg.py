"""TASK-2590: one streamed strict-reader turn on private PostgreSQL.

Synthetic identities only. The shared bot registry is cold and invalidated and
every shared read is fatal. The real named reader reads the admitted profile
from PostgreSQL; the real streaming path admits, streams and persists the turn
with the strict instance, whose client carries the tools-off seal. A foreign
profile exists in the same database and is never read.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from llm_bawt import bots
from llm_bawt.runtime_settings import BotProfileStore
from llm_bawt.service import strict_reader_execution as strict
from llm_bawt.service.schemas import ChatCompletionRequest
from tests.session_lifecycle_pg_fixtures import rig  # noqa: F401
from tests.test_message_native_upsert import capture_sql
from tests.test_message_native_upsert_pg import http_runtime, real_storage  # noqa: F401
from tests.test_native_selected_owner_pg import native_runtime  # noqa: F401

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("private_app_pg", "persisted_master_key"),
]


def _forbid_shared_reads(monkeypatch):
    from llm_bawt.bot_tags.store import BotTagStore

    def boom(*_a, **_k):
        raise AssertionError("shared bot registry was read")

    monkeypatch.setattr(bots, "_LOADED", False)
    monkeypatch.setattr(bots, "_FORCE_RELOAD", True)
    monkeypatch.setattr(bots, "_RETRY_AFTER", 0.0)
    for name in ("_load_bots_config", "_load_db_bot_profiles", "_load_bot_tags"):
        monkeypatch.setattr(bots, name, boom)
    monkeypatch.setattr(bots.BotManager, "__init__", boom)
    monkeypatch.setattr(BotProfileStore, "__init__", boom)
    monkeypatch.setattr(BotTagStore, "tags_by_bot", boom)


def _request(rig, **over):
    body = {"model": "fixture-native", "bot_id": rig.bot, "user": rig.user, "stream": True,
            "augment_memory": False, "user_message_id": str(uuid.uuid4()),
            "assistant_message_id": str(uuid.uuid4()),
            "messages": [{"role": "user", "content": "hello strict"}], **over}
    return ChatCompletionRequest.model_validate(body)


def _stream(service, request):
    async def run():
        return [chunk async for chunk in service.chat_completion_stream(request)]
    return "".join(asyncio.run(run()))


@pytest.fixture
def strict_runtime(native_runtime, rig, monkeypatch):  # noqa: F811
    profiles = BotProfileStore(rig.config)
    profiles.upsert({"slug": rig.bot, "name": "Sample Strict", "bot_type": "agent",
                     "agent_backend": "claude-code"})
    foreign = rig.bot + "x"  # created by http_runtime; must never be read
    llm = native_runtime.llm
    built = []

    def init(self, *_a, admitted_bot, **kwargs):
        # The real constructor is unit-tested; here the request-owned instance
        # reuses the harness's PG-backed history and fake model client.
        self._admitted_bot = admitted_bot
        self.__dict__.update({k: v for k, v in llm.__dict__.items() if k != "client"})
        self.bot = admitted_bot
        self.bot_id = admitted_bot.slug
        self.settings = None
        self.client = SimpleNamespace(**vars(llm.client))
        self._seal_tools()
        built.append(self)

    monkeypatch.setattr(strict.StrictReaderLLMBawt, "__init__", init)
    monkeypatch.setattr(strict, "validate_bound_endpoint",
                        lambda service, bot, model: model if bot.slug == rig.bot else None)
    native_runtime.service.strict_reader_bot = rig.bot
    native_runtime.service.config = rig.config  # the real PG config for the named reader
    native_runtime.service._get_llm_bawt.side_effect = AssertionError("ordinary instance used")
    _forbid_shared_reads(monkeypatch)
    return SimpleNamespace(rt=native_runtime, built=built, foreign=foreign)


def test_streamed_strict_turn_reads_only_admitted_profile_and_seals_tools(strict_runtime, rig):
    rt = strict_runtime.rt
    with capture_sql(rig.engine) as sql:
        text = _stream(rt.service, _request(rig))
    assert "native " in text and "answer" in text and "[DONE]" in text
    assert len(strict_runtime.built) == 1
    instance = strict_runtime.built[0]
    assert instance.bot.slug == rig.bot and instance.client.tools_disabled is True
    assert len(rt.model_calls) == 1
    assert [role for role, _, _ in rt.appends] == ["user", "assistant"]
    profile_reads = [s for s in sql.all_statements if "FROM bot_profiles" in s]
    assert profile_reads, sql.all_statements
    assert all("bot_profiles.slug =" in s or "slug =" in s for s in profile_reads), profile_reads
    assert not any(strict_runtime.foreign in s for s in sql.all_statements)
    assert not any("FROM bot_profile_tags GROUP BY" in s for s in sql.all_statements)


@pytest.mark.parametrize("over", [{"session_id": "explicit"}, {"inter_bot_delivery_id": "d-1"}],
                         ids=["explicit-thread", "inter-bot"])
def test_unsupported_strict_turns_fail_closed_before_model(strict_runtime, rig, over):
    rt = strict_runtime.rt
    if "session_id" in over:
        over = {"session_id": rig.session_id}
    with pytest.raises(Exception):
        _stream(rt.service, _request(rig, **over))
    assert rt.model_calls == [] and strict_runtime.built == []


def test_ordinary_turn_unchanged_when_strict_unset(native_runtime, rig):  # noqa: F811
    native_runtime.service.strict_reader_bot = None
    text = _stream(native_runtime.service, _request(rig, session_id=rig.session_id))
    assert "native " in text and "answer" in text
    assert native_runtime.service._get_llm_bawt.called
    assert getattr(native_runtime.llm.client, "tools_disabled", False) is False
