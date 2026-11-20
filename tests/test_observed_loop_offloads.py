"""The serving loop stays runnable while each observed offender is held."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from llm_bawt.service import (
    api,
    chat_nonstream,
    chat_streaming,
    stream_execution_selection,
)
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.schemas import ChatCompletionRequest


class _Done(Exception):
    pass


async def _while_blocked(start, *, complete=lambda: None):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    ran = []

    def blocking(*args, **kwargs):
        ran.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "blocking work kept the serving loop from running"
        return complete()

    work = asyncio.create_task(start(blocking))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert len(ran) == 1
        assert ran[0] != threading.get_ident()
        assert not work.done(), "the loop did not run while the worker was held"
    finally:
        release.set()
        await work


@pytest.mark.parametrize("stream", [True, False])
def test_chat_lazy_history_initialization_keeps_loop_runnable(monkeypatch, stream):
    class Chat(chat_streaming.ChatStreamingMixin, chat_nonstream.ChatNonStreamMixin):
        config = SimpleNamespace(DEFAULT_USER="fixture-user")
        _default_bot = "fixture-bot"
        _turn_log_store = None

        def _resolve_request_model(self, *_):
            return "fixture-model", []

    monkeypatch.setattr(chat_nonstream, "validate_inter_bot_claim", lambda *_: None)
    monkeypatch.setattr(chat_streaming, "validate_inter_bot_claim", lambda *_: None)
    monkeypatch.setattr(stream_execution_selection, "select_stream_execution",
                        lambda *_: ("fixture-model", [], None, None))

    async def start(blocking):
        svc = Chat()
        svc._get_llm_bawt = blocking
        req = ChatCompletionRequest(model="fixture-model", messages=[{"role": "user", "content": "hello"}])
        if stream:
            events = [item async for item in svc.chat_completion_stream(req)]
            assert events
        else:
            with pytest.raises(_Done):
                await svc.chat_completion(req)

    def stop_after_lookup():
        raise _Done

    asyncio.run(_while_blocked(start, complete=stop_after_lookup))


def test_delivery_renewal_commit_keeps_loop_runnable():
    async def start(blocking):
        def renew(delivery_id, claim_token):
            assert (delivery_id, claim_token) == ("delivery", "current-token")
            return blocking()

        dispatcher = SimpleNamespace(LEASE_RENEW_INTERVAL_SECONDS=0, store=SimpleNamespace(renew_lease=renew))
        await InterBotDeliveryDispatcher._heartbeat(dispatcher, "delivery", "current-token")

    # False means the token no longer owns the row; stop instead of retrying.
    asyncio.run(_while_blocked(start, complete=lambda: False))


def test_startup_warms_openapi_in_worker_before_other_initialization(monkeypatch):
    app = FastAPI(lifespan=api.lifespan)

    @app.get("/fixture")
    def route():
        return {"ok": True}

    original = app.openapi

    def stop_after_cache():
        assert app.openapi_schema and "/fixture" in app.openapi_schema["paths"]
        raise _Done

    monkeypatch.setattr(api, "Config", stop_after_cache)

    async def start(blocking):
        def schema():
            blocking()
            return original()

        monkeypatch.setattr(app, "openapi", schema)
        with pytest.raises(_Done):
            async with app.router.lifespan_context(app):
                pytest.fail("startup should stop before any service mutation")
        assert original() is app.openapi_schema

    asyncio.run(_while_blocked(start))
