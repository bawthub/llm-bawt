"""Registry-backed OPS_BLOCKED instructions without changing the wire envelope."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from llm_bawt import prompt_registry as registry
from llm_bawt.service import ops_blocked_notification as notice


@pytest.fixture
def resolver(monkeypatch):
    store = registry.PromptTemplateStore.__new__(registry.PromptTemplateStore)
    store.config = None
    store.connection_url = "sqlite://"
    store.engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(store.engine, tables=[
        registry.PromptTemplate.__table__, registry.PromptTemplateVersion.__table__,
    ])
    monkeypatch.setattr(registry, "DEFAULT_PROMPT_DEFINITIONS", {
        notice.PROMPT_KEY: registry.DEFAULT_PROMPT_DEFINITIONS[notice.PROMPT_KEY],
    })
    resolver = registry.PromptResolver.__new__(registry.PromptResolver)
    resolver.store = store
    monkeypatch.setattr(registry, "get_prompt_resolver", lambda: resolver)
    yield resolver
    store.engine.dispose()


def delivered_message():
    dispatcher = SimpleNamespace(
        store=SimpleNamespace(enqueue=Mock(return_value=(object(), False))),
        announce_admitted=Mock(),
    )
    notice.OpsBlockedNotifier(SimpleNamespace(_inter_bot_dispatcher=dispatcher)).notify(
        SimpleNamespace(id="approval-1", bot_id="fixture-bot", user_id="fixture-user"),
        {"job_id": "job-1", "operation": "fixture.op", "state": "queued",
         "error_text": "missing prerequisite"},
    )
    args = dispatcher.store.enqueue.call_args.kwargs
    message = args["message"]
    assert args["message_kind"] == "OPS_BLOCKED"
    assert args["payload"]["messages"] == [{"role": "user", "content": message}]
    header, facts, instructions = message.split("\n", 2)
    assert header == "[Ops preflight blocked — not a completion receipt]"
    assert facts == json.dumps({
        "job_id": "job-1", "operation": "fixture.op", "state": "queued",
        "error_text": "missing prerequisite",
    }, sort_keys=True)
    return instructions


def test_product_default_preserves_delivered_text(resolver):
    resolver.store.seed_defaults()
    assert delivered_message() == (
        "The approved job is still queued because preflight failed. No action "
        "was submitted by that preflight attempt. Do not submit another job or "
        "re-issue ops_run. Report the blocker; inspect this same job with "
        "ops_job_status. The existing job may run after prerequisites recover. "
        "Its terminal receipt will be delivered separately."
    )


@pytest.mark.parametrize("scope", ["global", "bot"])
def test_tenant_override_is_delivered_verbatim(resolver, scope):
    resolver.store.seed_defaults()
    resolver.store.upsert(
        key=notice.PROMPT_KEY, body="Tenant instructions\nwith {literal braces}. ",
        scope_type=scope, scope_id="fixture-bot" if scope == "bot" else None,
        updated_by="tenant",
    )
    resolver.store.seed_defaults()  # Startup must not replace tenant intent.
    assert delivered_message() == "Tenant instructions\nwith {literal braces}. "
    if scope == "bot":
        assert notice.blocked_notification_prompt("other-bot") == notice.DEFAULT_BLOCKED_INSTRUCTIONS


@pytest.mark.parametrize("body", [None, "", " \n "])
def test_empty_registry_text_falls_back(monkeypatch, body):
    monkeypatch.setattr(registry, "get_prompt_resolver", lambda: SimpleNamespace(
        resolve=lambda *a, **kw: SimpleNamespace(body=body),
    ))
    assert delivered_message() == notice.DEFAULT_BLOCKED_INSTRUCTIONS


def test_registry_failure_keeps_notification_deliverable(monkeypatch):
    def unavailable():
        raise RuntimeError("registry unavailable")
    monkeypatch.setattr(registry, "get_prompt_resolver", unavailable)
    assert delivered_message() == notice.DEFAULT_BLOCKED_INSTRUCTIONS
