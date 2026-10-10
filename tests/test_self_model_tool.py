"""Unit tests for the ``self_model`` MCP tool (TASK-1047).

Stubs the four HTTP boundaries (profile GET/PATCH, endpoint catalog, durable
delivery enqueue) plus the trusted-caller lookup, then calls the tool directly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from llm_bawt.mcp_server import self_model_tools as smt
from llm_bawt.task_turn_context import TaskTurnContext


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _ctx(bot: str, *, session: str | None = None, user: str = "nick",
         turn: str = "turn-abc") -> TaskTurnContext:
    """Trusted caller context; defaults to the bot's active default-user thread."""
    return TaskTurnContext(session_id=session or f"sess-{bot}", turn_id=turn,
                           trigger_message_id="msg-1", bot_id=bot, user_id=user, issued_at=0)


def _not_found(url: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", url)
    return httpx.HTTPStatusError(
        "404", request=request,
        response=httpx.Response(404, json={"detail": "not found"}, request=request),
    )


def _profile(slug: str, endpoint_id: int | None, model: str | None, harness: str) -> dict:
    backend = {"claude-code": "claude-code", "claude-proxy": "claude-code",
               "openclaw": "openclaw", "codex": "codex"}.get(harness)
    return {"slug": slug, "endpoint_id": endpoint_id, "default_model": model,
            "harness": harness, "agent_backend": backend}


_ENDPOINTS = {
    "claude-code": [
        {"id": 2158, "model_key": "claude-opus-5-5", "access_path_key": "anthropic-api",
         "context_window_override": None, "default_context_window": 200000},
        {"id": 2144, "model_key": "claude-opus-5", "access_path_key": "anthropic-api",
         "context_window_override": None, "default_context_window": 1000000},
    ],
    "claude-proxy": [
        {"id": 2150, "model_key": "gpt-6-sol", "access_path_key": "openai-oauth",
         "context_window_override": None, "default_context_window": 272000},
    ],
    "codex": [
        {"id": 2150, "model_key": "gpt-6-sol", "access_path_key": "openai-oauth",
         "context_window_override": None, "default_context_window": 272000},
    ],
}


class Harness:
    """Records boundary calls; profiles are mutated by PATCH like the server."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, profiles: dict[str, dict],
                 caller: TaskTurnContext | None = None) -> None:
        self.profiles = profiles
        # Active default-user ("nick") thread per bot, as GET /v1/sessions/active.
        self.active = {slug: {"id": f"sess-{slug}", "user_id": "nick"}
                       for slug in [*profiles, "al", "snark"]}
        self.patches: list[tuple[str, int]] = []
        self.enqueued: list[dict] = []
        self.patch_error: Exception | None = None
        self.enqueue_receipt: dict = {"delivery_id": "dlv-1", "status": "QUEUED"}
        # First harness wins, so 2150 heals to claude-proxy like the server does
        # for a claude-code bot (codex lists the same endpoint later).
        models: dict[int, tuple[str, str]] = {}
        for harness, rows in _ENDPOINTS.items():
            for row in rows:
                models.setdefault(row["id"], (row["model_key"], harness))

        async def get_profile(bot_id: str) -> dict:
            if bot_id not in self.profiles:
                raise _not_found(f"http://app/v1/bots/{bot_id}/profile")
            return dict(self.profiles[bot_id])

        async def get_active_session(bot_id: str) -> dict:
            if bot_id not in self.active:
                raise _not_found("http://app/v1/sessions/active")
            return dict(self.active[bot_id])

        async def patch_endpoint(bot_id: str, endpoint_id: int) -> dict:
            self.patches.append((bot_id, endpoint_id))
            if self.patch_error:
                raise self.patch_error
            model, harness = models[endpoint_id]
            self.profiles[bot_id] = _profile(bot_id, endpoint_id, model, harness)
            return dict(self.profiles[bot_id])

        async def list_endpoints(harness: str) -> list[dict]:
            return list(_ENDPOINTS.get(harness, []))

        async def enqueue(**kwargs: Any) -> dict:
            self.enqueued.append(kwargs)
            return self.enqueue_receipt

        monkeypatch.setattr(smt, "_get_profile", get_profile)
        monkeypatch.setattr(smt, "_patch_endpoint", patch_endpoint)
        monkeypatch.setattr(smt, "_list_endpoints", list_endpoints)
        monkeypatch.setattr(smt, "_enqueue_continue", enqueue)
        monkeypatch.setattr(smt, "_get_active_session", get_active_session)
        context = _ctx("al") if caller is None else caller
        monkeypatch.setattr(smt, "_trusted_caller", lambda: context)


def _rejected(status: int, detail: str) -> httpx.HTTPStatusError:
    request = httpx.Request("PATCH", "http://app/v1/bots/al/profile")
    return httpx.HTTPStatusError(
        str(status), request=request,
        response=httpx.Response(status, json={"detail": detail}, request=request),
    )


def test_view_lists_union_of_claude_harness_endpoints(monkeypatch):
    Harness(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")})

    res = _run(smt.self_model(bot_id="al"))

    assert res["current"] == {"endpoint_id": 2158, "model": "claude-opus-5-5",
                              "harness": "claude-code", "agent_backend": "claude-code"}
    ids = {o["endpoint_id"]: o for o in res["options"]}
    assert set(ids) == {2158, 2144, 2150}
    assert ids[2150]["harness"] == "claude-proxy"
    assert ids[2144]["context_window"] == 1000000


def test_view_for_codex_bot_offers_only_codex_endpoints(monkeypatch):
    Harness(monkeypatch, {"cx": _profile("cx", 2150, "gpt-6-sol", "codex")})

    res = _run(smt.self_model(bot_id="cx", action="view"))

    assert [(o["endpoint_id"], o["harness"]) for o in res["options"]] == [(2150, "codex")]


def test_switch_patches_only_endpoint_and_reports_previous(monkeypatch):
    h = Harness(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")})

    res = _run(smt.self_model(bot_id="al", action="switch", endpoint_id=2144))

    assert h.patches == [("al", 2144)]
    assert res["switched"] is True and res["changed"] is True
    assert res["previous"]["model"] == "claude-opus-5-5"
    assert res["current"]["model"] == "claude-opus-5"
    assert "next turn" in res["note"]
    assert "continue" not in res
    assert h.enqueued == []


def test_switch_reports_server_harness_heal(monkeypatch):
    h = Harness(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")})

    res = _run(smt.self_model(bot_id="al", action="switch", endpoint_id=2150))

    assert h.patches == [("al", 2150)]
    assert res["current"]["harness"] == "claude-proxy"
    assert res["current"]["agent_backend"] == "claude-code"


def test_switch_with_continue_prompt_queues_when_idle_self_delivery(monkeypatch):
    h = Harness(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")})

    res = _run(smt.self_model(bot_id="al", action="switch", endpoint_id=2144,
                              continue_prompt="  finish the review  "))

    assert res["continue"] == {"queued": True, "delivery_id": "dlv-1",
                               "status": "QUEUED", "delivery": "when_idle"}
    [sent] = h.enqueued
    # prefer_steer=False is applied inside _enqueue_continue; the caller never
    # passes a steer flag, so the follow-up can't land in the old-model turn.
    assert "prefer_steer" not in sent
    assert sent["target_bot_id"] == "al" and sent["sender_bot_id"] == "al"
    assert sent["message_kind"] == "MODEL_SWITCH_CONTINUE"
    assert sent["session_policy"] == "continue"
    assert sent["message"].endswith("finish the review")
    assert "claude-opus-5-5" in sent["message"] and "claude-opus-5" in sent["message"]
    assert sent["idempotency_key"].startswith("self_model:al:2144:turn-abc:")
    assert sent["metadata"]["self_model"]["to"]["endpoint_id"] == 2144


def test_enqueue_continue_never_steers(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_enqueue_durable(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return {"delivery_id": "dlv-x"}

    import llm_bawt.mcp_server.inter_bot_tools as ibt

    monkeypatch.setattr(ibt, "_enqueue_durable", fake_enqueue_durable)
    _run(smt._enqueue_continue(target_bot_id="al", message="m"))

    assert captured["prefer_steer"] is False


def test_continue_key_is_stable_within_a_turn_and_unique_across_turns():
    a = smt._continue_key("al", 2144, "turn-1", "go")
    assert a == smt._continue_key("al", 2144, "turn-1", "go")
    assert a != smt._continue_key("al", 2144, "turn-2", "go")
    assert a != smt._continue_key("al", 2144, "turn-1", "other prompt")


def test_cross_bot_switch_attributes_caller_as_sender(monkeypatch):
    h = Harness(monkeypatch, {"snark": _profile("snark", 2150, "gpt-6-sol", "claude-proxy")},
                caller=_ctx("al"))

    res = _run(smt.self_model(bot_id="snark", action="switch", endpoint_id=2158,
                              continue_prompt="report your model"))

    assert res["changed_by"] == "al"
    [sent] = h.enqueued
    assert sent["target_bot_id"] == "snark" and sent["sender_bot_id"] == "al"
    assert sent["message"].startswith("[self_model] al switched this bot")


def _untrusted(monkeypatch, profiles):
    h = Harness(monkeypatch, profiles)
    monkeypatch.setattr(smt, "_trusted_caller", lambda: None)
    return h


def test_untrusted_caller_can_switch_without_continuation(monkeypatch):
    h = _untrusted(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")})

    res = _run(smt.self_model(bot_id="al", action="switch", endpoint_id=2144))

    assert res["switched"] is True and res["changed_by"] == "al"
    assert h.patches == [("al", 2144)]


def test_untrusted_caller_continuation_refused_before_switch(monkeypatch):
    # Without a trusted turn there is no stable idempotency scope, so a retry
    # could queue a second follow-up (Snark P2): refuse up front instead.
    h = _untrusted(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")})

    res = _run(smt.self_model(bot_id="al", action="switch", endpoint_id=2144, continue_prompt="go"))

    assert res["switched"] is False and "trusted" in res["error"]
    assert h.patches == [] and h.enqueued == []


def test_non_default_user_continuation_refused_before_switch(monkeypatch):
    # Bot-authored deliveries run as DEFAULT_USER; another user's follow-up
    # would land in Nick's thread (Snark P1).
    h = Harness(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")},
                caller=_ctx("al", user="guest", session="sess-guest"))

    res = _run(smt.self_model(bot_id="al", action="switch", endpoint_id=2144, continue_prompt="go"))

    assert res["switched"] is False and "default user" in res["error"]
    assert h.patches == [] and h.enqueued == []


def test_task_thread_self_continuation_refused_before_switch(monkeypatch):
    # A task dispatch runs in a born-archived thread; the delivery would run in
    # the active main chat instead (Snark P1).
    h = Harness(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")},
                caller=_ctx("al", session="sess-task-thread"))

    res = _run(smt.self_model(bot_id="al", action="switch", endpoint_id=2144, continue_prompt="go"))

    assert res["switched"] is False and "active thread" in res["error"]
    assert h.patches == [] and h.enqueued == []


def test_cross_bot_continuation_from_task_thread_targets_other_bots_main_chat(monkeypatch):
    # Al in a task thread switching Snark: the follow-up belongs in Snark's
    # active default-user thread, which is exactly where deliveries run.
    h = Harness(monkeypatch, {"snark": _profile("snark", 2150, "gpt-6-sol", "claude-proxy")},
                caller=_ctx("al", session="sess-task-thread"))

    res = _run(smt.self_model(bot_id="snark", action="switch", endpoint_id=2158, continue_prompt="go"))

    assert res["continue"]["queued"] is True
    assert h.enqueued[0]["target_bot_id"] == "snark"


def test_unresolvable_active_thread_refuses_continuation(monkeypatch):
    h = Harness(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")})
    h.active.pop("al")

    res = _run(smt.self_model(bot_id="al", action="switch", endpoint_id=2144, continue_prompt="go"))

    assert res["switched"] is False and "404" in res["error"]
    assert h.patches == []


def test_codex_switch_warns_about_fresh_provider_session(monkeypatch):
    _ENDPOINTS_CODEX_ASTRA = {"id": 2145, "model_key": "gpt-6-astra", "access_path_key": "openai-oauth",
                              "context_window_override": None, "default_context_window": 1000000}
    monkeypatch.setitem(_ENDPOINTS, "codex", [*_ENDPOINTS["codex"], _ENDPOINTS_CODEX_ASTRA])
    Harness(monkeypatch, {"cx": _profile("cx", 2150, "gpt-6-sol", "codex")})

    res = _run(smt.self_model(bot_id="cx", action="switch", endpoint_id=2145))

    assert res["current"]["agent_backend"] == "codex"
    assert "fresh provider session" in res["note"]


def test_rejected_switch_does_not_queue_continuation(monkeypatch):
    h = Harness(monkeypatch, {"cx": _profile("cx", 2150, "gpt-6-sol", "codex")})
    h.patch_error = _rejected(422, "endpoint_id=2158 is incompatible with harness=codex")

    res = _run(smt.self_model(bot_id="cx", action="switch", endpoint_id=2158, continue_prompt="go"))

    assert res["switched"] is False
    assert "422" in res["error"] and "incompatible" in res["error"]
    assert res["previous"]["endpoint_id"] == 2150
    assert h.enqueued == []


def test_failed_enqueue_keeps_switch_and_reports_error(monkeypatch):
    h = Harness(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")})
    h.enqueue_receipt = {"success": False, "queued": False, "error": "db down"}

    res = _run(smt.self_model(bot_id="al", action="switch", endpoint_id=2144, continue_prompt="go"))

    assert res["switched"] is True
    assert res["continue"] == {"queued": False, "error": "db down"}


@pytest.mark.parametrize("bot_id", ["", "  ", "default", "DEFAULT"])
def test_default_and_blank_bot_refused(monkeypatch, bot_id):
    h = Harness(monkeypatch, {})

    res = _run(smt.self_model(bot_id=bot_id, action="switch", endpoint_id=2144))

    assert "error" in res and h.patches == []


def test_protected_bot_refused_for_other_callers(monkeypatch):
    h = Harness(monkeypatch, {"mira": _profile("mira", 2158, "claude-opus-5-5", "claude-code")},
                caller=_ctx("al"))

    for action in ("view", "switch"):
        res = _run(smt.self_model(bot_id="mira", action=action, endpoint_id=2144))
        assert "protected" in res["error"]
    assert h.patches == []


def test_protected_bot_may_switch_itself(monkeypatch):
    h = Harness(monkeypatch, {"mira": _profile("mira", 2158, "claude-opus-5-5", "claude-code")},
                caller=_ctx("mira", turn="turn-m"))

    res = _run(smt.self_model(bot_id="mira", action="switch", endpoint_id=2144))

    assert res["switched"] is True and h.patches == [("mira", 2144)]


def test_openclaw_bot_has_nothing_to_switch(monkeypatch):
    h = Harness(monkeypatch, {"oc": _profile("oc", None, None, "openclaw")})

    view = _run(smt.self_model(bot_id="oc"))
    switch = _run(smt.self_model(bot_id="oc", action="switch", endpoint_id=2144))

    assert view["options"] == [] and "OpenClaw" in view["note"]
    assert "OpenClaw" in switch["error"] and h.patches == []


@pytest.mark.parametrize("endpoint_id", [None, True])
def test_switch_requires_integer_endpoint(monkeypatch, endpoint_id):
    h = Harness(monkeypatch, {"al": _profile("al", 2158, "claude-opus-5-5", "claude-code")})

    res = _run(smt.self_model(bot_id="al", action="switch", endpoint_id=endpoint_id))

    assert "endpoint_id" in res["error"] and h.patches == []


def test_unknown_bot_reports_profile_error(monkeypatch):
    Harness(monkeypatch, {})

    res = _run(smt.self_model(bot_id="ghost"))

    assert res["error"].startswith("Could not read profile (404)")


def test_registered_on_shared_server():
    from llm_bawt.mcp_server.server import mcp

    tool = mcp._tool_manager.get_tool("self_model")
    assert tool is not None
    props = tool.parameters["properties"]
    assert set(props) == {"bot_id", "action", "endpoint_id", "continue_prompt"}
    assert props["action"]["enum"] == ["view", "switch"]
