from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from claude_code_bridge.proxy.request_context import ProxyCancellationRegistry
from claude_code_bridge.proxy.routes import messages
from claude_code_bridge.proxy.stream import responses_to_anthropic_sse
from claude_code_bridge.proxy.stream_cc import _clean_tool_arguments
from claude_code_bridge.proxy.tool_sanitizers import sanitize_tool_arguments
from claude_code_bridge.proxy.translate import anthropic_to_responses
from claude_code_bridge.proxy.translate_cc import anthropic_to_chat_completions


SKILLS = ("agent-system", "bawthub", "llm-bawt")
SKILL_SCHEMA = {
    "name": "Skill",
    "description": "Load a skill",
    "input_schema": {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Skill name."},
            "args": {"type": "string"},
        },
        "required": ["skill"],
    },
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("agent-system;", {"skill": "agent-system"}),
        ('bawthub;args="";', {"skill": "bawthub"}),
        ("llm-bawt\u202c", {"skill": "llm-bawt"}),
        (
            'agent-system;args="trace dispatch"',
            {"skill": "agent-system", "args": "trace dispatch"},
        ),
    ],
)
def test_skill_selector_repairs_are_catalog_exact(raw, expected) -> None:
    assert sanitize_tool_arguments(
        {"skill": raw}, "Skill", allowed_skill_names=SKILLS
    ) == expected


def test_removed_skill_is_not_invented() -> None:
    malformed = {"skill": 'claude-api;args=""'}
    assert sanitize_tool_arguments(
        malformed, "Skill", allowed_skill_names=SKILLS
    ) == malformed


def test_chat_completions_stream_uses_same_skill_repair() -> None:
    cleaned = _clean_tool_arguments(
        '{"skill":"bawthub;args=\\"\\";"}',
        tool_name="Skill",
        required_by_tool={"Skill": frozenset({"skill"})},
        allowed_skill_names=SKILLS,
    )
    assert cleaned == '{"skill":"bawthub"}'


def test_responses_stream_uses_skill_repair() -> None:
    async def run() -> list[dict]:
        item = SimpleNamespace(
            id="item-1", call_id="call-1", type="function_call", name="Skill"
        )

        async def events():
            yield SimpleNamespace(type="response.output_item.added", item=item)
            yield SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id=item.id,
                delta='{"skill":"agent-system;"}',
            )
            yield SimpleNamespace(
                type="response.function_call_arguments.done", item_id=item.id
            )

        payloads: list[dict] = []
        async for frame in responses_to_anthropic_sse(
            events(),
            anthropic_model="openai_chatgpt/gpt-6-astra",
            allowed_skill_names=SKILLS,
        ):
            data = next(
                line[6:]
                for line in frame.decode().splitlines()
                if line.startswith("data: ")
            )
            payloads.append(json.loads(data))
        return payloads

    payloads = asyncio.run(run())
    partials = [
        payload["delta"]["partial_json"]
        for payload in payloads
        if payload.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert partials == ['{"skill":"agent-system"}']


@pytest.mark.parametrize(
    "translate",
    [
        lambda body: anthropic_to_responses(
            body, "gpt-6-astra", allowed_skill_names=SKILLS
        ),
        lambda body: anthropic_to_chat_completions(
            body, "k3", allowed_skill_names=SKILLS
        ),
    ],
)
def test_skill_schema_is_constrained_without_mutating_source(translate) -> None:
    body = {
        "messages": [{"role": "user", "content": "work"}],
        "tools": [SKILL_SCHEMA],
    }
    payload = translate(body)
    tool = payload["tools"][0]
    params = tool.get("parameters") or tool["function"]["parameters"]
    assert params["properties"]["skill"]["enum"] == list(SKILLS)
    assert "enum" not in SKILL_SCHEMA["input_schema"]["properties"]["skill"]


def _twice_rejected_history() -> list[dict]:
    history: list[dict] = [{"role": "user", "content": "load it"}]
    for index in (1, 2):
        history.extend([
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": f"skill-{index}",
                    "name": "Skill",
                    "input": {"skill": 'claude-api;args=""'},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": f"skill-{index}",
                    "content": (
                        "<tool_use_error>Unknown skill: claude-api;args=\"\""
                        "</tool_use_error>"
                    ),
                }],
            },
        ])
    return history


@pytest.mark.parametrize("surface", ["responses", "chat_completions"])
def test_two_identical_unknown_skills_remove_tool_for_next_sample(surface) -> None:
    body = {
        "messages": _twice_rejected_history(),
        "tools": [SKILL_SCHEMA, {"name": "Bash", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "Skill"},
    }
    if surface == "responses":
        payload = anthropic_to_responses(
            body, "gpt-6-astra", allowed_skill_names=SKILLS
        )
        names = [tool["name"] for tool in payload["tools"]]
        reminder = payload["input"][-1]["content"][0]["text"]
    else:
        payload = anthropic_to_chat_completions(
            body, "k3", allowed_skill_names=SKILLS
        )
        names = [tool["function"]["name"] for tool in payload["tools"]]
        reminder = payload["messages"][-1]["content"]
    assert names == ["Bash"]
    assert "tool_choice" not in payload
    assert "Do not retry it" in reminder


def test_cancelled_request_is_rejected_before_adapter_call(monkeypatch) -> None:
    calls = 0

    class Adapter:
        async def call(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if False:
                yield b""

    registry = ProxyCancellationRegistry()
    registry.cancel("stopped-request")

    class Request:
        headers = {"X-LLM-Bawt-Request-ID": "stopped-request"}
        app = SimpleNamespace(
            state=SimpleNamespace(
                proxy_status_callback=None,
                proxy_cancellations=registry,
            )
        )

        async def json(self):
            return {
                "model": "openai_chatgpt/gpt-6-astra",
                "messages": [{"role": "user", "content": "continue"}],
                "stream": True,
            }

    monkeypatch.setattr(
        "claude_code_bridge.proxy.routes.lookup", lambda _provider: Adapter()
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(messages(Request()))
    assert exc.value.status_code == 499
    assert calls == 0


# TASK-1550: ambient runs also allow the skills the request advertises.
BUNDLED_LISTING = (
    "<system-reminder>\nThe following skills are available for use with the "
    "Skill tool:\n\n- agent-system: Task tools.\n- dataviz: Charts, mandatory "
    "for plots.\n- plugin:review: A plugin skill.\n</system-reminder>"
)


def _advertising_body(*, schema_enum: list[str] | None = None) -> dict:
    schema = json.loads(json.dumps(SKILL_SCHEMA))
    if schema_enum is not None:
        schema["input_schema"]["properties"]["skill"]["enum"] = schema_enum
    return {
        "model": "openai_chatgpt/gpt-6-astra",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": BUNDLED_LISTING},
            {"type": "text", "text": "chart it"},
        ]}],
        "tools": [schema],
        "stream": True,
    }


def test_ambient_catalog_adds_listed_bundled_and_plugin_skills() -> None:
    from claude_code_bridge.proxy.skill_catalog import callable_skill_names

    assert callable_skill_names(_advertising_body(), SKILLS, ambient=True) == (
        *SKILLS, "dataviz", "plugin:review",
    )


def test_ambient_catalog_keeps_existing_sdk_enum() -> None:
    from claude_code_bridge.proxy.skill_catalog import callable_skill_names

    body = _advertising_body(schema_enum=["sdk-only"])
    body["messages"] = [{"role": "user", "content": "no listing"}]
    assert callable_skill_names(body, SKILLS, ambient=True) == (*SKILLS, "sdk-only")


def test_managed_catalog_is_never_widened() -> None:
    from claude_code_bridge.proxy.skill_catalog import callable_skill_names

    body = _advertising_body(schema_enum=["sdk-only"])
    assert callable_skill_names(body, ("bawthub",), ambient=False) == ("bawthub",)


def test_tool_result_text_does_not_advertise_skills() -> None:
    from claude_code_bridge.proxy.skill_catalog import callable_skill_names

    body = {"messages": [{"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": "t", "content": BUNDLED_LISTING,
    }]}], "tools": [SKILL_SCHEMA]}
    assert callable_skill_names(body, SKILLS, ambient=True) == SKILLS


def test_scope_header_round_trips_only_for_ambient_runs() -> None:
    from claude_code_bridge.proxy.request_context import ProxyRequestContext, custom_header_env

    ambient = custom_header_env(ProxyRequestContext(
        request_id="r", provider="p", skill_names=SKILLS, skill_scope_ambient=True))
    managed = custom_header_env(ProxyRequestContext(
        request_id="r", provider="p", skill_names=SKILLS))
    assert "X-LLM-Bawt-Skill-Scope: ambient" in ambient
    assert "Skill-Scope" not in managed


def _route_skill_names(monkeypatch, headers: dict, body: dict) -> tuple[str, ...]:
    """Run the real route up to the adapter and return its resolved catalog."""
    from claude_code_bridge.proxy import routes

    seen: list[tuple[str, ...]] = []

    class Stop(Exception):
        pass

    def record(**kwargs):
        context = routes_context(**kwargs)
        seen.append(context.skill_names)
        raise Stop

    routes_context = routes.ProxyRequestContext
    monkeypatch.setattr(routes, "ProxyRequestContext", record)
    monkeypatch.setattr(routes, "lookup", lambda _provider: object())

    class Request:
        app = SimpleNamespace(state=SimpleNamespace(
            proxy_status_callback=None, proxy_cancellations=None))

        def __init__(self):
            self.headers = headers

        async def json(self):
            return body

    with pytest.raises(Stop):
        asyncio.run(messages(Request()))
    return seen[0]


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("ambient", (*SKILLS, "dataviz", "plugin:review")),
        (None, SKILLS),
    ],
)
def test_route_resolves_callable_skills_by_scope(monkeypatch, scope, expected) -> None:
    headers = {"X-LLM-Bawt-Skill-Names": ",".join(SKILLS)}
    if scope:
        headers["X-LLM-Bawt-Skill-Scope"] = scope
    assert _route_skill_names(monkeypatch, headers, _advertising_body()) == expected


@pytest.mark.parametrize("surface", ["responses", "chat_completions"])
def test_ambient_bundled_skill_reaches_both_translated_enums(surface) -> None:
    from claude_code_bridge.proxy.skill_catalog import callable_skill_names

    body = _advertising_body(schema_enum=["sdk-only"])
    names = callable_skill_names(body, SKILLS, ambient=True)
    if surface == "responses":
        tool = anthropic_to_responses(body, "gpt-6-astra", allowed_skill_names=names)["tools"][0]
        params = tool["parameters"]
    else:
        tool = anthropic_to_chat_completions(body, "k3", allowed_skill_names=names)["tools"][0]
        params = tool["function"]["parameters"]
    enum = params["properties"]["skill"]["enum"]
    assert {"dataviz", "plugin:review", "sdk-only", *SKILLS} == set(enum)
    assert sanitize_tool_arguments(
        {"skill": "dataviz;"}, "Skill", allowed_skill_names=names
    ) == {"skill": "dataviz"}
