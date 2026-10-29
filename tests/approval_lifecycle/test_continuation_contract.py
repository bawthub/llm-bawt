"""Exact authorization requires complete continuation inputs, not UI previews."""
import json

import pytest

from llm_bawt.service.approval_continuations import build_continuation_prompt, build_respond_prompt


def test_approved_subject_is_not_truncated():
    command = "printf " + "x" * 1000
    assert command in build_continuation_prompt(True, command, "Bash")


def test_approved_continuation_carries_all_original_arguments():
    arguments = {"command": "printf " + "x" * 1000, "timeout": 100, "cwd": "/repo"}
    prompt = build_continuation_prompt(
        True, "short policy field", "Bash", tool_arguments_json=json.dumps(arguments),
    )
    assert json.dumps(arguments, ensure_ascii=False, sort_keys=True) in prompt


def test_all_continuation_guidance_resolves_from_registry_at_call_time(monkeypatch):
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

    keys = {"approvals.approved_guidance", "approvals.denied_guidance",
            "approvals.responded_guidance", "approvals.no_response_guidance",
            "approvals.mcp_result_guidance", "approvals.continuation_reset_guidance"}
    assert keys <= DEFAULT_PROMPT_DEFINITIONS.keys()
    for key in keys:
        assert DEFAULT_PROMPT_DEFINITIONS[key].default_body()

    class Resolver:
        def render(self, *, key, variables, scope_type, scope_id):
            assert scope_type == "bot" and scope_id == "test-bot"
            return f"registry:{key}:{variables.get('shown', '')}:{variables.get('message', '')}"

    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", lambda: Resolver())
    assert build_continuation_prompt(True, "call", "Bash", bot_id="test-bot").startswith("registry:approvals.approved_guidance:call:\n")
    assert build_continuation_prompt(False, "call", "Bash", bot_id="test-bot").startswith("registry:approvals.denied_guidance:call:\n")
    assert build_respond_prompt("ok", "call", "Bash", bot_id="test-bot").startswith("registry:approvals.responded_guidance:call:ok\n")
    assert build_respond_prompt("", "call", "Bash", bot_id="test-bot").startswith("registry:approvals.no_response_guidance:call:\n")


@pytest.mark.parametrize("approved", [True, False])
def test_edited_approval_guidance_cannot_drop_original_arguments(monkeypatch, approved):
    class Resolver:
        def render(self, **_kwargs):
            return "Instruction without the original call"

    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", lambda: Resolver())
    with pytest.raises(ValueError, match="omitted the original call"):
        build_continuation_prompt(approved, "short", "Bash", tool_arguments_json='{"command": "full"}')


@pytest.mark.parametrize("body", ["shown", "reply"])
def test_edited_response_guidance_cannot_drop_original_request_or_reply(monkeypatch, body):
    class Resolver:
        def render(self, **_kwargs):
            return body

    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", lambda: Resolver())
    with pytest.raises(ValueError, match="omitted the original request or reply"):
        build_respond_prompt("reply", "shown", "Bash")


@pytest.mark.parametrize(
    ("kind", "build"),
    [
        ("approved", lambda: build_continuation_prompt(True, "call", "Bash")),
        ("denied", lambda: build_continuation_prompt(False, "call", "Bash")),
        ("responded", lambda: build_respond_prompt("reply", "call", "Bash")),
        ("reviewed", lambda: build_respond_prompt("", "call", "Bash")),
    ],
)
def test_harness_continuation_has_structural_metadata_without_changing_legacy_opening(kind, build):
    content = build()
    assert content.startswith("[The user ")
    marker, metadata = content.rsplit("\n", 2)[1:]
    assert marker == "[LLM_BAWT_HARNESS_CONTINUATION_V1]"
    assert json.loads(metadata) == {"kind": kind, "tool_name": "Bash"}


def test_preview_explains_structural_ops_fallback():
    from llm_bawt.service.routes.approval_policies import PolicyPreview, preview_policy
    result = preview_policy(PolicyPreview(backend="mcp", tool_name="ops_run", tool_input={}, policies=[]))
    assert result["action"] == "require_approval"
    assert result["reason"] == "No matching policy; default require_approval"
