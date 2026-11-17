"""TASK-668 — Playwright screenshots return durable Garage URLs to agents."""

from __future__ import annotations

import asyncio

import pytest

try:
    from claude_agent_sdk.types import ToolResultBlock, UserMessage
    from claude_code_bridge.bridge import ClaudeCodeBridge

    _SDK_OK = True
except Exception:  # noqa: BLE001
    _SDK_OK = False


class _FakePub:
    class _R:
        connection_pool = None

    _redis = _R()

    def close(self):
        pass


def _bridge() -> "ClaudeCodeBridge":
    if not _SDK_OK:
        pytest.skip("claude_agent_sdk not importable (run inside bridge container)")
    return ClaudeCodeBridge(
        _FakePub(),
        backend_name="claude-code",
        app_api_url="http://app:8642",
    )


def _artifact(asset_id: str = "ma_test") -> dict:
    return {
        "asset_id": asset_id,
        "kind": "image",
        "mime_type": "image/webp",
        "width": 1280,
        "height": 720,
        "urls": {
            "original": f"/v1/uploads/{asset_id}",
            "preview": f"/v1/uploads/{asset_id}/preview",
            "thumb": f"/v1/uploads/{asset_id}/thumb",
        },
    }


def _hook_input(*, tool_name: str = "mcp__playwright__browser_take_screenshot") -> dict:
    return {
        "tool_name": tool_name,
        "tool_input": {"type": "png", "fullPage": False},
        "tool_response": {
            "content": [
                {"type": "text", "text": "Screenshot captured"},
                {"type": "image", "data": "aW1hZ2U=", "mimeType": "image/png"},
            ]
        },
        "tool_use_id": "toolu_shot",
    }


def test_extract_image_block_supports_mcp_and_anthropic_shapes() -> None:
    bridge = _bridge()
    assert bridge._extract_image_block(
        {"type": "image", "data": "abc", "mimeType": "image/png"}
    ) == ("abc", "image/png")
    assert bridge._extract_image_block(
        {
            "type": "image",
            "source": {"data": "xyz", "media_type": "image/jpeg"},
        }
    ) == ("xyz", "image/jpeg")


def test_upload_data_url_returns_canonical_envelope(monkeypatch) -> None:
    bridge = _bridge()
    artifact = _artifact()
    captured: dict = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return artifact

    class Client:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return Response()

    monkeypatch.setattr("httpx.AsyncClient", Client)
    result = asyncio.run(bridge._upload_data_url(
        "data:image/png;base64,aW1hZ2U=", "operator", "toolu_shot"
    ))

    assert result == artifact
    assert captured["url"] == "http://app:8642/v1/uploads"
    assert captured["params"] == {"source": "agent_attachment"}
    assert captured["headers"] == {"X-Entity-Id": "operator"}
    assert captured["json"]["filename"] == "image-toolu_shot.png"


def test_upload_data_url_rejects_response_without_asset_id(monkeypatch) -> None:
    bridge = _bridge()

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"urls": {"original": "/not-real"}}

    class Client:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            return Response()

    monkeypatch.setattr("httpx.AsyncClient", Client)
    assert asyncio.run(bridge._upload_data_url(
        "data:image/png;base64,aW1hZ2U=", "operator", "toolu_shot"
    )) is None


def test_post_tool_hook_returns_curlable_urls_and_preserves_output() -> None:
    bridge = _bridge()
    artifacts_by_tool: dict[str, list[dict]] = {}
    uploads: list[tuple[list, str, str | None]] = []

    async def persist(content, session_key, tool_use_id):
        uploads.append((content, session_key, tool_use_id))
        return [_artifact()]

    bridge._persist_screenshot_artifacts = persist  # type: ignore[method-assign]
    hook = bridge._make_post_tool_use_hook(
        session_key="loopy:operator",
        screenshot_artifacts_by_tool_use_id=artifacts_by_tool,
    )

    result = asyncio.run(hook(_hook_input(), "toolu_shot", {"signal": None}))
    specific = result["hookSpecificOutput"]
    assert specific["hookEventName"] == "PostToolUse"
    context = specific["additionalContext"]
    assert "asset_id=ma_test" in context
    assert "original: http://app:8642/v1/uploads/ma_test" in context
    assert "preview: http://app:8642/v1/uploads/ma_test/preview" in context
    assert "thumb: http://app:8642/v1/uploads/ma_test/thumb" in context
    assert specific["updatedMCPToolOutput"]["content"][1] == _hook_input()["tool_response"]["content"][1]
    assert "untrusted_content=true" in context
    assert uploads == [(
        _hook_input()["tool_response"]["content"],
        "loopy:operator",
        "toolu_shot",
    )]


def test_post_tool_hook_failure_is_model_visible_without_fake_urls() -> None:
    bridge = _bridge()

    async def persist(content, session_key, tool_use_id):
        return []

    bridge._persist_screenshot_artifacts = persist  # type: ignore[method-assign]
    hook = bridge._make_post_tool_use_hook(
        session_key="loopy:operator",
        screenshot_artifacts_by_tool_use_id={},
    )
    result = asyncio.run(hook(_hook_input(), "toolu_shot", {"signal": None}))
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "persistence failed" in context
    assert "/v1/uploads/" not in context


def test_post_tool_hook_labels_non_screenshot_and_named_file_result() -> None:
    bridge = _bridge()
    hook = bridge._make_post_tool_use_hook(
        session_key="loopy:operator",
        screenshot_artifacts_by_tool_use_id={},
    )
    snapshot = asyncio.run(hook(
        _hook_input(tool_name="mcp__playwright__browser_snapshot"),
        "toolu_shot",
        {"signal": None},
    ))
    assert "untrusted_content=true" in snapshot["hookSpecificOutput"]["additionalContext"]
    assert "updatedMCPToolOutput" in snapshot["hookSpecificOutput"]
    named_result = _hook_input()
    named_result["tool_response"] = {
        "content": [{"type": "text", "text": "Saved to ./shot.png"}]
    }
    named = asyncio.run(hook(named_result, "toolu_named", {"signal": None}))
    assert "untrusted_content=true" in named["hookSpecificOutput"]["additionalContext"]


def test_tool_end_reuses_cached_upload_and_deduplicates_turn_refs() -> None:
    bridge = _bridge()
    uploads = 0
    published: list[dict] = []

    async def persist(content, session_key, tool_use_id):
        nonlocal uploads
        uploads += 1
        return [_artifact("ma_unexpected")]

    bridge._persist_screenshot_artifacts = persist  # type: ignore[method-assign]
    bridge._publish_event = lambda *args, **kwargs: published.append(kwargs)  # type: ignore[method-assign]
    cached = {"toolu_shot": [_artifact()]}
    turn_refs = [{"asset_id": "ma_test", "kind": "image"}]
    msg = UserMessage(
        content=[ToolResultBlock(
            tool_use_id="toolu_shot",
            content=_hook_input()["tool_response"]["content"],
        )],
        parent_tool_use_id=None,
    )

    seq = asyncio.run(bridge._on_user_message_tool_results(
        msg,
        request_id="req",
        session_key="loopy:operator",
        seq=0,
        tool_names_by_id={
            "toolu_shot": "mcp__playwright__browser_take_screenshot",
        },
        turn_screenshot_assets=turn_refs,
        screenshot_artifacts_by_tool_use_id=cached,
    ))

    assert seq == 1
    assert uploads == 0
    assert turn_refs == [{"asset_id": "ma_test", "kind": "image"}]
    assert published[-1]["attachments"] == [
        {"asset_id": "ma_test", "kind": "image"}
    ]


def test_tool_end_retries_after_post_tool_upload_failure() -> None:
    bridge = _bridge()
    uploads = 0

    async def persist(content, session_key, tool_use_id):
        nonlocal uploads
        uploads += 1
        return [_artifact()]

    bridge._persist_screenshot_artifacts = persist  # type: ignore[method-assign]
    bridge._publish_event = lambda *args, **kwargs: None  # type: ignore[method-assign]
    cached = {"toolu_shot": []}
    turn_refs: list[dict] = []
    msg = UserMessage(
        content=[ToolResultBlock(
            tool_use_id="toolu_shot",
            content=_hook_input()["tool_response"]["content"],
        )],
    )

    asyncio.run(bridge._on_user_message_tool_results(
        msg,
        request_id="req",
        session_key="loopy:operator",
        seq=0,
        tool_names_by_id={
            "toolu_shot": "mcp__playwright__browser_take_screenshot",
        },
        turn_screenshot_assets=turn_refs,
        screenshot_artifacts_by_tool_use_id=cached,
    ))

    assert uploads == 1
    assert cached["toolu_shot"] == [_artifact()]
    assert turn_refs == [{"asset_id": "ma_test", "kind": "image"}]


def test_agent_options_injects_task_capability_without_mutating_shared_mcp_config() -> None:
    bridge = _bridge()
    bridge._mcp_servers = {
        "bawthub": {"type": "http", "url": "http://app:8001/mcp"},
    }

    async def noop(*args, **kwargs):
        return {}

    options = bridge._build_agent_options(
        model="claude-opus-4-8",
        system_prompt="system",
        disallowed_tools=[],
        resume_id=None,
        sdk_env={},
        settings_path=None,
        bot_effort=None,
        bot_max_turns=None,
        can_use_tool_cb=noop,
        pre_tool_use_cb=noop,
        post_tool_use_cb=noop,
        stderr=lambda line: None,
        task_turn_capability="opaque-capability",
    )

    assert options.mcp_servers["bawthub"]["headers"] == {
        "X-LLM-Bawt-Task-Turn-Context": "opaque-capability",
    }
    assert "headers" not in bridge._mcp_servers["bawthub"]


def test_agent_options_registers_playwright_post_tool_hook() -> None:
    bridge = _bridge()

    async def noop(*args, **kwargs):
        return {}

    options = bridge._build_agent_options(
        model="claude-opus-4-8",
        system_prompt="system",
        disallowed_tools=[],
        resume_id=None,
        sdk_env={},
        settings_path=None,
        bot_effort=None,
        bot_max_turns=None,
        can_use_tool_cb=noop,
        pre_tool_use_cb=noop,
        post_tool_use_cb=noop,
        stderr=lambda line: None,
    )
    matcher = options.hooks["PostToolUse"][0]
    assert matcher.matcher is None
    assert matcher.hooks == [noop]
    assert options.hooks["PostToolUseFailure"][0].matcher is None
