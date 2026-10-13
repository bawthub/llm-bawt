from __future__ import annotations

from types import SimpleNamespace

from claude_code_bridge.send_errors import (
    AuthRetryPolicy,
    CLAUDE_CREDENTIAL_ERROR_MARKER,
    CODEX_CREDENTIAL_ERROR_MARKER,
    classify_terminal_error,
    result_message_error,
)
from claude_code_bridge.send_stream import ClaudeStreamMixin
from claude_code_bridge.turn_phase import TurnPhaseTracker


def _result(**overrides):
    values = {
        "is_error": True,
        "api_error_status": None,
        "errors": None,
        "result": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_snark_401_result_is_terminal_credential_failure():
    error = result_message_error(
        _result(
            api_error_status=401,
            errors=[
                "Failed to authenticate. API Error: 401 OAuth access token has been revoked."
            ],
        ),
        fallback="HTTP 401: authentication_failed",
    )

    assert error is not None
    assert error.credential_error is True
    text, raw = classify_terminal_error(error)
    assert text.startswith(CLAUDE_CREDENTIAL_ERROR_MARKER)
    assert raw == {"error_code": "credential_expired", "provider": "claude"}


def test_proxy_auth_failure_does_not_emit_claude_reconnect_marker():
    error = result_message_error(
        _result(api_error_status=401, errors=["authentication_failed"])
    )

    assert error is not None
    text, raw = classify_terminal_error(error, direct_anthropic=False)
    assert CLAUDE_CREDENTIAL_ERROR_MARKER not in text
    assert raw is None


def test_chatgpt_proxy_auth_failure_emits_codex_reconnect_marker():
    """TASK: expired ChatGPT OAuth behind the proxy must open the ChatGPT
    reconnect flow, not the Claude one (Nova 401 misroute)."""
    error = result_message_error(
        _result(
            api_error_status=401,
            errors=[
                "Failed to authenticate. API Error: 401 Proxy exhausted retries: "
                "Provided authentication token is expired."
            ],
        )
    )

    assert error is not None
    text, raw = classify_terminal_error(
        error, direct_anthropic=False, proxy_provider="openai_chatgpt"
    )
    assert text.startswith(CODEX_CREDENTIAL_ERROR_MARKER)
    assert CLAUDE_CREDENTIAL_ERROR_MARKER not in text
    assert raw == {"error_code": "credential_expired", "provider": "codex"}


def test_api_key_proxy_auth_failure_has_no_reconnect_marker():
    """API-key providers (xai, zai, …) have no UI reconnect flow — no marker."""
    error = result_message_error(
        _result(api_error_status=401, errors=["authentication_failed"])
    )

    assert error is not None
    text, raw = classify_terminal_error(
        error, direct_anthropic=False, proxy_provider="xai"
    )
    assert CLAUDE_CREDENTIAL_ERROR_MARKER not in text
    assert CODEX_CREDENTIAL_ERROR_MARKER not in text
    assert raw is None


def test_generic_upstream_exhaustion_is_error_but_not_credential_failure():
    error = result_message_error(
        _result(
            api_error_status=529,
            errors=["overloaded_error"],
        ),
        fallback="HTTP 529: overloaded_error",
    )

    assert error is not None
    assert error.credential_error is False
    text, raw = classify_terminal_error(error)
    assert "529" in text
    assert CLAUDE_CREDENTIAL_ERROR_MARKER not in text
    assert raw is None


def test_success_result_is_not_classified_as_error():
    assert result_message_error(
        _result(is_error=False, api_error_status=None, errors=None, result="ok")
    ) is None


def test_auth_retry_policy_allows_exactly_one_side_effect_free_direct_retry():
    policy = AuthRetryPolicy()

    assert policy.claim(
        is_auth_failure=True,
        direct_anthropic=True,
        model_side_effects=False,
    ) is True
    assert policy.claim(
        is_auth_failure=True,
        direct_anthropic=True,
        model_side_effects=False,
    ) is False


def test_auth_retry_policy_rejects_proxy_and_side_effectful_attempts():
    assert AuthRetryPolicy().claim(
        is_auth_failure=True,
        direct_anthropic=False,
        model_side_effects=False,
    ) is False
    assert AuthRetryPolicy().claim(
        is_auth_failure=True,
        direct_anthropic=True,
        model_side_effects=True,
    ) is False


class _RetryStatusHarness(ClaudeStreamMixin):
    def __init__(self):
        self.events: list[dict] = []

    def _publish_event(self, *_args, **kwargs):
        self.events.append(kwargs)


def _tracker() -> tuple[TurnPhaseTracker, list[dict]]:
    published: list[dict] = []
    tracker = TurnPhaseTracker(lambda _text, status: published.append(status))
    tracker.on_turn_start()
    published.clear()
    return tracker, published


def test_auth_retry_status_is_silent_before_self_heal():
    harness = _RetryStatusHarness()
    tracker, published = _tracker()

    attempt, error = harness._on_api_retry_status(
        {
            "attempt": 1,
            "max_retries": 10,
            "error_status": 401,
            "error": "authentication_failed",
        },
        session_key="snark:nick",
        phase_tracker=tracker,
    )

    assert (attempt, error) == (1, "HTTP 401: authentication_failed")
    assert published == []
    assert harness.events == []


def test_generic_retry_status_is_transient_phase_not_reply_text():
    harness = _RetryStatusHarness()
    tracker, published = _tracker()

    attempt, error = harness._on_api_retry_status(
        {
            "attempt": 1,
            "max_retries": 10,
            "error_status": 529,
            "error": "overloaded_error",
        },
        session_key="snark:nick",
        phase_tracker=tracker,
    )

    assert (attempt, error) == (1, "HTTP 529: overloaded_error")
    # TASK-1074: no ASSISTANT_DELTA — retry status never enters history.
    assert harness.events == []
    assert len(published) == 1
    assert published[0]["phase"] == "retrying"
    assert published[0]["attempt"] == 1
    assert published[0]["max_attempts"] == 10
    assert published[0]["error_status"] == 529
