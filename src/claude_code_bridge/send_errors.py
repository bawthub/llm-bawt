"""Terminal SDK failure classification and one-shot auth retry policy."""

from __future__ import annotations

from dataclasses import dataclass


CLAUDE_CREDENTIAL_ERROR_MARKER = "[credential_expired:claude]"

# ChatGPT-subscription OAuth (the ~/.codex/auth.json bundle) rides the bridge's
# Anthropic-compat proxy under the ``openai_chatgpt`` provider prefix. BawtHub's
# established recovery contract for that bundle is the codex marker → its
# "Reconnect ChatGPT" flow — reuse it so proxy 401s open the RIGHT reconnect.
CODEX_CREDENTIAL_ERROR_MARKER = "[credential_expired:codex]"

# Proxy provider prefix → (marker, provider) for credential errors that have a
# BawtHub reconnect flow. API-key providers (xai, zai, …) have nothing to
# reconnect in the UI and intentionally stay unmapped.
_PROXY_CREDENTIAL_CONTRACTS: dict[str, tuple[str, str]] = {
    "openai_chatgpt": (CODEX_CREDENTIAL_ERROR_MARKER, "codex"),
}


_AUTH_MARKERS = (
    "oauth access token has been revoked",
    "oauth token has expired",
    "failed to authenticate",
    "api error: 401",
    "http 401",
    "authentication_failed",
    "authentication_error",
    "invalid api key",
    "invalid token",
    "token has expired",
    "invalid_grant",
    "refresh token not found or invalid",
    "claude oauth refresh failed",
)


def is_auth_failure_text(text: str, *, status: int | None = None) -> bool:
    """Classify only concrete authentication failures, not generic upstream errors."""
    if status == 401:
        return True
    haystack = text.lower()
    return any(marker in haystack for marker in _AUTH_MARKERS)


def retry_exhausted_text(retry_count: int, last_error: str | None) -> str:
    """Reply text for a turn that ended empty after SDK API retries."""
    return (
        f"❌ Upstream error after {retry_count} retries: "
        f"{last_error or 'unknown'}. Try again in a moment."
    )


class TerminalSDKResultError(RuntimeError):
    """A Claude SDK ``ResultMessage`` that represents terminal API failure."""

    def __init__(self, detail: str, *, status: int | None, credential_error: bool):
        self.detail = detail
        self.status = status
        self.credential_error = credential_error
        status_label = f"HTTP {status}" if status is not None else "upstream API"
        super().__init__(f"{status_label}: {detail}")


def result_message_error(msg, *, fallback: str | None = None) -> TerminalSDKResultError | None:
    """Return a typed failure for an error-bearing SDK terminal result."""
    status = getattr(msg, "api_error_status", None)
    is_error = bool(getattr(msg, "is_error", False))
    errors = getattr(msg, "errors", None)
    result = getattr(msg, "result", None)

    if not is_error and status is None and not errors:
        return None

    details: list[str] = []
    if isinstance(errors, list):
        details.extend(str(item).strip() for item in errors if str(item).strip())
    if isinstance(result, str) and result.strip():
        details.append(result.strip())
    if not details and fallback:
        details.append(fallback.strip())
    detail = " | ".join(dict.fromkeys(details)) or "Claude SDK returned an error result"
    return TerminalSDKResultError(
        detail,
        status=status if isinstance(status, int) else None,
        credential_error=is_auth_failure_text(
            detail,
            status=status if isinstance(status, int) else None,
        ),
    )


@dataclass
class AuthRetryPolicy:
    """Allow one side-effect-free retry for a direct-Claude auth failure."""

    attempted: bool = False

    def claim(
        self,
        *,
        is_auth_failure: bool,
        direct_anthropic: bool,
        model_side_effects: bool,
    ) -> bool:
        if (
            self.attempted
            or not is_auth_failure
            or not direct_anthropic
            or model_side_effects
        ):
            return False
        self.attempted = True
        return True


def classify_terminal_error(
    exc: Exception,
    *,
    direct_anthropic: bool = True,
    proxy_provider: str | None = None,
) -> tuple[str, dict | None]:
    """Build the stable bridge ERROR contract consumed by BawtHub.

    ``proxy_provider`` is the proxy path's provider prefix (e.g.
    ``openai_chatgpt``) when ``direct_anthropic`` is False, so credential
    failures get tagged with the provider whose token actually died instead
    of defaulting BawtHub's recovery UI to the Claude reconnect.
    """
    text = str(exc)
    is_credential_failure = (
        isinstance(exc, TerminalSDKResultError) and exc.credential_error
    ) or is_auth_failure_text(text)
    if is_credential_failure:
        if direct_anthropic:
            return (
                f"{CLAUDE_CREDENTIAL_ERROR_MARKER} {text}",
                {"error_code": "credential_expired", "provider": "claude"},
            )
        contract = _PROXY_CREDENTIAL_CONTRACTS.get(proxy_provider or "")
        if contract:
            marker, provider = contract
            return (
                f"{marker} {text}",
                {"error_code": "credential_expired", "provider": provider},
            )
    return text, None
