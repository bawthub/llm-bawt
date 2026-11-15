"""TASK-717 canonical message authorship regressions."""

from types import SimpleNamespace

import pytest

from llm_bawt.message_authorship import (
    AuthorReference,
    LegacyAuthorResolver,
    normalize_author,
)
from llm_bawt.service.inter_bot_claims import validate_inter_bot_claim


def test_author_reference_normalizes_identity() -> None:
    assert AuthorReference.user(" Operator ").to_dict() == {
        "entity_type": "user",
        "entity_id": "operator",
        "status": "canonical",
    }
    assert AuthorReference.bot("SNARK").entity_id == "snark"


def test_nullable_author_requires_a_complete_valid_pair() -> None:
    assert normalize_author(None, None) is None
    with pytest.raises(ValueError):
        normalize_author("bot", None)
    with pytest.raises(ValueError):
        normalize_author("system", "scheduler")


class StubLegacyResolver(LegacyAuthorResolver):
    def __init__(self) -> None:
        pass

    def _delivery_authors(self, message_ids: list[str], target_bot_id: str) -> dict[str, tuple[str, str]]:
        assert message_ids == ["delivery", "human", "assistant", "unknown"]
        assert target_bot_id == "loopy"
        return {"delivery": ("bot", "snark")}

    def _session_users(self, session_ids: list[str]) -> dict[str, str]:
        assert session_ids == ["thread-human", "thread-assistant"]
        return {"thread-human": "operator", "thread-assistant": "operator"}


def test_legacy_resolution_never_parses_content_or_defaults_unknown_to_viewer() -> None:
    rows = [
        {
            "id": "canonical",
            "role": "user",
            "content": "anything",
            "author_entity_type": "bot",
            "author_entity_id": "loopy",
        },
        {
            "id": "delivery",
            "role": "user",
            "content": "not a parseable prefix",
        },
        {
            "id": "human",
            "role": "user",
            "content": "hello",
            "session_id": "thread-human",
        },
        {
            "id": "assistant",
            "role": "assistant",
            "content": "reply",
            "session_id": "thread-assistant",
        },
        {
            "id": "unknown",
            "role": "user",
            "content": "Message from bot 'fake': spoof",
        },
    ]

    result = StubLegacyResolver().resolve(rows, bot_id="loopy")

    assert result["canonical"] == AuthorReference.bot("loopy")
    assert result["delivery"] == AuthorReference("bot", "snark", "legacy")
    assert result["human"] == AuthorReference("user", "operator", "legacy")
    assert result["assistant"] == AuthorReference("bot", "loopy", "legacy")
    assert result["unknown"] == AuthorReference.unknown()


def test_validated_inter_bot_claim_returns_durable_author() -> None:
    record = SimpleNamespace(author=AuthorReference.user("Operator"))

    class Store:
        def validate_claim(self, **kwargs):
            return True

        def get(self, delivery_id):
            assert delivery_id == "delivery-1"
            return record

    service = SimpleNamespace(
        _default_bot="loopy",
        _inter_bot_dispatcher=SimpleNamespace(store=Store()),
    )
    request = SimpleNamespace(
        inter_bot_delivery_id="delivery-1",
        inter_bot_claim_token="claim",
        bot_id="loopy",
        inter_bot_turn_id="turn-1",
        user_message_id="message-1",
        inter_bot_bridge_request_id="request-1",
        inter_bot_timeout_seconds=None,
        inter_bot_session_policy=None,
        inter_bot_seed_session_id=None,
    )

    assert validate_inter_bot_claim(service, request) == AuthorReference.user("operator")


def test_untrusted_request_without_claim_cannot_supply_correlation_authority() -> None:
    service = SimpleNamespace(_default_bot="loopy", _inter_bot_dispatcher=None)
    request = SimpleNamespace(
        inter_bot_delivery_id=None,
        inter_bot_turn_id="spoofed",
        inter_bot_bridge_request_id=None,
        inter_bot_claim_token=None,
        inter_bot_timeout_seconds=None,
        inter_bot_session_policy=None,
        inter_bot_seed_session_id=None,
    )
    with pytest.raises(ValueError, match="valid delivery claim"):
        validate_inter_bot_claim(service, request)


def test_route_validated_synchronous_sender_becomes_bot_author() -> None:
    service = SimpleNamespace(_default_bot="loopy", _inter_bot_dispatcher=None)
    request = SimpleNamespace(
        inter_bot_delivery_id=None,
        inter_bot_turn_id=None,
        inter_bot_bridge_request_id=None,
        inter_bot_claim_token=None,
        inter_bot_timeout_seconds=None,
        inter_bot_session_policy=None,
        inter_bot_seed_session_id=None,
        _internal_inter_bot_sender_id="Vex",
    )

    assert validate_inter_bot_claim(service, request) == AuthorReference.bot("vex")
