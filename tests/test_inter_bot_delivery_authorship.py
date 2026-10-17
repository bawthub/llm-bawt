"""TASK-792 UI-originated delivery authorship regressions."""

import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.routes.inter_bot_deliveries import (
    DeliveryCreate, _format_delivery_message, create_delivery,
)


def test_user_authored_delivery_is_not_wrapped_as_a_bot_message() -> None:
    message = "Repository workspace decision request: TASK-792"

    assert _format_delivery_message(message, AuthorReference.user("nick")) == message


def test_bot_authored_delivery_keeps_the_bot_wrapper() -> None:
    assert _format_delivery_message(
        "Review this change",
        AuthorReference.bot("loopy"),
    ) == "Message from bot 'loopy': Review this change"


def test_waited_delivery_tells_target_final_answer_is_the_reply() -> None:
    formatted = _format_delivery_message(
        "Choose SPLIT or HOLD", AuthorReference.bot("nova"), "waited",
    )
    assert formatted.startswith("Message from bot 'nova': Choose SPLIT or HOLD")
    assert "Your final response on this turn is the answer" in formatted
    assert "Do not send a separate message back merely to repeat that answer" in formatted
    assert "intentional, distinct follow-up" in formatted
    assert "[Reply channel:" not in _format_delivery_message(
        "Choose SPLIT or HOLD", AuthorReference.bot("nova"), "async",
    )


@pytest.mark.parametrize("options", [
    {"prefer_steer": True},
    {"prefer_steer": False, "author_entity_type": "user", "author_entity_id": "nick"},
    {"prefer_steer": False, "sender_bot_id": "unknown"},
])
def test_waited_contract_rejects_steers_and_non_bot_senders(options) -> None:
    body = DeliveryCreate(target_bot_id="target", message="choose", reply_mode="waited",
                          **{"sender_bot_id": "sender", **options})
    with pytest.raises(HTTPException, match="Waited replies require"):
        asyncio.run(create_delivery(body))


def test_unknown_reply_mode_is_rejected_at_boundary() -> None:
    with pytest.raises(ValidationError):
        DeliveryCreate(target_bot_id="target", message="choose", reply_mode="steer")
