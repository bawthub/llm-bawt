"""Talk webhooks subscribe once; the dispatcher owns eventual answers."""
import json
from types import SimpleNamespace

import pytest

from llm_bawt.integrations.nextcloud import webhook
from llm_bawt.service import dependencies


class FakeRequest:
    headers = {"X-Nextcloud-Talk-Backend": "https://talk.example"}

    def __init__(self, message_id="42"):
        self.message_id = message_id

    async def body(self):
        return json.dumps({
            "target": {"id": "room"}, "type": "Create",
            "actor": {"name": "Nick", "id": "users/nick"},
            "object": {"id": self.message_id, "content": json.dumps({"message": "Help"})},
        }).encode()


@pytest.fixture
def talk(monkeypatch):
    monkeypatch.setattr(webhook, "verify_nextcloud_signature", lambda *args: True)
    monkeypatch.setattr(webhook, "get_nextcloud_manager", lambda: SimpleNamespace(
        get_bot_by_conversation=lambda token: SimpleNamespace(llm_bawt_bot="nova", secret="secret")
    ))
    calls = []

    async def send(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(webhook, "send_nextcloud_message", send)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_kind,source_type,reference", [
    ("chat.completion.queued", "delivery", "delivery-talk"),
    ("chat.completion.in_flight", "turn", "turn-original"),
])
async def test_accepted_talk_registers_exact_reply_before_ack_and_notifies_once(
    monkeypatch, talk, receipt_kind, source_type, reference,
):
    subscriptions = []
    wakes = []

    async def complete(request):
        assert request.user == "nick"
        return {"object": receipt_kind, "delivery_id" if source_type == "delivery" else "turn_id": reference,
                "user_message_id": request.user_message_id}

    def register(subscription):
        subscriptions.append(subscription)
        return "PENDING", len(subscriptions) == 1

    dispatcher = SimpleNamespace(talk_replies=SimpleNamespace(register=register), wake=wakes.append)
    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(
        chat_completion=complete, _inter_bot_dispatcher=dispatcher,
    ))
    first = await webhook.handle_nextcloud_webhook(FakeRequest())
    second = await webhook.handle_nextcloud_webhook(FakeRequest())
    assert first["reference"] == second["reference"] == reference
    assert first["reply_state"] == "PENDING"
    assert first["status"] == ("queued" if source_type == "delivery" else "in_flight")
    assert subscriptions[0] == subscriptions[1]
    assert subscriptions[0].source_type == source_type
    assert subscriptions[0].source_id == reference
    assert subscriptions[0].reply_to == 42
    assert subscriptions[0].user_id == "nick"
    assert len(talk) == 1
    assert talk[0]["reply_to"] == 42
    assert wakes == ["nova", "nova"]


@pytest.mark.asyncio
async def test_webhook_retry_reuses_original_message_identity(monkeypatch, talk):
    ids = []

    async def complete(request):
        ids.append(request.user_message_id)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])

    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(chat_completion=complete))
    await webhook.handle_nextcloud_webhook(FakeRequest("42"))
    await webhook.handle_nextcloud_webhook(FakeRequest("42"))
    await webhook.handle_nextcloud_webhook(FakeRequest("43"))
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


@pytest.mark.asyncio
async def test_completed_turn_and_retried_webhook_share_one_outbound_answer(monkeypatch, talk):
    from datetime import datetime, timezone

    subscriptions = []
    wakes = []

    async def complete(request):
        if not subscriptions:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])
        return {"object": "chat.completion.in_flight", "turn_id": subscriptions[0].source_id,
                "user_message_id": request.user_message_id}

    def get_turn(turn_id):
        import uuid
        return SimpleNamespace(
            bot_id="nova", user_id="nick",
            trigger_message_id=str(uuid.UUID(hex=turn_id.removeprefix("turn-"))),
            ended_at=datetime.now(timezone.utc), response_text="answer",
        )

    def register(subscription):
        subscriptions.append(subscription)
        return "PENDING", len(subscriptions) == 1

    # A response choice must bind to the persisted exact turn before it can
    # become an outbound reply subscription.
    dispatcher = SimpleNamespace(talk_replies=SimpleNamespace(register=register), wake=wakes.append)
    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(
        chat_completion=complete, _turn_log_store=SimpleNamespace(get_turn=get_turn),
        _inter_bot_dispatcher=dispatcher,
    ))
    first = await webhook.handle_nextcloud_webhook(FakeRequest())
    second = await webhook.handle_nextcloud_webhook(FakeRequest())
    assert first["status"] == "reply_pending" and second["status"] == "in_flight"
    assert first["reference"] == second["reference"] == subscriptions[0].source_id
    assert subscriptions[0] == subscriptions[1]
    assert talk == []  # only the worker may post the actual answer
    assert wakes == ["nova", "nova"]


@pytest.mark.asyncio
async def test_agent_completion_without_exact_persisted_turn_never_posts_unguarded_answer(monkeypatch, talk):
    import llm_bawt.bots as bots

    monkeypatch.setattr(bots, "BotManager", lambda config: SimpleNamespace(
        get_bot=lambda bot: SimpleNamespace(agent_backend="claude-code", harness=None)
    ))

    async def complete(request):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])

    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(
        config=object(), chat_completion=complete,
        _turn_log_store=SimpleNamespace(get_turn=lambda turn_id: None),
    ))
    result = await webhook.handle_nextcloud_webhook(FakeRequest())
    assert result["status"] == "accepted_unverified"
    assert len(talk) == 1
    assert talk[0]["message"] != "answer"


@pytest.mark.asyncio
async def test_completed_answer_with_reply_store_failure_does_not_send_false_error(monkeypatch, talk):
    async def complete(request):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])

    def unavailable(turn_id):
        raise ConnectionError("turn database unavailable")

    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(
        chat_completion=complete, _turn_log_store=SimpleNamespace(get_turn=unavailable),
    ))
    result = await webhook.handle_nextcloud_webhook(FakeRequest())
    assert result["status"] == "accepted_unverified"
    assert len(talk) == 1
    assert talk[0]["message"] != "answer"


@pytest.mark.asyncio
async def test_failed_registration_after_accepted_ack_is_not_a_false_processing_error(monkeypatch, talk):
    async def complete(request):
        return {"object": "chat.completion.queued", "delivery_id": "delivery-talk",
                "user_message_id": request.user_message_id}

    def unavailable(_subscription):
        raise ConnectionError("reply database unavailable")

    dispatcher = SimpleNamespace(talk_replies=SimpleNamespace(register=unavailable))
    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(
        chat_completion=complete, _inter_bot_dispatcher=dispatcher,
    ))
    result = await webhook.handle_nextcloud_webhook(FakeRequest())
    assert result["status"] == "accepted_unverified"
    assert len(talk) == 1
    assert "accepted" in talk[0]["message"]
    assert "Sorry" not in talk[0]["message"]


@pytest.mark.asyncio
async def test_outbound_uncertainty_never_posts_second_message(monkeypatch, talk):
    async def send(**kwargs):
        talk.append(kwargs)
        raise ConnectionError("outbound result unknown")

    async def complete(request):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])

    monkeypatch.setattr(webhook, "send_nextcloud_message", send)
    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(chat_completion=complete))
    result = await webhook.handle_nextcloud_webhook(FakeRequest())
    assert result["status"] == "outbound_uncertain"
    assert len(talk) == 1
    assert talk[0]["message"] == "answer"
