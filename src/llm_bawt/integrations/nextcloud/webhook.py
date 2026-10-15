"""Nextcloud Talk webhook handler."""

import hmac
import hashlib
import logging
import secrets
import json
import uuid
from datetime import datetime
from typing import Optional

import httpx
from fastapi import Request, HTTPException

from ...bot_types import durable_agent_backend
from ...service.delivery_reply import QUEUED_RECEIPT, is_accepted_receipt
from .manager import get_nextcloud_manager

log = logging.getLogger(__name__)


def verify_nextcloud_signature(
    signature: str,
    random: str,
    body: bytes,
    secret: str
) -> bool:
    """Verify HMAC-SHA256 signature from Nextcloud Talk."""
    expected = hmac.new(
        secret.encode('utf-8'),
        random.encode('utf-8') + body,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature.lower())


async def send_nextcloud_message(
    nextcloud_url: str,
    conversation_token: str,
    message: str,
    secret: str,
    reply_to: Optional[int] = None,
) -> bool:
    """
    Send a message to Nextcloud Talk.

    Args:
        nextcloud_url: Nextcloud instance URL
        conversation_token: The conversation/room token
        message: The message text to send
        secret: Bot secret for signing
        reply_to: Optional message ID to reply to

    Returns:
        True if successful, False otherwise
    """
    url = f"{nextcloud_url}/ocs/v2.php/apps/spreed/api/v1/bot/{conversation_token}/message"

    # Build payload
    payload = {"message": message}
    if reply_to:
        payload["replyTo"] = reply_to

    body = json.dumps(payload).encode('utf-8')

    # Generate random nonce
    random_string = secrets.token_hex(32)

    # Calculate signature: HMAC-SHA256(random + message_text, secret)
    # NOTE: Sign the MESSAGE TEXT, not the full JSON body!
    signature = hmac.new(
        secret.encode('utf-8'),
        (random_string + message).encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    # Set headers
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'OCS-APIRequest': 'true',
        'X-Nextcloud-Talk-Bot-Random': random_string,
        'X-Nextcloud-Talk-Bot-Signature': signature,
    }

    # Log outgoing message (truncated preview)
    preview = message[:100] + "..." if len(message) > 100 else message
    log.info(f"📤 Sending ({len(message)} chars): {preview!r}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, content=body)

            if response.status_code == 201:
                log.info(f"✓ Message sent to {conversation_token} ({len(message)} chars)")
                return True
            else:
                log.error(f"Failed to send message: {response.status_code} {response.text}")
                log.error(f"Message was ({len(message)} chars): {message[:200]}...")
                return False
    except Exception as e:
        log.error(f"Error sending message: {e}")
        return False


async def handle_nextcloud_webhook(request: Request) -> dict:
    """
    Handle incoming Nextcloud Talk webhooks.
    Routes to appropriate bot based on conversation token.
    """
    # Get signature headers
    signature = request.headers.get('X-Nextcloud-Talk-Signature', '')
    random = request.headers.get('X-Nextcloud-Talk-Random', '')
    backend = request.headers.get('X-Nextcloud-Talk-Backend', '')

    # Get raw body
    body = await request.body()

    # Parse payload
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    conversation_token = payload.get('target', {}).get('id', '')

    # Log the webhook
    log.info("=" * 80)
    log.info(f"Nextcloud Talk webhook received at {datetime.now()}")
    log.info(f"Backend: {backend}")
    log.info(f"Conversation: {conversation_token}")

    # Look up bot configuration
    manager = get_nextcloud_manager()
    bot_config = manager.get_bot_by_conversation(conversation_token)

    if not bot_config:
        log.warning(f"No bot configured for conversation {conversation_token}")
        log.info("=" * 80)
        return {"status": "no_bot_configured", "conversation": conversation_token}

    log.info(f"Routing to bot: {bot_config.llm_bawt_bot}")

    # Verify signature with this bot's secret
    if not verify_nextcloud_signature(signature, random, body, bot_config.secret):
        log.error(f"Invalid signature for bot {bot_config.llm_bawt_bot}")
        log.info("=" * 80)
        raise HTTPException(status_code=401, detail="Invalid signature")

    log.info("✓ Signature verified")

    # Extract message info
    msg_type = payload.get('type', 'unknown')
    actor_name = payload.get('actor', {}).get('name', 'unknown')
    user_id = payload.get('actor', {}).get('id', 'users/unknown')

    # Clean up user_id (remove "users/" prefix)
    if user_id.startswith('users/'):
        user_id = user_id[6:]

    # Parse the actual message text from the content field
    content = payload.get('object', {}).get('content', '')
    try:
        content_json = json.loads(content) if content else {}
        message = content_json.get('message', '')
    except json.JSONDecodeError:
        message = payload.get('object', {}).get('name', '')

    short_msg = message[:80] + "..." if len(message) > 80 else message
    log.info(f"📨 {actor_name}: \"{short_msg}\"")

    # Process message with appropriate bot
    if message and msg_type == "Create":
        accepted_reference: str | None = None
        sending_reply = False
        try:
            # Import here to avoid circular imports
            from ...service.dependencies import get_service
            from ...service.schemas import ChatCompletionRequest, ChatMessage

            service = get_service()
            # Talk may resend the same Create webhook after a timeout. Bind its
            # message id to the ordinary admission key so retries cannot start
            # a second agent turn. Older payloads without an id remain unkeyed.
            talk_message_id = payload.get('object', {}).get('id')
            if isinstance(talk_message_id, (str, int)) and str(talk_message_id).strip():
                user_message_id = str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"nextcloud-talk:{backend.rstrip('/')}:{conversation_token}:{user_id}:{talk_message_id}",
                ))
            else:
                user_message_id = str(uuid.uuid4())
            reply_to = int(str(talk_message_id)) if str(talk_message_id).isdigit() else None

            # Mint the original identity at ingress, before ordinary admission.
            request_obj = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content=message)],
                user=user_id,
                bot_id=bot_config.llm_bawt_bot,
                user_message_id=user_message_id,
                stream=False,
            )

            response = await service.chat_completion(request_obj)
            receipt_kind = response.get("object") if isinstance(response, dict) else None
            source_type = None
            source_id = None
            if is_accepted_receipt(response):
                source_type = "delivery" if receipt_kind == QUEUED_RECEIPT else "turn"
                source_id = response.get("delivery_id" if source_type == "delivery" else "turn_id")
                accepted_reference = source_id if isinstance(source_id, str) and source_id else None
                if not accepted_reference or response.get("user_message_id") != user_message_id:
                    raise ValueError("Nextcloud receipt does not match the submitted message")
            elif getattr(response, "choices", None) and getattr(service, "_turn_log_store", None) is not None:
                # A completed ordinary agent turn has the same deterministic ID
                # as its retry. Route its reply through the outbound claim too;
                # otherwise a Talk webhook retry could POST the answer twice.
                from ...service.ordinary_turn_queue import stable_ordinary_ids
                import asyncio

                candidate, _, _ = stable_ordinary_ids(
                    request_obj, bot_config.llm_bawt_bot, user_id,
                )
                accepted_reference = candidate  # failure here is uncertain, not a failed inference
                persisted = await asyncio.to_thread(service._turn_log_store.get_turn, candidate)
                if (persisted is not None and persisted.bot_id == bot_config.llm_bawt_bot
                        and persisted.user_id == user_id
                        and persisted.trigger_message_id == user_message_id):
                    source_type, source_id = "turn", candidate
                    accepted_reference = candidate
                else:
                    from ...bots import BotManager
                    configured = BotManager(service.config).get_bot(bot_config.llm_bawt_bot)
                    if configured and durable_agent_backend(configured):
                        # Accepted agent work without its expected durable row
                        # is uncertain; never fall back to an untracked POST.
                        accepted_reference = candidate
                        raise RuntimeError("Completed agent turn lacks its exact durable reply row")
            if source_type is not None:
                from ...service.talk_reply_outbox import TalkReplySubscription

                dispatcher = service._inter_bot_dispatcher
                if dispatcher is None or dispatcher.talk_replies is None:
                    raise RuntimeError("Talk reply outbox unavailable")
                # Register before returning an ACK. A finished turn is also
                # picked up by the next sweep, including after process restart.
                import asyncio
                state, newly_registered = await asyncio.to_thread(
                    dispatcher.talk_replies.register,
                    TalkReplySubscription(
                        message_id=user_message_id, bot_id=bot_config.llm_bawt_bot,
                        user_id=user_id, source_type=source_type, source_id=source_id,
                        backend=backend.rstrip('/'), conversation_token=conversation_token,
                        reply_to=reply_to,
                    ),
                )
                if newly_registered and receipt_kind is not None:
                    # One visible pending notice per original Talk message; a
                    # webhook redelivery cannot produce repeated notices.
                    await send_nextcloud_message(
                        nextcloud_url=backend.rstrip('/'),
                        conversation_token=conversation_token,
                        message=f"Your request was accepted (reference {source_id}); I'll reply here when it finishes.",
                        secret=bot_config.secret, reply_to=reply_to,
                    )
                if state == "PENDING":
                    dispatcher.wake(bot_config.llm_bawt_bot)
                # Never send a second answer from this HTTP request: the outbox
                # owns the only outbound attempt, even if the turn just finished.
                return {"status": ("queued" if source_type == "delivery" else
                                   "in_flight" if receipt_kind is not None else "reply_pending"),
                        "reference": source_id, "reply_state": state,
                        "bot": bot_config.llm_bawt_bot, "message": message, "from": actor_name}
            elif getattr(response, "choices", None):
                llm_response = response.choices[0].message.content
            else:
                raise RuntimeError("No assistant response or queued delivery receipt")

            if llm_response:
                resp_preview = llm_response[:100] + "..." if len(llm_response) > 100 else llm_response
                log.info(f"🤖 Response ({len(llm_response)} chars): {resp_preview!r}")
                sending_reply = True
                await send_nextcloud_message(
                    nextcloud_url=backend.rstrip('/'),
                    conversation_token=conversation_token,
                    message=llm_response,
                    secret=bot_config.secret,
                    reply_to=reply_to,
                )

        except Exception as e:
            log.error(f"Error processing message with bot {bot_config.llm_bawt_bot}: {e}", exc_info=True)
            if sending_reply:
                # The outbound POST might have succeeded before its connection
                # failed. A second message would make an ambiguous send worse.
                return {"status": "outbound_uncertain", "bot": bot_config.llm_bawt_bot,
                        "message": message, "from": actor_name}
            if accepted_reference:
                await send_nextcloud_message(
                    nextcloud_url=backend.rstrip('/'),
                    conversation_token=conversation_token,
                    message=f"Your request was accepted (reference {accepted_reference}), but I can't verify its answer yet.",
                    secret=bot_config.secret,
                )
                return {"status": "accepted_unverified", "reference": accepted_reference,
                        "bot": bot_config.llm_bawt_bot, "message": message, "from": actor_name}
            await send_nextcloud_message(
                nextcloud_url=backend.rstrip('/'),
                conversation_token=conversation_token,
                message="Sorry, I encountered an error processing your message.",
                secret=bot_config.secret,
            )

    return {
        "status": "received",
        "bot": bot_config.llm_bawt_bot,
        "message": message,
        "from": actor_name
    }
