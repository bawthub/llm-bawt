"""The peer endpoint: a small MCP server other installs call (TASK-2517).

It is a separate listener, never the internal ``:8001`` catalog. Routes:

* ``GET  /peer/identity``: this install's public identity (ID, label, public
  key, fingerprint). Public: it is what an invite's fingerprint is checked
  against.
* ``POST /peer/pair``: consume an invite (``pairing.accept_pair_request``).
* ``/peer/mcp``: MCP over streamable HTTP, only for an ``active`` peer whose
  ``Authorization: Bearer <assertion>`` verifies. Its only tool in this slice
  is ``peer_bots_list``: the bots this install exports to *that* peer, with
  minimal metadata and never a private bot.

The authenticated peer is bound per request; tools read its current grant, not
anything the caller sends.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from . import identity, pairing, store

log = logging.getLogger(__name__)

MAX_BODY = 64 * 1024
_current_peer: ContextVar[store.PeerLink | None] = ContextVar("peer_link", default=None)

EngineFactory = Callable[[], Any]
BotLister = Callable[..., list[dict]]


def exported_bots(slugs: tuple[str, ...], *, deliverable_only: bool = True) -> list[dict]:
    """Minimal metadata for exported slugs that exist and are not private.

    Discovery lists only bots that take queued delivery. Sending passes
    ``deliverable_only=False``: admission itself refuses a fresh message to a
    bot that can't take one, but still acknowledges a replay it already admitted.
    """
    from .. import bots
    from ..bot_types import durable_agent_backend
    from ..campus_events import visibility

    visible = visibility()
    out = []
    for slug in slugs:
        bot = bots.get_bot(slug)
        # A shared bot that can no longer take queued delivery isn't listed.
        if bot is None or not visible(bot.slug) or (
                deliverable_only and not durable_agent_backend(bot)):
            continue
        out.append({"slug": bot.slug, "name": bot.name,
                    "description": (bot.description or "")[:200]})
    return out


async def admit_local(link: store.PeerLink, target: str, sender: str, message: str,
                      key: str) -> dict:
    """Admit a peer's message as an ordinary durable delivery to a local bot."""
    from ..inter_bot_send import (DeliveryCreate, DeliveryRefused, DeliverySender,
                                  admission_refusals)
    from ..service.routes.inter_bot_deliveries import _dispatcher
    from ..service.dependencies import get_service
    from . import messaging

    body = DeliveryCreate(
        target_bot_id=target, sender_bot_id="system", author_entity_type="user",
        author_entity_id="system", message=messaging.inbound_text(link, sender, message),
        idempotency_key=f"peer:{link.install_id}:{key}"[:256],
        metadata={messaging.ORIGIN_KEY: messaging.origin(link, sender)},
        prefer_steer=False,
        # Explicit, so a same-key retry never recomputes a reset from the recipient's
        # current context health and conflicts with the admission it is replaying.
        session_policy="continue")
    try:
        # _admit_one directly: peer_origin is server-owned, written only here.
        status, result = await DeliverySender(get_service, dispatcher=_dispatcher)._admit_one(body)
    except DeliveryRefused as exc:
        retry = exc.status_code >= 500 or exc.status_code == 429
        return {"accepted": False, "retry": retry, "error": str(exc.detail)[:300]}
    delivery_id = result.get("delivery_id")
    # A replay of a row failed on arrival repeats that refusal; any other
    # duplicate was admitted before, whatever its state now.
    if result.get("duplicate") and result.get("status") == "FAILED" \
            and result.get("last_error") in admission_refusals(target):
        return {"accepted": False, "retry": False, "error": str(result["last_error"])[:300]}
    if delivery_id and (status == 202 or result.get("duplicate")):
        return {"accepted": True, "delivery_id": delivery_id,
                "duplicate": bool(result.get("duplicate"))}
    return {"accepted": False, "retry": False,
            "error": str(result.get("last_error") or result.get("detail") or status)[:300]}


Admitter = Callable[..., Any]


def build_app(engine_factory: EngineFactory, *, list_bots: BotLister = exported_bots,
              admit: Admitter = admit_local):
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer(name="bawthub-peer",
                    instructions="BawtHub peer endpoint: bots shared with your installation.")

    def own_identity(conn) -> identity.Identity:
        return identity.load_or_create(conn)

    @mcp.tool(name="peer_bots_list",
              description="List the bots this installation shares with you.")
    def peer_bots_list() -> dict:
        link = _current_peer.get()
        if link is None:  # the auth wrapper guarantees this; fail closed anyway
            raise PermissionError("unauthenticated")
        with engine_factory().begin() as conn:
            own = own_identity(conn)
        return {"install_id": own.install_id, "label": own.label,
                "bots": list_bots(link.export_bots)}

    @mcp.tool(name="peer_message_send",
              description="Send a message from one of your bots to a bot shared with you.")
    async def peer_message_send(target: str, sender: str, message: str,
                                idempotency_key: str) -> dict:
        from . import messaging

        link = _current_peer.get()
        if link is None:
            raise PermissionError("unauthenticated")
        try:
            target, sender, message, key = messaging.check_inbound(
                link, {"target": target, "sender": sender, "message": message,
                       "idempotency_key": idempotency_key})
        except messaging.OutboundRefused as exc:
            return {"accepted": False, "error": exc.detail}
        if not list_bots((target,), deliverable_only=False):  # gone, private or inactive since it was shared
            return {"accepted": False, "error": "that bot is not available"}
        return await admit(link, target, sender, message, key)

    @mcp.custom_route("/peer/identity", methods=["GET"])
    async def get_identity(request: Request):
        challenge = request.query_params.get("challenge", "")
        if not 16 <= len(challenge) <= 128:
            return JSONResponse({"error": "challenge required"}, status_code=400)
        with engine_factory().begin() as conn:
            own = own_identity(conn)
        return JSONResponse({"install_id": own.install_id, "label": own.label,
                             "public_key": own.public_key, "fingerprint": own.fingerprint,
                             "proof": pairing.identity_proof(own, challenge)})

    @mcp.custom_route("/peer/pair", methods=["POST"])
    async def pair(request: Request):
        raw = await request.body()
        if len(raw) > MAX_BODY:
            return JSONResponse({"error": "too large"}, status_code=413)
        try:
            body = json.loads(raw)
            with engine_factory().begin() as conn:
                own = own_identity(conn)
                link = pairing.accept_pair_request(conn, own, body)
        except pairing.PairingError as exc:
            return JSONResponse({"error": exc.reason}, status_code=exc.status)
        except (ValueError, TypeError):
            return JSONResponse({"error": "malformed pairing request"}, status_code=400)
        log.info("peer pairing request from %s (%s), pending owner confirmation",
                 link.label, link.install_id)
        return JSONResponse({"state": link.state, "install_id": own.install_id,
                             "label": own.label, "ack": pairing.pair_ack(own, body)})

    inner = mcp.streamable_http_app(streamable_http_path="/peer/mcp", stateless_http=True,
                                    json_response=True, host="0.0.0.0")
    return PeerAuth(inner, engine_factory)


class PeerAuth:
    """ASGI wrapper: ``/peer/mcp`` requires a verified, active peer."""

    def __init__(self, app, engine_factory: EngineFactory):
        self.app = app
        self.engine_factory = engine_factory

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/peer/mcp"):
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return await JSONResponse({"error": "peer assertion required"},
                                      status_code=401)(scope, receive, send)
        try:
            with self.engine_factory().begin() as conn:
                own = identity.load_or_create(conn)
                link = identity.verify_assertion(conn, auth[7:].strip(),
                                                 own_install_id=own.install_id)
        except identity.PeerAuthError as exc:
            log.info("peer call refused: %s", exc.reason)
            return await JSONResponse({"error": exc.reason},
                                      status_code=exc.status)(scope, receive, send)
        token = _current_peer.set(link)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_peer.reset(token)
