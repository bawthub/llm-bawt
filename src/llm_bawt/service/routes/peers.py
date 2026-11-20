"""Owner controls for peer installations (TASK-2517).

Every route takes the BawtHub service bearer (``require_service_token``):
BawtHub's server is the only caller and decides that the signed-in person is
the owner; bots never hold the token, whatever the general API auth mode. They create invites, join with an invite after checking the
peer's fingerprint, confirm or revoke peers, choose which local bots each peer
may see, and read a peer's shared bots. The peer-facing endpoint is separate
(``llm_bawt.peer.server``).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .service_auth import require_service_token

router = APIRouter(dependencies=[Depends(require_service_token)])


def _require_enabled():
    from ...peer import runtime

    if not runtime.enabled() or not runtime.public_url():
        raise HTTPException(status_code=409, detail="Peer networking is off on this install")
    return runtime


class JoinBody(BaseModel):
    code: str = Field(max_length=2048)
    # Absent means preview; a supplied value must be a real confirmation.
    confirm_fingerprint: str | None = Field(default=None, min_length=1, max_length=64)


class ConfirmBody(BaseModel):
    confirm_fingerprint: str = Field(min_length=1, max_length=64)


class ShareBody(BaseModel):
    """One bot's sharing, so an edit from a stale page can't undo another."""
    bot: str = Field(min_length=1, max_length=64)
    shared: bool


@router.get("/v1/peers", tags=["Peers"])
async def list_peers():
    from ...peer import identity, store

    rt = _require_enabled()
    with rt.engine().begin() as conn:
        own = identity.load_or_create(conn)
        links = store.list_links(conn)
    return {"self": {"install_id": own.install_id, "label": own.label,
                     "fingerprint": own.fingerprint, "url": rt.public_url()},
            "peers": [dict(link.public(), fingerprint=identity.fingerprint(link.public_key))
                      for link in links]}


@router.post("/v1/peers/invites", tags=["Peers"])
async def create_invite():
    from ...peer import identity, pairing

    rt = _require_enabled()
    with rt.engine().begin() as conn:
        invite, expires = pairing.create_invite(conn, identity.load_or_create(conn),
                                                rt.public_url())
    return {"code": invite.code(), "expires_at": expires, "fingerprint": invite.fingerprint}


async def _pinned_certificate(invite) -> str | None:
    """Read an HTTPS peer's certificate and keep it only if its key is the invite's."""
    from ...peer import client, identity, pairing, tls

    if not invite.base_url.startswith("https://"):
        return None
    cert = await client.fetch_certificate(invite.base_url)
    try:
        cert = tls.canonical(cert)
        matches = identity.fingerprint(tls.public_key_of(cert)) == invite.fingerprint
    except tls.CertificateError:
        matches = False
    if not matches:
        raise pairing.PairingError("the peer's TLS certificate does not match the invite", 409)
    return cert


@router.post("/v1/peers/join", tags=["Peers"])
async def join(body: JoinBody):
    """Without ``confirm_fingerprint``: preview who the invite names. With it: pair."""
    from ...peer import client, identity, pairing, store

    rt = _require_enabled()
    try:
        invite = pairing.parse_code(body.code)
        cert = await _pinned_certificate(invite)
        challenge = pairing.new_challenge()
        remote = await client.fetch_identity(invite.base_url, challenge, cert)
        pairing.check_remote_identity(invite, remote, challenge)
        cert = pairing.peer_certificate(invite.base_url, cert, remote["public_key"])
    except pairing.PairingError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.reason) from None
    except client.PeerUnavailable as exc:
        raise HTTPException(status_code=502, detail=exc.reason) from None
    preview = {"install_id": remote["install_id"], "label": str(remote.get("label"))[:80],
               "fingerprint": invite.fingerprint, "url": invite.base_url}
    if body.confirm_fingerprint is None:
        return {"preview": preview}
    if body.confirm_fingerprint.strip().upper() != invite.fingerprint:
        raise HTTPException(status_code=409, detail="Fingerprint confirmation does not match")
    with rt.engine().begin() as conn:
        own = identity.load_or_create(conn)
    try:
        reply = await client.send_pair(invite.base_url,
                                       pairing.pair_request_body(own, invite, rt.public_url()),
                                       cert)
        pairing.check_pair_ack(own, invite, remote["public_key"], reply)
    except client.PeerUnavailable as exc:
        raise HTTPException(status_code=502, detail=exc.reason) from None
    except pairing.PairingError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.reason) from None
    with rt.engine().begin() as conn:
        store.upsert_pending(conn, install_id=preview["install_id"], label=preview["label"],
                             base_url=invite.base_url, public_key=remote["public_key"],
                             key_id=identity.key_id(remote["public_key"]), tls_cert=cert)
        store.set_state(conn, preview["install_id"], "active")
    return {"paired": preview, "note": "The other install's owner must also confirm."}


@router.post("/v1/peers/{install_id}/confirm", tags=["Peers"])
async def confirm(install_id: str, body: ConfirmBody):
    from ...peer import identity, store

    rt = _require_enabled()
    with rt.engine().begin() as conn:
        link = store.get_link(conn, install_id)
        if link is None or link.state != "pending":
            raise HTTPException(status_code=404, detail="No pending peer with that ID")
        want = identity.fingerprint(link.public_key)
        if body.confirm_fingerprint.strip().upper() != want:
            raise HTTPException(status_code=409, detail="Fingerprint confirmation does not match")
        store.set_state(conn, install_id, "active")
    return {"install_id": install_id, "state": "active"}


@router.post("/v1/peers/{install_id}/revoke", tags=["Peers"])
async def revoke(install_id: str):
    from ...peer import store

    rt = _require_enabled()
    with rt.engine().begin() as conn:
        if not store.set_state(conn, install_id, "revoked"):
            raise HTTPException(status_code=404, detail="Unknown peer")
        store.set_export(conn, install_id, [])
    return {"install_id": install_id, "state": "revoked"}


@router.put("/v1/peers/{install_id}/export", tags=["Peers"])
async def set_shared(install_id: str, body: ShareBody):
    from ... import bots
    from ...campus_events import visibility
    from ...peer import store

    rt = _require_enabled()
    slug = body.bot.strip().lower()
    if body.shared:
        from ...bot_types import durable_agent_backend

        bot = bots.get_bot(slug)
        if bot is None or not visibility()(slug):
            raise HTTPException(status_code=400, detail=f"Cannot share: {body.bot}")
        # Peers reach a bot only by queued delivery; a bot that can't take
        # one would be listed and then refuse every message.
        if not durable_agent_backend(bot):
            raise HTTPException(status_code=400,
                                detail=f"{body.bot} can't receive messages from other bots")
    with rt.engine().begin() as conn:
        link = store.get_link(conn, install_id)
        if link is None or link.state == "revoked":
            raise HTTPException(status_code=404, detail="Unknown or revoked peer")
        shared = store.set_shared(conn, install_id, slug, body.shared)
    return {"install_id": install_id, "export_bots": shared}


@router.get("/v1/peers/{install_id}/bots", tags=["Peers"])
async def remote_bots(install_id: str):
    """The bots that peer shares with us, qualified by its location."""
    from ...peer import discovery, identity, store

    rt = _require_enabled()
    with rt.engine().begin() as conn:
        own = identity.load_or_create(conn)
        link = store.get_link(conn, install_id)
    if link is None:
        raise HTTPException(status_code=404, detail="Unknown peer")
    return {"peer": link.public(), **await discovery.peer_bots(own, link)}
