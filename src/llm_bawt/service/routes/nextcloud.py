"""Nextcloud webhook and admin routes."""

import asyncio

from fastapi import APIRouter, HTTPException, Request

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import NextcloudProvisionRequest, NextcloudProvisionResponse

router = APIRouter()
log = get_service_logger(__name__)

@router.post("/webhook/nextcloud", tags=["Webhooks"])
async def nextcloud_talk_webhook(request: Request):
    """Handle incoming Nextcloud Talk webhooks."""
    from llm_bawt.integrations.nextcloud.webhook import handle_nextcloud_webhook
    return await handle_nextcloud_webhook(request)

# -------------------------------------------------------------------------
# Admin: Nextcloud Talk Provisioning
# -------------------------------------------------------------------------

@router.post("/admin/nextcloud-talk/provision", response_model=NextcloudProvisionResponse, tags=["Admin"])
async def provision_nextcloud_talk(request: NextcloudProvisionRequest):
    """Provision a Nextcloud Talk room and bot for an llm_bawt bot."""
    from llm_bawt.integrations.nextcloud.provisioner import get_provisioner_client
    from llm_bawt.integrations.nextcloud.manager import get_nextcloud_manager

    manager = get_nextcloud_manager()

    # Check if already configured
    if manager.get_bot(request.bot_id):
        raise HTTPException(
            status_code=400,
            detail=f"Bot '{request.bot_id}' already has Nextcloud config"
        )

    # Defaults
    room_name = request.room_name or request.bot_id.title()
    bot_name = request.bot_name or request.bot_id.title()

    try:
        # Resolved per request (a blocking credential-store read), so a saved
        # or rotated token applies without a restart.
        provisioner = await asyncio.to_thread(get_provisioner_client, get_service().config)

        # Provision via service
        result = await provisioner.provision_talk_room_and_bot(
            room_name=room_name,
            bot_name=bot_name,
            owner_user_id=request.owner_user_id,
        )

        # Save config
        manager.add_bot(
            llm_bawt_bot=request.bot_id,
            nextcloud_bot_id=result.bot_id,
            secret=result.bot_secret,
            conversation_token=result.room_token,
        )

        return NextcloudProvisionResponse(
            bot_id=request.bot_id,
            room_token=result.room_token,
            room_url=result.room_url,
            nextcloud_bot_id=result.bot_id,
            nextcloud_bot_name=result.bot_name,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Provisioning failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/admin/nextcloud-talk/reload", tags=["Admin"])
def reload_nextcloud_bots():
    """Force reload Nextcloud bot configuration from disk."""
    from llm_bawt.integrations.nextcloud.manager import get_nextcloud_manager
    manager = get_nextcloud_manager()
    manager.reload()
    bots = manager.list_bots()
    log.info(f"🔄 Reloaded {len(bots)} Nextcloud bots: {[b.llm_bawt_bot for b in bots]}")
    return {
        "status": "reloaded",
        "bots_count": len(bots),
        "bots": [b.llm_bawt_bot for b in bots],
    }

# -------------------------------------------------------------------------
# OpenAI-Compatible Endpoints
# -------------------------------------------------------------------------
