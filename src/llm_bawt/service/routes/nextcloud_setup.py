"""Browser-bound Nextcloud setup API; separate from Google-specific handlers."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from llm_bawt.integrations.nextcloud.setup import NextcloudSetup, SetupError, LOGIN_TTL
from llm_bawt.integrations.nextcloud.setup_bindings import NextcloudBindings
from ..dependencies import get_service

router = APIRouter(prefix="/v1/integrations/nextcloud-talk", tags=["Integrations"])
COOKIE = "__Secure-bawthub-nextcloud-login"
COOKIE_PATH = "/api/chat/proxy/v1/integrations/nextcloud-talk"


def manager() -> NextcloudSetup:
    config = get_service().config
    return NextcloudSetup(config, bindings=NextcloudBindings(config))


async def call(fn, *args):
    try:
        result = await run_in_threadpool(fn, *args)
    except SetupError as exc:
        raise HTTPException(exc.status, {"code": exc.code, "message": exc.message},
                            headers={"Cache-Control": "no-store"}) from None
    return JSONResponse(result, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


class LoginBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_url: str = Field(min_length=1, max_length=2048)


class BindingBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bot_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    room_token: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9]+$")
    nextcloud_bot_id: int = Field(gt=0)
    user_id: str = Field(min_length=1, max_length=128)


@router.get("/status")
async def status():
    return await call(manager().status)


@router.post("/login")
async def login(body: LoginBody):
    service = manager()
    try:
        result = await run_in_threadpool(service.start, body.instance_url)
    except SetupError as exc:
        raise HTTPException(exc.status, {"code": exc.code, "message": exc.message}, headers={"Cache-Control": "no-store"}) from None
    browser = result.pop("browser")
    response = JSONResponse(result, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
    response.set_cookie(COOKIE, browser, max_age=LOGIN_TTL, secure=True, httponly=True,
                        samesite="strict", path=COOKIE_PATH)
    return response


@router.post("/login/poll")
async def poll(request: Request):
    return await call(manager().poll, request.cookies.get(COOKIE, ""))


@router.get("/rooms")
async def rooms():
    return await call(manager().rooms)


@router.get("/bots")
async def bots():
    return await call(manager().bots)


@router.post("/bindings")
async def connect_binding(body: BindingBody):
    return await call(manager().bindings.connect, body.bot_id, body.room_token, body.nextcloud_bot_id, body.user_id)


@router.delete("/bindings/{bot_id}")
async def disconnect_binding(bot_id: str):
    return await call(manager().bindings.disconnect, bot_id)


@router.delete("/account")
@router.delete("")
async def disconnect_account():
    response = await call(manager().disconnect)
    response.delete_cookie(COOKIE, path=COOKIE_PATH, secure=True, httponly=True, samesite="strict")
    return response
