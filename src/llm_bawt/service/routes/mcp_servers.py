"""Remote MCP servers: add by URL, sign in, grant to bots or tags (TASK-2610).

Mounted under ``/v1/integrations/mcp-servers`` so BawtHub's integration proxy
applies its same-origin guard and un-followed OAuth redirects. Writes and the
OAuth callback need the service bearer, which BawtHub adds only after its admin
check. Each server's sign-in has its own browser-binding cookie, scoped to that
server's callback path, so concurrent sign-ins never clobber each other.
Bodies are parsed by hand: a FastAPI 422 would echo a bearer token.
"""
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse

from llm_bawt.integrations.connections import IntegrationError
from llm_bawt.integrations.mcp_servers.oauth import RETURN_PATHS, callback_path
from llm_bawt.integrations.mcp_servers.service import McpServers
from llm_bawt.integrations.mcp_servers.store import McpServerError, validate_slug
from ..dependencies import get_service
from .service_auth import require_service_token

router = APIRouter(prefix="/v1/integrations/mcp-servers")
_SERVICE = [Depends(require_service_token)]
SETTINGS_PATH = next(iter(RETURN_PATHS))
_NO_STORE = {"Cache-Control": "no-store"}
COOKIE = "__Secure-bawthub-mcp-login"
_COOKIE_FLAGS = {"secure": True, "httponly": True, "samesite": "lax"}


def cookie_path(slug: str) -> str:
    """This server's callback path in the browser (behind BawtHub's integration proxy)."""
    return callback_path(validate_slug(slug))


def servers() -> McpServers:
    return McpServers(get_service().config)


async def call(fn, *args, **kwargs):
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except McpServerError as exc:
        raise HTTPException(404 if str(exc) == "not_found" else 400, str(exc)) from exc
    except IntegrationError as exc:
        raise HTTPException(400, str(exc)) from exc


async def body_of(request: Request, allowed: set[str]) -> dict:
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict) or not set(body) <= allowed:
        raise HTTPException(400, "invalid_field")
    return body


def _str(body: dict, key: str, default: str | None = None) -> str | None:
    value = body.get(key, default)
    if value is not None and not isinstance(value, str):
        raise HTTPException(400, "invalid_field")
    return value


def _actor(request: Request) -> str | None:
    return request.query_params.get("user_id") or None


@router.get("")
async def list_servers():
    return JSONResponse({"servers": await call(servers().list)}, headers=_NO_STORE)


@router.post("", dependencies=_SERVICE)
async def create_server(request: Request):
    body = await body_of(request, {"slug", "name", "url", "transport", "auth_kind", "bearer_token"})
    result = await call(servers().create, slug=_str(body, "slug", ""), name=_str(body, "name", ""),
                        url=_str(body, "url", ""), transport=_str(body, "transport", "streamable_http"),
                        auth_kind=_str(body, "auth_kind", "oauth"), bearer_token=_str(body, "bearer_token"),
                        actor=_actor(request))
    return JSONResponse(result, status_code=201, headers=_NO_STORE)


@router.patch("/{slug}", dependencies=_SERVICE)
async def update_server(slug: str, request: Request):
    body = await body_of(request, {"name", "enabled"})
    enabled = body.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise HTTPException(400, "invalid_field")
    return await call(servers().update, slug, name=_str(body, "name"), enabled=enabled, actor=_actor(request))


@router.delete("/{slug}", dependencies=_SERVICE)
async def delete_server(slug: str):
    return {"deleted": await call(servers().delete, slug)}


@router.put("/{slug}/bearer", dependencies=_SERVICE)
async def set_bearer(slug: str, request: Request):
    body = await body_of(request, {"token"})
    service = servers()
    await call(service.set_bearer, slug, _str(body, "token", ""))
    return await call(lambda: service.describe(service.get(slug)))


@router.put("/{slug}/grants", dependencies=_SERVICE)
async def set_grants(slug: str, request: Request):
    body = await body_of(request, {"grants"})
    return await call(servers().set_grants, slug, body.get("grants"), _actor(request))


@router.post("/{slug}/tools", dependencies=_SERVICE)
async def list_tools(slug: str):
    """Connect now and list the server's tools: the Settings "Test" button."""
    return {"tools": await call(servers().list_tools, slug)}


@router.post("/{slug}/connect", dependencies=_SERVICE)
async def connect(slug: str, request: Request):
    service = servers()
    server = await call(service.get, slug)
    body = await body_of(request, {"return_to"})
    result = await call(service.oauth.start, server, request.headers.get("origin", ""),
                        _str(body, "return_to", SETTINGS_PATH))
    browser = result.pop("browser")
    response = JSONResponse(result, headers=_NO_STORE)
    response.set_cookie(COOKIE, browser, max_age=600, path=cookie_path(server.slug), **_COOKIE_FLAGS)
    return response


@router.get("/{slug}/callback", dependencies=_SERVICE)
async def callback(slug: str, request: Request, state: str = "", code: str = "", error: str = "", iss: str = ""):
    service = servers()
    try:
        server = await run_in_threadpool(service.get, slug)
        return_to = await run_in_threadpool(service.oauth.finish, server, state, request.cookies.get(COOKIE, ""),
                                            code, error, iss)
        location = f"{return_to}?{urlencode({'mcp_server': slug, 'mcp_result': 'connected'})}"
    except (IntegrationError, McpServerError) as exc:
        location = f"{SETTINGS_PATH}?{urlencode({'mcp_server': slug, 'mcp_error': str(exc)})}"
    response = RedirectResponse(location, status_code=303,
                                headers={**_NO_STORE, "Referrer-Policy": "no-referrer"})
    try:
        response.delete_cookie(COOKIE, path=cookie_path(slug), **_COOKIE_FLAGS)
    except McpServerError:
        pass  # a malformed slug never had a cookie
    return response


@router.delete("/{slug}/connection", dependencies=_SERVICE)
async def sign_out(slug: str):
    """Forget the server's tokens and client registration; the server row stays."""
    return {"disconnected": await call(servers().sign_out, slug)}
