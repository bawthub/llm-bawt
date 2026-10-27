"""Connection configuration for external integrations; secrets stay server-side."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from llm_bawt.integrations.connections import IntegrationConnections, IntegrationError, REGISTRY
from llm_bawt.integrations.credential_integrations import CREDENTIAL_INTEGRATIONS, CredentialIntegrations
from ..dependencies import get_service

router = APIRouter(prefix="/v1/integrations")
COOKIE = "__Secure-bawthub-integration"
COOKIE_PATH = "/api/chat/proxy/v1/integrations"


def manager(integration_id: str) -> IntegrationConnections:
    # These handlers implement Google contracts, not the Nextcloud adapter.
    if integration_id != "google-keep":
        raise HTTPException(404, "Unknown integration")
    return IntegrationConnections(get_service().config)


class ClientConfig(BaseModel):
    client_id: str = Field(min_length=1, max_length=512)
    client_secret: str | None = Field(default=None, max_length=2048)
    redirect_uri: str = Field(min_length=1, max_length=2048)


class WorkspaceConfig(BaseModel):
    service_account_json: str = Field(min_length=1, max_length=32768, repr=False)
    subject: str = Field(min_length=3, max_length=254)


@router.get("/{integration_id}/notes")
async def list_keep_notes(integration_id: str):
    return await call(manager(integration_id).list_keep_notes, integration_id)


@router.get("/{integration_id}/notes/{note_id}")
async def read_keep_list(integration_id: str, note_id: str):
    return await call(manager(integration_id).read_keep_list, integration_id, f"notes/{note_id}")


@router.put("/{integration_id}/workspace")
async def configure_workspace(integration_id: str, body: WorkspaceConfig):
    return await call(manager(integration_id).configure_workspace, integration_id,
                      body.service_account_json, body.subject)


@router.post("/{integration_id}/workspace/verify")
async def verify_workspace(integration_id: str):
    return await call(manager(integration_id).verify_workspace, integration_id)


class StartConfig(BaseModel):
    return_to: str = "/tools/settings/integrations"


async def call(fn, *args):
    try:
        return await run_in_threadpool(fn, *args)
    except IntegrationError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("")
async def list_integrations():
    from .nextcloud_setup import manager as nextcloud_manager
    service = IntegrationConnections(get_service().config)
    descriptors = [await run_in_threadpool(service.descriptor, key) for key in REGISTRY if key != "nextcloud-talk"]
    descriptors.append(await run_in_threadpool(nextcloud_manager().descriptor))
    credentials = CredentialIntegrations(get_service().config)
    descriptors.extend([await run_in_threadpool(credentials.descriptor, key) for key in CREDENTIAL_INTEGRATIONS])
    return {"integrations": descriptors}


class CredentialValues(BaseModel):
    # Field names and limits are checked against the integration's definition;
    # a blank secret keeps the stored one.
    values: dict[str, str] = Field(max_length=16, repr=False)


def credential_manager(integration_id: str) -> CredentialIntegrations:
    if integration_id not in CREDENTIAL_INTEGRATIONS:
        raise HTTPException(404, "Unknown integration")
    return CredentialIntegrations(get_service().config)


@router.put("/{integration_id}/credentials")
async def configure_credentials(integration_id: str, body: CredentialValues):
    result = await call(credential_manager(integration_id).configure, integration_id, body.values)
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.put("/{integration_id}/client")
async def configure_client(integration_id: str, body: ClientConfig):
    return await call(manager(integration_id).configure, integration_id,
                      body.client_id.strip(), body.client_secret, body.redirect_uri.strip())


@router.post("/{integration_id}/connect")
async def connect(integration_id: str, body: StartConfig, request: Request):
    result = await call(manager(integration_id).start, integration_id, body.return_to,
                        request.headers.get("origin", ""))
    browser = result.pop("browser")
    response = JSONResponse(result, headers={"Cache-Control": "no-store"})
    response.set_cookie(COOKIE, browser, max_age=600, secure=True, httponly=True,
                        samesite="lax", path=COOKIE_PATH)
    return response


@router.get("/{integration_id}/callback")
async def callback(integration_id: str, request: Request, state: str = "", code: str = "", error: str = ""):
    service = manager(integration_id)
    try:
        return_to = await run_in_threadpool(service.finish, integration_id, state,
                                           request.cookies.get(COOKIE, ""), code, error)
        location = f"{return_to}?integration_result=connected"
    except IntegrationError as exc:
        location = f"/tools/settings/integrations?integration_error={exc}"
    response = RedirectResponse(location, status_code=303,
                                headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
    response.delete_cookie(COOKIE, path=COOKIE_PATH, secure=True, httponly=True, samesite="lax")
    return response


@router.delete("/{integration_id}")
async def disconnect(integration_id: str):
    if integration_id in CREDENTIAL_INTEGRATIONS:
        return {"disconnected": await call(credential_manager(integration_id).disconnect, integration_id)}
    return {"disconnected": await call(manager(integration_id).disconnect, integration_id)}


# ── Keep notes management (TASK-908) ─────────────────────────────────────────
# Full Keep v1 surface under /keep/* so the importer's /notes routes stay stable.

class KeepItem(BaseModel):
    text: str = Field(max_length=1000)
    checked: bool = False
    children: list["KeepItem"] = Field(default_factory=list, max_length=1000)


class KeepNoteBody(BaseModel):
    title: str = Field(default="", max_length=1000)
    text: str | None = Field(default=None, max_length=20000)
    items: list[KeepItem] | None = Field(default=None, max_length=1000)
    collaborators: list[str] = Field(default_factory=list, max_length=50)


class KeepShareBody(BaseModel):
    emails: list[str] = Field(min_length=1, max_length=50)


class KeepUnshareBody(BaseModel):
    names: list[str] = Field(min_length=1, max_length=50)


def keep(integration_id: str):
    from llm_bawt.integrations.keep_notes import KeepNotes
    if integration_id != "google-keep":
        raise HTTPException(404, "Unknown integration")
    return KeepNotes(manager(integration_id), integration_id)


def _items(body: KeepNoteBody):
    return [item.model_dump() for item in body.items] if body.items is not None else None


@router.get("/{integration_id}/keep/notes")
async def keep_list(integration_id: str, trashed: bool = False):
    return await call(lambda: keep(integration_id).list(trashed=trashed))


@router.post("/{integration_id}/keep/notes")
async def keep_create(integration_id: str, body: KeepNoteBody):
    return await call(keep(integration_id).create, body.title, body.text, _items(body), body.collaborators)


@router.get("/{integration_id}/keep/notes/{note_id}")
async def keep_get(integration_id: str, note_id: str):
    return await call(keep(integration_id).get, note_id)


@router.delete("/{integration_id}/keep/notes/{note_id}")
async def keep_delete(integration_id: str, note_id: str):
    return await call(keep(integration_id).delete, note_id)


@router.post("/{integration_id}/keep/notes/{note_id}/replace")
async def keep_replace(integration_id: str, note_id: str, body: KeepNoteBody):
    return await call(keep(integration_id).replace, note_id, body.title, body.text, _items(body))


@router.post("/{integration_id}/keep/notes/{note_id}/permissions")
async def keep_share(integration_id: str, note_id: str, body: KeepShareBody):
    return await call(keep(integration_id).share, note_id, body.emails)


@router.post("/{integration_id}/keep/notes/{note_id}/permissions/remove")
async def keep_unshare(integration_id: str, note_id: str, body: KeepUnshareBody):
    return await call(keep(integration_id).unshare, note_id, body.names)


@router.get("/{integration_id}/keep/notes/{note_id}/attachments/{attachment_id}")
async def keep_attachment(integration_id: str, note_id: str, attachment_id: str, mime_type: str):
    from fastapi.responses import Response
    content, mime = await call(keep(integration_id).attachment, note_id, attachment_id, mime_type)
    ext = mime.split("/", 1)[1].split("+", 1)[0][:10]
    return Response(content, media_type=mime, headers={
        "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        "Content-Disposition": f'{"inline" if mime in INLINE_MIME else "attachment"}; filename="keep-{attachment_id[:40]}.{ext}"'})


INLINE_MIME = {"image/png", "image/jpeg", "image/gif", "image/webp"}
