"""Google Keep notes management over Workspace delegation (TASK-908).

Exposes every operation the official Keep v1 API offers: list (active or
trashed), get, create (text or checklist, one nesting level), delete, share /
unshare collaborators, and attachment download. The API has no update method,
so ``replace`` composes create → re-share → delete and reports partial
outcomes honestly instead of pretending it was an in-place edit.

Upstream bodies are never surfaced; failures map to bounded IntegrationError
codes. Only the configured Workspace subject is ever impersonated.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Iterator

import httpx

from llm_bawt.integrations.connections import IntegrationConnections, IntegrationError, keep_failure_code
from llm_bawt.integrations.keep_workspace import KEEP_URL, WorkspaceAuthError, delegated_token

ID = re.compile(r"[A-Za-z0-9_-]{1,200}")
EMAIL = re.compile(r"[^\s@]{1,64}@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}")
MIME = re.compile(r"[a-z]+/[a-z0-9.+-]{1,100}")
MAX_TITLE, MAX_TEXT, MAX_ITEM, MAX_ITEMS, MAX_PAGES = 999, 19999, 999, 999, 20


def _id(value: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise IntegrationError("invalid_keep_note")
    return value


def _text(value) -> str:
    return value.get("text", "") if isinstance(value, dict) and isinstance(value.get("text"), str) else ""


def _items(raw, depth: int = 0) -> list[dict]:
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        out.append({"text": _text(item.get("text")), "checked": bool(item.get("checked")),
                    "children": _items(item.get("childListItems"), depth + 1) if depth == 0 else []})
    return out


def normalize(note: dict) -> dict:
    """Stable, frontend-friendly shape. Unknown/malformed parts degrade to empty."""
    name = note.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"notes/[A-Za-z0-9_-]+", name):
        raise IntegrationError("keep_invalid_response")
    body = note.get("body") if isinstance(note.get("body"), dict) else {}
    is_list = isinstance(body.get("list"), dict)
    items = _items(body["list"].get("listItems")) if is_list else []
    flat = [i for top in items for i in (top, *top["children"])]
    attachments = []
    for att in note.get("attachments") or []:
        if isinstance(att, dict) and isinstance(att.get("name"), str) and att["name"].startswith(f"{name}/attachments/"):
            mimes = [m for m in att.get("mimeType") or [] if isinstance(m, str) and MIME.fullmatch(m)]
            attachments.append({"id": att["name"].rsplit("/", 1)[1], "mime_types": mimes})
    collaborators = []
    for perm in note.get("permissions") or []:
        if not isinstance(perm, dict) or not isinstance(perm.get("name"), str):
            continue
        kind = "group" if perm.get("group") else "family" if perm.get("family") else "user"
        member = perm.get(kind) if isinstance(perm.get(kind), dict) else {}
        email = perm.get("email") or member.get("email")
        collaborators.append({"name": perm["name"], "role": perm.get("role", ""), "email": email or "",
                              "kind": kind, "deleted": bool(perm.get("deleted"))})
    return {
        "name": name, "id": name.removeprefix("notes/"),
        "title": note.get("title") if isinstance(note.get("title"), str) else "",
        "type": "list" if is_list else "text", "text": _text(body.get("text")), "items": items,
        "item_count": len(flat), "unchecked_count": sum(not i["checked"] for i in flat),
        "create_time": note.get("createTime"), "update_time": note.get("updateTime"),
        "trash_time": note.get("trashTime"), "trashed": bool(note.get("trashed")),
        "attachments": attachments, "collaborators": collaborators,
    }


def build_body(title: str, text: str | None, items: list[dict] | None) -> dict:
    """Validate against documented Keep limits before any network call."""
    title = (title or "").strip()
    if len(title) > MAX_TITLE:
        raise IntegrationError("keep_title_too_long")
    if (text is None) == (items is None):
        raise IntegrationError("keep_invalid_content")
    if text is not None:
        if len(text) > MAX_TEXT:
            raise IntegrationError("keep_text_too_long")
        if not title and not text.strip():
            raise IntegrationError("keep_empty_note")
        return {"title": title, "body": {"text": {"text": text}}}

    def item(raw: dict, depth: int) -> dict:
        value = (raw.get("text") or "").strip() if isinstance(raw, dict) else ""
        if not value:
            raise IntegrationError("keep_empty_item")
        if len(value) > MAX_ITEM:
            raise IntegrationError("keep_item_too_long")
        out = {"text": {"text": value}, "checked": bool(raw.get("checked"))}
        children = raw.get("children") or []
        if children:
            if depth:
                raise IntegrationError("keep_nesting_too_deep")
            out["childListItems"] = [item(child, 1) for child in children]
        return out

    built = [item(raw, 0) for raw in items or []]
    if not built:
        raise IntegrationError("keep_empty_note")
    if len(built) >= MAX_ITEMS + 1 or sum(1 + len(i.get("childListItems", [])) for i in built) > MAX_ITEMS:
        raise IntegrationError("keep_too_many_items")
    return {"title": title, "body": {"list": {"listItems": built}}}


def _emails(values: list[str]) -> list[str]:
    cleaned = sorted({v.strip().lower() for v in values or [] if isinstance(v, str) and v.strip()})
    if not cleaned or len(cleaned) > 50 or any(not EMAIL.fullmatch(v) for v in cleaned):
        raise IntegrationError("invalid_collaborator_email")
    return cleaned


class KeepNotes:
    """Authorized Keep operations for the one connected Workspace subject."""

    def __init__(self, connections: IntegrationConnections, integration_id: str = "google-keep"):
        self.connections = connections
        self.integration_id = integration_id

    @contextmanager
    def _session(self) -> Iterator[tuple[httpx.Client, dict]]:
        store = self.connections.store
        connection = store.load(self.integration_id)
        config = store.load(f"workspace:{self.integration_id}")
        if (not connection or connection.status != "connected"
                or connection.auth_method != "service_account_delegation"
                or not config or not config.secret.get("private_key")
                or config.meta.get("subject") != connection.account):
            raise IntegrationError("keep_not_connected")
        try:
            with self.connections.client_factory(timeout=30, follow_redirects=False) as http:
                token, _ = delegated_token(http, config.meta, config.secret)
                yield http, {"Authorization": f"Bearer {token}"}
        except WorkspaceAuthError as exc:
            raise IntegrationError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise IntegrationError("google_unavailable") from exc

    @staticmethod
    def _check(response: httpx.Response) -> httpx.Response:
        if response.status_code == 404:
            raise IntegrationError("keep_note_not_found")
        if response.status_code == 400:
            raise IntegrationError("keep_request_rejected")
        if response.status_code >= 300:
            raise IntegrationError(keep_failure_code(response))
        return response

    @staticmethod
    def _json(response: httpx.Response) -> dict:
        try:
            data = response.json()
        except ValueError as exc:
            raise IntegrationError("keep_invalid_response") from exc
        if not isinstance(data, dict):
            raise IntegrationError("keep_invalid_response")
        return data

    def list(self, *, trashed: bool = False) -> dict:
        notes, seen, token = [], set(), None
        with self._session() as (http, headers):
            for _ in range(MAX_PAGES):
                # Live API rejects documented `trashed = true` comparisons (400); the
                # default returns active notes and `trashed OR -trashed` returns all,
                # which we split on each note's own `trashed` flag below.
                params = {"pageSize": 100, **({"filter": "trashed OR -trashed"} if trashed else {})}
                if token:
                    params["pageToken"] = token
                page = self._json(self._check(http.get(KEEP_URL, params=params, headers=headers)))
                for note in page.get("notes") or []:
                    if isinstance(note, dict) and bool(note.get("trashed")) == trashed:
                        try:
                            notes.append(normalize(note))
                        except IntegrationError:
                            continue
                token = page.get("nextPageToken")
                if not token:
                    notes.sort(key=lambda n: n.get("update_time") or "", reverse=True)
                    return {"notes": notes, "total": len(notes), "trashed": trashed}
                if not isinstance(token, str) or token in seen:
                    raise IntegrationError("keep_invalid_response")
                seen.add(token)
        raise IntegrationError("keep_too_many_pages")

    def get(self, note_id: str) -> dict:
        with self._session() as (http, headers):
            return normalize(self._json(self._check(http.get(f"{KEEP_URL}/{_id(note_id)}", headers=headers))))

    def _create(self, http, headers, body: dict) -> dict:
        return normalize(self._json(self._check(http.post(KEEP_URL, json=body, headers=headers))))

    def create(self, title: str, text: str | None = None, items: list[dict] | None = None,
               collaborators: list[str] | None = None) -> dict:
        body = build_body(title, text, items)
        emails = _emails(collaborators) if collaborators else []
        with self._session() as (http, headers):
            note = self._create(http, headers, body)
            if not emails:
                return {"note": note}
            try:
                return {"note": self._share(http, headers, note["id"], emails)}
            except IntegrationError as exc:
                return {"note": note, "warning": "collaborators_not_added", "error": str(exc)}
            except httpx.HTTPError:
                # The note exists; whether sharing applied is unknown.
                return {"note": note, "warning": "collaborators_unknown", "error": "google_unavailable"}

    def delete(self, note_id: str) -> dict:
        with self._session() as (http, headers):
            self._check(http.delete(f"{KEEP_URL}/{_id(note_id)}", headers=headers))
        return {"deleted": f"notes/{note_id}"}

    def _share(self, http, headers, note_id: str, emails: list[str]) -> dict:
        parent = f"notes/{note_id}"
        self._check(http.post(f"{KEEP_URL}/{note_id}/permissions:batchCreate", headers=headers, json={
            "requests": [{"parent": parent, "permission": {"email": e, "role": "WRITER"}} for e in emails]}))
        return normalize(self._json(self._check(http.get(f"{KEEP_URL}/{note_id}", headers=headers))))

    def share(self, note_id: str, emails: list[str]) -> dict:
        cleaned = _emails(emails)
        with self._session() as (http, headers):
            return {"note": self._share(http, headers, _id(note_id), cleaned)}

    def unshare(self, note_id: str, permission_names: list[str]) -> dict:
        note_id = _id(note_id)
        prefix = f"notes/{note_id}/permissions/"
        names = sorted(set(permission_names or []))
        if not names or len(names) > 50 or any(
                not isinstance(n, str) or not n.startswith(prefix) or not ID.fullmatch(n[len(prefix):]) for n in names):
            raise IntegrationError("invalid_permission")
        with self._session() as (http, headers):
            self._check(http.post(f"{KEEP_URL}/{note_id}/permissions:batchDelete", headers=headers,
                                  json={"names": names}))
            return {"note": normalize(self._json(self._check(http.get(f"{KEEP_URL}/{note_id}", headers=headers))))}

    def attachment(self, note_id: str, attachment_id: str, mime_type: str) -> tuple[bytes, str]:
        note_id, attachment_id = _id(note_id), _id(attachment_id)
        if not isinstance(mime_type, str) or not MIME.fullmatch(mime_type):
            raise IntegrationError("invalid_mime_type")
        with self._session() as (http, headers):
            response = self._check(http.get(f"{KEEP_URL}/{note_id}/attachments/{attachment_id}", headers=headers,
                                             params={"mimeType": mime_type, "alt": "media"}))
            if len(response.content) > 50 * 1024 * 1024:
                raise IntegrationError("keep_attachment_too_large")
            return response.content, mime_type

    def replace(self, note_id: str, title: str, text: str | None = None, items: list[dict] | None = None) -> dict:
        """Create an edited copy, restore collaborators, then delete the original.

        The original is deleted only after the copy exists and sharing succeeded,
        so a failure never loses content. Keep-only metadata (labels, color,
        pins, reminders, attachments, voice-assistant list linkage) cannot be
        carried over by the API.
        """
        body = build_body(title, text, items)
        with self._session() as (http, headers):
            original = normalize(self._json(self._check(http.get(f"{KEEP_URL}/{_id(note_id)}", headers=headers))))
            if original["trashed"]:
                raise IntegrationError("keep_note_trashed")
            emails = [c["email"] for c in original["collaborators"]
                      if c["role"] != "OWNER" and not c["deleted"] and c["email"] and c["kind"] != "family"]
            copy = self._create(http, headers, body)
            if emails:
                try:
                    copy = self._share(http, headers, copy["id"], _emails(emails))
                except IntegrationError:
                    return {"note": copy, "original_deleted": False, "warning": "collaborators_not_copied"}
                except httpx.HTTPError:
                    return {"note": copy, "original_deleted": False, "warning": "collaborators_unknown"}
            try:
                self._check(http.delete(f"{KEEP_URL}/{original['id']}", headers=headers))
            except IntegrationError:
                return {"note": copy, "original_deleted": False, "warning": "original_not_deleted"}
            except httpx.HTTPError:
                # The DELETE may have been applied; don't claim the original survived.
                return {"note": copy, "original_deleted": None, "warning": "original_delete_unknown"}
            return {"note": copy, "original_deleted": True}
