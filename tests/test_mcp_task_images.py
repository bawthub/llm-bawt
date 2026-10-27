"""TASK-1526: tasks_create/tasks_update attach chat images to a task.

An update sends ``imageUrlsAdd``/``imageUrlsRemove``, never the whole array:
BawtHub applies them under the row lock (its taskImageEdits.pg.test.ts holds
the race). An older BawtHub ignores them, and the tool says so.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_bawt.mcp_server import task_association, task_images, task_tools
from llm_bawt.task_turn_context import TaskTurnContext, TaskTurnContextError

ID = "ma_01M5X91YHQ3ZTFTCSV9VRA3GS2"
OTHER = "ma_01M5X91YHQ3ZTFTCSV9VRA3GS3"
STORED = f"/api/chat/uploads/{ID}"
STORED_OTHER = f"/api/chat/uploads/{OTHER}"
UI_UPLOAD = "/api/uploads/serve/originals/24/f1/24f1abc.png"


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture
def assets(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    rows = {ID: {"id": ID, "kind": "image"}, OTHER: {"id": OTHER, "kind": "image"}}

    def lookup(asset_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {asset_id: rows[asset_id] for asset_id in asset_ids if asset_id in rows}

    monkeypatch.setattr(task_images, "_lookup_assets", lookup)
    return rows


class FakeTasks:
    """Task API double over one stored image list, applying edits as BawtHub does.

    ``old_server`` answers like a BawtHub release without image edits: the
    edit fields are ignored and the rest of the PATCH applies.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, urls: list[str], *, old_server: bool = False) -> None:
        self.urls = list(urls)
        self.old_server = old_server
        self.patches: list[dict[str, Any]] = []
        self.posts: list[dict[str, Any]] = []
        monkeypatch.setattr(task_tools, "_api_get", self.get)
        monkeypatch.setattr(task_tools, "_api_patch", self.patch)
        monkeypatch.setattr(task_tools, "_api_post", self.post)

    def _task(self, **fields: Any) -> dict[str, Any]:
        return {"id": "task-id", "shortId": "TASK-9", "title": "t", "status": "QUEUED", **fields,
                "imageUrls": list(self.urls)}

    async def get(self, path: str, **_kwargs: Any) -> dict[str, Any]:
        return self._task()

    async def patch(self, path: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        self.patches.append(json)
        if not self.old_server:
            removed = set(json.get("imageUrlsRemove", []))
            self.urls = [url for url in self.urls if url not in removed]
            self.urls += [url for url in json.get("imageUrlsAdd", []) if url not in self.urls]
        rest = {key: value for key, value in json.items() if not key.startswith("imageUrls")}
        return self._task(**rest)

    async def post(self, path: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        self.posts.append(json)
        return {"id": "task-id", "shortId": "TASK-9", **json}


@pytest.mark.parametrize("ref", [
    ID,
    f"/v1/uploads/{ID}",
    f"http://app:8642/v1/uploads/{ID}/thumb",
    f"/api/chat/uploads/{ID}",
    f"https://app.bawthub.com/api/chat/uploads/{ID}?download=1",
])
def test_every_upload_form_is_stored_as_the_proxy_path(ref: str) -> None:
    assert task_images.parse_image_ref(ref) == (ID, STORED)


def test_task_ui_upload_kept_origin_relative() -> None:
    assert task_images.parse_image_ref(f"https://dev.bawthub.com{UI_UPLOAD}") == (None, UI_UPLOAD)


@pytest.mark.parametrize("ref", ["", "ma_short", "https://example.com/cat.png",
                                 f"https://example.com{UI_UPLOAD}", "ftp://x/v1/uploads/" + ID,
                                 "/api/uploads/serve/../secrets"])
def test_non_upload_refs_are_refused(ref: str) -> None:
    with pytest.raises(task_images.ImageRefError):
        task_images.parse_image_ref(ref)


def test_create_attaches_images(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])

    result = _run(task_tools.create_task("t", image_urls=[ID, f"/v1/uploads/{ID}", UI_UPLOAD]))

    assert api.posts[0]["imageUrls"] == [STORED, UI_UPLOAD]
    assert result["imageUrls"] == [STORED, UI_UPLOAD]


def test_create_refuses_unknown_upload_and_writes_nothing(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])
    del assets[OTHER]

    result = _run(task_tools.create_task("t", image_urls=[ID, OTHER]))

    assert "unknown upload id" in result["error"] and OTHER in result["error"]
    assert api.posts == []


def test_non_image_upload_refused(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])
    assets[ID]["kind"] = "file"

    result = _run(task_tools.update_task("TASK-9", add_image_urls=[ID]))

    assert "not image uploads" in result["error"]
    assert api.patches == []


def test_add_appends_to_the_current_list(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [UI_UPLOAD])

    result = _run(task_tools.update_task("TASK-9", add_image_urls=[ID], bot_id="vex"))

    assert api.patches == [{"imageUrlsAdd": [STORED]}]
    assert api.urls == [UI_UPLOAD, STORED]
    assert result["ok"] is True and result["updated"] == ["imageUrlsAdd"]


def test_add_and_remove_are_idempotent(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [STORED])

    added = _run(task_tools.update_task("TASK-9", add_image_urls=[ID]))
    removed = _run(task_tools.update_task("TASK-9", remove_image_urls=[OTHER]))

    # The request is sent whatever the read showed; BawtHub's apply is a no-op.
    assert api.patches == [{"imageUrlsAdd": [STORED]}, {"imageUrlsRemove": [STORED_OTHER, OTHER]}]
    assert api.urls == [STORED]
    assert added["ok"] is True and removed["ok"] is True


def test_adding_an_upload_held_in_a_legacy_form_leaves_one_canonical_entry(
        monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    legacy = f"https://app.bawthub.com/api/chat/uploads/{ID}"
    api = FakeTasks(monkeypatch, [legacy, UI_UPLOAD])

    result = _run(task_tools.update_task("TASK-9", add_image_urls=[ID]))

    assert api.patches == [{"imageUrlsAdd": [STORED], "imageUrlsRemove": [legacy]}]
    assert api.urls == [UI_UPLOAD, STORED]
    assert result["ok"] is True


def _race(monkeypatch: pytest.MonkeyPatch, api: FakeTasks, competing) -> None:
    """Run ``competing`` on the list after the tool's GET returns and before its PATCH."""
    real_get = api.get

    async def read_then_compete(path: str, **kwargs: Any) -> dict[str, Any]:
        task = await real_get(path, **kwargs)
        competing(api.urls)
        return task

    monkeypatch.setattr(task_tools, "_api_get", read_then_compete)


def test_an_add_the_read_already_held_survives_a_concurrent_remove(
        monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [STORED])
    _race(monkeypatch, api, lambda urls: urls.remove(STORED))

    result = _run(task_tools.update_task("TASK-9", title="new", add_image_urls=[ID]))

    assert api.patches == [{"title": "new", "imageUrlsAdd": [STORED]}]
    assert api.urls == [STORED]
    assert result["ok"] is True


def test_a_remove_the_read_lacked_survives_a_concurrent_add(
        monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])
    _race(monkeypatch, api, lambda urls: urls.append(STORED))

    result = _run(task_tools.update_task("TASK-9", title="new", remove_image_urls=[ID]))

    assert api.patches == [{"title": "new", "imageUrlsRemove": [STORED, ID]}]
    assert api.urls == []
    assert result["ok"] is True


@pytest.mark.parametrize(("current", "add", "remove", "expected"), [
    ([STORED], [STORED], [], task_images.ImageEdit(add=[STORED], remove=[], fresh=[])),
    ([], [], [ID], task_images.ImageEdit(add=[], remove=[STORED, ID], fresh=[])),
    ([], [STORED], [ID], task_images.ImageEdit(add=[], remove=[STORED, ID], fresh=[])),
    ([], [], [], None),
])
def test_the_plan_keeps_the_whole_request(current: list[str], add: list[str], remove: list[str],
                                          expected: task_images.ImageEdit | None) -> None:
    assert task_images.plan_image_edit(current, add, remove) == expected


def test_remove_matches_any_form_even_for_a_deleted_upload(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [STORED, STORED_OTHER, "blob:legacy"])
    assets.clear()

    result = _run(task_tools.update_task("TASK-9", remove_image_urls=[f"/v1/uploads/{ID}", "blob:legacy"]))

    # Removes name the canonical spelling, the string as given and every stored
    # entry for that upload: BawtHub matches exact strings.
    assert api.patches == [{"imageUrlsRemove": [STORED, "blob:legacy", f"/v1/uploads/{ID}"]}]
    assert api.urls == [STORED_OTHER]
    assert result["ok"] is True


def test_a_concurrent_upload_is_kept_because_only_the_edit_is_sent(
        monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])
    real_get = api.get

    async def read_then_ui_upload(path: str, **kwargs: Any) -> dict[str, Any]:
        task = await real_get(path, **kwargs)
        api.urls.append(UI_UPLOAD)  # lands after the tool read the list
        return task

    monkeypatch.setattr(task_tools, "_api_get", read_then_ui_upload)

    result = _run(task_tools.update_task("TASK-9", add_image_urls=[ID], title="new"))

    assert "imageUrls" not in api.patches[0]
    assert api.urls == [UI_UPLOAD, STORED]
    assert result["ok"] is True


def test_images_travel_with_other_fields(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])

    _run(task_tools.update_task("TASK-9", title="new", add_image_urls=[ID]))

    assert api.patches == [{"title": "new", "imageUrlsAdd": [STORED]}]


def test_an_older_bawthub_is_reported_not_called_ok(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [STORED_OTHER], old_server=True)
    turn = Turn(monkeypatch, [_image(ID)])

    mixed = _run(task_tools.update_task("TASK-9", title="new", remove_image_urls=[OTHER],
                                        associate_current_turn=True))
    alone = _run(task_tools.update_task("TASK-9", add_image_urls=[ID], attach_turn_images=False))

    assert "images were not changed" in mixed["error"] and "other fields were applied" in mixed["error"]
    assert mixed["updated"] == ["title"]
    assert "turnImagesAttached" not in mixed
    assert turn.associated == ["TASK-9"]
    assert "images were not changed" in alone["error"] and "other fields" not in alone["error"]
    assert alone["updated"] == []
    assert api.urls == [STORED_OTHER]


@pytest.mark.parametrize("ref", [f"https://example.com/v1/uploads/{ID}",
                                 f"https://evil.example/api/chat/uploads/{ID}",
                                 "https://other/api/uploads/serve/example.png",
                                 f"http://intranet:8642/v1/uploads/{ID}"])
def test_an_upload_path_on_a_foreign_host_is_refused(ref: str) -> None:
    with pytest.raises(task_images.ImageRefError):
        task_images.parse_image_ref(ref)


class Turn:
    """Signed-turn double: the trigger message's attachments and the associations made."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, attachments: list[dict[str, Any]]) -> None:
        self.attachments = attachments
        self.reads: list[tuple[str, str]] = []
        self.associated: list[str] = []
        context = TaskTurnContext(session_id="s", turn_id="turn-1", trigger_message_id="m-1",
                                  bot_id="vex", user_id="u", issued_at=0)
        monkeypatch.setattr(task_association, "current_task_turn_context", lambda: context)
        monkeypatch.setattr(task_images, "_trigger_attachments", self.read)
        monkeypatch.setattr(task_tools, "associate_current_task", self.associate)

    def read(self, bot_id: str, message_id: str) -> list[dict[str, Any]]:
        self.reads.append((bot_id, message_id))
        return self.attachments

    async def associate(self, task_ref: str) -> dict[str, Any]:
        self.associated.append(task_ref)
        return {"ok": True}


def _image(asset_id: str) -> dict[str, str]:
    return {"kind": "image", "asset_id": asset_id}


def test_linking_attaches_the_trigger_images(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [UI_UPLOAD])
    turn = Turn(monkeypatch, [_image(ID), _image(OTHER)])

    result = _run(task_tools.update_task("TASK-9", associate_current_turn=True))

    assert turn.reads == [("vex", "m-1")]
    assert api.patches == [{"imageUrlsAdd": [STORED, STORED_OTHER]}]
    assert api.urls == [UI_UPLOAD, STORED, STORED_OTHER]
    assert result["turnImagesAttached"] == [STORED, STORED_OTHER]
    assert turn.associated == ["TASK-9"]


def test_opt_out_attaches_nothing(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])
    turn = Turn(monkeypatch, [_image(ID)])

    result = _run(task_tools.update_task("TASK-9", associate_current_turn=True, attach_turn_images=False))

    assert turn.reads == [] and api.patches == []
    assert "turnImagesAttached" not in result
    assert turn.associated == ["TASK-9"]


def test_relinking_the_same_turn_adds_no_duplicate(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    # The first link already stored this upload (here in its legacy absolute form).
    api = FakeTasks(monkeypatch, [f"https://app.bawthub.com/api/chat/uploads/{ID}"])
    turn = Turn(monkeypatch, [_image(ID)])

    result = _run(task_tools.update_task("TASK-9", associate_current_turn=True))

    # Re-sent and normalized to one canonical entry; nothing is reported as new.
    assert api.urls == [STORED]
    assert "turnImagesAttached" not in result
    assert turn.associated == ["TASK-9"]


def test_turn_without_images_is_a_no_op(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])
    # A file upload and a deleted image are not attached either.
    Turn(monkeypatch, [{"kind": "file", "asset_id": "ma_01M5X91YHQ3ZTFTCSV9VRA3GS4"}])
    assets["ma_01M5X91YHQ3ZTFTCSV9VRA3GS4"] = {"id": "x", "kind": "file"}

    result = _run(task_tools.update_task("TASK-9", associate_current_turn=True))

    assert api.patches == []
    assert "turnImagesAttached" not in result and "turnImagesError" not in result


def test_create_with_link_attaches_trigger_images_once(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])
    turn = Turn(monkeypatch, [_image(ID), _image(OTHER)])

    result = _run(task_tools.create_task("t", image_urls=[ID], associate_current_turn=True))

    assert api.posts[0]["imageUrls"] == [STORED, STORED_OTHER]
    assert result["turnImagesAttached"] == [STORED_OTHER]
    assert turn.associated == ["TASK-9"]


def test_image_lookup_failure_never_blocks_the_write(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])
    turn = Turn(monkeypatch, [])

    def broken(bot_id: str, message_id: str) -> list[dict[str, Any]]:
        raise RuntimeError("history store down")

    monkeypatch.setattr(task_images, "_trigger_attachments", broken)

    result = _run(task_tools.update_task("TASK-9", title="new", associate_current_turn=True))

    assert api.patches == [{"title": "new"}]
    assert "history store down" in result["turnImagesError"]
    assert turn.associated == ["TASK-9"]


def test_no_trusted_turn_attaches_nothing(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])

    def untrusted() -> TaskTurnContext:
        raise TaskTurnContextError("no signed turn")

    monkeypatch.setattr(task_association, "current_task_turn_context", untrusted)

    result = _run(task_tools.create_task("t", associate_current_turn=True))

    assert "imageUrls" not in api.posts[0]
    assert "turnImagesError" not in result
