"""TASK-1526: tasks_create/tasks_update attach chat images to a task.

BawtHub stores ``imageUrls`` and its PATCH replaces the whole array, so the
tool reads, edits, re-reads and refuses a list that moved in between.
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
    """Task API double: GETs return successive image lists, PATCH/POST are recorded."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *lists: list[str]) -> None:
        self.lists = list(lists)
        self.patches: list[dict[str, Any]] = []
        self.posts: list[dict[str, Any]] = []
        monkeypatch.setattr(task_tools, "_api_get", self.get)
        monkeypatch.setattr(task_tools, "_api_patch", self.patch)
        monkeypatch.setattr(task_tools, "_api_post", self.post)

    async def get(self, path: str, **_kwargs: Any) -> dict[str, Any]:
        urls = self.lists.pop(0) if len(self.lists) > 1 else self.lists[0]
        return {"id": "task-id", "shortId": "TASK-9", "title": "t", "status": "QUEUED", "imageUrls": urls}

    async def patch(self, path: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        self.patches.append(json)
        return {"id": "task-id", "shortId": "TASK-9", "title": "t", "status": "QUEUED", **json}

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

    assert api.patches == [{"imageUrls": [UI_UPLOAD, STORED]}]
    assert result["ok"] is True and result["updated"] == ["imageUrls"]


def test_add_and_remove_are_idempotent(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    # A legacy absolute entry counts as the same upload.
    api = FakeTasks(monkeypatch, [f"https://app.bawthub.com/api/chat/uploads/{ID}"])

    added = _run(task_tools.update_task("TASK-9", add_image_urls=[ID]))
    removed = _run(task_tools.update_task("TASK-9", remove_image_urls=[OTHER]))

    assert api.patches == []
    assert added["ok"] is True and added["updated"] == []
    assert removed["ok"] is True and removed["updated"] == []


def test_remove_matches_any_form_even_for_a_deleted_upload(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [STORED, STORED_OTHER, "blob:legacy"])
    assets.clear()

    result = _run(task_tools.update_task("TASK-9", remove_image_urls=[f"/v1/uploads/{ID}", "blob:legacy"]))

    assert api.patches == [{"imageUrls": [STORED_OTHER]}]
    assert result["ok"] is True


def test_list_changed_before_write_refuses_lost_update(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    # First read [], then a concurrent UI upload lands before the write.
    api = FakeTasks(monkeypatch, [], [UI_UPLOAD])

    result = _run(task_tools.update_task("TASK-9", add_image_urls=[ID], title="new"))

    assert "nothing was saved" in result["error"]
    assert api.patches == []


def test_images_travel_with_other_fields(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = FakeTasks(monkeypatch, [])

    _run(task_tools.update_task("TASK-9", title="new", add_image_urls=[ID]))

    assert api.patches == [{"title": "new", "imageUrls": [STORED]}]


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
    assert api.patches == [{"imageUrls": [UI_UPLOAD, STORED, STORED_OTHER]}]
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

    assert api.patches == []
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
