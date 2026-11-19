"""TASK-2494: tasks_create/tasks_update label task images by role and caption.

Labels go to BawtHub as ``imageLabelsSet`` keyed by stored URL, and a new
acceptance round is the explicit ``replaceResults``; BawtHub applies both
under the row lock (taskImageLabels.pg.test.ts). An older BawtHub returns no
``imageLabels`` and the tool says the labels were not changed.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_bawt.mcp_server import task_tools
from tests.test_mcp_task_images import ID, OTHER, STORED, STORED_OTHER, UI_UPLOAD, FakeTasks, _run, assets  # noqa: F401


class LabelTasks(FakeTasks):
    """FakeTasks that also answers with the stored labels, as BawtHub does."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, urls: list[str], *, old_server: bool = False) -> None:
        super().__init__(monkeypatch, urls, old_server=old_server)
        self.labels: dict[str, dict[str, str]] = {}

    async def patch(self, path: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        task = await super().patch(path, {k: v for k, v in json.items()}, headers)
        if self.old_server:
            task.pop("imageLabelsSet", None)
            task.pop("replaceResults", None)
            return task
        for url, label in json.get("imageLabelsSet", {}).items():
            self.labels[url] = {**self.labels.get(url, {}), **label}
        task.pop("imageLabelsSet", None)
        task.pop("replaceResults", None)
        return {**task, "imageLabels": self.labels}


def test_added_results_are_labelled_with_their_captions(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = LabelTasks(monkeypatch, [UI_UPLOAD])

    result = _run(task_tools.update_task(
        "TASK-9", add_image_urls=[ID, f"/v1/uploads/{OTHER}"], image_role="result",
        image_captions={ID: " Card, phone 390px, dev ", STORED_OTHER: "Card, desktop, dev"}))

    assert api.patches == [{
        "imageUrlsAdd": [STORED, STORED_OTHER],
        "imageLabelsSet": {STORED: {"role": "result", "caption": "Card, phone 390px, dev"},
                           STORED_OTHER: {"role": "result", "caption": "Card, desktop, dev"}},
    }]
    assert result["ok"] is True


def test_a_new_round_is_explicit_and_never_removes(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = LabelTasks(monkeypatch, [STORED_OTHER])

    _run(task_tools.update_task("TASK-9", add_image_urls=[ID], image_role="result", replace_current_results=True))
    _run(task_tools.update_task("TASK-9", add_image_urls=[ID], image_role="result"))

    assert api.patches[0] == {"imageUrlsAdd": [STORED], "imageLabelsSet": {STORED: {"role": "result"}},
                              "replaceResults": True}
    assert "replaceResults" not in api.patches[1]
    assert all("imageUrlsRemove" not in body for body in api.patches)


def test_caption_alone_relabels_an_attached_image(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = LabelTasks(monkeypatch, [STORED])

    result = _run(task_tools.update_task("TASK-9", image_captions={ID: ""}))

    assert api.patches == [{"imageLabelsSet": {STORED: {"caption": ""}}}]
    assert result["updated"] == ["imageLabelsSet"]


def test_a_removed_image_gets_no_role(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = LabelTasks(monkeypatch, [])

    _run(task_tools.update_task("TASK-9", add_image_urls=[ID, OTHER], remove_image_urls=[OTHER], image_role="history"))

    assert api.patches[0]["imageLabelsSet"] == {STORED: {"role": "history"}}


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"image_role": "result"}, "image_role labels add_image_urls"),
    ({"add_image_urls": [ID], "image_role": "final"}, "image_role must be one of"),
    ({"image_captions": {ID: "x" * 201}}, "at most 200"),
    ({"image_captions": {"https://example.com/a.png": "x"}}, "not a BawtHub upload URL"),
])
def test_bad_labels_write_nothing(monkeypatch: pytest.MonkeyPatch, assets: dict,
                                  kwargs: dict[str, Any], message: str) -> None:
    api = LabelTasks(monkeypatch, [STORED])

    result = _run(task_tools.update_task("TASK-9", **kwargs))

    assert message in result["error"]
    assert api.patches == []


def test_an_older_bawthub_reports_labels_unchanged(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    LabelTasks(monkeypatch, [STORED], old_server=True)

    result = _run(task_tools.update_task("TASK-9", title="new", image_captions={ID: "Phone, dev"}))

    assert "image roles or captions" in result["error"] and "other fields were applied" in result["error"]
    assert result["updated"] == ["title"]


def test_create_labels_its_images_as_requirements_by_default(monkeypatch: pytest.MonkeyPatch, assets: dict) -> None:
    api = LabelTasks(monkeypatch, [])

    _run(task_tools.create_task("t", image_urls=[ID, UI_UPLOAD], image_captions={ID: "Bug capture, phone"}))
    _run(task_tools.create_task("t", image_urls=[ID], image_role="result"))
    refused = _run(task_tools.create_task("t", image_urls=[ID], image_captions={OTHER: "x"}))
    bare = _run(task_tools.create_task("t", image_captions={ID: "x"}))

    assert api.posts[0]["imageLabelsSet"] == {
        STORED: {"role": "requirement", "caption": "Bug capture, phone"},
        UI_UPLOAD: {"role": "requirement"},
    }
    assert api.posts[1]["imageLabelsSet"] == {STORED: {"role": "result"}}
    assert "not attached" in refused["error"] and "attach them" in bare["error"]
    assert len(api.posts) == 2
