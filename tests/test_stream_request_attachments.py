"""Focused tests for extracted streaming attachment preparation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.stream_request_attachments import (
    prepare_stream_request_attachments,
)


def _run(request, log):
    return asyncio.run(
        prepare_stream_request_attachments(request, user_id="nick", log=log)
    )


def test_plain_text_survives_unavailable_media_store(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.service.stream_request_attachments.get_media_store",
        Mock(side_effect=RuntimeError("offline")),
    )
    log = SimpleNamespace(error=Mock(), warning=Mock())
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")]
    )

    prompt, llm_images, durable_refs, store = _run(request, log)

    assert prompt == "hello"
    assert llm_images == []
    assert durable_refs == []
    assert store is None
    log.error.assert_called_once()


def test_last_user_text_parts_ignore_trailing_assistant(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.service.stream_request_attachments.get_media_store",
        Mock(return_value=None),
    )
    log = SimpleNamespace(error=Mock(), warning=Mock())
    request = ChatCompletionRequest(messages=[
        ChatMessage(role="user", content=[
            {"type": "text", "text": " /new\n"},
            {"type": "image_url", "image_url": {"url": "not-an-inline-image"}},
            {"type": "text", "text": "keep bytes  "},
        ]),
        ChatMessage(role="assistant", content="not the command"),
    ])

    prompt, images, refs, _ = _run(request, log)
    assert prompt == " /new\nkeep bytes  "
    assert images == [] and refs == []


def test_legacy_inline_image_is_forwarded_and_uploaded(monkeypatch):
    asset = SimpleNamespace(id="ma_uploaded")
    store = SimpleNamespace(upload=Mock(return_value=asset))
    monkeypatch.setattr(
        "llm_bawt.service.stream_request_attachments.get_media_store",
        Mock(return_value=store),
    )
    log = SimpleNamespace(error=Mock(), warning=Mock())
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ],
            )
        ]
    )

    prompt, llm_images, durable_refs, returned_store = _run(request, log)

    assert prompt == "inspect"
    assert llm_images == [{"mimeType": "image/png", "content": "aGVsbG8="}]
    assert durable_refs == [{"asset_id": "ma_uploaded", "kind": "image"}]
    assert returned_store is store
    assert store.upload.call_args.kwargs["raw_bytes"] == b"hello"


def test_asset_id_uses_preview_and_keeps_durable_ref(monkeypatch):
    store = SimpleNamespace(
        read_preview_as_data_url=Mock(
            return_value="data:image/webp;base64,cHJldmlldw=="
        )
    )
    monkeypatch.setattr(
        "llm_bawt.service.stream_request_attachments.get_media_store",
        Mock(return_value=store),
    )
    log = SimpleNamespace(error=Mock(), warning=Mock())
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content="look",
                attachment_ids=["ma_existing"],
            )
        ]
    )

    prompt, llm_images, durable_refs, returned_store = _run(request, log)

    assert prompt == "look"
    assert llm_images == [{"mimeType": "image/webp", "content": "cHJldmlldw=="}]
    assert durable_refs == [{"asset_id": "ma_existing", "kind": "image"}]
    assert returned_store is store


def test_file_kind_asset_id_is_persisted_but_not_inlined(monkeypatch):
    """TASK-847: a PDF attachment keeps its durable ref; no vision payload."""
    store = SimpleNamespace(
        stat=Mock(return_value=SimpleNamespace(id="ma_pdf", kind="file")),
        read_preview_as_data_url=Mock(side_effect=AssertionError("must not inline files")),
    )
    monkeypatch.setattr(
        "llm_bawt.service.stream_request_attachments.get_media_store",
        Mock(return_value=store),
    )
    log = SimpleNamespace(error=Mock(), warning=Mock())
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="read this", attachment_ids=["ma_pdf"])]
    )

    prompt, llm_images, durable_refs, returned_store = _run(request, log)

    assert prompt == "read this"
    assert llm_images == []
    assert durable_refs == [{"asset_id": "ma_pdf", "kind": "file"}]
    assert returned_store is store
    store.read_preview_as_data_url.assert_not_called()


def test_image_kind_asset_id_still_inlines_preview(monkeypatch):
    store = SimpleNamespace(
        stat=Mock(return_value=SimpleNamespace(id="ma_img", kind="image")),
        read_preview_as_data_url=Mock(return_value="data:image/webp;base64,cHJldmlldw=="),
    )
    monkeypatch.setattr(
        "llm_bawt.service.stream_request_attachments.get_media_store",
        Mock(return_value=store),
    )
    log = SimpleNamespace(error=Mock(), warning=Mock())
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="look", attachment_ids=["ma_img"])]
    )
    _, llm_images, durable_refs, _ = _run(request, log)
    assert llm_images == [{"mimeType": "image/webp", "content": "cHJldmlldw=="}]
    assert durable_refs == [{"asset_id": "ma_img", "kind": "image"}]
