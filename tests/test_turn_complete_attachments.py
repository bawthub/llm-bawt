"""Regression coverage for live turn_complete screenshot attachments."""

from __future__ import annotations

from types import SimpleNamespace

from llm_bawt.service.turn_stream_finalize import TurnStreamFinalizer
from llm_bawt.service.turn_stream_worker import TurnStreamWorker


def test_worker_passes_attachment_enricher_to_terminal_coordinator() -> None:
    """The worker captures its mixin helper before rebinding ``self`` to the service."""
    import inspect

    source = inspect.getsource(TurnStreamWorker._stream_to_queue)
    assert "_enrich_attachment_refs = self._enrich_attachment_refs" in source
    assert "TurnStreamFinalizer(" in source
    assert "enrich_attachment_refs=_enrich_attachment_refs" in source


def test_finalizer_uses_injected_attachment_enricher() -> None:
    calls: list[list[dict]] = []
    ctx = SimpleNamespace(agent_attachments_holder=[])
    finalizer = TurnStreamFinalizer(
        ctx,
        publish_event_direct=lambda _event: None,
        enrich_attachment_refs=lambda refs: calls.append(refs) or refs,
    )
    refs = [{"asset_id": "ma_test", "kind": "image"}]

    assert finalizer._enrich_attachment_refs(refs) == refs
    assert calls == [refs]


def test_finalizer_carries_response_text_on_completed_turn_complete() -> None:
    """TASK-779: the streaming turn_complete emit MUST carry response_text
    on successful completion so the frontend can fall back to
    server-authoritative bytes when the client-side text_delta partial is
    empty (subscriber gap, packed flush, dropped delta). Symmetric with
    the nonstream inter-bot emit in background_service.py.

    TASK-714 preserves error text; TASK-952 also preserves cancelled partials
    for clients that missed the last delta. Empty timeout paths carry no text.
    """
    import inspect

    source = inspect.getsource(TurnStreamFinalizer._finalize)
    # Field is present in the emit dict:
    assert '"response_text": _response_text' in source, (
        "streaming turn_complete emit must carry response_text (TASK-779 safety net)"
    )
    # TASK-952: cancellation preserves the same server-authoritative partial.
    assert 'if status in ("completed", "error", "cancelled")' in source
    # The value pulls from the accumulated streaming text holder, not empty:
    assert "ctx.full_response_holder[0]" in source, (
        "response_text must pull from full_response_holder[0] (the accumulated "
        "streaming text)"
    )
