"""App must reject unowned local-video requests before persisting generation jobs."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.media.schemas import MediaGenerationRequest
from llm_bawt.service.routes import media


@pytest.mark.parametrize("owner,phase", [("unknown", "idle"), ("voice", "idle"),
                                         ("video", "offered"), ("video", "recovery_required")])
def test_local_video_without_stable_owner_creates_no_job(monkeypatch, owner, phase):
    inserts = []
    monkeypatch.setattr(media, "_get_store", lambda: SimpleNamespace(insert=lambda row: inserts.append(row)))
    monkeypatch.setattr(media, "_get_storage", lambda: object())
    monkeypatch.setattr(media, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(media, "get_shared_engine", lambda config: object())
    monkeypatch.setattr(media, "GpuHandoffStore", lambda engine: SimpleNamespace(
        status=lambda: SimpleNamespace(owner=owner, phase=phase)))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(media.create_generation(MediaGenerationRequest(
            prompt="running dog", provider="local-video", media_type="video")))
    assert exc.value.status_code == 409
    assert inserts == []
