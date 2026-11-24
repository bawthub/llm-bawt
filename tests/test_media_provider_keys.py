"""Bring-your-own-key image providers: off until a key exists, with a Settings hint."""

from __future__ import annotations

import asyncio

import pytest

from llm_bawt.media.clients import grok_media, openai_images
from llm_bawt.service.routes import media


@pytest.fixture
def keyless(monkeypatch):
    monkeypatch.setattr(grok_media, "resolve_media_api_key", lambda *a, **k: "")
    monkeypatch.setattr(openai_images, "resolve_media_api_key", lambda *a, **k: "")
    monkeypatch.setattr(media, "resolve_media_api_key", lambda *a, **k: "")


@pytest.mark.parametrize("client_cls", [grok_media.GrokMediaClient, openai_images.OpenAIImageClient])
def test_keyless_generate_raises_settings_hint(keyless, client_cls):
    with pytest.raises(RuntimeError, match="Settings"):
        asyncio.run(client_cls().generate("x", "image", "m"))


def test_provider_list_reports_keyless_byok_providers_off(keyless):
    providers = asyncio.run(media.list_media_providers(None)).providers
    configured = {p.provider: p.configured for p in providers}
    assert configured["grok"] is False and configured["openai"] is False
    # Providers that need no key stay on.
    assert all(configured[p] for p in configured if p not in ("grok", "openai"))


def test_provider_list_reports_keyed_provider_on(monkeypatch):
    monkeypatch.setattr(media, "resolve_media_api_key",
                        lambda _c, provider_id, *a, **k: "k" if provider_id == "xai" else "")
    configured = {p.provider: p.configured for p in asyncio.run(media.list_media_providers(None)).providers}
    assert configured["grok"] is True and configured["openai"] is False
