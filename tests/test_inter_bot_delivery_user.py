"""Delivery owner resolution is independent of a runner's default user."""

from types import SimpleNamespace

import pytest

from llm_bawt.inter_bot_delivery_user import resolve_delivery_user
from llm_bawt.utils.config import Config


@pytest.mark.parametrize("configured", [None, "", "   "])
def test_missing_delivery_owner_fails_closed(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("LLM_BAWT_DEFAULT_USER", raising=False)
    else:
        monkeypatch.setenv("LLM_BAWT_DEFAULT_USER", configured)
    config = Config(_env_file=None)
    assert config.DEFAULT_USER == configured

    with pytest.raises(ValueError, match="Delivery user is required"):
        resolve_delivery_user(config)
    with pytest.raises(ValueError, match="Delivery user is required"):
        resolve_delivery_user(config, "  ")


@pytest.mark.parametrize("configured", [None, "", "  other  "])
def test_explicit_delivery_owner_takes_precedence(configured):
    assert resolve_delivery_user(SimpleNamespace(DEFAULT_USER=configured), "  casey  ") == "casey"


def test_configured_delivery_owner_is_used_when_payload_is_blank():
    config = SimpleNamespace(DEFAULT_USER="  casey  ")
    assert resolve_delivery_user(config, None) == "casey"
    assert resolve_delivery_user(config, "  ") == "casey"
