"""TASK-1180: MCP task tools must not depend on retired BawtHub hostname aliases."""

import importlib
import re

from llm_bawt.mcp_server import task_api
from llm_bawt.utils import config

LEGACY_ALIAS = re.compile(r"(echo|bawthub)\.(lan\.)?(ferreri\.us|zenoran\.com)", re.I)


def test_default_task_api_url_is_internal_service_name(monkeypatch):
    monkeypatch.delenv("LLM_BAWT_TASK_API_URL", raising=False)
    module = importlib.reload(task_api)
    assert module.BASE_URL == config.DEFAULT_TASK_API_URL == "http://frontend-prod:3000"
    assert config.Config.model_fields["TASK_API_URL"].default == config.DEFAULT_TASK_API_URL
    assert not LEGACY_ALIAS.search(module.BASE_URL)


def test_task_api_url_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_TASK_API_URL", "http://example.internal:9000")
    try:
        assert importlib.reload(task_api).BASE_URL == "http://example.internal:9000"
    finally:
        monkeypatch.delenv("LLM_BAWT_TASK_API_URL", raising=False)
        importlib.reload(task_api)
