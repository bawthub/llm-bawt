"""Unit tests for the tenant seed engine (TASK-350 / TASK-355).

Pure in-memory: the SQLAlchemy engine and the stores are mocked, so these
tests exercise the seeding control flow (ordering, insert-if-missing
branches, report shape) without any database.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_bawt.memory.model_catalog_migration import STANDARD_ACCESS_PATHS
from llm_bawt.seeding.data import (
    FIRST_AGENT_PROMPT_SEED,
    EndpointSeed,
    ModelSeed,
)
from llm_bawt.seeding.engine import TenantSeeder


# ---------------------------------------------------------------------------
# Seed data integrity
# ---------------------------------------------------------------------------


def test_standard_paths_include_openai_platform_responses_route():
    by_key = {p.key: p for p in STANDARD_ACCESS_PATHS}
    path = by_key["openai-responses"]

    assert path.vendor == "openai"
    assert path.protocol == "responses"
    assert path.base_url == "https://api.openai.com/v1"
    assert path.auth_mechanism == "api-key"


def test_first_agent_prompt_seed_is_generic():
    """No owner name, no personal traits — it's a tenant starter, not Nick's bot."""
    lowered = FIRST_AGENT_PROMPT_SEED.lower()
    for forbidden in ("nick", "nova", "mira", "personal ai assistant"):
        assert forbidden not in lowered
    assert FIRST_AGENT_PROMPT_SEED.strip()


def test_seed_data_carries_no_credentials():
    import inspect
    import re

    from llm_bawt.seeding import data

    source = inspect.getsource(data)
    # Word-anchored so prose like "TASK-355" (…sk-3…) doesn't trip the check.
    for pattern in (r"\bsk-[A-Za-z0-9]{8,}", r"\bxai-[A-Za-z0-9]{16,}", r"api_key\s*=\s*[\"']", r"(?i)bearer "):
        assert not re.search(pattern, source), f"seed data must not carry credentials ({pattern!r})"


# ---------------------------------------------------------------------------
# Engine control flow (mocked engine / stores)
# ---------------------------------------------------------------------------


def _seeder_with_mock_engine():
    engine = MagicMock(name="engine")
    conn = MagicMock(name="conn")
    engine.begin.return_value.__enter__.return_value = conn
    engine.connect.return_value.__enter__.return_value = conn
    seeder = TenantSeeder(config=MagicMock(name="config"), engine=engine)
    return seeder, engine, conn


_MODELS = (
    ModelSeed(key="grok-4.3", vendor="xai", display_name="grok-4.3", default_context_window=1_000_000),
    ModelSeed(key="grok-4.5", vendor="xai", display_name="Grok 4.5", default_context_window=500_000),
)
_ENDPOINTS = (
    EndpointSeed(model_key="grok-4.3", access_path_key="xai-chat", upstream_model_id="grok-4.3"),
    EndpointSeed(model_key="grok-4.5", access_path_key="xai-chat", upstream_model_id="grok-4.5"),
)


def test_seed_models_reports_created_vs_existing():
    seeder, _, conn = _seeder_with_mock_engine()
    conn.execute.return_value.first.side_effect = [("1",), None]
    report = seeder._seed_models(conn, _MODELS)
    assert report["created"] == ["grok-4.3"]
    assert report["existing"] == ["grok-4.5"]


def test_seed_endpoints_noop_when_all_exist():
    seeder, _, conn = _seeder_with_mock_engine()
    conn.execute.return_value.first.return_value = None
    report = seeder._seed_endpoints(conn, _ENDPOINTS)
    assert report["created"] == []
    assert report["existing"] == ["grok-4.3@xai-chat", "grok-4.5@xai-chat"]


def test_resolve_endpoint_id_raises_when_missing():
    seeder, _, conn = _seeder_with_mock_engine()
    conn.execute.return_value.scalar.return_value = None
    with pytest.raises(RuntimeError, match="not found"):
        seeder._resolve_endpoint_id(conn, "grok-4.3", "xai-chat")


def test_apply_seeds_infra_only_and_bootstraps_bot_profiles_first():
    """No bots, no models: schema + access paths + prompt defaults, in order."""
    seeder, _, _ = _seeder_with_mock_engine()
    calls: list[str] = []

    def _bot_store():
        calls.append("bot_store")
        return MagicMock(name="bot_store")

    def _schema():
        calls.append("schema")
        return {"access_paths": {"created": ["xai-chat"]}}

    def _prompts():
        calls.append("prompts")
        return {"created": [], "existing": ["x"]}

    def _ops():
        calls.append("ops")
        return {"created": ["llm-bawt.restart-app"], "existing": []}

    def _policies():
        calls.append("policies")
        return {"created": 3, "total": 9}

    with (
        patch.object(seeder, "_bot_profile_store", side_effect=_bot_store),
        patch.object(seeder, "_ensure_catalog_schema", side_effect=_schema),
        patch.object(seeder, "_seed_prompt_defaults", side_effect=_prompts),
        patch.object(seeder, "_seed_ops_catalog", side_effect=_ops),
        patch.object(seeder, "_seed_approval_policies", side_effect=_policies),
        patch.object(seeder, "_seed_models") as seed_models,
        patch.object(seeder, "_seed_endpoints") as seed_endpoints,
    ):
        report = seeder.apply()

    assert calls == ["bot_store", "schema", "prompts", "ops", "policies"]
    assert set(report) == {
        "catalog_migration", "prompts", "ops_operations", "approval_policies",
    }
    seed_models.assert_not_called()
    seed_endpoints.assert_not_called()


def test_register_model_inserts_model_and_endpoint_then_resolves_id():
    seeder, _, conn = _seeder_with_mock_engine()
    with (
        patch.object(seeder, "_seed_models", return_value={"created": ["grok-4.3"], "existing": []}) as sm,
        patch.object(seeder, "_seed_endpoints", return_value={"created": ["grok-4.3@xai-chat"], "existing": []}) as se,
        patch.object(seeder, "_resolve_endpoint_id", return_value=7) as resolve,
    ):
        report = seeder.register_model(
            vendor="xai",
            model_id=" grok-4.3 ",
            access_path_key="xai-chat",
            display_name=None,
            context_window=131072,
        )

    (model,) = sm.call_args.args[1]
    assert model == ModelSeed(key="grok-4.3", vendor="xai", display_name="grok-4.3", default_context_window=131072)
    (endpoint,) = se.call_args.args[1]
    assert endpoint == EndpointSeed(model_key="grok-4.3", access_path_key="xai-chat", upstream_model_id="grok-4.3")
    resolve.assert_called_once_with(conn, "grok-4.3", "xai-chat")
    assert report == {
        "models": {"created": ["grok-4.3"], "existing": []},
        "endpoints": {"created": ["grok-4.3@xai-chat"], "existing": []},
        "endpoint_id": 7,
    }


@pytest.mark.parametrize(
    ("model_id", "window", "message"),
    [("", 1000, "model_id"), ("   ", 1000, "model_id"), ("grok-4.3", 0, "context_window")],
)
def test_register_model_rejects_bad_input_before_writes(model_id, window, message):
    seeder, engine, _ = _seeder_with_mock_engine()
    with pytest.raises(ValueError, match=message):
        seeder.register_model(vendor="xai", model_id=model_id, access_path_key="xai-chat", context_window=window)
    engine.begin.assert_not_called()
