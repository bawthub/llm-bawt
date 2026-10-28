"""The all-providers usage view lists only providers with a catalog model (TASK-1571)."""

import asyncio
from types import SimpleNamespace

import pytest

from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service import usage
from llm_bawt.service.routes import usage as usage_routes
from llm_bawt.service.usage.base import UsageAdapter
from llm_bawt.service.usage.canonical import STATUS_OK


class _Probe(UsageAdapter):
    provider = "probe"
    display_name = "Probe"
    backend = "test"
    access_path_keys = ("probe-path",)

    async def fetch(self):
        return self._base(available=True, status=STATUS_OK, limits=[])


def _endpoint(endpoint_id: int, path_key: str) -> ModelEndpoint:
    return ModelEndpoint(
        id=endpoint_id,
        model=ModelIdentity(id=endpoint_id, key=f"model-{endpoint_id}", vendor="test", display_name="M"),
        access_path=AccessPath(
            id=endpoint_id, key=path_key, vendor="test", protocol="chat-completions",
            base_url=None, auth_mechanism="none",
        ),
        upstream_model_id=None,
    )


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(usage, "_REGISTRY", {"probe": _Probe()})
    monkeypatch.setattr(usage, "_ACCESS_PATH_ADAPTERS", {})
    usage.invalidate_cache("probe")
    yield
    usage.invalidate_cache("probe")


def _listed(catalog) -> list[str]:
    report = asyncio.run(usage.get_all(catalog=catalog))
    return [snap.provider for snap in report.providers]


def test_provider_without_models_is_hidden(probe):
    assert _listed(ModelCatalog([_endpoint(1, "other-path")])) == []
    assert _listed(None) == []


def test_provider_with_one_model_is_shown(probe):
    assert _listed(ModelCatalog([_endpoint(1, "probe-path")])) == ["probe"]


def test_provider_hides_again_after_its_last_model_is_removed(probe):
    assert _listed(ModelCatalog([_endpoint(1, "probe-path"), _endpoint(2, "other-path")])) == ["probe"]
    assert _listed(ModelCatalog([_endpoint(2, "other-path")])) == []


def test_usage_route_filters_by_the_installed_catalog(probe, monkeypatch):
    catalog = ModelCatalog([_endpoint(1, "probe-path")])
    service = SimpleNamespace(config=SimpleNamespace(ensure_model_catalog=lambda: catalog))
    monkeypatch.setattr(usage_routes, "get_service", lambda: service)
    report = asyncio.run(usage_routes.get_usage_endpoint(provider=None, bot_id=None, access_path=None, force=False))
    assert [snap.provider for snap in report.providers] == ["probe"]
