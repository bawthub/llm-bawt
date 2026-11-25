"""TASK-2370: the dangerous llm-bawt writes need BawtHub's service bearer.

Approval resolution and policy writes, ops catalog writes and dispatch, model
catalog writes, integration credentials and provider connections were open to
any LAN caller. The set below is the contract BawtHub's proxy allowlist mirrors
(frontend/server/lib/auth/operatorBearer.ts); a route added to or dropped from
it must change both sides.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from llm_bawt.bawthub_auth.service_token import TokenCheck
from llm_bawt.service.routes import (
    approval_policies,
    guidance,
    integrations,
    mcp_servers,
    model_catalog,
    ops,
    providers,
    service_auth,
    skills,
)
from llm_bawt.service.routes.service_auth import require_service_token

ROUTERS = (approval_policies, ops, model_catalog, mcp_servers, integrations, providers, guidance, skills)

PROTECTED = {
    ("POST", "/v1/tool-approval-policies"),
    ("POST", "/v1/tool-approval-policies/seed-defaults"),
    ("PATCH", "/v1/tool-approval-policies/{policy_id}"),
    ("DELETE", "/v1/tool-approval-policies/{policy_id}"),
    ("POST", "/v1/admin/reload-tool-approval-policies"),
    ("GET", "/v1/tool-approval-requests"),
    ("POST", "/v1/chat/approvals/{request_id}/resolve"),
    ("POST", "/v1/ops/operations"),
    ("PATCH", "/v1/ops/operations/{slug}"),
    ("POST", "/v1/ops/operations/{slug}/soft-delete"),
    ("POST", "/v1/ops/operations/{slug}/enable"),
    ("POST", "/v1/ops/seed-defaults"),
    ("POST", "/v1/ops/catalog-proposals/{batch_id}/decide"),
    ("POST", "/v1/ops/jobs"),
    ("POST", "/v1/ops/jobs/{job_id}/cancel"),
    ("POST", "/v1/ops/operator-jobs"),
    ("POST", "/v1/ops/host-access"),
    ("PATCH", "/v1/ops/host-access/{host_id}"),
    ("DELETE", "/v1/ops/host-access/{host_id}"),
    ("POST", "/v1/ops/host-access/{host_id}/confirm-host-key"),
    ("PUT", "/v1/models/catalog/models/{model_key}"),
    ("DELETE", "/v1/models/catalog/models/{model_key}"),
    ("PUT", "/v1/models/catalog/access-paths/{access_key}"),
    ("DELETE", "/v1/models/catalog/access-paths/{access_key}"),
    ("PUT", "/v1/models/catalog/endpoints/{model_key}/{access_key}"),
    ("PATCH", "/v1/models/catalog/endpoints/{model_key}/{access_key}"),
    ("DELETE", "/v1/models/catalog/endpoints/{endpoint_id}"),
    ("PUT", "/v1/integrations/{integration_id}/workspace"),
    ("POST", "/v1/integrations/{integration_id}/workspace/verify"),
    ("PUT", "/v1/integrations/{integration_id}/credentials"),
    ("PUT", "/v1/integrations/{integration_id}/client"),
    ("DELETE", "/v1/integrations/{integration_id}"),
    ("POST", "/v1/integrations/google-tasks/changed"),
    ("POST", "/v1/integrations/google-tasks/sync"),
    ("POST", "/v1/integrations/mcp-servers"),
    ("PATCH", "/v1/integrations/mcp-servers/{slug}"),
    ("DELETE", "/v1/integrations/mcp-servers/{slug}"),
    ("PUT", "/v1/integrations/mcp-servers/{slug}/bearer"),
    ("PUT", "/v1/integrations/mcp-servers/{slug}/grants"),
    ("POST", "/v1/integrations/mcp-servers/{slug}/tools"),
    ("POST", "/v1/integrations/mcp-servers/{slug}/connect"),
    ("GET", "/v1/integrations/mcp-servers/{slug}/callback"),
    ("DELETE", "/v1/integrations/mcp-servers/{slug}/connection"),
    ("POST", "/v1/providers/{provider_id}/connect/api-key"),
    ("POST", "/v1/providers/{provider_id}/connect/endpoint"),
    ("DELETE", "/v1/providers/{provider_id}"),
    ("PUT", "/v1/providers/github/app-config"),
    ("POST", "/v1/providers/github/repos"),
    ("GET", "/v1/guidance"),
    ("POST", "/v1/guidance/proposals/{proposal_id}/decide"),
    ("POST", "/v1/guidance/runs"),
    ("POST", "/v1/skills/sources"),
    ("POST", "/v1/skills/sources/{source_id}/check"),
    ("POST", "/v1/skills/sources/{source_id}/update"),
    ("POST", "/v1/skills/sources/{source_id}/rollback"),
    ("DELETE", "/v1/skills/sources/{source_id}"),
}


def _app() -> FastAPI:
    app = FastAPI()
    for module in ROUTERS:
        app.include_router(module.router)
    return app


def _guarded(route: APIRoute) -> bool:
    return any(dep.call is require_service_token for dep in route.dependant.dependencies)


def test_exactly_the_dangerous_routes_need_the_bearer():
    guarded = {(method, route.path) for route in _app().routes if isinstance(route, APIRoute)
               for method in route.methods if _guarded(route)}
    assert guarded == PROTECTED


def test_bridge_and_browser_reads_stay_open():
    # Bridges fetch the policy bundle; the UI and agents read the catalogs.
    open_reads = {("GET", "/v1/tool-approval-policies/bundle"), ("GET", "/v1/ops/operations"),
                  ("GET", "/v1/models/catalog/endpoints"), ("GET", "/v1/integrations")}
    routes = {(method, route.path): route for route in _app().routes if isinstance(route, APIRoute)
              for method in route.methods}
    assert all(not _guarded(routes[key]) for key in open_reads)


@pytest.mark.parametrize("method, path", sorted(PROTECTED))
def test_protected_route_refuses_a_caller_without_the_bearer(monkeypatch, method, path):
    monkeypatch.setattr(service_auth, "service_token_source",
                        lambda: SimpleNamespace(check=lambda header: TokenCheck.INVALID))
    url = path.format(policy_id="p", request_id="r", slug="s", job_id="j", model_key="m",
                      access_key="a", endpoint_id="e", integration_id="unraid", provider_id="openai-api", host_id="h",
                      proposal_id="1", source_id="s", batch_id="b")
    response = TestClient(_app()).request(method, url)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
