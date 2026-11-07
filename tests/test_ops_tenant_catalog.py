"""TASK-2128: the tenant installer's restart-row classifier."""

import json

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from llm_bawt.ops import OpsStore, tenant_catalog
from llm_bawt.ops.executor import validate_spec
from llm_bawt.ops.seeds import seed_all


def services(project):
    names = ["app", "agent-bridge", "frontend", "redis", "postgres", "pocket-tts", "house-h1", "house-h2"]
    return {s: f"{project}-{s}" for s in names}


def _store():
    store = object.__new__(OpsStore)
    store.events = None
    store.engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                                 poolclass=StaticPool)
    store._ensure_tables_exist()
    seed_all(store)
    return store


def test_plan_specs_are_valid_and_target_tenant_containers():
    planned = tenant_catalog.plan("t1", services("t1"))
    assert set(planned) == {
        "llm-bawt.restart-app", "llm-bawt.restart-bridge", "bawthub.restart-frontend",
        "llm-bawt.restart-redis", "llm-bawt.restart-aux", "llm-bawt.restart-house"}
    for fields in planned.values():
        validate_spec(fields["command_script"])
    assert json.loads(planned["llm-bawt.restart-app"]["command_script"])["container_name"] == "t1-app"
    house = json.loads(planned["llm-bawt.restart-house"]["command_script"])
    assert house["compose_project"] == "t1" and house["compose_service_from_arg"] == "house"
    aux = json.loads(planned["llm-bawt.restart-aux"]["args_schema_json"])
    assert aux["properties"]["service"]["enum"] == ["t1-pocket-tts"]


def test_absent_services_are_not_planned():
    assert set(tenant_catalog.plan("t1", {"app": "t1-app"})) == {"llm-bawt.restart-app"}


# "llm-bawt" makes the app and house targets identical to the seed's scripts.
@pytest.mark.parametrize("project", ["t1", "llm-bawt"])
def test_classify_enables_once_and_never_overrides_the_operator(project):
    store = _store()
    first = tenant_catalog.classify(store, project, services(project))
    assert len(first["classified"]) == 6 and not first["unchanged"]
    for slug in first["classified"]:
        row = store.get_operation_by_slug(slug)
        assert row.enabled and row.updated_by == tenant_catalog.ACTOR
    assert not store.get_operation_by_slug("llm-bawt.reconcile-app-config").enabled

    for slug in ("llm-bawt.restart-app", "llm-bawt.restart-house"):
        store.update_operation(slug, {"enabled": False}, actor="operator")
    again = tenant_catalog.classify(store, project, services(project))
    assert again["classified"] == [] and len(again["unchanged"]) == 6
    for slug in ("llm-bawt.restart-app", "llm-bawt.restart-house"):
        assert not store.get_operation_by_slug(slug).enabled


def test_compare_and_set_refuses_a_row_changed_after_it_was_read():
    store = _store()
    row = store.get_operation_by_slug("llm-bawt.restart-redis")
    store.update_operation("llm-bawt.restart-redis", {"enabled": False}, actor="operator")
    assert store.update_operation("llm-bawt.restart-redis", {"enabled": True}, actor="x",
                                  expect_version=row.version) is None
    assert not store.get_operation_by_slug("llm-bawt.restart-redis").enabled


def test_soft_deleted_row_is_reported_missing():
    store = _store()
    store.soft_delete_operation("llm-bawt.restart-app", actor="operator")
    assert tenant_catalog.classify(store, "t1", {"app": "t1-app"})["missing"] == ["llm-bawt.restart-app"]
