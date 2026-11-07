"""TASK-2128: the tenant installer's restart-row classifier."""

import json
from types import SimpleNamespace

from llm_bawt.ops import tenant_catalog
from llm_bawt.ops.executor import validate_spec
from llm_bawt.ops.seeds import SEEDS

SERVICES = {
    "app": "t1-app", "agent-bridge": "t1-agent-bridge", "frontend": "t1-frontend",
    "redis": "t1-redis", "postgres": "t1-postgres", "pocket-tts": "t1-pocket-tts",
    "house-h1": "t1-house-h1", "house-h2": "t1-house-h2",
}


class FakeStore:
    def __init__(self):
        self.rows = {s["slug"]: SimpleNamespace(**{**s, "soft_deleted_at": None}) for s in SEEDS}
        self.updates = []

    def get_operation_by_slug(self, slug):
        return self.rows.get(slug)

    def update_operation(self, slug, data, *, actor=None):
        self.updates.append((slug, actor))
        for k, v in data.items():
            setattr(self.rows[slug], k, v)
        return self.rows[slug]


def test_plan_specs_are_valid_and_target_tenant_containers():
    planned = tenant_catalog.plan("t1", SERVICES)
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


def test_classify_enables_once_and_respects_operator_edits():
    store = FakeStore()
    first = tenant_catalog.classify(store, "t1", SERVICES)
    assert len(first["classified"]) == 6 and not first["unchanged"]
    assert all(store.rows[s].enabled for s in first["classified"])
    assert all(actor == tenant_catalog.ACTOR for _, actor in store.updates)
    # Unclassified echo-only rows stay disabled.
    assert not store.rows["llm-bawt.reconcile-app-config"].enabled

    store.rows["llm-bawt.restart-redis"].enabled = False  # operator turns one off
    store.updates.clear()
    again = tenant_catalog.classify(store, "t1", SERVICES)
    assert again["classified"] == [] and len(again["unchanged"]) == 6
    assert store.updates == [] and not store.rows["llm-bawt.restart-redis"].enabled


def test_soft_deleted_row_is_reported_missing():
    store = FakeStore()
    store.rows["llm-bawt.restart-app"].soft_deleted_at = "x"
    assert tenant_catalog.classify(store, "t1", {"app": "t1-app"})["missing"] == ["llm-bawt.restart-app"]
