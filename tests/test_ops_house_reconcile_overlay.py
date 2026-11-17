"""TASK-2406: same-image reconcile of an agent house from the ops catalog.

A house's Compose label ends in a deployment overlay outside the checkout, its
service declares a read-only root filesystem, and recreating it must hold the
house against new assignments. Pure functions and SQLite, no Docker."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from llm_bawt.house_retirement import HouseAssignmentRetiring, assert_assignable, ensure_schema, reservations
from llm_bawt.ops.app_config_reconcile import approval_text, validate_label_compose_config
from llm_bawt.ops.compose_config import BUILT_PROFILE, apply_desired, canonical_hash, needs_read_only_worker, translate
from llm_bawt.ops.compose_render import (ComposeError, compose_source, overlay_files_valid, overlay_mounts,
                                         overlay_paths)
from llm_bawt.ops.house_guard import hold_id, house_targets, recreate_houses, release_houses, reserve_houses
from llm_bawt.ops.sealed_container import sealed_body
from llm_bawt.ops.source_state import SourceStateError, source_plan
from tests.test_ops_init_memory_profile import IMAGE, NAME, SERVICE, SPEC, _container, _drift, _rendered

PROJECT = "/srv/dev/llm-bawt"
ROOT = "/srv/dev/deployment/llm-bawt/houses"
HOUSE_FILE = f"{ROOT}/houses.compose.json"
ENV_FILE = f"{ROOT}/frontend.env"
BASE = [f"{PROJECT}/docker-compose.yml", f"{PROJECT}/docker-compose.dev.yml"]
PIN = "/srv/dev/.agent-work/artifacts/TASK-2405/caid/activation/to-h3.yml"
RENDERER = "docker@sha256:" + "a" * 64


def _config(**over):
    config = {"renderer_image": RENDERER, "overlay_root": ROOT,
              "overlay_files": ["houses.compose.json", "frontend.env"]}
    config.update(over)
    return {k: v for k, v in config.items() if v is not None}


def _spec(**over):
    return {"compose_config": _config(**over)}


def _labelled(files, working_dir=PROJECT):
    return {"Config": {"Labels": {"com.docker.compose.project.working_dir": working_dir,
                                  "com.docker.compose.project.config_files": ",".join(files)}}}


# ── read-only root filesystem on the built kind ─────────────────────────────

def _apply(old, desired):
    return apply_desired({"Labels": {}, "HostConfig": dict(old["HostConfig"])}, old, desired,
                         container_name=NAME, service=SERVICE, rendered_hash="h", image=IMAGE)


def test_undeclared_read_only_keeps_the_desired_configuration_and_its_hash():
    desired = translate(_rendered(), SPEC, "unused", BUILT_PROFILE)
    assert "read_only" not in desired and not needs_read_only_worker(desired)
    assert canonical_hash(desired) == canonical_hash(translate(_rendered(read_only=False), SPEC, "unused",
                                                               BUILT_PROFILE))
    assert _drift(_rendered(), _container()) == []
    assert _apply(_container(), desired)["HostConfig"]["ReadonlyRootfs"] is False


def test_declared_read_only_is_modelled_compared_and_applied():
    rendered = _rendered(read_only=True)
    desired = translate(rendered, SPEC, "unused", BUILT_PROFILE)
    assert desired["read_only"] is True and needs_read_only_worker(desired)
    assert _drift(rendered, _container()) == ["read_only:~"]
    assert _drift(rendered, _container(ReadonlyRootfs=True)) == []
    assert _apply(_container(), desired)["HostConfig"]["ReadonlyRootfs"] is True


def test_an_undeclared_read_only_container_is_drift_and_is_cleared():
    assert _drift(_rendered(), _container(ReadonlyRootfs=True)) == ["read_only:~"]
    assert needs_read_only_worker({}, ["read_only:~"])


# ── overlay declaration ──────────────────────────────────────────────────────

def test_overlay_paths_split_compose_and_env_files():
    assert overlay_paths(_config()) == ([HOUSE_FILE], [ENV_FILE])
    assert overlay_paths({"renderer_image": RENDERER}) == ([], [])
    validate_label_compose_config({"compose_config": _config()})


@pytest.mark.parametrize("over", [
    {"overlay_root": "/etc/houses"},                      # a trusted renderer path
    {"overlay_root": "/run/compose-secrets"},             # the secrets mount
    {"overlay_root": "/srv/../etc"},                      # not clean
    {"overlay_root": None},                               # files without a root
    {"overlay_files": None},                              # a root without files
    {"overlay_files": ["frontend.env"]},                  # no Compose file
    {"overlay_files": ["a.yml", "b.yml", "c.json"]},      # too many Compose files
    {"overlay_files": ["../x.yml"]},                      # not a plain name
    {"overlay_files": ["houses.compose.json", "houses.compose.json"]},
    {"overlay_files": ["secrets.txt", "houses.compose.json"]},
])
def test_an_invalid_overlay_declaration_is_refused(over):
    with pytest.raises(ValueError):
        validate_label_compose_config({"compose_config": _config(**over)})


def test_the_overlay_may_not_overlap_the_project():
    with pytest.raises(ValueError):
        overlay_mounts(_config(overlay_root=f"{PROJECT}/deploy"), PROJECT)
    assert overlay_mounts(_config(), PROJECT) == (HOUSE_FILE, ENV_FILE)


def test_adopt_files_need_an_overlay_and_clean_paths():
    with pytest.raises(ValueError):
        validate_label_compose_config({"compose_config": {"renderer_image": RENDERER, "adopt_files": [PIN]}})
    with pytest.raises(ValueError):
        validate_label_compose_config({"compose_config": _config(adopt_files=["relative.yml"])})


# ── compose source: label tail and adoption ─────────────────────────────────

def test_a_label_ending_in_the_declared_overlay_renders_from_it():
    source = compose_source(_spec(), _labelled([*BASE, HOUSE_FILE]))
    assert source == {"project_dir": PROJECT, "compose_files": [*BASE, HOUSE_FILE]}
    assert overlay_files_valid(_config(), PROJECT, source["compose_files"])


@pytest.mark.parametrize("files", [
    [*BASE],                                   # overlay missing
    [*BASE, f"{ROOT}/other.compose.json"],     # another external file
    [*BASE, HOUSE_FILE, PIN],                  # an extra tail without adoption
    [HOUSE_FILE, *BASE],                       # overlay not at the end
    ["/elsewhere/docker-compose.yml", HOUSE_FILE],
])
def test_any_other_label_is_refused(files):
    with pytest.raises(ComposeError) as caught:
        compose_source(_spec(), _labelled(files))
    assert caught.value.code == "compose_source_invalid"


def test_the_exact_adopted_label_renders_from_the_catalog_list_without_the_pin():
    old = [*BASE, HOUSE_FILE, PIN]
    source = compose_source(_spec(adopt_files=old), _labelled(old))
    assert source["compose_files"] == [*BASE, HOUSE_FILE]
    with pytest.raises(ComposeError):
        compose_source(_spec(adopt_files=old), _labelled([*BASE, PIN, HOUSE_FILE]))


def test_a_sibling_checkout_rebases_only_the_project_files():
    other = "/srv/dev/llm-bawt-next"
    source = compose_source(_spec(), _labelled([*BASE, HOUSE_FILE]), other)
    assert source["compose_files"] == [f"{other}/docker-compose.yml", f"{other}/docker-compose.dev.yml", HOUSE_FILE]


def test_the_bound_source_must_keep_the_overlay_tail():
    assert not overlay_files_valid(_config(), PROJECT, [*BASE])
    assert not overlay_files_valid(_config(), PROJECT, [*BASE, HOUSE_FILE, ENV_FILE])
    assert overlay_files_valid({"renderer_image": RENDERER}, PROJECT, BASE)


def test_the_card_names_the_overlay_and_the_adopted_label():
    from llm_bawt.ops.app_config_reconcile import _overlay_text
    text = _overlay_text(_spec(adopt_files=[*BASE, HOUSE_FILE, PIN]))
    assert HOUSE_FILE in text and ENV_FILE in text and PIN in text and "never read" in text
    assert _overlay_text({"compose_config": {"renderer_image": RENDERER}}) == ""
    assert callable(approval_text)


# ── renderer container ──────────────────────────────────────────────────────

def test_overlay_files_are_single_read_only_binds_never_created():
    body = sealed_body("img", entrypoint=["env"], cmd=[], env=[], directory=PROJECT, labels={}, role="r",
                       ro_bind_files=(HOUSE_FILE, ENV_FILE))
    host = body["HostConfig"]
    assert host["Binds"] == [f"{PROJECT}:{PROJECT}:ro"]
    binds = [m for m in host["Mounts"] if m["Type"] == "bind"]
    assert [(m["Source"], m["Target"], m["ReadOnly"]) for m in binds] == [
        (HOUSE_FILE, HOUSE_FILE, True), (ENV_FILE, ENV_FILE, True)]
    assert host["NetworkMode"] == "none" and host["ReadonlyRootfs"] is True and host["CapDrop"] == ["ALL"]


def test_no_overlay_leaves_the_renderer_body_unchanged():
    body = sealed_body("img", entrypoint=["env"], cmd=[], env=[], directory=PROJECT, labels={}, role="r")
    assert "Mounts" not in body["HostConfig"]


# ── git source guard covers only in-project files ───────────────────────────

def _plan(compose_files, spec_over=None):
    spec = {"compose_config": _config(), "source_mounts": [], "data_mounts": [], "env_names": [],
            **(spec_over or {})}
    desired = {"mounts": {}, "env": {}}
    return source_plan(spec, {"project_dir": PROJECT, "compose_files": compose_files},
                       {"Mounts": []}, {"Config": {"Env": []}}, desired)


def test_the_overlay_is_not_git_checked_but_project_files_are():
    plan = _plan([*BASE, HOUSE_FILE])
    assert plan["compose_files"] == ["docker-compose.yml", "docker-compose.dev.yml"]


def test_an_undeclared_external_compose_file_is_refused():
    with pytest.raises(SourceStateError) as caught:
        _plan([*BASE, "/srv/dev/elsewhere/x.yml"])
    assert "source_compose_outside" in str(caught.value)


# ── house guard and reservation ─────────────────────────────────────────────

APP_SPEC = {"action": "reconcile_app_config", "container_name": "llm-bawt-house-h3",
            "compose_project": "llm-bawt", "compose_service": "house-h3"}


def test_a_house_reconcile_is_guarded_and_recreates_the_house():
    assert house_targets(APP_SPEC, {}) == ("h3",)
    assert recreate_houses(APP_SPEC, {}) == ("h3",)
    assert recreate_houses({**APP_SPEC, "action": "restart"}, {}) == ()
    assert recreate_houses({**APP_SPEC, "container_name": "llm-bawt-app", "compose_service": "app"}, {}) == ()


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://")
    with eng.begin() as conn:
        ensure_schema(conn)
    return eng


def test_reserve_holds_a_free_house_and_blocks_assignment_until_released(engine):
    owner = hold_id("0" * 32)
    assert reserve_houses(engine, lambda h: [], ["h3"], owner) is None
    with engine.connect() as conn:
        assert reservations(conn) == {"h3": owner}
        with pytest.raises(HouseAssignmentRetiring):
            assert_assignable(conn, "h3", "vex")
        assert_assignable(conn, "h2", "vex")
    assert reserve_houses(engine, lambda h: [], ["h3"], owner) is None  # idempotent for the owner
    release_houses(engine, ["h3"], owner)
    with engine.connect() as conn:
        assert reservations(conn) == {}


def test_an_occupied_or_foreign_held_house_is_refused_and_nothing_is_reserved(engine):
    assert "occupied (assigned to vex)" in reserve_houses(engine, lambda h: ["assigned to vex"], ["h3"], "ops-a")
    with engine.connect() as conn:
        assert reservations(conn) == {}
    assert reserve_houses(engine, lambda h: [], ["h3"], "ops-a") is None
    assert "reserved by ops-a" in reserve_houses(engine, lambda h: [], ["h3"], "ops-b")
    release_houses(engine, ["h3"], "ops-b")  # another owner's hold is never touched
    with engine.connect() as conn:
        assert reservations(conn) == {"h3": "ops-a"}
