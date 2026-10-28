"""TASK-1476: the v3 process profile (healthcheck, tmpfs, entrypoint argv) and
the same-image reconcile of a service run by a public TAGGED image (crawl4ai).
No live Docker or network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_bawt.ops.app_config_reconcile import (
    BUILT_PROFILE_LABEL, PROCESS_PROFILE_LABEL, SOURCE_GUARD_LABEL, WORKER_CAPABILITY_LABEL, uses_process_profile,
)
from llm_bawt.ops.compose_config import BUILT_PROFILE, PINNED_PROCESS_PROFILE, PINNED_PROFILE, translate
from llm_bawt.ops.compose_render import ComposeError, ComposeRenderer
from llm_bawt.ops.config_reconcile import WORKER_CAPABILITY_LABEL as CONFIG_WORKER_LABEL, valid_config_binding
from llm_bawt.ops.executor import ConfigCheckError, validate_spec
from llm_bawt.ops.image_deploy import DeployFailed
from llm_bawt.ops.service import _valid_config_snapshot
from llm_bawt.ops.tagged_config_reconcile import (
    ACTION, SCHEMA, TaggedConfigReconciler, approval_text, tagged_config_binding, valid_tagged_binding,
)
from llm_bawt.ops.worker import run as worker_run
from tests.test_ops_app_config_reconcile import (  # noqa: F401  (source: autouse stub of the sealed git check)
    APP_DIR, APP_ID, APP_NAME, EXECUTION, REDIS, SPEC as APP_SPEC, TAG as APP_TAG, AppDocker, _executor,
    _write_request, app_compose, binding as app_binding, reconciler as app_reconciler, snapshot as app_snapshot,
    source,
)
from tests.test_ops_image_deploy import CONFIG_HASH, JOB_ID, RENDERER, RENDERER_ID

TAG = "unclecode/crawl4ai:latest"
NAME = "llm-bawt-crawl4ai"
CRAWL_ID, NEWER = "sha256:" + "7" * 64, "sha256:" + "8" * 64
FILES = [f"{APP_DIR}/docker-compose.yml"]
SECOND = 1_000_000_000
IMAGE_HEALTH = {"Test": ["CMD-SHELL", "redis-cli ping && curl -f http://localhost:11235/health"],
                "Interval": 30 * SECOND, "Timeout": 10 * SECOND, "StartPeriod": 5 * SECOND, "Retries": 3}
SCRIPT = "cp /app/supervisord.conf /tmp/supervisord.conf\nexec /tmp/entrypoint.sh\n"
ENTRY = ["/bin/bash", "-c", SCRIPT]
PORT = {"mode": "ingress", "target": 11235, "published": "11235", "protocol": "tcp"}
TMPFS = {"/data": "uid=999,gid=999"}
SPEC = {"action": ACTION, "compose_config": {"renderer_image": RENDERER}, "display_values": {}, "stat_paths": [],
        "stop_grace_seconds": 10, "health_timeout_seconds": 120, "health_value": "ok", "container_name": NAME,
        "compose_project": "llm-bawt", "compose_service": "crawl4ai",
        "health_url": "http://127.0.0.1:11235/health", "health_field": "status", "image_tag": TAG}
TAGGED_WORKER = {CONFIG_WORKER_LABEL: "1", PROCESS_PROFILE_LABEL: "1"}


def crawl_compose(**service):
    """``docker compose config --format json crawl4ai`` for the shipped block: no drift by default."""
    body = {"image": TAG, "container_name": NAME, "ports": [dict(PORT)],
            "environment": {"CRAWL4AI_API_TOKEN": ""}, "tmpfs": ["/data:uid=999,gid=999"],
            "entrypoint": list(ENTRY), "networks": {"default": None}, "restart": "unless-stopped"}
    body.update(service)
    return {"name": "llm-bawt", "services": {"crawl4ai": {k: v for k, v in body.items() if v is not None}},
            "networks": {"default": {"name": "llm-bawt_default"}}}


def _crawl_image(image_id, refs):
    # The image's own entrypoint and command: a declared entrypoint replaces both.
    return {"Id": image_id, "RepoDigests": [], "_health": "healthy", "_refs": refs,
            "Config": {"Env": ["PATH=/usr/bin"], "Entrypoint": ["/app/entrypoint.sh"], "Cmd": ["supervisord"],
                       "ExposedPorts": {"6379/tcp": {}, "11235/tcp": {}},
                       "Labels": {"maintainer": "unclecode"}, "Healthcheck": dict(IMAGE_HEALTH)}}


class CrawlDocker(AppDocker):
    """crawl4ai as Compose created it from the tag, plus the app fixture's redis."""

    def __init__(self, *, compose=None, tag_moved=False, **kwargs):
        super().__init__(compose=crawl_compose() if compose is None else compose, **kwargs)
        self.images = {CRAWL_ID: _crawl_image(CRAWL_ID, [] if tag_moved else [TAG]),
                       RENDERER_ID: {"Id": RENDERER_ID, "RepoDigests": [RENDERER], "Config": {}}}
        if tag_moved:
            self.images[NEWER] = _crawl_image(NEWER, [TAG])
        crawl = "d" * 64
        self.containers = {crawl: {
            "Id": crawl, "Name": f"/{NAME}", "Image": CRAWL_ID,
            "Config": {"Image": TAG, "Env": ["PATH=/usr/bin", "CRAWL4AI_API_TOKEN="], "Cmd": None,
                       "Entrypoint": list(ENTRY), "ExposedPorts": {"6379/tcp": {}, "11235/tcp": {}},
                       "Healthcheck": dict(IMAGE_HEALTH),
                       "Labels": {"com.docker.compose.project": "llm-bawt",
                                  "com.docker.compose.service": "crawl4ai",
                                  "com.docker.compose.project.working_dir": APP_DIR,
                                  "com.docker.compose.project.config_files": ",".join(FILES),
                                  "maintainer": "unclecode"}},
            "HostConfig": {"RestartPolicy": {"Name": "unless-stopped"}, "Binds": None, "Tmpfs": dict(TMPFS),
                           "PortBindings": {"11235/tcp": [{"HostIp": "", "HostPort": "11235"}]}},
            "NetworkSettings": {"Networks": {"llm-bawt_default": {"Aliases": [NAME, "crawl4ai", crawl[:12]]}}},
            "State": {"Running": True, "Status": "running", "Health": {"Status": "healthy"}}},
            REDIS: self.containers[REDIS]}
        self.old_id = crawl


def binding(api, spec=SPEC, *, moved=False):
    return tagged_config_binding(spec, api.by_name(NAME), api.images[CRAWL_ID],
                                 ComposeRenderer(api, spec, tag="crawl1"), tag_moved=lambda _tag: moved)[0]


def snapshot(api, **kwargs):
    return {"spec": SPEC, "resolved_args": {}, "config": binding(api, **kwargs), "execution": EXECUTION}


def reconciler(api, tmp_path, snap=None):
    return TaggedConfigReconciler(api, snap or snapshot(api), JOB_ID, lock_root=tmp_path / "locks",
                                  sleep=lambda _: None, clock=iter(range(0, 10000, 5)).__next__)


def run(api, tmp_path, snap=None):
    return json.loads(reconciler(api, tmp_path, snap).run())


def _desired(**service):
    return translate(crawl_compose(**service), SPEC, TAG, PINNED_PROCESS_PROFILE)


# ── spec ────────────────────────────────────────────────────────────────────

def test_spec_requires_an_explicit_tag_and_the_label_compose_source():
    assert validate_spec(json.dumps(SPEC)) == SPEC
    for key, value in (("image_tag", "unclecode/crawl4ai@sha256:" + "a" * 64),
                       ("image_tag", "unclecode/crawl4ai"), ("image_tag", "unclecode//crawl4ai:latest"),
                       ("image_tag", "unclecode/./crawl4ai:latest"),
                       ("compose_config", {**SPEC["compose_config"], "compose_files": ["docker-compose.yml"]}),
                       ("compose_config", {"renderer_image": "docker:latest"}),
                       ("health_url", "http://crawl4ai:11235/health"), ("health_value", "ok now"),
                       ("container_name", "../x")):
        with pytest.raises(ValueError):
            validate_spec(json.dumps({**SPEC, key: value}))
    with pytest.raises(ValueError):
        validate_spec(json.dumps({**SPEC, "source_repository": "x/y"}))


# ── the v3 process profile: healthcheck (the built app models it) ───────────

APP_TEST = ["CMD", "curl", "-f", "http://localhost:8642/health"]
APP_HEALTH = {"Test": APP_TEST, "Interval": 30 * SECOND, "Timeout": 0, "StartPeriod": 0, "Retries": 3}


def _app_health(health):
    return translate(app_compose(healthcheck=health), APP_SPEC, APP_TAG, BUILT_PROFILE)["healthcheck"]


def _app_docker(health):
    """The app fixture with an image healthcheck that sets some timing fields."""
    api = AppDocker(compose=app_compose(healthcheck=health))
    api.images[APP_ID]["Config"]["Healthcheck"] = dict(APP_HEALTH)
    api.by_name(APP_NAME)["Config"]["Healthcheck"] = dict(APP_HEALTH)
    return api


@pytest.mark.parametrize("value, nanos", [
    ("30s", 30 * SECOND), ("1m30s", 90 * SECOND), ("1500ms", 1_500_000_000), ("2h", 7200 * SECOND),
    ("10us", 10_000), ("5ns", 5), (None, 0),
])
def test_healthcheck_durations_are_docker_nanoseconds(value, nanos):
    assert _app_health({"test": ["CMD", "true"], **({"interval": value} if value else {})})["Interval"] == nanos


@pytest.mark.parametrize("health, name", [
    ({"test": ["CMD", "true"], "interval": "30"}, "healthcheck.interval"),
    ({"test": ["CMD", "true"], "interval": "1.5s"}, "healthcheck.interval"),
    ({"test": ["CMD", "true"], "timeout": "-1s"}, "healthcheck.timeout"),
    ({"test": ["CMD", "true"], "start_period": "1d"}, "healthcheck.start_period"),
    ({"test": ["CMD", "true"], "retries": "3"}, "healthcheck.retries"),
    ({"test": ["CMD", "true"], "retries": True}, "healthcheck.retries"),
    ({"test": "curl -f localhost"}, "healthcheck.test"),
    ({"test": ["CMD-SHELL", "a", "b"]}, "healthcheck.test"),
    ({"test": ["NONE", "x"]}, "healthcheck.test"),
    ({"test": ["SHELL", "x"]}, "healthcheck.test"),
    ({"test": ["CMD", "true"], "disable": True}, "healthcheck"),
])
def test_unmodelled_healthchecks_refuse(health, name):
    with pytest.raises(ComposeError) as caught:
        _app_health(health)
    assert caught.value.code == "unsupported_compose_key" and caught.value.names == [name]


def test_a_declared_healthcheck_equal_to_the_images_is_none_declared():
    same = {"test": APP_TEST, "interval": "30s", "retries": 3}
    assert app_binding(_app_docker(same))["drift"] == []
    # Unset (zero) fields are the image's, as Docker merges them at create.
    assert app_binding(_app_docker({"test": APP_TEST}))["drift"] == []
    assert app_binding(_app_docker({"test": APP_TEST, "interval": "1m"}))["drift"] == ["healthcheck:~"]


def test_a_changed_healthcheck_is_drift_and_applied_with_unset_fields_left_to_docker(tmp_path):
    api = _app_docker({"test": ["CMD", "curl", "-f", "http://x/"], "timeout": "5s"})
    snap = app_snapshot(api)
    assert snap["config"]["drift"] == ["healthcheck:~"]
    app_reconciler(api, tmp_path, snap).run()
    assert api.create_bodies[0]["Healthcheck"] == {"Test": ["CMD", "curl", "-f", "http://x/"], "Retries": 0,
                                                   "Interval": 0, "Timeout": 5 * SECOND, "StartPeriod": 0,
                                                   "StartInterval": 0}


def test_disabling_the_healthcheck_is_drift_and_applied(tmp_path):
    api = _app_docker({"test": ["NONE"]})
    snap = app_snapshot(api)
    assert snap["config"]["drift"] == ["healthcheck:~"]
    app_reconciler(api, tmp_path, snap).run()
    assert api.create_bodies[0]["Healthcheck"]["Test"] == ["NONE"]


def test_the_tagged_kind_keeps_the_images_healthcheck_and_refuses_a_declared_one():
    with pytest.raises(ComposeError) as caught:
        _desired(healthcheck={"test": ["NONE"]})
    assert caught.value.code == "unsupported_compose_key" and caught.value.names == ["healthcheck"]


def test_an_undeclared_running_healthcheck_is_removed_so_the_images_applies(tmp_path):
    api = CrawlDocker()
    api.by_name(NAME)["Config"]["Healthcheck"] = {**IMAGE_HEALTH, "Interval": SECOND}
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["healthcheck:~"]
    run(api, tmp_path, snap)
    assert "Healthcheck" not in api.create_bodies[0]


# ── tmpfs ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tmpfs, names, applied", [
    (None, ["tmpfs:-/data"], None),
    (["/data:uid=999,gid=999", "/scratch"], ["tmpfs:+/scratch"], {**TMPFS, "/scratch": ""}),
    (["/data:uid=1000"], ["tmpfs:~/data"], {"/data": "uid=1000"}),
    ("/data", ["tmpfs:~/data"], {"/data": ""}),
], ids=["remove", "add", "change", "string-form"])
def test_tmpfs_changes_are_drift_and_applied(tmp_path, tmpfs, names, applied):
    api = CrawlDocker(compose=crawl_compose(tmpfs=tmpfs))
    snap = snapshot(api)
    assert snap["config"]["drift"] == names
    run(api, tmp_path, snap)
    assert api.create_bodies[0]["HostConfig"]["Tmpfs"] == applied
    assert snap["config"]["tmpfs"] == sorted(applied or {})


@pytest.mark.parametrize("tmpfs", [
    ["data"], ["/a/../etc"], ["/data", "/data"], ["/data:mode=1777;rm"], ["/data:uid=$(id)"], [1],
    {"/data": "uid=999"},
])
def test_unmodelled_tmpfs_refuses(tmpfs):
    with pytest.raises(ComposeError) as caught:
        _desired(tmpfs=tmpfs)
    assert caught.value.names == ["tmpfs"]


def test_tmpfs_over_a_volume_target_refuses():
    volume = {"type": "bind", "source": "/srv/crawl", "target": "/data", "bind": {"create_host_path": True}}
    with pytest.raises(ComposeError) as caught:
        _desired(volumes=[volume])
    assert caught.value.names == ["tmpfs"]


def test_built_profile_models_healthcheck_and_tmpfs_only_when_declared():
    plain = translate(app_compose(), APP_SPEC, APP_TAG, BUILT_PROFILE)
    assert "healthcheck" not in plain and "tmpfs" not in plain and not uses_process_profile(plain)
    declared = translate(app_compose(tmpfs=["/run/x"]), APP_SPEC, APP_TAG, BUILT_PROFILE)
    assert declared["tmpfs"] == {"/run/x": ""} and uses_process_profile(declared)


# ── entrypoint argv ────────────────────────────────────────────────────────

def test_the_shipped_service_is_modelled_with_no_drift(tmp_path):
    api = CrawlDocker()
    bound = binding(api)
    # A declared entrypoint runs alone: the image's command is not merged in.
    assert bound["drift"] == [] and bound["ports"] == ["*:11235->11235/tcp"] and bound["tmpfs"] == ["/data"]
    assert bound["image_ref"] == TAG and bound["image_id"] == CRAWL_ID and bound["image_tag_moved"] is False
    assert bound["source"] == {"project_dir": APP_DIR, "compose_files": FILES} and bound["depends_on"] == []
    result = run(api, tmp_path)
    assert result["unchanged"] and result["health"]["status"] == "matched" and not api.create_bodies


def test_a_changed_entrypoint_is_drift_and_applied_without_the_images_command(tmp_path):
    api = CrawlDocker(compose=crawl_compose(entrypoint=["/bin/bash", "-c", "exec /tmp/other.sh"]))
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["entrypoint:~"]
    run(api, tmp_path, snap)
    body = api.create_bodies[0]
    assert body["Entrypoint"] == ["/bin/bash", "-c", "exec /tmp/other.sh"] and "Cmd" not in body


def test_dropping_the_entrypoint_restores_the_images_own(tmp_path):
    api = CrawlDocker(compose=crawl_compose(entrypoint=None))
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["command:~", "entrypoint:~"]
    run(api, tmp_path, snap)
    assert "Entrypoint" not in api.create_bodies[0] and "Cmd" not in api.create_bodies[0]


def test_a_string_entrypoint_or_command_refuses():
    for service in ({"entrypoint": "/bin/bash -c x"}, {"command": "supervisord -n"}):
        with pytest.raises(ComposeError) as caught:
            _desired(**service)
        assert caught.value.code == "unsupported_compose_key"


def test_the_frontend_profile_still_refuses_the_v3_keys():
    ref = TAG
    for key, value in (("entrypoint", ENTRY), ("tmpfs", ["/data"]), ("healthcheck", {"test": ["NONE"]}),
                       ("ports", [dict(PORT)])):
        rendered = {"services": {"crawl4ai": {"image": ref, key: value}}}
        with pytest.raises(ComposeError) as caught:
            translate(rendered, SPEC, ref, PINNED_PROFILE)
        assert caught.value.names == [key]


# ── the tagged kind: the running image, never the tag ──────────────────────

def test_drift_swaps_to_the_running_image_id_and_keeps_the_process(tmp_path):
    api = CrawlDocker(compose=crawl_compose(environment={"CRAWL4AI_API_TOKEN": "", "NEW": "1"}))
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["env:+NEW"]
    result = run(api, tmp_path, snap)
    assert result["schema"] == SCHEMA and result["phase"] == "done" and result["deployed"]["image_id"] == CRAWL_ID
    body = api.create_bodies[0]
    assert body["Image"] == CRAWL_ID and body["Entrypoint"] == ENTRY and "Cmd" not in body
    assert body["HostConfig"]["Tmpfs"] == TMPFS and "Healthcheck" not in body
    assert body["HostConfig"]["PortBindings"] == {"11235/tcp": [{"HostIp": "", "HostPort": "11235"}]}
    assert body["Labels"]["com.docker.compose.config-hash"] == CONFIG_HASH
    assert api.exec_cmds[-1] == ["curl", "-fsS", "--max-time", "10", SPEC["health_url"]]
    assert not any("/images/create" in p for _m, p in api.calls) and not api.renderers_left()


def test_moved_tag_is_shown_and_the_running_image_is_still_what_runs(tmp_path):
    api = CrawlDocker(compose=crawl_compose(environment={"CRAWL4AI_API_TOKEN": "", "NEW": "1"}), tag_moved=True)
    approved = _executor(api, worker_labels=TAGGED_WORKER).config_approval(SPEC)
    assert approved["image_tag_moved"] is True and approved["drift"] == ["env:+NEW"]
    text = approval_text(SPEC, approved, None)
    assert "now names a different local image" in text and "keeps the RUNNING image" in text
    snap = {"spec": SPEC, "resolved_args": {}, "config": approved, "execution": EXECUTION}
    result = run(api, tmp_path, snap)
    assert api.create_bodies[0]["Image"] == CRAWL_ID and result["config"]["image_tag_moved"] is True
    assert not any(TAG in p for _m, p in api.calls if p.startswith("/images/"))  # never re-resolved


def test_a_target_already_running_by_id_still_binds():
    api = CrawlDocker()
    assert _executor(api, worker_labels=TAGGED_WORKER).config_approval(SPEC)["image_tag_moved"] is False
    api.by_name(NAME)["Config"]["Image"] = CRAWL_ID
    assert binding(api)["image_ref"] == CRAWL_ID


def test_a_foreign_running_image_or_compose_image_refuses():
    api = CrawlDocker()
    api.by_name(NAME)["Config"]["Image"] = "unclecode/crawl4ai:0.6"
    with pytest.raises(ValueError, match="declared image tag"):
        binding(api)
    with pytest.raises(ComposeError, match="compose_image_mismatch"):
        binding(CrawlDocker(compose=crawl_compose(image="unclecode/crawl4ai:0.7")))


def test_card_shows_names_ports_tmpfs_and_no_values():
    secret = "tok-never-shown"
    bound = binding(CrawlDocker(compose=crawl_compose(environment={"CRAWL4AI_API_TOKEN": secret})))
    text = approval_text(SPEC, bound, None)
    assert bound["drift"] == ["env:~CRAWL4AI_API_TOKEN"] and secret not in json.dumps(bound) + text
    assert "Ports after: *:11235->11235/tcp" in text and "Tmpfs after: /data" in text
    assert f"Unchanged image: {CRAWL_ID} (tag {TAG})" in text and "now names" not in text


@pytest.mark.parametrize("tamper", [
    lambda s: s["config"].update(env={"CRAWL4AI_API_TOKEN": "x"}),
    lambda s: s["config"].update(image_tag_moved="no"),
    lambda s: s["config"].update(image_tag="evil/crawl:latest"),
    lambda s: s["config"].update(image_ref="evil/crawl:latest"),
    lambda s: s["config"].update(tmpfs=["data"]),
    lambda s: s["config"].update(tmpfs=[1]),
    lambda s: s["config"]["source"].update(compose_files=["/etc/compose.yml"]),
    lambda s: s["config"].update(display={"X": "1"}),
    lambda s: s["config"].pop("tmpfs"),
    lambda s: s.update(resolved_args={"x": 1}),
    lambda s: s.update(spec={**SPEC, "image_tag": "evil/crawl:latest"}),
])
def test_binding_shape_is_validated(tamper):
    snap = snapshot(CrawlDocker())
    assert valid_tagged_binding(snap) and _valid_config_snapshot(snap)
    tamper(snap)
    assert not valid_tagged_binding(snap) and not _valid_config_snapshot(snap)


def test_other_kinds_never_accept_a_tagged_binding():
    snap = snapshot(CrawlDocker())
    assert not valid_config_binding(snap)
    assert not _valid_config_snapshot({**snap, "spec": APP_SPEC})


def test_change_since_approval_refuses_before_create(tmp_path):
    api = CrawlDocker(compose=crawl_compose(environment={"CRAWL4AI_API_TOKEN": "", "NEW": "1"}))
    snap = snapshot(api)
    api.compose = crawl_compose(tmpfs=["/data:uid=1000"])
    with pytest.raises(DeployFailed, match=r"config_changed_since_approval \(desired_hash, drift") as caught:
        run(api, tmp_path, snap)
    assert caught.value.outcome_known and not api.create_bodies


def test_failed_health_restores_the_previous_container(tmp_path):
    api = CrawlDocker(compose=crawl_compose(environment={"CRAWL4AI_API_TOKEN": "", "NEW": "1"}),
                      health_status="degraded")
    with pytest.raises(DeployFailed, match="health status did not match expected 'ok'") as caught:
        run(api, tmp_path)
    assert caught.value.detail["restored"] and api.by_name(NAME)["Id"] == api.old_id


def test_a_declared_dependency_is_a_precondition(tmp_path):
    depends = {"redis": {"condition": "service_healthy", "required": True}}
    api = CrawlDocker(compose=crawl_compose(environment={"CRAWL4AI_API_TOKEN": "", "NEW": "1"},
                                            depends_on=depends))
    snap = snapshot(api)
    assert snap["config"]["depends_on"] == ["redis:service_healthy"]
    api.containers[REDIS]["State"]["Health"]["Status"] = "unhealthy"
    with pytest.raises(DeployFailed, match="dependency_unhealthy: redis"):
        run(api, tmp_path, snap)
    assert not api.create_bodies


# ── worker capability: v3 labels ───────────────────────────────────────────

@pytest.mark.parametrize("missing", [CONFIG_WORKER_LABEL, PROCESS_PROFILE_LABEL])
def test_tagged_reconcile_requires_a_v3_worker(missing):
    api = CrawlDocker()
    executor = _executor(api, worker_labels={k: v for k, v in TAGGED_WORKER.items() if k != missing})
    for call in (lambda: executor.config_approval(SPEC),
                 lambda: executor.preflight({"spec": SPEC, "execution": executor.execution_settings()})):
        with pytest.raises(ConfigCheckError) as caught:
            call()
        assert caught.value.code == "config_reconcile_unavailable" and missing in str(caught.value)
    assert not api.render_bodies


APP_V2_WORKER = {WORKER_CAPABILITY_LABEL: "1", SOURCE_GUARD_LABEL: "1", BUILT_PROFILE_LABEL: "1"}


def test_app_needs_the_v3_worker_only_once_it_declares_a_v3_key():
    assert _executor(AppDocker(), worker_labels=APP_V2_WORKER).config_approval(APP_SPEC)["drift"] == []
    # A tmpfs is a mount destination: the install classifies it like any other.
    tmpfs_spec = {**APP_SPEC, "data_mounts": sorted([*APP_SPEC["data_mounts"], "/run/x"])}
    for service, spec in (({"tmpfs": ["/run/x"]}, tmpfs_spec), ({"healthcheck": {"test": ["CMD", "true"]}}, APP_SPEC),
                          ({"entrypoint": ["tini", "--"]}, APP_SPEC)):
        api = AppDocker(compose=app_compose(**service))
        with pytest.raises(ConfigCheckError) as caught:
            _executor(api, worker_labels=APP_V2_WORKER).config_approval(spec)
        assert caught.value.code == "config_reconcile_unavailable" and PROCESS_PROFILE_LABEL in str(caught.value)
        approved = _executor(api, worker_labels={**APP_V2_WORKER, PROCESS_PROFILE_LABEL: "1"}).config_approval(spec)
        assert approved["drift"]


def test_worker_runs_the_tagged_kind_from_a_receipt(tmp_path):
    api = CrawlDocker(compose=crawl_compose(environment={"CRAWL4AI_API_TOKEN": "", "NEW": "1"}))
    root = tmp_path / "receipts"
    snap = {**snapshot(api), "execution": {**EXECUTION, "receipt_root": str(root)}}
    path, digest = _write_request(root, snap)
    assert not worker_run(path, digest, api_factory=lambda _: api, sleep=lambda _: None)
    fresh = _executor(CrawlDocker(), worker_labels=TAGGED_WORKER)
    fresh._get_worker = lambda _name: None
    result = fresh.reconcile(job_id=JOB_ID, snapshot=snap)
    assert result.state == "succeeded" and json.loads(result.output_tail)["schema"] == SCHEMA
    assert api.create_bodies[0]["Image"] == CRAWL_ID


def test_worker_image_carries_the_module_and_the_v3_label():
    dockerfile = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.ops-worker").read_text()
    assert "src/llm_bawt/ops/tagged_config_reconcile.py" in dockerfile
    assert f'LABEL {PROCESS_PROFILE_LABEL}="1"' in dockerfile
