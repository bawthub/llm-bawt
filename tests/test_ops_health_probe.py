"""TASK-1801: catalog-declared health probes, adopted Compose sources, the
pinned-hardened profile, the ci_publish binding and the worker-capability gate.

No network, no Docker: the deployer runs against the TASK-997 FakeDocker, the
verifier against a fake GitHub opener.
"""
from __future__ import annotations

import copy
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.ops.compose_config import (
    HARDENED_PROFILE, HARDENED_PROFILE_LABEL, apply_desired, drift, normalized, running, selected_profile, translate,
)
from llm_bawt.ops.compose_render import ComposeError, compose_source
from llm_bawt.ops.executor import Executor, ExecutorError, spec_capability_labels, validate_spec
from llm_bawt.ops.health_probe import WORKER_LABEL as HEALTH_PROBE_LABEL
from llm_bawt.ops.health_probe import http_exec_argv, validate_health_probe
from llm_bawt.ops.image_deploy import DeployFailed, ImageDeployer, StepError
from llm_bawt.ops.release import (BINDING_CI_PUBLISH, GitHubReleaseVerifier, ReleaseVerificationError,
                                  binding_matches_args)
from tests.test_ops_config_reconcile import WORKER, _executor
from tests.test_ops_image_deploy import (
    DIGEST, JOB_ID, NAME, NEW_ID, OLD_ID, PROJECT_DIR, SEED, SOURCE, SPEC, FakeDocker, _labels, _release, _Resp,
)

POCKET = {slug: json.loads(SEED[slug]["command_script"]) for slug in (
    "bawthub.deploy-pocket-tts", "bawthub.rollback-pocket-tts", "bawthub.reconcile-pocket-tts-config")}
POCKET_DEPLOY = POCKET["bawthub.deploy-pocket-tts"]
POCKET_REPO = POCKET_DEPLOY["image_repository"]
POCKET_REF = f"{POCKET_REPO}@{DIGEST}"
RUN_ID = "987654"
PUBLISH_ARGS = {"workflow_run_id": RUN_ID, "source_sha": SOURCE, "digest": DIGEST}
FRONTEND_PROBE = {"type": "http_exec", "port": 3000, "path": "/api/health"}


# ── the probe field ────────────────────────────────────────────────────────

@pytest.mark.parametrize("probe", [
    {"type": "docker_health"}, {"type": "http_exec", "port": 8091, "path": "/health"}, FRONTEND_PROBE,
])
def test_valid_probes(probe):
    assert validate_health_probe(probe) == probe


@pytest.mark.parametrize("probe", [
    None, "docker_health", {}, {"type": "tcp"},
    {"type": "docker_health", "port": 1},
    {"type": "http_exec", "port": 3000},
    {"type": "http_exec", "port": 0, "path": "/x"},
    {"type": "http_exec", "port": 70000, "path": "/x"},
    {"type": "http_exec", "port": True, "path": "/x"},
    {"type": "http_exec", "port": "3000", "path": "/x"},
    {"type": "http_exec", "port": 3000, "path": "x"},
    {"type": "http_exec", "port": 3000, "path": "/a/../b"},
    {"type": "http_exec", "port": 3000, "path": "//evil"},
    {"type": "http_exec", "port": 3000, "path": "/a?b=1"},
    {"type": "http_exec", "port": 3000, "path": "/a b"},
    {"type": "http_exec", "port": 3000, "path": "/x", "host": "10.0.0.1"},
])
def test_invalid_probes_refuse(probe):
    with pytest.raises(ValueError):
        validate_health_probe(probe)


def test_http_exec_argv_uses_only_the_declared_port_and_path():
    assert http_exec_argv({"type": "http_exec", "port": 8091, "path": "/health"}) == [
        "curl", "-fsS", "--max-time", "10", "http://127.0.0.1:8091/health"]


def test_every_image_family_seed_declares_a_valid_probe():
    """The frontend rows migrate onto the field; nothing relies on a built-in port."""
    image_family = [s for s in SEED.values() if json.loads(s["command_script"]).get("action") in {
        "deploy_image", "rollback_image", "reconcile_frontend_config", "reconcile_frontend_routing"}]
    assert {s["slug"] for s in image_family} >= {
        "bawthub.deploy-prod-image", "bawthub.deploy-pocket-tts", "bawthub.rollback-pocket-tts",
        "bawthub.reconcile-pocket-tts-config"}
    for seed in image_family:
        spec = json.loads(seed["command_script"])
        validate_health_probe(spec["health_probe"])
        if spec["container_name"] == NAME:
            assert spec["health_probe"] == FRONTEND_PROBE


# ── the deployer verifies with the row's probe ─────────────────────────────

class ProbeDocker(FakeDocker):
    """FakeDocker that also records exec create bodies."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exec_bodies = []

    def request(self, method, path, *, body=None, headers=None, raw=False):
        if method == "POST" and path.endswith("/exec") and path.startswith("/containers/"):
            self.exec_bodies.append(body)
        return super().request(method, path, body=body, headers=headers, raw=raw)


def _spec(**over):
    spec = copy.deepcopy(SPEC)
    spec.update(over)
    return spec


def _run(api, tmp_path, spec):
    return json.loads(ImageDeployer(api, {"spec": spec, "release": _release()}, JOB_ID, lock_root=tmp_path / "l",
                                    sleep=lambda _s: None, clock=iter(range(0, 10_000, 5)).__next__).run())


@pytest.mark.parametrize("probe", [None, {"type": "tcp"}, {"type": "http_exec", "port": 3000}])
def test_deploy_without_a_valid_probe_refuses_before_the_lock_or_any_docker_call(tmp_path, probe):
    spec = _spec()
    spec.pop("health_probe")
    if probe is not None:
        spec["health_probe"] = probe
    api = ProbeDocker()
    with pytest.raises(DeployFailed, match="health_probe") as caught:
        _run(api, tmp_path, spec)
    assert caught.value.outcome_known and api.calls == [] and not (tmp_path / "l").exists()


def test_http_exec_probe_requests_the_declared_port_and_path(tmp_path):
    api = ProbeDocker()
    _run(api, tmp_path, _spec(health_probe={"type": "http_exec", "port": 8091, "path": "/ready"}))
    assert [b["Cmd"] for b in api.exec_bodies] == [
        ["curl", "-fsS", "--max-time", "10", "http://127.0.0.1:8091/ready"]]


def test_docker_health_probe_checks_the_approved_image_and_its_release_labels(tmp_path):
    api = ProbeDocker()
    record = _run(api, tmp_path, _spec(health_probe={"type": "docker_health"}))
    assert api.exec_bodies == []  # no in-container request
    assert record["health"] == {"status": "healthy", "probe": "docker_health", "image_id": NEW_ID,
                                "release": {"version": "1.4.2", "source_sha": SOURCE, "workflow_run_id": "123456"}}
    assert api.by_name(NAME)["Image"] == NEW_ID


@pytest.mark.parametrize("labels", [_labels(source="9" * 40), _labels(run="1"), {}])
def test_docker_health_deploy_refuses_a_foreign_release_before_the_swap(tmp_path, labels):
    api = ProbeDocker(new_labels=labels)
    with pytest.raises(DeployFailed, match="release labels do not match") as caught:
        _run(api, tmp_path, _spec(health_probe={"type": "docker_health"}))
    assert caught.value.outcome_known and not api.create_bodies
    live = api.by_name(NAME)
    assert live["Id"] == api.old_id and live["Image"] == OLD_ID


def _verifier(api, tmp_path, approved):
    deployer = ImageDeployer(api, {"spec": _spec(health_probe={"type": "docker_health"}), "release": _release()},
                             JOB_ID, lock_root=tmp_path / "l", sleep=lambda _s: None)
    deployer._approved_image_id = approved
    return deployer


EXPECT = {"version": "1.4.2", "source_sha": SOURCE, "workflow_run_id": "123456"}


def test_docker_health_verification_checks_identity_on_the_container(tmp_path):
    api = ProbeDocker()
    old = api.old_id
    assert _verifier(api, tmp_path, OLD_ID)._verify_release(old, None)["image_id"] == OLD_ID
    with pytest.raises(StepError, match="approved image"):
        _verifier(api, tmp_path, NEW_ID)._verify_release(old, EXPECT)  # runs OLD, approved NEW
    with pytest.raises(StepError, match="no approved image"):
        _verifier(api, tmp_path, None)._verify_release(old, EXPECT)
    with pytest.raises(StepError, match="running release"):
        _verifier(api, tmp_path, OLD_ID)._verify_release(old, EXPECT)  # OLD image carries no release labels
    api.by_name(NAME)["State"]["Health"]["Status"] = "starting"
    with pytest.raises(StepError, match="not running healthy"):
        _verifier(api, tmp_path, OLD_ID)._verify_release(old, None)


def test_docker_health_probe_refuses_an_unhealthy_container_and_restores(tmp_path):
    api = ProbeDocker(new_health="unhealthy")
    with pytest.raises(DeployFailed):
        _run(api, tmp_path, _spec(health_probe={"type": "docker_health"}))
    assert api.by_name(NAME)["Id"] == api.old_id


# ── adoption: an earlier file list is relabelled onto the catalog's ────────

def _adopting_spec():
    files = SPEC["compose_config"]["compose_files"]
    return _spec(compose_config={**SPEC["compose_config"], "adopt_files": files[:1]})


def _relabel(api, files):
    labels = api.by_name(NAME)["Config"]["Labels"]
    labels["com.docker.compose.project.config_files"] = ",".join(f"{PROJECT_DIR}/{n}" for n in files)


def test_compose_source_adopts_only_the_declared_earlier_list():
    spec = _adopting_spec()
    expected = [f"{PROJECT_DIR}/{n}" for n in spec["compose_config"]["compose_files"]]
    for files in (spec["compose_config"]["adopt_files"], spec["compose_config"]["compose_files"]):
        api = FakeDocker()
        _relabel(api, files)
        assert compose_source(spec, api.by_name(NAME)) == {"project_dir": PROJECT_DIR, "compose_files": expected}
    for files in (["docker-compose.image.yml"], ["docker-compose.yml", "docker-compose.override.yml"]):
        api = FakeDocker()
        _relabel(api, files)
        with pytest.raises(ComposeError, match="compose_source_invalid"):
            compose_source(spec, api.by_name(NAME))


def test_without_adopt_files_the_earlier_list_still_refuses():
    api = FakeDocker()
    _relabel(api, SPEC["compose_config"]["compose_files"][:1])
    with pytest.raises(ComposeError, match="compose_source_invalid"):
        compose_source(SPEC, api.by_name(NAME))


def test_adopted_target_is_rendered_from_and_relabelled_with_the_catalog_files(tmp_path):
    spec = _adopting_spec()
    api = ProbeDocker()
    _relabel(api, spec["compose_config"]["adopt_files"])
    _run(api, tmp_path, spec)
    catalog = [f"{PROJECT_DIR}/{n}" for n in spec["compose_config"]["compose_files"]]
    rendered = api.render_bodies[0]["Cmd"]
    assert [rendered[i + 1] for i, a in enumerate(rendered) if a == "-f"] == catalog
    assert api.create_bodies[0]["Labels"]["com.docker.compose.project.config_files"] == ",".join(catalog)


# ── the pinned-hardened profile (pocket-tts) ───────────────────────────────

def _pocket_rendered(**changes):
    """The pocket-tts service as ``docker compose config --format json`` prints it."""
    service = {
        "cap_drop": ["ALL"], "cpus": 2, "command": None, "entrypoint": None,
        "environment": {"CUDA_VISIBLE_DEVICES": "", "HF_TOKEN": "fixture-token",
                        "POCKET_CACHE_DIR": "/var/cache/pocket-tts",
                        "POCKET_VOICES_DIR": "/var/lib/pocket-tts/voices"},
        "healthcheck": {"test": ["CMD", "python", "-c", "pass"], "timeout": "5s", "interval": "30s",
                        "retries": 3, "start_period": "3m0s"},
        "image": POCKET_REF, "mem_limit": "4294967296", "memswap_limit": "8589934592",
        "networks": {"default": None}, "pids_limit": 128, "pull_policy": "missing", "read_only": True,
        "restart": "unless-stopped", "security_opt": ["no-new-privileges:true"],
        "tmpfs": ["/tmp:size=256m,mode=1777"],
        "volumes": [{"type": "volume", "source": "pocket-model-cache", "target": "/var/cache/pocket-tts", "volume": {}},
                    {"type": "volume", "source": "pocket-voice-states", "target": "/var/lib/pocket-tts/voices",
                     "volume": {}}]}
    service.update(changes)
    for key in [k for k, v in changes.items() if v is _DROP]:
        service.pop(key)
    return {"name": "bawthub", "services": {"pocket-tts": service},
            "networks": {"default": {"name": "bawthub_default"}},
            "volumes": {"pocket-model-cache": {"name": "bawthub_pocket-model-cache"},
                        "pocket-voice-states": {"name": "bawthub_pocket-voice-states"}}}


_DROP = object()
IMAGE = {"Config": {"Env": [], "Labels": {}, "ExposedPorts": {"8091/tcp": {}}}}


def _pocket_desired(**changes):
    return translate(_pocket_rendered(**changes), POCKET_DEPLOY, POCKET_REF, HARDENED_PROFILE)


def _container_from(body):
    """What Docker reports for a container created from ``body``."""
    config = {k: body[k] for k in ("Env", "Labels", "Cmd", "Entrypoint", "Healthcheck", "ExposedPorts",
                                   "OpenStdin", "Tty", "StopTimeout") if k in body}
    return {"Config": config, "HostConfig": body["HostConfig"],
            "NetworkSettings": {"Networks": dict.fromkeys(body["NetworkingConfig"]["EndpointsConfig"], {})}}


def _pocket_created(desired):
    base = {"Image": POCKET_REF, "Labels": {}, "HostConfig": {"NetworkMode": "bawthub_default"},
            "NetworkingConfig": {"EndpointsConfig": {"bawthub_default": {}}}}
    return apply_desired(base, {}, desired, container_name="bawthub-pocket-tts-1", service="pocket-tts",
                         rendered_hash="h", image=IMAGE)


def test_pocket_seed_selects_the_hardened_profile():
    assert selected_profile(POCKET_DEPLOY["compose_config"]) is HARDENED_PROFILE
    assert POCKET_DEPLOY["health_probe"] == {"type": "docker_health"}


def test_hardened_profile_models_the_live_confinement():
    desired = _pocket_desired()
    assert desired["hardening"] == {"CapDrop": ["ALL"], "CapAdd": [], "NanoCpus": 2_000_000_000,
                                    "PidsLimit": 128, "ReadonlyRootfs": True}
    assert desired["memory"] == {"Memory": 4294967296, "MemorySwap": 8589934592}
    # The live container's HostConfig as docker inspect reports it.
    live = running({"HostConfig": {"CapDrop": ["ALL"], "CapAdd": None, "NanoCpus": 2000000000,
                                   "PidsLimit": 128, "ReadonlyRootfs": True}})
    assert live["hardening"] == desired["hardening"]


def test_hardened_create_body_round_trips_with_zero_drift():
    desired = _pocket_desired()
    created = _pocket_created(desired)
    host = created["HostConfig"]
    assert (host["CapDrop"], host["CapAdd"], host["NanoCpus"], host["PidsLimit"], host["ReadonlyRootfs"]) == (
        ["ALL"], None, 2_000_000_000, 128, True)
    assert drift(normalized(desired, IMAGE), normalized(running(_container_from(created)), IMAGE)) == []


@pytest.mark.parametrize("mutate, name", [
    (lambda h: h.update(CapDrop=None), "hardening:~CapDrop"),
    (lambda h: h.update(CapAdd=["SYS_ADMIN"]), "hardening:~CapAdd"),
    (lambda h: h.update(NanoCpus=0), "hardening:~NanoCpus"),
    (lambda h: h.update(PidsLimit=None), "hardening:~PidsLimit"),
    (lambda h: h.update(ReadonlyRootfs=False), "hardening:~ReadonlyRootfs"),
])
def test_loosened_confinement_is_drift_by_name(mutate, name):
    desired = _pocket_desired()
    container = _container_from(_pocket_created(desired))
    mutate(container["HostConfig"])
    assert drift(normalized(desired, IMAGE), normalized(running(container), IMAGE)) == [name]


@pytest.mark.parametrize("changes, names", [
    ({"cap_drop": _DROP}, ["hardening:~CapDrop"]),
    ({"read_only": _DROP}, ["hardening:~ReadonlyRootfs"]),
    ({"pids_limit": _DROP}, ["hardening:~PidsLimit"]),
    ({"cpus": _DROP}, ["hardening:~NanoCpus"]),
])
def test_a_compose_file_dropping_confinement_is_drift_against_the_live_container(changes, names):
    live = _container_from(_pocket_created(_pocket_desired()))
    desired = _pocket_desired(**changes)
    assert drift(normalized(desired, IMAGE), normalized(running(live), IMAGE)) == names


@pytest.mark.parametrize("changes", [
    {"cap_add": ["NET_ADMIN"]}, {"privileged": True}, {"devices": ["/dev/kmsg"]},
    {"cap_drop": "ALL"}, {"cap_drop": ["ALL", "CAP_ALL"]}, {"cpus": 0}, {"cpus": 65},
    {"pids_limit": 0}, {"pids_limit": True}, {"read_only": "yes"},
])
def test_hardened_profile_refuses_what_it_does_not_model(changes):
    with pytest.raises(ComposeError, match="unsupported_compose_key"):
        _pocket_desired(**changes)


def test_other_profiles_do_not_model_hardening():
    with pytest.raises(ComposeError, match="unsupported_compose_key"):
        translate(_pocket_rendered(), POCKET_DEPLOY, POCKET_REF, selected_profile({}))


# ── ci_publish: verification of the CI publish run ─────────────────────────

TRIGGER_ID = "123456"
# A workflow_run run's own head_sha is the branch head when it was queued, not
# the commit it built (live: publish 41075630797 for CI 41074381903).
QUEUED_HEAD = "4" * 40


class PublishGitHub:
    def __init__(self, *, run=None, receipt=None, trigger=None, artifact="pocket-tts-publish-receipt-1",
                 filename="publish-receipt.json"):
        api = "https://api.github.com/repos/bawthub/bawthub"
        run = {"status": "completed", "conclusion": "success",
               "path": POCKET_DEPLOY["workflow_path"] + "@refs/heads/main", "head_branch": "main",
               "event": "workflow_run", "head_sha": QUEUED_HEAD, "run_attempt": 1,
               "repository": {"full_name": "bawthub/bawthub"}, "html_url": "https://run", **(run or {})}
        trigger = {"status": "completed", "conclusion": "success",
                   "path": POCKET_DEPLOY["trigger_workflow_path"] + "@refs/heads/main", "head_branch": "main",
                   "event": "push", "head_sha": SOURCE, "run_attempt": 1,
                   "repository": {"full_name": "bawthub/bawthub"},
                   "head_repository": {"full_name": "bawthub/bawthub"}, **(trigger or {})}
        receipt = {"schema": "bawthub.image-publish-receipt/v1", "repository": "bawthub/bawthub",
                   "workflow_run_id": RUN_ID, "workflow_run_attempt": "1", "source_sha": SOURCE,
                   "trigger_run_id": TRIGGER_ID,
                   "image_repository": POCKET_REPO, "digest": DIGEST, "status": "complete", **(receipt or {})}
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(filename, json.dumps(receipt))
        self.seen = []
        self.routes = {
            f"{api}/actions/runs/{RUN_ID}": _Resp(body=json.dumps(run).encode()),
            f"{api}/actions/runs/{TRIGGER_ID}": _Resp(body=json.dumps(trigger).encode()),
            f"{api}/actions/runs/{RUN_ID}/artifacts?name={artifact}": _Resp(body=json.dumps({"artifacts": [
                {"name": artifact, "expired": False, "archive_download_url": f"{api}/zip"}]}).encode()),
            f"{api}/zip": _Resp(302, location="https://blob.example/signed"),
            "https://blob.example/signed": _Resp(body=archive.getvalue()),
        }

    def open(self, request, timeout):
        self.seen.append((request.full_url, request.get_header("Authorization")))
        resp = self.routes.get(request.full_url)
        if resp is None and "/artifacts?name=" in request.full_url:
            return _Resp(body=b'{"total_count": 0, "artifacts": []}')
        if resp is None:
            raise AssertionError(f"unexpected GitHub request {request.full_url}")
        return resp


def _verify(**gh):
    return GitHubReleaseVerifier("tok", opener=PublishGitHub(**gh)).verify_publish(POCKET_DEPLOY, PUBLISH_ARGS)


def test_verify_publish_binds_run_sha_and_digest():
    release = _verify()
    assert release["binding"] == BINDING_CI_PUBLISH and release["image_ref"] == POCKET_REF
    assert release["source_sha"] == SOURCE and release["version"] == f"sha-{SOURCE}"
    assert release["workflow_run_id"] == RUN_ID and release["workflow_run_attempt"] == "1"
    assert release["trigger_run_id"] == TRIGGER_ID and release["trigger_sha"] == SOURCE
    assert "release_run_id" not in release


def test_verify_publish_reads_the_latest_attempt_artifact_only():
    release = _verify(run={"run_attempt": 2}, artifact="pocket-tts-publish-receipt-2",
                      receipt={"workflow_run_attempt": "2"})
    assert release["workflow_run_attempt"] == "2"
    with pytest.raises(ReleaseVerificationError, match="pocket-tts-publish-receipt-2"):
        _verify(run={"run_attempt": 2})  # attempt 1's artifact never satisfies attempt 2


@pytest.mark.parametrize("gh, message", [
    ({"run": {"event": "workflow_dispatch"}}, "event"),
    ({"run": {"event": "push"}}, "event"),
    ({"trigger": {"head_sha": "3" * 40}}, "trigger run .* head_sha"),
    ({"trigger": {"event": "pull_request"}}, "event"),
    ({"trigger": {"head_branch": "feature"}}, "branch"),
    ({"trigger": {"conclusion": "failure"}}, "conclusion"),
    ({"trigger": {"status": "in_progress"}}, "status"),
    ({"trigger": {"path": POCKET_DEPLOY["workflow_path"]}}, "workflow path"),
    ({"trigger": {"repository": {"full_name": "evil/bawthub"}}}, "repository"),
    ({"trigger": {"head_repository": {"full_name": "evil/bawthub"}}}, "head_repository"),
    ({"receipt": {"trigger_run_id": None}}, "trigger_run_id"),
    ({"receipt": {"trigger_run_id": 123456}}, "trigger_run_id"),
    ({"receipt": {"trigger_run_id": RUN_ID}}, "trigger_run_id"),
    ({"receipt": {"trigger_run_id": "../x"}}, "trigger_run_id"),
    ({"run": {"head_branch": "feature"}}, "branch"),
    ({"run": {"conclusion": "failure"}}, "conclusion"),
    ({"run": {"path": ".github/workflows/release-prod.yml"}}, "workflow path"),
    ({"run": {"repository": {"full_name": "evil/bawthub"}}}, "repository"),
    ({"run": {"run_attempt": None}}, "run attempt"),
    ({"artifact": "frontend-publish-receipt-1"}, "pocket-tts-publish-receipt-1"),
    ({"filename": "release.json"}, "publish-receipt.json"),
    ({"receipt": {"schema": "bawthub.release-receipt/v1"}}, "schema"),
    ({"receipt": {"digest": "sha256:" + "e" * 64}}, "digest"),
    ({"receipt": {"source_sha": "3" * 40}}, "source_sha"),
    ({"receipt": {"image_repository": "ghcr.io/bawthub/frontend"}}, "image_repository"),
    ({"receipt": {"workflow_run_id": "1"}}, "workflow_run_id"),
    ({"receipt": {"workflow_run_attempt": 1}}, "workflow_run_attempt"),
    ({"receipt": {"repository": "evil/bawthub"}}, "repository"),
    ({"receipt": {"status": "partial"}}, "status"),
])
def test_verify_publish_refuses_mismatches(gh, message):
    with pytest.raises(ReleaseVerificationError, match=message):
        _verify(**gh)


@pytest.mark.parametrize("args", [
    {"workflow_run_id": RUN_ID, "source_sha": SOURCE},
    {**PUBLISH_ARGS, "release_run_id": "e" * 32},
    {**PUBLISH_ARGS, "source_sha": "HEAD"},
    {**PUBLISH_ARGS, "digest": "latest"},
])
def test_verify_publish_refuses_malformed_args_before_any_request(args):
    gh = PublishGitHub()
    with pytest.raises(ReleaseVerificationError):
        GitHubReleaseVerifier("tok", opener=gh).verify_publish(POCKET_DEPLOY, args)
    assert gh.seen == []


def _snapshot(release=None, args=None, **extra):
    return {"spec": POCKET_DEPLOY, "resolved_args": dict(args or PUBLISH_ARGS),
            "release": _verify() if release is None else release, **extra}


def test_ci_publish_binding_matches_its_args():
    assert binding_matches_args(_snapshot())


@pytest.mark.parametrize("snapshot", [
    lambda: _snapshot(args={**PUBLISH_ARGS, "digest": "sha256:" + "e" * 64}),
    lambda: _snapshot(args={**PUBLISH_ARGS, "source_sha": "3" * 40}),
    lambda: _snapshot(args={**PUBLISH_ARGS, "workflow_run_id": "1"}),
    lambda: _snapshot(args={**PUBLISH_ARGS, "extra": "1"}),
    lambda: _snapshot(release={**_verify(), "binding": "release_run"}),
    lambda: _snapshot(release={**_verify(), "release_run_id": "e" * 32}),
    lambda: _snapshot(release={**_verify(), "version": "1.4.2"}),
    lambda: _snapshot(release={**_verify(), "image_ref": f"ghcr.io/bawthub/frontend@{DIGEST}"}),
    lambda: _snapshot(rollback={"deploy_job_id": JOB_ID}),
    lambda: _snapshot(release={}),
])
def test_ci_publish_binding_refuses_inconsistent_snapshots(snapshot):
    assert not binding_matches_args(snapshot())


# ── catalog and worker capability gates ────────────────────────────────────

def test_pocket_seeds_ship_disabled_and_validate():
    for slug, spec in POCKET.items():
        assert SEED[slug]["enabled"] is False
        assert validate_spec(SEED[slug]["command_script"])["container_name"] == "bawthub-pocket-tts-1"
        assert spec["compose_config"]["adopt_files"] == ["docker-compose.yml", "docker-compose.pocket.yml"]
    assert POCKET_DEPLOY["binding"] == BINDING_CI_PUBLISH
    assert POCKET_DEPLOY["trigger_workflow_path"] == ".github/workflows/ci.yml"
    assert "trigger_workflow_path" not in POCKET["bawthub.rollback-pocket-tts"]
    assert set(json.loads(SEED["bawthub.deploy-pocket-tts"]["args_schema_json"])["properties"]) == set(PUBLISH_ARGS)


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(binding="anything"),
    lambda s: s.update(health_probe={"type": "tcp"}),
    lambda s: s["compose_config"].update(profile="privileged"),
    lambda s: s["compose_config"].update(adopt_files=["/etc/compose.yml"]),
    lambda s: s.pop("trigger_workflow_path"),
    lambda s: s.update(trigger_workflow_path="../ci.yml"),
    lambda s: s.update(trigger_workflow_path=None),
    lambda s: s.pop("binding"),  # a release_run row never names a trigger workflow
])
def test_pocket_spec_drift_is_refused_by_validation(mutate):
    spec = copy.deepcopy(POCKET_DEPLOY)
    mutate(spec)
    with pytest.raises(ValueError):
        validate_spec(json.dumps(spec))


def test_capability_labels_follow_the_optional_fields():
    assert spec_capability_labels(POCKET_DEPLOY) == (HEALTH_PROBE_LABEL, HARDENED_PROFILE_LABEL)
    assert spec_capability_labels(SPEC) == (HEALTH_PROBE_LABEL,)
    bare = copy.deepcopy(SPEC)
    bare.pop("health_probe")
    assert spec_capability_labels(bare) == ()


def test_base_executor_refuses_specs_needing_worker_capabilities():
    fake = SimpleNamespace(kind=lambda: "fake")
    with pytest.raises(ExecutorError, match=HEALTH_PROBE_LABEL):
        Executor.check_spec_capabilities(fake, POCKET_DEPLOY)


@pytest.mark.parametrize("labels, missing", [
    ({HEALTH_PROBE_LABEL: "1"}, HARDENED_PROFILE_LABEL),
    ({HARDENED_PROFILE_LABEL: "1"}, HEALTH_PROBE_LABEL),
    ({HEALTH_PROBE_LABEL: "1", HARDENED_PROFILE_LABEL: "0"}, HARDENED_PROFILE_LABEL),
    ({}, HEALTH_PROBE_LABEL),
])
def test_docker_executor_refuses_a_worker_without_the_labels(labels, missing):
    executor, _pulls = _executor(FakeDocker(), worker_labels=labels)
    with pytest.raises(ExecutorError, match=missing):
        executor.check_spec_capabilities(POCKET_DEPLOY)


def test_docker_executor_accepts_a_worker_with_both_labels():
    executor, _pulls = _executor(FakeDocker(), worker_labels={HEALTH_PROBE_LABEL: "1", HARDENED_PROFILE_LABEL: "1"})
    executor.check_spec_capabilities(POCKET_DEPLOY)
    assert executor.execution_settings(POCKET_DEPLOY)["worker_image"] == WORKER


def test_worker_dockerfile_ships_the_probe_module_and_both_labels():
    text = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.ops-worker").read_text()
    assert "src/llm_bawt/ops/health_probe.py" in text
    assert f'LABEL {HEALTH_PROBE_LABEL}="1"' in text and f'LABEL {HARDENED_PROFILE_LABEL}="1"' in text

