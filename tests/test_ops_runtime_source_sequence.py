"""TASK-1800 sequence integration: explicit null exemptions and child break-glass disclosure (Codex controls)."""
# ruff: noqa: F811 - the imported repo fixture is a parameter by design
import json

import pytest

from llm_bawt.ops import OpsDispatchError
from llm_bawt.ops.execution_mode import with_card_lines
from llm_bawt.ops.runtime_source import bind_runtime_source, valid_runtime_binding
from llm_bawt.ops.sequence_approval import approval_text
from llm_bawt.ops import worker
from tests.test_ops_runtime_source import (
    container_for, extended_schema, extended_spec, repo,  # noqa: F401 - pytest fixture
)
from tests.test_ops_source_state import GitDocker, git
from tests.test_ops_sequence_admission import HEALTH, child, prepare, rig, step
from tests.test_ops_sequence_worker import Docker, invoke, receipt, request_file


def sequence_rig(repo, tmp_path, *, mixed, slug="test.first"):
    spec = {**extended_spec(repo, mixed=mixed), **HEALTH}
    schema = extended_schema(mixed=mixed)
    r = rig(
        [child(slug, spec=spec, args_schema_json=json.dumps(schema),
               args_defaults_json='{"allow_dirty":false}')],
        release_verifier=type("Published", (), {"branch_contains": lambda *_: True})(),
    )
    r.executor.add("app")
    r.executor.add("exempt")
    container = container_for(repo)
    container["Id"] = r.executor.containers["app"].id
    scratch = tmp_path / "git-scratch"
    scratch.mkdir()
    api = GitDocker(scratch)
    r.executor.runtime_source_approval = lambda spec, args: bind_runtime_source(
        api, spec, args, container, tag="review")
    return r


def test_restart_app_child_uses_its_own_schema_and_guarded_sha(repo, tmp_path):
    revision = git(repo, "rev-parse", "HEAD").strip()
    r = sequence_rig(repo, tmp_path, mixed=False, slug="llm-bawt.restart-app")
    args = {"expected_runtime_sha": revision}
    snapshot = prepare(r, [step("llm-bawt.restart-app", args)])
    nested = snapshot["sequence"]["steps"][0]["snapshot"]
    assert nested["input_args"] == args
    assert nested["resolved_args"] == {**args, "allow_dirty": False}
    assert nested["runtime_source"]["expected_runtime_sha"] == revision
    assert nested["runtime_source"]["source_commit"]["head"] == revision
    assert nested["runtime_source"]["publication"] == "published"
    assert valid_runtime_binding(nested)
    assert nested == r.service.prepare_invocation("llm-bawt.restart-app", args)
    assert r.service._verify_snapshot(snapshot, r.parent.slug, snapshot["input_args"]) == snapshot
    assert r.executor.dispatch_calls == [] and r.store.list_jobs() == []


@pytest.mark.parametrize("args,match", [
    ({}, "required property 'expected_runtime_sha'"),
    ({"expected_runtime_sha": "not-a-sha"}, "does not match pattern"),
    ({"expected_runtime_sha": "a" * 40, "unknown": True}, "unknown property 'unknown'"),
    ({"expected_runtime_sha": "a" * 40, "allow_dirty": "false"}, "expected boolean"),
])
def test_restart_app_child_refuses_invalid_args_without_dispatch(repo, tmp_path, args, match):
    r = sequence_rig(repo, tmp_path, mixed=False, slug="llm-bawt.restart-app")
    with pytest.raises(OpsDispatchError, match=match) as caught:
        prepare(r, [step("llm-bawt.restart-app", args)])
    assert caught.value.code == "args_invalid"
    assert r.executor.checked == [] and r.executor.preflighted == []
    assert r.executor.dispatch_calls == [] and r.store.list_jobs() == []


def test_explicit_null_target_runs_in_sequence(repo, tmp_path, monkeypatch):
    r = sequence_rig(repo, tmp_path, mixed=True)
    snapshot = prepare(r, [step(args={"target": "exempt"})])
    nested = snapshot["sequence"]["steps"][0]["snapshot"]
    assert nested["runtime_source"] is None
    assert valid_runtime_binding(nested)
    assert r.service._verify_snapshot(snapshot, r.parent.slug, snapshot["input_args"]) == snapshot
    path, digest = request_file(tmp_path, snapshot)
    api = Docker(snapshot, path)
    monkeypatch.setattr(worker, "wait_healthy", lambda *a, **k: "healthy")
    original = worker.verify_runtime_source
    errors = []

    def verify(api, snapshot, container, **kwargs):
        try:
            return original(api, snapshot, container, **kwargs)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            raise

    monkeypatch.setattr(worker, "verify_runtime_source", verify)
    result = invoke(path, digest, api)
    final = receipt(path)
    assert not errors, errors
    assert result == 0, final
    assert final["state"] == "succeeded"
    assert api.events == [("submit", "exempt")]


def test_sequence_card_discloses_dirty_source(repo, tmp_path):
    revision = git(repo, "rev-parse", "HEAD").strip()
    (repo / "src/pkg/module.py").write_text("UNREVIEWED = True\n")
    r = sequence_rig(repo, tmp_path, mixed=False)
    snapshot = prepare(r, [step(args={"expected_runtime_sha": revision, "allow_dirty": True})])
    nested = snapshot["sequence"]["steps"][0]["snapshot"]
    assert nested["runtime_source"]["dirty_paths"] == ["src/pkg/module.py"]
    assert valid_runtime_binding(nested)
    assert r.service._verify_snapshot(snapshot, r.parent.slug, snapshot["input_args"]) == snapshot
    text = with_card_lines(approval_text(snapshot, None), snapshot)
    assert "BREAK-GLASS / UNTESTED" in text, text
    assert "src/pkg/module.py" in text, text
