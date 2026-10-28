"""TASK-1169 S1b: one Compose service identity shared by every ops consumer."""

from pathlib import Path

import pytest

from llm_bawt.ops import executor
from llm_bawt.ops.service_identity import (PROJECT_PATTERN, SERVICE_PATTERN, SPEC_PATTERNS,
                                           ComposeServiceRef)
from llm_bawt.ops.worker import target_selector


def test_ref_accepts_compose_names_and_round_trips_its_id():
    ref = ComposeServiceRef("llm-bawt", "claude-code.bridge_1")
    assert ref.service_id == "llm-bawt.claude-code.bridge_1"
    assert ComposeServiceRef.from_id(ref.service_id) == ref
    assert ref.labels() == {"com.docker.compose.project": "llm-bawt",
                            "com.docker.compose.service": "claude-code.bridge_1"}


@pytest.mark.parametrize("project,service", [
    ("", "app"), ("App", "app"), ("-proj", "app"), ("p" * 64, "app"), ("proj.x", "app"),
    ("proj", ""), ("proj", ".app"), ("proj", "a" * 64), ("proj", "app;rm"), ("proj", "a/b"),
    (None, "app"), ("proj", 7),
])
def test_ref_refuses_values_outside_the_grammar(project, service):
    with pytest.raises(ValueError):
        ComposeServiceRef(project, service)


@pytest.mark.parametrize("service_id", ["noservice", ".app", "proj.", "Proj.app"])
def test_from_id_refuses_malformed_ids(service_id):
    with pytest.raises(ValueError):
        ComposeServiceRef.from_id(service_id)


def test_from_spec_reads_fixed_and_argument_selectors():
    assert ComposeServiceRef.from_spec({"compose_project": "p", "compose_service": "app"}) \
        == ComposeServiceRef("p", "app")
    spec = {"compose_project": "p", "compose_service_from_arg": "target"}
    assert ComposeServiceRef.from_spec(spec, {"target": "redis"}) == ComposeServiceRef("p", "redis")
    with pytest.raises(ValueError):
        ComposeServiceRef.from_spec(spec, {})
    with pytest.raises(ValueError):
        ComposeServiceRef.from_spec({"container_name": "x"})


@pytest.mark.parametrize("spec,args", [
    ({"action": "restart", "compose_project": "p", "compose_service": "app"}, {}),
    ({"action": "restart", "compose_project": "p", "compose_service_from_arg": "s"}, {"s": "redis"}),
])
def test_label_filters_select_exactly_what_the_worker_selects(spec, args):
    # The worker's selector runs in the immutable worker image; until it moves
    # onto the helper, both must build the same Docker filter.
    assert target_selector(spec, args) == (None, ComposeServiceRef.from_spec(spec, args).label_filters())


def test_matches_requires_both_labels():
    ref = ComposeServiceRef("p", "app")
    assert ref.matches({"com.docker.compose.project": "p", "com.docker.compose.service": "app"})
    assert not ref.matches({"com.docker.compose.project": "p", "com.docker.compose.service": "web"})
    assert not ref.matches({"com.docker.compose.service": "app"})
    assert not ref.matches(None)


def test_image_spec_validation_uses_the_shared_grammar():
    assert executor._IMAGE_SPEC_PATTERNS["compose_project"] is PROJECT_PATTERN
    assert executor._IMAGE_SPEC_PATTERNS["compose_service"] is SERVICE_PATTERN
    assert set(SPEC_PATTERNS) == {"compose_project", "compose_service"}


def test_worker_image_carries_the_helper():
    dockerfile = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.ops-worker").read_text()
    assert "src/llm_bawt/ops/service_identity.py" in dockerfile
