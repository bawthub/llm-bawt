"""TASK-1476 slice 1: the swap seams a multi-target plan composes.

``ImageDeployer._replace`` is now ``_commit_swap(_swap(...))``; a plan swaps
several targets, keeps each previous container until every target is verified,
and reverts completed swaps in reverse order on a later failure. These tests pin
the seams themselves against the same fake Docker the deploy tests use, plus the
shared Compose identity check the reconcilers now route through.
"""
from __future__ import annotations

import fcntl
import json

import pytest

from llm_bawt.ops.config_reconcile import check_compose_identity
from llm_bawt.ops.image_deploy import DeployFailed, StepError, Swap
from llm_bawt.ops.routing_reconcile import routing_binding
from llm_bawt.ops.seeds import SEEDS
from tests.test_ops_image_deploy import JOB_ID, NAME, NEW_ID, OLD_ID, SPEC, FakeDocker, _deployer

PREV = f"{NAME}-prev-{JOB_ID[:8]}"


def _swapped(api, tmp_path):
    """Run a full deploy but stop before the commit; return the deployer and swap."""
    deployer = _deployer(api, tmp_path)
    held: list[Swap] = []
    deployer._commit_swap = held.append
    deployer.run()
    del deployer._commit_swap  # later calls reach the real method
    assert len(held) == 1
    return deployer, held[0]


def test_swap_leaves_the_previous_container_stopped_until_commit(tmp_path):
    api = FakeDocker()
    deployer, swap = _swapped(api, tmp_path)
    assert swap == Swap(api.old_id, swap.new_id, PREV)
    live, prev = api.by_name(NAME), api.by_name(PREV)
    assert live["Id"] == swap.new_id and live["Image"] == NEW_ID and live["State"]["Running"]
    assert prev["Id"] == api.old_id and not prev["State"]["Running"]
    assert not any(m == "DELETE" and api.old_id in p for m, p in api.calls)

    deployer._commit_swap(swap)
    assert api.old_id not in api.containers and api.by_name(NAME)["Id"] == swap.new_id
    assert ("DELETE", f"/containers/{api.old_id}?v=0") in api.calls


def test_commit_swap_failure_is_only_a_warning_naming_the_previous_container(tmp_path):
    api = FakeDocker()
    deployer, swap = _swapped(api, tmp_path)
    deployer._commit_swap(Swap("0" * 64, swap.new_id, swap.prev))  # no such container
    [warning] = deployer.detail["warnings"]
    assert warning.startswith(f"previous container {PREV} not removed: ")
    assert api.by_name(NAME)["Id"] == swap.new_id  # the deploy itself stands


def test_revert_puts_the_previous_container_back_healthy_and_removes_the_new_one(tmp_path):
    api = FakeDocker()
    deployer, swap = _swapped(api, tmp_path)
    deployer._revert(swap)
    live = api.by_name(NAME)
    assert live["Id"] == api.old_id and live["Image"] == OLD_ID
    assert live["State"]["Running"] and live["State"]["Health"]["Status"] == "healthy"
    assert swap.new_id not in api.containers
    assert ("POST", f"/containers/{swap.new_id}/stop?t=5") in api.calls
    assert ("DELETE", f"/containers/{swap.new_id}?force=1&v=0") in api.calls


def test_revert_raises_when_the_previous_container_is_not_healthy(tmp_path):
    api = FakeDocker()
    deployer, swap = _swapped(api, tmp_path)
    api.images[OLD_ID]["_health"] = "unhealthy"
    with pytest.raises(StepError, match="unhealthy"):
        deployer._revert(swap)
    assert swap.new_id in api.containers  # never deleted unless the old one is back


def test_target_lock_is_held_for_the_whole_block_and_released_after(tmp_path):
    api = FakeDocker()
    first, second = _deployer(api, tmp_path), _deployer(api, tmp_path)
    with first._target_lock():
        with pytest.raises(DeployFailed, match="holds this target") as caught:
            with second._target_lock():
                pytest.fail("a second holder entered a held target lock")
        assert caught.value.outcome_known
    with second._target_lock():
        pass
    assert not api.calls


def test_target_lock_busy_from_another_process_fails_known_with_no_calls(tmp_path):
    api = FakeDocker()
    (tmp_path / "locks").mkdir()
    with (tmp_path / "locks" / f"{NAME}.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(DeployFailed, match="holds this target") as caught:
            with _deployer(api, tmp_path)._target_lock():
                pytest.fail("entered a busy target lock")
    assert caught.value.outcome_known and not api.calls


# ── one Compose identity check for every consumer ─────────────────────────

CONFIG_SPEC = json.loads(next(s for s in SEEDS if s["slug"] == "bawthub.reconcile-prod-config")["command_script"])
ROUTING_SPEC = json.loads(next(s for s in SEEDS if s["slug"] == "bawthub.reconcile-prod-routing")["command_script"])


@pytest.mark.parametrize("label, value", [
    ("com.docker.compose.project", "other"),
    ("com.docker.compose.service", "frontend"),
    ("com.docker.compose.service", None),
])
def test_every_consumer_refuses_a_target_with_the_wrong_compose_identity(tmp_path, label, value):
    api = FakeDocker()
    labels = api.by_name(NAME)["Config"]["Labels"]
    if value is None:
        del labels[label]
    else:
        labels[label] = value
    container = api.by_name(NAME)

    with pytest.raises(DeployFailed, match="expected Compose identity") as caught:
        _deployer(api, tmp_path).run()
    assert caught.value.outcome_known and not api.side_effect_started
    with pytest.raises(ValueError, match="config target Compose identity mismatch"):
        check_compose_identity(CONFIG_SPEC, container)
    with pytest.raises(ValueError, match="routing target Compose identity mismatch"):
        routing_binding(ROUTING_SPEC, container, api.images[OLD_ID])


def test_every_consumer_accepts_the_matching_compose_identity():
    api = FakeDocker()
    container = api.by_name(NAME)
    assert SPEC["compose_project"] == CONFIG_SPEC["compose_project"] == ROUTING_SPEC["compose_project"]
    check_compose_identity(CONFIG_SPEC, container)  # no raise
