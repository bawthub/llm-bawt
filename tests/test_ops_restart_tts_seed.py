"""TASK-1171: the catalogued TTS restart and the backend's lost Docker access."""
from __future__ import annotations

import json

from llm_bawt.ops.executor import validate_spec
from llm_bawt.ops.health_collector import catalog_targets
from llm_bawt.ops.seeds import SEEDS

BY_SLUG = {seed["slug"]: seed for seed in SEEDS}


def test_restart_tts_is_a_typed_disabled_no_args_restart():
    seed = BY_SLUG["bawthub.restart-tts"]
    assert seed["enabled"] is False
    assert seed["executor_kind"] == "docker" and seed["target_host"] == ""
    assert validate_spec(seed["command_script"]) == {
        "action": "restart", "compose_project": "bawthub", "compose_service": "tts"}
    assert json.loads(seed["args_schema_json"])["properties"] == {}
    assert seed["risk_level"] == "high" and seed["timeout_seconds"] == 120
    assert seed["start_delay_seconds"] == 0 and seed.get("max_concurrent", 1) == 1


def test_enabled_restart_tts_is_a_restart_capability_of_bawthub_tts():
    seed = BY_SLUG["bawthub.restart-tts"]
    targets = {t.service_id: t for t in catalog_targets(
        [(seed["slug"], seed["command_script"], seed["args_schema_json"])])}
    assert [c.to_api() for c in targets["bawthub.tts"].capabilities] == [
        {"operation": "bawthub.restart-tts", "action": "restart", "args": {}}]


def test_backend_reconcile_row_no_longer_carries_docker_access():
    spec = json.loads(BY_SLUG["bawthub.reconcile-backend-config"]["command_script"])
    assert not any("docker.sock" in mount for mount in spec["data_mounts"])
    assert not {"BAWTHUB_TTS_DOCKER_CONTAINER", "BAWTHUB_COMPOSE_PROJECT"} & set(spec["env_names"])
