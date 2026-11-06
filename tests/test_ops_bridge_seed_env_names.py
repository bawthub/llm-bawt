"""TASK-2088: every environment name a bridge's Compose service declares is
classified by its reconcile seed, or the binder refuses the live bridge with
``source_env_unclassified`` (as BAWT_WT_DEV_ROOT did after TASK-2072)."""
import json
from pathlib import Path

import pytest
import yaml

from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.source_state import PRODUCT_ENV

ROOT = Path(__file__).resolve().parents[1]


def _compose_names(service: str) -> set[str]:
    env = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"][service].get("environment") or []
    if isinstance(env, dict):
        return set(env)
    return {item.split("=", 1)[0] for item in env}


@pytest.mark.parametrize("bridge", ["codex-bridge", "claude-code-bridge"])
def test_bridge_seed_classifies_every_compose_env_name(bridge):
    seed = next(s for s in SEEDS if s["slug"] == f"llm-bawt.reconcile-{bridge}-config")
    spec = seed["command_script"]
    spec = json.loads(spec) if isinstance(spec, str) else spec
    unclassified = _compose_names(bridge) - set(PRODUCT_ENV) - set(spec["env_names"])
    assert not unclassified, f"add to the {bridge} seed env_names: {sorted(unclassified)}"
