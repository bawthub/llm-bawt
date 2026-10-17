"""TASK-915: GPU access must survive systemd reapplying device policy.

Legacy NVIDIA prestart-hook reservations change cgroups behind runc's back.
Native CDI records the devices in the OCI configuration instead. These are
source guards; live DeviceAllow and GPU probes remain an activation gate.
"""

from pathlib import Path

import yaml


COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def test_local_model_gpu_uses_native_cdi_without_legacy_hook() -> None:
    service = yaml.safe_load(COMPOSE.read_text())["services"]["local-model-bridge"]

    assert service["devices"] == ["nvidia.com/gpu=all"]
    assert service["runtime"] == "runc"
    assert not service.get("gpus")
    reservations = service.get("deploy", {}).get("resources", {}).get("reservations", {})
    assert not reservations.get("devices")
    assert not service.get("privileged", False)


def test_cdi_does_not_grant_gpu_to_app_or_agent_bridges() -> None:
    services = yaml.safe_load(COMPOSE.read_text())["services"]

    for name in ("app", "claude-code-bridge", "openclaw-bridge", "codex-bridge"):
        assert not services[name].get("devices"), name
        assert not services[name].get("gpus"), name
