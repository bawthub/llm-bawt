"""TASK-1802: the compose `preview` profile stays isolated from production.

Static guard over docker-compose.yml. Every preview-profile service:

* joins only its declared internal networks (``preview`` by default; the
  credential gateway only ``preview-broker-net`` + ``preview-credential-net``),
  and no prod service joins a preview network except ``app`` on the broker net;
* publishes no port and runs no privileged/host-namespace mode;
* mounts no host path other than the read-only seed artifact (no checkout,
  no credentials, no sockets);
* uses a digest-pinned image or the locally built preview image;
* receives only preview connection names, never a production credential.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llm_bawt.preview import topology

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
SERVICES = COMPOSE["services"]
PREVIEW = {name: svc for name, svc in SERVICES.items() if "preview" in svc.get("profiles", [])}
ALLOWED_BINDS = {"./.preview/seed:/seed:ro"}
ALLOWED_ENV_PREFIXES = ("LLM_BAWT_POSTGRES_", "POSTGRES_", "LLM_BAWT_REDIS_URL", "LLM_BAWT_PREVIEW_", "LLM_BAWT_SERVICE_")
BUILT_IMAGES = {"llm-bawt-preview"}


def test_preview_services_declared():
    assert {"preview-postgres", "preview-redis", "preview-seed"} <= set(PREVIEW)


def _networks(svc) -> set[str]:
    networks = svc.get("networks", [])
    return set(networks if isinstance(networks, list) else networks.keys())


PREVIEW_NETWORKS = set(topology.NETWORK_NAMES)


def test_preview_networks_are_internal_with_fixed_names():
    declared = COMPOSE.get("networks", {})
    assert "preview" in declared
    for key in PREVIEW_NETWORKS & set(declared):
        assert declared[key]["internal"] is True, key
        # The live probe matches containers by these names.
        assert declared[key]["name"] == topology.NETWORK_NAMES[key], key


def test_preview_profile_and_service_prefix_coincide():
    # The probe finds preview containers by the prefix; the profile and the
    # prefix must select the same services.
    assert set(PREVIEW) == {n for n in SERVICES if topology.is_preview_service(n)}


def test_preview_services_join_only_their_networks():
    for name, svc in PREVIEW.items():
        assert _networks(svc) == topology.networks_for(name), name


def test_prod_services_stay_off_preview_networks():
    for name, svc in SERVICES.items():
        if name in PREVIEW:
            continue
        joined = _networks(svc) & PREVIEW_NETWORKS
        assert joined <= topology.PROD_JOINERS.get(name, set()), f"{name} joins {joined}"


def test_no_ports_privilege_or_host_namespaces():
    for name, svc in PREVIEW.items():
        for key in ("ports", "privileged", "network_mode", "pid", "ipc", "cap_add", "devices", "extra_hosts", "env_file"):
            assert key not in svc, f"{name}.{key}"


def test_only_seed_artifact_is_bind_mounted():
    named = set(COMPOSE.get("volumes", {}))
    for name, svc in PREVIEW.items():
        for volume in svc.get("volumes", []):
            source = volume.split(":", 1)[0]
            if source in named:
                assert source.startswith("preview-"), f"{name} mounts shared volume {source}"
            else:
                assert volume in ALLOWED_BINDS, f"{name} binds {volume}"


def test_images_are_pinned():
    for name, svc in PREVIEW.items():
        image = svc["image"]
        assert "@sha256:" in image or (image in BUILT_IMAGES and "build" in svc), f"{name}: {image}"


def test_environment_is_preview_only():
    for name, svc in PREVIEW.items():
        for entry in svc.get("environment", []):
            key, _, value = entry.partition("=")
            assert key.startswith(ALLOWED_ENV_PREFIXES), f"{name}: {key}"
            assert "PASSWORD" not in key or key.endswith("_FILE"), f"{name}: inline {key}"
            assert "${" not in value and "$" not in value, f"{name}: {key} interpolates host env"
        for secret in svc.get("secrets", []):
            assert secret.startswith("preview_"), f"{name}: {secret}"
    for secret, spec in COMPOSE.get("secrets", {}).items():
        if secret.startswith("preview_"):
            assert spec["file"].startswith("./.preview/"), secret
