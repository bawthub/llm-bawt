"""Offline deployment rendering: never contact Docker, Redis, bots or auth."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from agent_bridge.house_pool import render_compose, render_routes, validate_pool

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return {
        "image": "example-house:test",
        "base_image": "example-bridge@sha256:" + "a" * 64,
        "memory": "4g",
        "preview_network": "example-frontend",
        "service_token_volume": "example-service-token",
        "frontend_env_file": "/private/house-frontend.env",
        "houses": [
            {"id": "h1", "harness": "claude-code", "origin": "https://dev1.example.test"},
            {"id": "h2", "harness": "codex", "origin": "https://dev2.example.test"},
        ],
    }


@pytest.fixture
def template():
    return {"http": {"routers": {
        "frontend": {
            "rule": "Host(`dev.example.test`)", "priority": 10,
            "entryPoints": ["web"], "service": "frontend",
        },
        "frontend-api": {
            "rule": "(Host(`dev.example.test`)) && (PathPrefix(`/api/auth`) || PathPrefix(`/api/chat`))",
            "priority": 200, "entryPoints": ["web"], "service": "frontend",
        },
        "backend": {"rule": "PathPrefix(`/api`)", "priority": 100, "service": "backend"},
    }}}


def test_zero_pool_is_opt_in_and_resource_free():
    assert render_compose({}, ROOT / "docker-compose.yml") == {"services": {}}
    assert render_routes({}, {}) == {"http": {"routers": {}, "services": {}}}


def test_one_definition_supports_either_command(config):
    before = deepcopy(config)
    rendered = render_compose(config, ROOT / "docker-compose.yml")
    claude, codex = rendered["services"].values()
    assert claude["command"] == ["python", "-m", "claude_code_bridge"]
    assert codex["command"] == ["python", "-m", "codex_bridge"]
    for name, service in rendered["services"].items():
        assert service["extends"] == {
            "file": str(ROOT / "docker-compose.yml"), "service": "claude-code-bridge",
        }
        assert service["environment"]["BAWT_HOUSE"] == name.removeprefix("house-")
        assert service["environment"]["CODEX_MODEL"] == ""
        assert service["environment"]["CODEX_BIN"].endswith("/codex}")
        assert service["environment"]["CODEX_BRIDGE_HEALTH_PORT"] == "8681"
        assert service["image"] == "example-house:test"
        assert service["build"]["args"]["AGENT_BRIDGE_IMAGE"] == config["base_image"]
        assert service["init"] is True
        assert service["mem_limit"] == "4g"
        assert "cgroup_parent" not in service
        assert "ports" not in service
        assert service["networks"] == ["default", "house-previews"]
        assert "${HOME}/.codex/sessions:/home/bridge/.codex/sessions" in service["volumes"]
        assert f"{ROOT}/src:/app/src" in service["volumes"]
        assert service["env_file"] == ["/private/house-frontend.env"]
    assert rendered["networks"]["house-previews"]["external"] is True
    assert rendered["volumes"]["house-service-token"]["external"] is True
    assert config == before


def test_mansion_parent_supplies_identical_session_paths():
    base = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    parent = base["services"]["claude-code-bridge"]
    assert "${HOME}/dev:/home/bridge/dev" in parent["volumes"]
    assert "${HOME}/dev:${HOME}/dev" in parent["volumes"]
    assert "${HOME}/.config/claude-code-bridge:/home/bridge/.claude" in parent["volumes"]
    assert "${HOME}/.config/claude-code-bridge/.claude.json:/home/bridge/.claude.json" in parent["volumes"]


def test_effective_dev_account_mount_preserved(config):
    dev = yaml.safe_load((ROOT / "docker-compose.dev.yml").read_text())
    account = "${BRIDGE_GIT_CONFIG_DIR:-${HOME}/.config/bawthub-git/bridge}:/home/bridge/.config/bawthub-git:ro"
    assert account in dev["services"]["claude-code-bridge"]["volumes"]
    assert account in dev["services"]["codex-bridge"]["volumes"]
    for service in render_compose(config, ROOT / "docker-compose.yml")["services"].values():
        assert account in service["volumes"]


def test_parent_limits_and_per_house_override(config):
    config["cgroup_parent"] = "tenant-agents.slice"
    config["houses"][1]["memory"] = "6144m"
    services = render_compose(config, ROOT / "docker-compose.yml")["services"]
    assert services["house-h1"]["cgroup_parent"] == "tenant-agents.slice"
    assert services["house-h2"]["mem_limit"] == "6144m"


def test_preview_auth_is_per_house_not_shared_origin(config):
    services = render_compose(config, ROOT / "docker-compose.yml")["services"]
    for index, service in enumerate(services.values(), start=1):
        env = service["environment"]
        assert env["BAWTHUB_AUTH_URL"] == f"https://dev{index}.example.test"
        assert env["BAWTHUB_AUTH_SESSIONS"] == "on"
        assert env["BAWTHUB_LLM_URL"] == "http://app:8642"
        assert env["BAWTHUB_SERVICE_TOKEN_FILE"] == "/run/bawthub-service-token/token"
        assert "DATABASE_URL" not in env  # only a deployment-local env_file reference


def test_routes_clone_owning_api_prefixes_and_leave_voice_shared(config, template):
    before = deepcopy(template)
    routes = render_routes(config, template)["http"]
    assert len(routes["routers"]) == 4
    assert len(routes["services"]) == 2
    for index in (1, 2):
        prefix = f"house-h{index}"
        page = routes["routers"][prefix + "-frontend"]
        api = routes["routers"][prefix + "-frontend-api"]
        assert page["rule"] == f"Host(`dev{index}.example.test`)"
        assert api["priority"] == 200
        assert api["service"] == prefix
        assert "PathPrefix(`/api/auth`)" in api["rule"]
        assert "PathPrefix(`/api/chat`)" in api["rule"]
        assert "PathPrefix(`/api`)" not in api["rule"]
        assert routes["services"][prefix]["loadBalancer"]["servers"] == [
            {"url": f"http://{prefix}:5173"},
        ]
    assert template == before


def test_routes_require_unambiguous_host_rule(config, template):
    template["http"]["routers"]["frontend-api"]["rule"] = "PathPrefix(`/api/auth`)"
    with pytest.raises(ValueError, match="exactly one Host"):
        render_routes(config, template)


@pytest.mark.parametrize("house_id", ["", "H1", "h_1", "-h", "a" * 32, "h\n", None])
def test_invalid_house_ids(config, house_id):
    config["houses"][0]["id"] = house_id
    with pytest.raises(ValueError, match="house id"):
        validate_pool(config)


@pytest.mark.parametrize("origin", [
    "http://dev1.example.test", "https://dev1.example.test/path",
    "https://dev1.example.test:443", "https://user@dev1.example.test",
    "https://dev1.example.test?x=1", "https://dev1.example.test/#x",
    "https://dev1..example.test", "https://-dev.example.test", None,
])
def test_invalid_preview_origin(config, origin):
    config["houses"][0]["origin"] = origin
    with pytest.raises(ValueError):
        validate_pool(config)


@pytest.mark.parametrize("memory", ["0", 0, -1, "", None, "unlimited", "0g", "4g\n"])
def test_uncapped_or_invalid_house_rejected(config, memory):
    config["memory"] = memory
    with pytest.raises(ValueError, match="memory"):
        validate_pool(config)


@pytest.mark.parametrize("field", ["id", "origin"])
def test_duplicate_slot_or_hostname_rejected(config, field):
    config["houses"][1][field] = config["houses"][0][field]
    with pytest.raises(ValueError, match="unique"):
        validate_pool(config)


def test_bad_harness_and_unknown_config_fail_closed(config):
    config["houses"][0]["harness"] = "python -m arbitrary"
    with pytest.raises(ValueError, match="harness"):
        validate_pool(config)
    config["houses"][0]["harness"] = "codex"
    config["house_count_typo"] = 3
    with pytest.raises(ValueError, match="documented keys"):
        validate_pool(config)


def test_launcher_has_no_schema_or_backend_startup():
    script = (ROOT / "docker" / "house-preview").read_text()
    assert "--strictPort" in script
    assert "vite --host 0.0.0.0" in script
    assert "tsx watch server/index.ts" in script
    assert "BAWTHUB_AUTH_SESSIONS" in script
    assert "exec pnpm exec concurrently" in script
    assert "db:push" not in script
    assert "db:migrate" not in script
    assert "uvicorn" not in script


def test_image_keeps_tests_separate_from_harness_environment():
    dockerfile = (ROOT / "Dockerfile.house").read_text()
    assert "FROM ${AGENT_BRIDGE_IMAGE}" in dockerfile
    assert "uv venv /opt/house-tests" in dockerfile
    assert '--python /opt/house-tests/bin/python -e "/app[dev,service,mcp,search]"' in dockerfile
    assert "gpt-" not in dockerfile
    assert "USER bridge" in dockerfile
