"""TASK-2590: the opt-in tools-off reader house (renderer and consumer fence).

Offline only: no Docker, Redis, bots, profiles or auth are contacted.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_bridge import houses
from agent_bridge.house_pool import (
    render_compose, render_routes, validate_houses, validate_pool,
)
from agent_bridge.houses import EPOCH_FIELD, HouseSlot, gate_send, tools_off_from_env
from agent_bridge.run_contract import REFUSED

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docker-compose.yml"


def reader_row(**extra):
    row = {
        "id": "reader-a", "harness": "claude-code", "profile": "tools-off",
        "home": "/srv/example/reader-a/home",
        "memory": "2g",
    }
    row.update(extra)
    return row


def pool(*rows, **extra):
    config = {
        "image": "example-house:test",
        "base_image": "example-bridge@sha256:" + "a" * 64,
        "memory": "4g",
        "houses": list(rows),
    }
    config.update(extra)
    return config


# --- renderer ----------------------------------------------------------------

def _service():
    return render_compose(pool(reader_row()), BASE)["services"]["house-reader-a"]


def test_tools_off_house_is_standalone_and_private():
    service = _service()
    assert "extends" not in service
    assert service["user"] == "1001:1001"
    env = service["environment"]
    assert env["BAWT_HOUSE"] == "reader-a"
    assert env["BAWT_HOUSE_TOOLS_OFF"] == "on"
    assert env["HOME"] == "/var/lib/bawt-reader/home"
    assert env["CLAUDE_CODE_CWD"] == "/var/lib/bawt-reader/home/work"
    assert "BAWT_HOUSE_PREVIEW" not in env
    for absent in ("CLAUDE_CODE_SKILLS_PATH", "SKILL_SOURCES_CURRENT", "SKILL_SOURCES_ROOT",
                   "BRIDGE_GIT_CREDENTIAL_TOKEN_FILE", "BRIDGE_VAULT_TOKEN_FILE",
                   "CRAWL4AI_API_TOKEN_FILE", "ANTHROPIC_API_KEY", "BAWT_WT_DEV_ROOT",
                   "CLAUDE_CODE_BAWTHUB_MCP_URL"):
        assert absent not in env
    assert service["read_only"] is True
    assert service["networks"] == ["default"]
    assert "env_file" not in service


def test_tools_off_house_mounts_only_its_role_inputs():
    volumes = _service()["volumes"]
    assert volumes == [
        f"{ROOT}/src:/app/src:ro",
        "/srv/example/reader-a/home:/var/lib/bawt-reader/home",
        "reader-broker-token:/run/llm-bawt-reader-broker-token:ro",
        "api-key-bridge-claude-code:/run/llm-bawt-api-key:ro",
    ]
    joined = "\n".join(volumes)
    for forbidden in ("${HOME}", "/home/bridge/.claude", ".claude.json", "/repos", ".ssh",
                      "agent-skills", "skill-sources", "service-token"):
        assert forbidden not in joined
    # The full shared token volume is never mounted, only the broker-only one.
    assert not any(v.startswith("bridge-tokens:") for v in volumes)


def test_tools_off_house_has_no_route_or_preview_resources():
    rendered = render_compose(pool(reader_row()), BASE)
    assert "networks" not in rendered
    assert rendered["volumes"] == {"reader-broker-token": {}}
    assert render_routes(pool(reader_row()), {}) == {"http": {"routers": {}, "services": {}}}
    # Preview-only pool keys are not required for a tools-off-only pool.
    assert validate_pool(pool(reader_row()))["houses"][0]["host"] is None


def test_default_houses_are_unchanged_beside_a_tools_off_house():
    preview = {"id": "h1", "harness": "claude-code", "origin": "https://dev1.example.test"}
    extras = {"preview_network": "example-frontend", "service_token_volume": "example-token",
              "frontend_env_file": "/private/house-frontend.env"}
    alone = render_compose(pool(preview, **extras), BASE)
    mixed = render_compose(pool(preview, reader_row(), **extras), BASE)
    assert mixed["services"]["house-h1"] == alone["services"]["house-h1"]
    assert mixed["networks"] == alone["networks"]
    assert mixed["volumes"] == {**alone["volumes"], "reader-broker-token": {}}


@pytest.mark.parametrize("change, message", [
    ({"profile": "other"}, "profile must be tools-off"),
    ({"harness": "codex"}, "requires harness claude-code"),
    ({"origin": "https://r.example.test"}, "no preview origin"),
    ({"home": "relative/home"}, "normalized absolute"),
    ({"home": "/srv/../etc"}, "normalized absolute"),
    ({"home": "/"}, "normalized absolute"),
    ({"home": ""}, "home must be"),
    ({"broker_token_volume": "x"}, "must contain only"),
])
def test_tools_off_house_rejects_invalid_rows(change, message):
    with pytest.raises(ValueError, match=message):
        validate_pool(pool(reader_row(**change)))


def test_tools_off_keys_require_the_profile_and_no_aliases():
    plain = {"id": "h1", "harness": "claude-code", "origin": "https://dev1.example.test",
             "home": "/srv/example/home"}
    with pytest.raises(ValueError, match="requires profile tools-off"):
        validate_houses({"houses": [plain]})
    with pytest.raises(ValueError, match="no preview aliases"):
        validate_pool(pool(reader_row(), preview_aliases={"reader-a": ["r.example.test"]}))
    with pytest.raises(ValueError, match="unique"):
        validate_pool(pool(reader_row(), reader_row()))


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_rendered_overlay_passes_docker_compose_config(tmp_path):
    overlay = tmp_path / "reader.json"
    overlay.write_text(json.dumps(render_compose(pool(reader_row()), BASE)))
    env = tmp_path / "empty.env"
    env.write_text("")
    result = subprocess.run(
        ["docker", "compose", "--env-file", str(env), "-f", str(BASE), "-f", str(overlay),
         "config", "--no-interpolate", "--format", "json", "house-reader-a"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 and "Cannot connect" in result.stderr:
        pytest.skip("docker compose unavailable")
    assert result.returncode == 0, result.stderr
    service = json.loads(result.stdout)["services"]["house-reader-a"]
    targets = {v["target"] for v in service["volumes"]}
    assert targets == {"/app/src", "/var/lib/bawt-reader/home",
                       "/run/llm-bawt-reader-broker-token", "/run/llm-bawt-api-key"}
    assert all(v.get("read_only") for v in service["volumes"]
               if v["target"] != "/var/lib/bawt-reader/home")


# --- consumer fence ----------------------------------------------------------

class Redis:
    def __init__(self):
        self.acks = []

    async def xack(self, stream, group, msg_id):
        self.acks.append((stream, group, msg_id))


class Done:
    def __init__(self):
        self.done = []

    def publish_run_done(self, request_id, *, teardown):
        self.done.append((request_id, teardown))


def _slot(tools_off=True):
    slot = HouseSlot("reader-a", "claude-code", object(), tools_off=tools_off)
    slot.epoch = "7"
    return slot


def _send(**fields):
    base = {"action": "chat.send", "request_id": "req-1", "session_key": "s",
            EPOCH_FIELD: "7"}
    base.update(fields)
    return base


def _gate(slot, fields):
    redis, done = Redis(), Done()
    slot.track = lambda _fields: None  # no Redis turn record in this unit test
    admitted = asyncio.run(gate_send(slot, fields, "1-0", redis, group="g", publisher=done))
    return admitted, redis.acks, done.done


@pytest.mark.parametrize("fields", [
    {},                                         # ordinary send misrouted here
    {"tools_disabled": "1"},                    # tools-off but no strict run proof
    {"strict_run": "run-token"},                # proof without the tools-off stamp
    {"tools_disabled": "0", "strict_run": "x"},
])
def test_tools_off_house_refuses_unmarked_sends_before_running(fields):
    admitted, acks, done = _gate(_slot(), _send(**fields))
    assert admitted is False
    assert acks == [(houses.commands_stream("reader-a"), "g", "1-0")]
    assert done == [("req-1", REFUSED)]


def test_tools_off_house_admits_a_strict_tools_off_send():
    admitted, acks, done = _gate(_slot(), _send(tools_disabled="1", strict_run="run-token"))
    assert admitted is True and acks == [] and done == []


def test_tools_off_house_still_fences_a_stale_epoch():
    fields = _send(tools_disabled="1", strict_run="run-token", **{EPOCH_FIELD: "6"})
    admitted, _acks, done = _gate(_slot(), fields)
    assert admitted is False and done == [("req-1", REFUSED)]


def test_ordinary_house_is_unchanged():
    admitted, _acks, _done = _gate(_slot(tools_off=False), _send())
    assert admitted is True


def test_tools_off_env_requires_a_named_house():
    assert tools_off_from_env("reader-a", {}) is False
    assert tools_off_from_env("reader-a", {"BAWT_HOUSE_TOOLS_OFF": "on"}) is True
    with pytest.raises(ValueError, match="requires a named"):
        tools_off_from_env("", {"BAWT_HOUSE_TOOLS_OFF": "on"})
    with pytest.raises(ValueError, match="on or off"):
        tools_off_from_env("reader-a", {"BAWT_HOUSE_TOOLS_OFF": "maybe"})
    with pytest.raises(ValueError, match="requires a named"):
        HouseSlot("", "claude-code", object(), tools_off=True)


def test_tools_off_house_reads_the_broker_only_copy():
    env = _service()["environment"]
    assert env["BRIDGE_BROKER_TOKEN_FILE"] == "/run/llm-bawt-reader-broker-token/broker"
    assert env["LLM_BAWT_API_KEY_FILE"] == "/run/llm-bawt-api-key/api-key"


READER_OVERLAY = BASE.parent / "docker-compose.reader-broker.yml"


def test_default_compose_has_no_reader_export():
    """Default app service unchanged: no reader mount, env or volume anywhere."""
    assert "reader-broker" not in BASE.read_text()
    assert "READER_EXPORT" not in BASE.read_text()


def test_reader_overlay_adds_only_the_app_mount_env_and_volume():
    import yaml

    overlay = yaml.safe_load(READER_OVERLAY.read_text())
    assert set(overlay) == {"services", "volumes"}
    assert set(overlay["services"]) == {"app"}
    assert overlay["services"]["app"] == {
        "environment": ["BRIDGE_BROKER_TOKEN_READER_EXPORT_FILE=/run/llm-bawt-reader-broker-token/broker"],
        "volumes": ["reader-broker-token:/run/llm-bawt-reader-broker-token"],
    }
    assert overlay["volumes"] == {"reader-broker-token": None}


def test_tools_off_house_never_generates_an_ssh_identity():
    assert _service()["environment"]["BRIDGE_SSH_IDENTITY"] == "0"
