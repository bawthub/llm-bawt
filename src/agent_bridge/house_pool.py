"""Render opt-in house services and preview routes from deployment-local config.

This is an offline renderer, not a daemon or a bot allocator. The default pool
is empty. It never writes a profile, Redis lease, route, or deployment file.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

HOUSE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,30}\Z")
HOSTNAME = re.compile(
    r"(?=.{1,253}\Z)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*\Z"
)
MEMORY = re.compile(r"[1-9][0-9]*(?:[kKmMgG](?:[bB])?)?\Z")
HARNESSES = {"claude-code": "claude_code_bridge", "codex": "codex_bridge"}
POOL_KEYS = {
    "image", "base_image", "memory", "cgroup_parent", "preview_network",
    "service_token_volume", "frontend_env_file", "houses",
}
HOUSE_KEYS = {"id", "harness", "origin", "memory"}


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{name} must be a nonempty single-line string")
    return value


def _memory(value: object) -> str:
    value = _text(value, "memory")
    if not MEMORY.fullmatch(value):
        raise ValueError("memory must be a positive Docker memory limit, e.g. 4g")
    return value


def _origin(value: object) -> tuple[str, str]:
    value = _text(value, "origin")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not HOSTNAME.fullmatch(parsed.hostname)
        or parsed.netloc != parsed.hostname
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("origin must be a bare HTTPS hostname (no port or credentials)")
    return f"https://{parsed.hostname}", parsed.hostname


def validate_pool(config: object) -> dict:
    """Fail before rendering any services when deployment input is ambiguous."""
    if not isinstance(config, dict) or config.keys() - POOL_KEYS:
        raise ValueError("pool must be an object containing only documented keys")
    houses = config.get("houses", [])
    if not isinstance(houses, list):
        raise ValueError("houses must be a list")
    result = copy.deepcopy(config)
    result["houses"] = []
    ids: set[str] = set()
    hosts: set[str] = set()
    for row in houses:
        if not isinstance(row, dict) or row.keys() - HOUSE_KEYS:
            raise ValueError("house must contain only id, harness, origin and memory")
        house_id = row.get("id")
        if not isinstance(house_id, str) or not HOUSE_ID.fullmatch(house_id):
            raise ValueError("house id must match ^[a-z0-9][a-z0-9-]{0,30}$")
        harness = row.get("harness")
        if harness not in HARNESSES:
            raise ValueError("harness must be claude-code or codex")
        origin, host = _origin(row.get("origin"))
        if house_id in ids or host in hosts:
            raise ValueError("house ids and preview hostnames must be unique")
        ids.add(house_id)
        hosts.add(host)
        result["houses"].append({
            "id": house_id, "harness": harness, "origin": origin, "host": host,
            "memory": _memory(row.get("memory", config.get("memory"))),
        })
    # With zero houses nothing else is required and no resources are produced.
    if houses:
        for key in (
            "image", "base_image", "preview_network", "service_token_volume",
            "frontend_env_file",
        ):
            _text(config.get(key), key)
        if not Path(config["frontend_env_file"]).is_absolute():
            raise ValueError("frontend_env_file must be an absolute deployment-local path")
    parent = config.get("cgroup_parent", "")
    if not isinstance(parent, str) or "\n" in parent:
        raise ValueError("cgroup_parent must be a string (empty means engine default)")
    return result


def render_compose(config: object, base_compose: Path) -> dict:
    """Overlay the mansion Compose definition; inherit its credentials/mounts."""
    pool = validate_pool(config)
    services = {}
    for house in pool["houses"]:
        house_id = house["id"]
        service = {
            # One definition for either harness. Inherit the existing mansion
            # mount paths, broker files, source, Redis/app/MCP URLs and SSH setup.
            "extends": {
                "file": str(base_compose.resolve()), "service": "claude-code-bridge",
            },
            "image": pool["image"],
            "build": {
                "context": str(base_compose.resolve().parent),
                "dockerfile": "Dockerfile.house",
                "args": {"AGENT_BRIDGE_IMAGE": pool["base_image"]},
            },
            # Override the mansion's fixed container name when extending it.
            "container_name": f"${{COMPOSE_PROJECT_NAME:-llm-bawt}}-house-{house_id}",
            "command": ["python", "-m", HARNESSES[house["harness"]]],
            "init": True,
            "mem_limit": house["memory"],
            "restart": "unless-stopped",
            "env_file": [pool["frontend_env_file"]],
            "environment": {
                "BAWT_HOUSE": house_id,
                "CODEX_HOME": "/home/bridge/.codex",
                # Never inherit the obsolete mansion model fallback. Requests
                # carry their model from the bot/endpoint configuration.
                "CODEX_MODEL": "",
                "CODEX_BRIDGE_CWD": "/home/bridge/dev",
                "CODEX_BRIDGE_HEALTH_PORT": "8681",
                "CODEX_BIN": "${CODEX_BIN:-/app/.venv/lib/python3.12/site-packages/codex_cli_bin/bin/codex}",
                "CODEX_BAWTHUB_MCP_URL": "http://app:8001/mcp",
                "CODEX_CRAWL4AI_MCP_URL": "http://crawl4ai:11235/mcp/sse",
                "CODEX_LOCAL_PLUGINS_SRC": "/home/bridge/dev/agent-skills/codex",
                "BAWTHUB_AUTH_SESSIONS": "on",
                "BAWTHUB_AUTH_URL": house["origin"],
                "BAWTHUB_SERVICE_TOKEN_FILE": "/run/bawthub-service-token/token",
                "BAWTHUB_LLM_URL": "http://app:8642",
            },
            # Claude state comes from the mansion parent. Add Codex state in
            # the identical locations, regardless of the selected start command.
            "volumes": [
                f"{base_compose.resolve().parent}/src:/app/src",
                "${BRIDGE_GIT_CONFIG_DIR:-${HOME}/.config/bawthub-git/bridge}:/home/bridge/.config/bawthub-git:ro",
                "${HOME}/.codex/config.toml:/home/bridge/.codex/config.toml",
                "${HOME}/.codex/sessions:/home/bridge/.codex/sessions",
                "house-service-token:/run/bawthub-service-token:ro",
            ],
            "networks": ["default", "house-previews"],
        }
        if pool.get("cgroup_parent"):
            service["cgroup_parent"] = pool["cgroup_parent"]
        services[f"house-{house_id}"] = service
    output: dict = {"services": services}
    if services:
        output["networks"] = {
            "house-previews": {"external": True, "name": pool["preview_network"]},
        }
        output["volumes"] = {
            "house-service-token": {"external": True, "name": pool["service_token_volume"]},
        }
    return output


def render_routes(config: object, template: dict) -> dict:
    """Clone the owning frontend route rules, preserving shared voice routing."""
    pool = validate_pool(config)
    routers = {}
    services = {}
    for house in pool["houses"]:
        service_name = f"house-{house['id']}"
        for source in ("frontend", "frontend-api"):
            router = copy.deepcopy(template["http"]["routers"][source])
            rule, count = re.subn(
                r"Host\(`[^`]+`\)", f"Host(`{house['host']}`)", router["rule"],
            )
            if count != 1:
                raise ValueError(f"{source} template must contain exactly one Host rule")
            router["rule"] = rule
            router["service"] = service_name
            routers[f"{service_name}-{source}"] = router
        services[service_name] = {
            "loadBalancer": {"servers": [{"url": f"http://{service_name}:5173"}]},
        }
    return {"http": {"routers": routers, "services": services}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("compose", "routes"))
    parser.add_argument("config", type=Path, help="nonsecret JSON pool configuration")
    parser.add_argument("--base-compose", type=Path)
    parser.add_argument("--route-template", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        if args.kind == "compose":
            if not args.base_compose:
                parser.error("compose requires --base-compose")
            output = render_compose(config, args.base_compose)
        else:
            if not args.route_template:
                parser.error("routes requires --route-template")
            template = yaml.safe_load(args.route_template.read_text())
            output = render_routes(config, template)
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(2, f"invalid house configuration: {exc}\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
