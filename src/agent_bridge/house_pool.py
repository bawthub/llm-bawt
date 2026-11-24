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

from agent_bridge.bridge_token import BROKER

HOUSE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,30}\Z")
HOSTNAME = re.compile(
    r"(?=.{1,253}\Z)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*\Z"
)
MEMORY = re.compile(r"[1-9][0-9]*(?:[kKmMgG](?:[bB])?)?\Z")
HARNESSES = {"claude-code": "claude_code_bridge", "codex": "codex_bridge"}
_BRIDGE_OWNED = "uid=1001,gid=1001"
# The only container-local writable paths of a house (its root is read-only),
# emptied by every restart. Shared state (~/.claude, Codex config/sessions,
# dev, skills) is bind mounted and unaffected; ~/.local holds image tools and
# is read-only. Pages count against the house memory limit; caps total 1.3 GiB.
HOUSE_TMPFS = (
    "/tmp:size=512m,mode=1777",
    # exec: tools run executables they cache here (pre-commit hook envs, uv);
    # Docker's tmpfs default is noexec (TASK-2175).
    f"/home/bridge/.cache:size=512m,{_BRIDGE_OWNED},mode=0755,exec",
    f"/home/bridge/.config:size=16m,{_BRIDGE_OWNED},mode=0700",
    f"/home/bridge/.codex:size=64m,{_BRIDGE_OWNED},mode=0755",
    f"/home/bridge/.npm:size=256m,{_BRIDGE_OWNED},mode=0755",
)
POOL_KEYS = {
    "image", "base_image", "memory", "cgroup_parent", "preview_network",
    "service_token_volume", "frontend_env_file", "houses", "preview_aliases",
    # Optional: the shared dev site the mansion serves, linked from Campus.
    "mansion_origin",
}
HOUSE_KEYS = {"id", "harness", "origin", "memory", "profile", "home"}
# TASK-2590: opt-in profile for a strict tools-off reader house. It is rendered
# standalone (no mansion ``extends``), with no preview, so it inherits none of
# the shared SDK state, dev tree, SSH or bridge-token mounts.
TOOLS_OFF = "tools-off"
TOOLS_OFF_HOME = "/var/lib/bawt-reader/home"
TOOLS_OFF_UID = "1001:1001"
_TOOLS_OFF_KEYS = {"home"}
# The app's opt-in broker-only export directory (docker-compose.yml, the
# BRIDGE_BROKER_TOKEN_READER_EXPORT_FILE setting), mounted read-only here.
READER_BROKER_VOLUME = "reader-broker-token"
READER_BROKER_DIR = "/run/llm-bawt-reader-broker-token"


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
    """A bare http(s) origin, optionally with a port: ``(origin, host[:port])``.

    A self-hosted tenant serves each preview on a published port of its own
    origin (TASK-2129); the Traefik route renderer accepts only bare HTTPS.
    """
    value = _text(value, "origin")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError:
        port = -1
    host = parsed.hostname or ""
    netloc = f"{host}:{port}" if port else host
    if (
        parsed.scheme not in ("http", "https")
        or not HOSTNAME.fullmatch(host)
        # Default ports are written bare, so one origin has one spelling.
        or port in (-1, 0) or port == {"http": 80, "https": 443}.get(parsed.scheme)
        or parsed.netloc != netloc
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("origin must be a bare http(s) origin, e.g. https://dev1.example.test or "
                         "http://host:3101 (no path or credentials)")
    return f"{parsed.scheme}://{netloc}", netloc


def _tools_off_row(row: dict, house_id: str, harness: str, profile: object,
                   config: dict) -> dict:
    """Validate one ``profile: tools-off`` house (TASK-2590)."""
    if profile != TOOLS_OFF:
        raise ValueError("profile must be tools-off")
    if harness != "claude-code":
        raise ValueError("profile tools-off requires harness claude-code")
    if "origin" in row:
        raise ValueError("profile tools-off has no preview origin")
    home = _text(row.get("home"), "home")
    path = Path(home)
    if not path.is_absolute() or ".." in path.parts or str(path) != home or home == "/":
        raise ValueError("home must be a normalized absolute deployment-local directory")
    return {
        "id": house_id, "harness": harness, "profile": TOOLS_OFF,
        "origin": None, "host": None, "home": home,
        "memory": _memory(row.get("memory", config.get("memory"))),
    }


def is_tools_off(house: dict) -> bool:
    return house.get("profile") == TOOLS_OFF


def validate_houses(config: object) -> list[dict]:
    """The pool's house rows; all a reader of the pool (the app) needs checked."""
    return validate_pool(config, renderer=False)["houses"]


def validate_pool(config: object, *, renderer: bool = True) -> dict:
    """Fail before rendering any services when deployment input is ambiguous.

    ``renderer=False`` skips the image/network/volume/env-file keys only the
    echo-style renderers below use; a self-hosted tenant's installer renders
    its own services and ships a pool file with just ``houses``.
    """
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
            raise ValueError("house must contain only id, harness, origin and memory "
                             "(plus profile and home for tools-off)")
        house_id = row.get("id")
        if not isinstance(house_id, str) or not HOUSE_ID.fullmatch(house_id):
            raise ValueError("house id must match ^[a-z0-9][a-z0-9-]{0,30}$")
        harness = row.get("harness")
        if harness not in HARNESSES:
            raise ValueError("harness must be claude-code or codex")
        profile = row.get("profile")
        if profile is not None:
            result["houses"].append(_tools_off_row(row, house_id, harness, profile, config))
            if house_id in ids:
                raise ValueError("house ids and preview hostnames must be unique")
            ids.add(house_id)
            continue
        if row.keys() & _TOOLS_OFF_KEYS:
            raise ValueError("home requires profile tools-off")
        origin, host = _origin(row.get("origin"))
        if house_id in ids or host in hosts:
            raise ValueError("house ids and preview hostnames must be unique")
        ids.add(house_id)
        hosts.add(host)
        result["houses"].append({
            "id": house_id, "harness": harness, "origin": origin, "host": host,
            "memory": _memory(row.get("memory", config.get("memory"))),
        })
    aliases = config.get("preview_aliases", {})
    if not isinstance(aliases, dict):
        raise ValueError("preview_aliases must map house ids to hostname lists")
    for house_id, names in aliases.items():
        if not isinstance(house_id, str) or not HOUSE_ID.fullmatch(house_id):
            raise ValueError("preview_aliases requires valid house ids")
        if any(h["id"] == house_id and is_tools_off(h) for h in result["houses"]):
            raise ValueError("a tools-off house has no preview aliases")
        if not isinstance(names, list):
            raise ValueError("preview_aliases requires hostname lists")
        for name in names:
            if not isinstance(name, str) or not HOSTNAME.fullmatch(name):
                raise ValueError("preview_aliases requires bare lowercase hostnames")
            if name in hosts:
                raise ValueError("preview hostnames and aliases must be unique")
            hosts.add(name)
    # With zero houses nothing else is required and no resources are produced.
    preview_houses = [h for h in result["houses"] if not is_tools_off(h)]
    if houses and renderer:
        for key in ("image", "base_image"):
            _text(config.get(key), key)
    if preview_houses and renderer:
        for key in ("preview_network", "service_token_volume", "frontend_env_file"):
            _text(config.get(key), key)
        if not Path(config["frontend_env_file"]).is_absolute():
            raise ValueError("frontend_env_file must be an absolute deployment-local path")
    if config.get("mansion_origin") is not None:
        result["mansion_origin"], _ = _origin(config["mansion_origin"])
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
        if is_tools_off(house):
            services[f"house-{house_id}"] = _tools_off_service(house, pool, base_compose)
            continue
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
                # TASK-2135: a preview authenticates exactly like the shared
                # dev site, so sign-in settings come only from the same
                # frontend env file (no sessions there means none here).
                # The house serves its occupant's preview itself.
                "BAWT_HOUSE_PREVIEW": "on",
                "npm_config_store_dir": "/home/bridge/dev/.pnpm-store",
                "BAWTHUB_SERVICE_TOKEN_FILE": "/run/bawthub-service-token/token",
                # TASK-2400: the voice backend requires the frontend's proxy
                # token on /api/v1 (TASK-2235); same file as the dev frontend.
                "BAWTHUB_VOICE_PROXY_TOKEN_FILE": "/run/bawthub-service-token/token",
                "BAWTHUB_LLM_URL": "http://app:8642",
                # TASK-2184: the preview's Repos page reads the dev tree from
                # /repos (read-only below) and shows host paths from here.
                "REPOS_HOST_ROOT": "${HOME}/dev",
                # TASK-2386: the preview's DB URL is injected at creation from
                # the root-only secrets volume (scripts/compose-secrets), never
                # read from the frontend env file. Null takes it from Compose's
                # environment and, unset, removes the env-file value.
                "DATABASE_URL": None,
            },
            # Claude state comes from the mansion parent. Add Codex state in
            # the identical locations, regardless of the selected start command.
            "volumes": [
                f"{base_compose.resolve().parent}/src:/app/src",
                "${BRIDGE_GIT_CONFIG_DIR:-${HOME}/.config/bawthub-git/bridge}:/home/bridge/.config/bawthub-git:ro",
                "${HOME}/.codex/config.toml:/home/bridge/.codex/config.toml",
                "${HOME}/.codex/sessions:/home/bridge/.codex/sessions",
                "house-service-token:/run/bawthub-service-token:ro",
                # Same dev tree as the mansion frontend, read-only so the
                # Repos viewer can't modify it. The host-identical path that
                # linked worktrees need is already inherited from the mansion.
                "${HOME}/dev:/repos:ro",
            ],
            # TASK-2080: a restart keeps the container layer, so the root is
            # read-only. A tenant writes only to the shared mounts or to the
            # size-capped tmpfs below, which a restart empties (nested binds
            # still mount on top). Any other write fails instead of persisting.
            "read_only": True,
            "tmpfs": list(HOUSE_TMPFS),
            "networks": ["default", "house-previews"],
        }
        if pool.get("cgroup_parent"):
            service["cgroup_parent"] = pool["cgroup_parent"]
        services[f"house-{house_id}"] = service
    output: dict = {"services": services}
    if any(is_tools_off(h) for h in pool["houses"]):
        # TASK-2590: the broker-only copy the app writes when
        # docker-compose.reader-broker.yml is in the stack. Same project
        # volume; declaring it here too keeps the reader overlay standalone.
        output["volumes"] = {READER_BROKER_VOLUME: {}}
    if any(not is_tools_off(h) for h in pool["houses"]):
        output["networks"] = {
            "house-previews": {"external": True, "name": pool["preview_network"]},
        }
        output.setdefault("volumes", {}).update({
            "house-service-token": {"external": True, "name": pool["service_token_volume"]},
        })
    return output


def _tools_off_service(house: dict, pool: dict, base_compose: Path) -> dict:
    """A standalone strict tools-off reader house (TASK-2590).

    Deliberately not ``extends``: every mount and variable is listed here, so
    the shared ``~/.claude``/``.claude.json``, the dev tree, SSH, agent scripts,
    skills and the full bridge-token set can never be inherited. The SDK home
    (its ``.claude``, ``.claude.json`` and project base) is a private,
    operator-created host directory owned by the bridge UID. Credentials: the
    Claude token comes from the broker (broker token only, read-only), and app
    calls use this bridge role's API key directory, read-only.
    """
    house_id = house["id"]
    root = base_compose.resolve().parent
    home = TOOLS_OFF_HOME
    service = {
        "image": pool["image"],
        "build": {
            "context": str(root),
            "dockerfile": "Dockerfile.house",
            "args": {"AGENT_BRIDGE_IMAGE": pool["base_image"]},
        },
        "container_name": f"${{COMPOSE_PROJECT_NAME:-llm-bawt}}-house-{house_id}",
        "command": ["python", "-m", HARNESSES["claude-code"]],
        "user": TOOLS_OFF_UID,
        "init": True,
        "mem_limit": house["memory"],
        "restart": "unless-stopped",
        "working_dir": "/app",
        "environment": {
            "BAWT_HOUSE": house_id,
            "BAWT_HOUSE_TOOLS_OFF": "on",
            "BRIDGE_SSH_IDENTITY": "0",
            "HOME": home,
            "CLAUDE_CODE_CWD": f"{home}/work",
            # No skills path or MCP URLs: the image default skills path is not
            # mounted (bootstrap links nothing), and a tools-off turn passes
            # an explicit empty MCP set with --strict-mcp-config.
            "CLAUDE_CODE_BRIDGE_LOG_LEVEL": "${CLAUDE_CODE_BRIDGE_LOG_LEVEL:-INFO}",
            "PYTHONPYCACHEPREFIX": "/tmp/llm-bawt-pycache",
            "LLM_BAWT_API_URL": "http://app:8642",
            "LLM_BAWT_SERVICE_HOST": "app",
            # Passed through by name from the deployment .env, as the mansion does.
            "LLM_BAWT_REDIS_URL": None,
            BROKER.file_env: f"{READER_BROKER_DIR}/broker",
            "LLM_BAWT_API_KEY_FILE": "/run/llm-bawt-api-key/api-key",
        },
        "volumes": [
            f"{root}/src:/app/src:ro",
            f"{house['home']}:{home}",
            f"{READER_BROKER_VOLUME}:{READER_BROKER_DIR}:ro",
            "api-key-bridge-claude-code:/run/llm-bawt-api-key:ro",
        ],
        "read_only": True,
        "tmpfs": ["/tmp:size=512m,mode=1777"],
        "networks": ["default"],
        "depends_on": {
            "redis": {"condition": "service_healthy"},
            "app": {"condition": "service_healthy"},
        },
        "healthcheck": {
            "test": ["CMD", "curl", "-f", "http://localhost:8681/health"],
            "interval": "15s",
        },
    }
    if pool.get("cgroup_parent"):
        service["cgroup_parent"] = pool["cgroup_parent"]
    return service


def render_routes(config: object, template: dict) -> dict:
    """Clone the owning frontend route rules, preserving shared voice routing."""
    pool = validate_pool(config)
    routers = {}
    services = {}
    for house in pool["houses"]:
        if is_tools_off(house):
            continue  # TASK-2590: no preview, so no route or DNS name.
        if not house["origin"].startswith("https://") or ":" in house["host"]:
            raise ValueError("Traefik preview routes need a bare HTTPS origin (no port)")
        service_name = f"house-{house['id']}"
        hosts = [house["host"], *pool.get("preview_aliases", {}).get(house["id"], [])]
        host_rule = " || ".join(f"Host(`{host}`)" for host in hosts)
        if len(hosts) > 1:
            host_rule = f"({host_rule})"
        for source in ("frontend", "frontend-api"):
            router = copy.deepcopy(template["http"]["routers"][source])
            rule, count = re.subn(
                r"Host\(`[^`]+`\)", host_rule, router["rule"],
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


def count_pool(config: dict, count: int, origin_template: str) -> dict:
    """Expand HOUSES without a second, hand-maintained list of house IDs."""
    if type(count) is not int or not 0 <= count <= 16:
        raise ValueError("HOUSES must be an integer from 0 to 16")
    if not isinstance(config, dict) or "houses" in config:
        raise ValueError("count renderer settings must not include a houses list")
    if not isinstance(origin_template, str):
        raise ValueError("origin_template must be a string")
    result = copy.deepcopy(config)
    result["houses"] = [
        {"id": f"h{i}", "harness": "claude-code",
         "origin": origin_template.replace("{id}", f"h{i}"),
         "memory": config.get("memory")}
        for i in range(1, count + 1)
    ]
    # Return the portable source schema, not validator-only `host` fields.
    validate_pool(result)
    return result


def render_bundle(request: object) -> dict:
    """App-side renderer transport; stdin contains deployment data, not secrets."""
    keys = {"count", "pool", "origin_template", "base_compose", "route_template"}
    if not isinstance(request, dict) or request.keys() != keys:
        raise ValueError("invalid count renderer request")
    pool = count_pool(request["pool"], request["count"], request["origin_template"])
    base = request["base_compose"]
    if not isinstance(base, str) or not Path(base).is_absolute():
        raise ValueError("base_compose must be an absolute path")
    template = yaml.safe_load(request["route_template"])
    return {"pool": pool, "compose": render_compose(pool, Path(base)),
            "routes": render_routes(pool, template)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("compose", "routes", "bundle"))
    parser.add_argument("config", type=Path, help="nonsecret JSON pool configuration")
    parser.add_argument("--base-compose", type=Path)
    parser.add_argument("--route-template", type=Path)
    args = parser.parse_args()
    try:
        if str(args.config) == "-":
            import sys
            raw = sys.stdin.read(131073)
            if len(raw) > 131072:
                raise ValueError("renderer request too large")
            config = json.loads(raw)
        else:
            config = json.loads(args.config.read_text())
        if args.kind == "bundle":
            output = render_bundle(config)
        elif args.kind == "compose":
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
