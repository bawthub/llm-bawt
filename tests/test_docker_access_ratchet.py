"""Direct Docker access stays inside the ops adapter (TASK-1169 S1a).

Only the ops boundary may reach the Docker daemon. Consumers go through ops
catalog actions and the service observation contract instead. This test is
an executable ratchet over ``src/`` and the compose files:

- a new offender fails immediately;
- every current offender is listed in ``OFFENDERS`` with the task that
  removes it;
- an allowlist entry that no longer matches anything fails too, so removing
  an offender also removes its entry.

It enforces these bounded patterns (it cannot prove arbitrary dynamic
strings): importing ``docker`` (``import``, ``from``,
``importlib``/``__import__``); a non-docstring string naming the daemon
socket, ``DOCKER_HOST`` or the Engine API (ports 2375/2376,
``http+docker://``, ``/containers/...`` paths, any versioned
``/v1.NN/...`` endpoint);
a subprocess argv, ``sh -c`` wrapper, shell string or container
entrypoint/command that runs ``docker``; a compose service that mounts the
socket or its parent directory (bind paths normalized) or sets
``DOCKER_HOST``. Host tooling outside ``src/`` (``scripts/``,
``tenant.sh``, the Makefile) is not product runtime and is out of scope;
the compose files it ships are covered.
"""
from __future__ import annotations

import ast
import posixpath
import re
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SOCKET = "docker.sock"
DAEMON_ENV = re.compile(r"\bDOCKER_HOST\b(?!_)")
#: The Docker Engine API reached without the socket path: the daemon's TCP
#: ports, the http+docker scheme, or Engine API endpoint paths.
ENGINE_API = re.compile(
    r":237[56]\b|\bhttp\+docker://|/v1\.\d{2,}/[A-Za-z_]"
    r"|/containers/(json|create)\b|/containers/[^/\s\"']+/(json|start|stop|restart|kill|exec|logs)\b")
#: A shell command line that runs the docker CLI.
SHELL_DOCKER = re.compile(r"(^|[;&|(]|\$\()\s*(sudo\s+)?(\S*/)?docker(-compose)?(\s|$)")
SHELLS = frozenset({"sh", "bash", "dash", "zsh"})
#: A file skipped here cannot hold any finding: every access kind needs
#: "docker" (any case) or an ENGINE_API match, so this is built from them.
#: The AST scan matches ENGINE_API against an f-string's constant parts, so
#: f"/containers/{quote(cid, safe='')}/exec" is a finding even though its raw
#: source text is not an ENGINE_API match; "/containers/" admits that form.
PREFILTER = re.compile(r"docker|/containers/|" + ENGINE_API.pattern, re.IGNORECASE)
#: Mounting a socket's parent directory exposes the socket too.
SOCKET_PARENTS = frozenset({"/run", "/var/run", "/var/run/docker", "/run/docker"})
DOCKER_MODULES = frozenset({"docker", "aiodocker"})
SUBPROCESS_CALLS = frozenset({
    "run", "call", "check_call", "check_output", "Popen", "getoutput", "getstatusoutput",
    "create_subprocess_exec", "create_subprocess_shell", "system", "popen",
    "execv", "execvp", "execvpe", "spawnv", "spawnvp",
})

#: The ops Docker adapter: the one place that talks to the daemon. Each file
#: may hold only the access kinds listed; a new kind fails, and every listed
#: kind must still be found, or the entry is stale.
OPS_ADAPTER = {
    "llm_bawt/ops/executor.py": ({"import docker", "daemon socket"}, "submits workers, runs typed actions"),
    "llm_bawt/ops/health_collector.py": ({"import docker"}, "samples container state for /v1/health"),
    "llm_bawt/ops/worker.py": ({"daemon socket", "engine api"}, "the one-shot worker's daemon connection"),
    "llm_bawt/ops/sealed_container.py": ({"engine api"}, "worker-side sealed container create/start/exec"),
    "llm_bawt/ops/image_deploy.py": ({"engine api"}, "worker-side pinned image deploy/rollback"),
    "llm_bawt/ops/app_config_reconcile.py": ({"engine api"}, "worker-side container lookup for app-config apply"),
    "llm_bawt/ops/config_reconcile.py": ({"engine api"}, "worker-side exec for config apply"),
    "llm_bawt/ops/routing_reconcile.py": ({"engine api"}, "worker-side exec for routing apply"),
    "llm_bawt/ops/plan_reconcile.py": ({"engine api"}, "worker-side plan volumes and container inspect"),
    "llm_bawt/ops/compose_render.py": ({"runs docker"}, "sealed `docker compose config` render"),
    # Catalog data only: the deployed mounts a sealed recreate preserves.
    "llm_bawt/ops/seeds.py": ({"daemon socket"}, "data_mounts strings, no client"),
}
#: Compose services inside the ops boundary.
OPS_COMPOSE = {
    ("docker-compose.ops.yml", "ops-reconciler"): "the ops reconciler owns the socket",
}

#: Current offenders outside the boundary, each with the task that removes it.
OFFENDERS = {
    ("compose", "docker-compose.yml", "app"): "TASK-1173",
}


def _docstring_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _call_name(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _text(node):
    """A constant string, or the constant parts of an f-string joined."""
    if isinstance(node, ast.JoinedStr):
        return "".join(_const_str(v) or "{}" for v in node.values)
    return _const_str(node)


def _argv_runs_docker(node):
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        return False
    head = _const_str(node.elts[0])
    if head is None:
        return False
    if Path(head).name == "docker":
        return True
    # sh -c "docker ..."
    args = [_text(e) for e in node.elts[1:]]
    return Path(head).name in SHELLS and "-c" in args and any(
        a and SHELL_DOCKER.search(a) for a in args[args.index("-c") + 1:])


def _runs_docker(call):
    # A container spec whose entrypoint/command is the docker CLI.
    if any(k.arg in ("entrypoint", "command") and _argv_runs_docker(k.value) for k in call.keywords):
        return True
    if _call_name(call.func) not in SUBPROCESS_CALLS or not call.args:
        return False
    first = call.args[0]
    if _argv_runs_docker(first):
        return True
    text = _text(first)
    return text is not None and SHELL_DOCKER.search(text) is not None


def scan_source(tree):
    """Return the kinds of Docker access in one parsed module."""
    found = set()
    docstrings = _docstring_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in DOCKER_MODULES for alias in node.names):
                found.add("import docker")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and (node.module or "").split(".")[0] in DOCKER_MODULES:
                found.add("import docker")
        elif isinstance(node, ast.Call):
            if _call_name(node.func) in ("import_module", "__import__") and node.args:
                name = _const_str(node.args[0])
                if name and name.split(".")[0] in DOCKER_MODULES:
                    found.add("import docker")
            if _runs_docker(node):
                found.add("runs docker")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            if SOCKET in node.value:
                found.add("daemon socket")
            if DAEMON_ENV.search(node.value):
                found.add("DOCKER_HOST")
            if ENGINE_API.search(node.value):
                found.add("engine api")
        elif isinstance(node, ast.JoinedStr) and ENGINE_API.search(_text(node)):
            found.add("engine api")  # f"/containers/{cid}/restart"
    return found


def scan_text(text, filename="<src>"):
    """A file's findings: the prefilter, then the AST scan. Self-tests use this too."""
    return scan_source(ast.parse(text, filename=filename)) if PREFILTER.search(text) else set()


@cache
def _source_findings():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        found.extend((rel, kind) for kind in sorted(scan_text(path.read_text(), str(path))))
    return tuple(found)


def _volume_source(volume):
    source = str(volume.get("source", "")) if isinstance(volume, dict) else str(volume).split(":", 1)[0]
    return posixpath.normpath(source) if source.startswith("/") else source


def scan_service(service):
    """True when a compose service reaches the daemon (socket mount or DOCKER_HOST)."""
    volumes = service.get("volumes") or []
    sources = [_volume_source(v) for v in volumes]
    if any(SOCKET in src or src.rstrip("/") in SOCKET_PARENTS for src in sources):
        return True
    env = service.get("environment") or {}
    keys = env.keys() if isinstance(env, dict) else (str(e).split("=", 1)[0] for e in env)
    return "DOCKER_HOST" in keys


@cache
def _compose_findings():
    found = []
    for path in sorted(ROOT.glob("docker-compose*.yml")):
        services = (yaml.safe_load(path.read_text()) or {}).get("services") or {}
        found.extend((path.name, name) for name, service in sorted(services.items())
                     if scan_service(service or {}))
    return tuple(found)


def test_no_new_direct_docker_access_in_src():
    offenders = {("src", rel, kind) for rel, kind in _source_findings()
                 if kind not in OPS_ADAPTER.get(rel, ((), ""))[0]}
    assert offenders - OFFENDERS.keys() == set(), "route Docker access through ops (TASK-1169)"


def test_no_new_compose_service_mounts_the_daemon():
    offenders = {("compose", f, s) for f, s in _compose_findings() if (f, s) not in OPS_COMPOSE}
    assert offenders - OFFENDERS.keys() == set(), "only the ops boundary mounts the Docker socket"


def test_every_allowlist_entry_still_matches():
    src = set(_source_findings())
    compose = set(_compose_findings())
    live = {("src", rel, kind) for rel, kind in src} | {("compose", f, s) for f, s in compose}
    assert set(OFFENDERS) - live == set(), "an offender is gone: delete its OFFENDERS entry"
    allowed = {(rel, kind) for rel, (kinds, _) in OPS_ADAPTER.items() for kind in kinds}
    assert allowed - src == set(), "an adapter access kind is gone: narrow OPS_ADAPTER"
    assert set(OPS_COMPOSE) - compose == set(), "ops compose service no longer mounts the daemon"


def test_every_offender_names_its_owning_task():
    assert all(task.startswith("TASK-") and task[5:].isdigit() for task in OFFENDERS.values())


def _parse(code):
    return scan_text(code)


def test_scanner_detects_each_access_kind():
    assert _parse("import docker.errors") == {"import docker"}
    assert _parse("from docker import from_env") == {"import docker"}
    assert _parse("import importlib\nimportlib.import_module('docker')") == {"import docker"}
    assert _parse("x = '/var/run/docker.sock'") == {"daemon socket"}
    assert _parse("import os\nos.environ['DOCKER_HOST']") == {"DOCKER_HOST"}
    assert _parse("import subprocess\nsubprocess.run(['docker', 'ps'])") == {"runs docker"}
    assert _parse("import subprocess\nsubprocess.run(['/usr/bin/docker', 'ps'])") == {"runs docker"}
    assert _parse("import os\nos.system('docker compose up -d')") == {"runs docker"}
    assert _parse("import asyncio\nasyncio.create_subprocess_exec('docker', 'ps')") == {"runs docker"}
    assert _parse("sealed.run(entrypoint=['docker'], cmd=argv)") == {"runs docker"}
    # r1 B1: shell wrappers and the Engine API without the socket path.
    assert _parse("import subprocess\nsubprocess.run(['sh', '-c', 'docker ps'])") == {"runs docker"}
    assert _parse("import subprocess\nsubprocess.run(['/bin/bash', '-c', 'set -e; docker restart x'])") == {"runs docker"}
    assert _parse("import subprocess\nsubprocess.run(f'cd {d} && docker compose up', shell=True)") == {"runs docker"}
    assert _parse("import httpx\nhttpx.get('http://127.0.0.1:2375/containers/json')") == {"engine api"}
    assert _parse("import httpx\nhttpx.post(f'{base}/v1.43/containers/{cid}/restart')") == {"engine api"}
    assert _parse("s = 'http+docker://localhost/info'") == {"engine api"}
    assert _parse("api.request('POST', f'/containers/{cid}/restart')") == {"engine api"}
    # r2: the file-level prefilter must admit every Engine API form.
    assert _parse("import httpx\nhttpx.get(f'{engine_url}/v1.43/images/json')") == {"engine api"}
    assert _parse("import httpx\nhttpx.post(f'{u}/v1.43/networks/create')") == {"engine api"}
    assert _parse("import httpx\nhttpx.get(u + '/v1.43/volumes')") == {"engine api"}
    # Loopy r1: any versioned Engine endpoint, through the file prefilter.
    assert _parse("import httpx\nhttpx.get(f'{engine_url}/v1.43/info')") == {"engine api"}
    assert _parse("import httpx\nhttpx.post(f'{engine_url}/v1.43/build')") == {"engine api"}
    # TASK-1476: a quoted placeholder hides the raw form; no "docker" in the file.
    assert _parse("""api.request("POST", f"/containers/{quote(cid, safe='')}/exec")""") == {"engine api"}
    assert _parse("url = 'https://graph.microsoft.com/v1.0/me'") == set()


def test_scanner_ignores_docstrings_and_lookalikes():
    assert _parse('"""Talks to /var/run/docker.sock."""') == set()
    assert _parse('def f():\n    """Uses DOCKER_HOST."""') == set()
    assert _parse("from .docker_names import x\nimport dockerfile_parse") == set()
    assert _parse("import subprocess\nsubprocess.run(['git', 'docker'])") == set()
    assert _parse("import subprocess\nsubprocess.run(['sh', '-c', 'echo docker'])") == set()
    assert _parse("import os\nos.environ.get('DOCKER_HOST_IP')") == set()
    assert _parse("url = 'http://h:23750/x'") == set()


def test_adapter_exemption_is_per_kind():
    # r1 B2: a new access kind in a boundary file is still an offender.
    assert "import docker" not in OPS_ADAPTER["llm_bawt/ops/seeds.py"][0]
    assert all(kinds and reason for kinds, reason in OPS_ADAPTER.values())


def test_compose_scanner_detects_socket_and_daemon_env():
    assert scan_service({"volumes": ["/var/run/docker.sock:/var/run/docker.sock"]})
    assert scan_service({"volumes": [{"type": "bind", "source": "/run/docker.sock", "target": "/x"}]})
    assert scan_service({"volumes": ["/var/run/:/host-run:ro"]})
    assert scan_service({"volumes": ["/run/../var/run:/host-run:ro"]})  # r1 B1: normalized.
    assert scan_service({"volumes": [{"type": "bind", "source": "/var//run/./docker.sock", "target": "/s"}]})
    assert scan_service({"environment": {"DOCKER_HOST": "tcp://h:2375"}})
    assert scan_service({"environment": ["DOCKER_HOST=unix:///x"]})
    assert not scan_service({"volumes": ["./src:/app/src:ro"], "environment": {"A": "1"}})
