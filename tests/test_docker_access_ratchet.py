"""Direct Docker access stays inside the ops adapter (TASK-1169 S1a).

Only the ops boundary may reach the Docker daemon. Consumers go through ops
catalog actions and the service observation contract instead. This test is
an executable ratchet over ``src/`` and the compose files:

- a new offender fails immediately;
- every current offender is listed in ``OFFENDERS`` with the task that
  removes it;
- an allowlist entry that no longer matches anything fails too, so removing
  an offender also removes its entry.

Access means any of: importing ``docker`` (``import``, ``from``,
``importlib``/``__import__``), a non-docstring string naming the daemon
socket or ``DOCKER_HOST``, a subprocess argv, shell string or container
entrypoint that runs ``docker``, or a compose service that mounts the
socket (or its parent directory) or sets ``DOCKER_HOST``. Host tooling
outside ``src/`` (``scripts/``, ``tenant.sh``, the Makefile) is not product
runtime and is out of scope; the compose files it ships are covered.
"""
from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SOCKET = "docker.sock"
DAEMON_ENV = "DOCKER_HOST"
#: Mounting a socket's parent directory exposes the socket too.
SOCKET_PARENTS = frozenset({"/run", "/var/run", "/var/run/docker", "/run/docker"})
DOCKER_MODULES = frozenset({"docker", "aiodocker"})
SUBPROCESS_CALLS = frozenset({
    "run", "call", "check_call", "check_output", "Popen", "getoutput", "getstatusoutput",
    "create_subprocess_exec", "create_subprocess_shell", "system", "popen",
    "execv", "execvp", "execvpe", "spawnv", "spawnvp",
})

#: The ops Docker adapter: the one place that talks to the daemon. Every
#: finding in these files is allowed; each file must still have one, or it
#: no longer belongs here.
OPS_ADAPTER = {
    "llm_bawt/ops/executor.py": "submits workers and runs typed actions",
    "llm_bawt/ops/health_collector.py": "samples container state for /v1/health",
    "llm_bawt/ops/worker.py": "the one-shot worker's daemon connection",
    "llm_bawt/ops/compose_render.py": "sealed `docker compose config` render (entrypoint docker)",
    "llm_bawt/ops/seeds.py": "catalog data: the deployed mounts a sealed recreate preserves",
}
#: Compose services inside the ops boundary.
OPS_COMPOSE = {
    ("docker-compose.ops.yml", "ops-reconciler"): "the ops reconciler owns the socket",
}

#: Current offenders outside the boundary, each with the task that removes it.
OFFENDERS = {
    ("src", "llm_bawt/media/gpu_handoff_adapter.py", "import docker"): "TASK-1171",
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


def _argv_runs_docker(node):
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        head = _const_str(node.elts[0])
        return head is not None and Path(head).name == "docker"
    return False


def _runs_docker(call):
    # A container spec whose entrypoint/command is the docker CLI.
    if any(k.arg in ("entrypoint", "command") and _argv_runs_docker(k.value) for k in call.keywords):
        return True
    if _call_name(call.func) not in SUBPROCESS_CALLS or not call.args:
        return False
    first = call.args[0]
    if _argv_runs_docker(first):
        return True
    text = _const_str(first)
    if text is None and isinstance(first, ast.JoinedStr) and first.values:
        text = _const_str(first.values[0])
    return text is not None and text.lstrip().split(" ", 1)[0] in ("docker", "docker-compose")


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
            if DAEMON_ENV in node.value:
                found.add("DOCKER_HOST")
    return found


@cache
def _source_findings():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text()
        # Every access kind spells "docker" or "DOCKER"; skip parsing the rest.
        if "docker" not in text.lower():
            continue
        rel = path.relative_to(SRC).as_posix()
        found.extend((rel, kind) for kind in sorted(scan_source(ast.parse(text, filename=str(path)))))
    return tuple(found)


def _volume_source(volume):
    if isinstance(volume, dict):
        return str(volume.get("source", ""))
    return str(volume).split(":", 1)[0]


def scan_service(service):
    """True when a compose service reaches the daemon (socket mount or DOCKER_HOST)."""
    volumes = service.get("volumes") or []
    sources = [_volume_source(v) for v in volumes]
    if any(SOCKET in src or src.rstrip("/") in SOCKET_PARENTS for src in sources):
        return True
    env = service.get("environment") or {}
    keys = env.keys() if isinstance(env, dict) else (str(e).split("=", 1)[0] for e in env)
    return DAEMON_ENV in keys


@cache
def _compose_findings():
    found = []
    for path in sorted(ROOT.glob("docker-compose*.yml")):
        services = (yaml.safe_load(path.read_text()) or {}).get("services") or {}
        found.extend((path.name, name) for name, service in sorted(services.items())
                     if scan_service(service or {}))
    return tuple(found)


def test_no_new_direct_docker_access_in_src():
    offenders = {("src", rel, kind) for rel, kind in _source_findings() if rel not in OPS_ADAPTER}
    assert offenders - OFFENDERS.keys() == set(), "route Docker access through ops (TASK-1169)"


def test_no_new_compose_service_mounts_the_daemon():
    offenders = {("compose", f, s) for f, s in _compose_findings() if (f, s) not in OPS_COMPOSE}
    assert offenders - OFFENDERS.keys() == set(), "only the ops boundary mounts the Docker socket"


def test_every_allowlist_entry_still_matches():
    src = set(_source_findings())
    compose = set(_compose_findings())
    live = {("src", rel, kind) for rel, kind in src} | {("compose", f, s) for f, s in compose}
    assert set(OFFENDERS) - live == set(), "an offender is gone: delete its OFFENDERS entry"
    assert set(OPS_ADAPTER) - {rel for rel, _ in src} == set(), "adapter file has no Docker access"
    assert set(OPS_COMPOSE) - compose == set(), "ops compose service no longer mounts the daemon"


def test_every_offender_names_its_owning_task():
    assert all(task.startswith("TASK-") and task[5:].isdigit() for task in OFFENDERS.values())


def _parse(code):
    return scan_source(ast.parse(code))


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


def test_scanner_ignores_docstrings_and_lookalikes():
    assert _parse('"""Talks to /var/run/docker.sock."""') == set()
    assert _parse('def f():\n    """Uses DOCKER_HOST."""') == set()
    assert _parse("from .docker_names import x\nimport dockerfile_parse") == set()
    assert _parse("import subprocess\nsubprocess.run(['git', 'docker'])") == set()


def test_compose_scanner_detects_socket_and_daemon_env():
    assert scan_service({"volumes": ["/var/run/docker.sock:/var/run/docker.sock"]})
    assert scan_service({"volumes": [{"type": "bind", "source": "/run/docker.sock", "target": "/x"}]})
    assert scan_service({"volumes": ["/var/run/:/host-run:ro"]})
    assert scan_service({"environment": {"DOCKER_HOST": "tcp://h:2375"}})
    assert scan_service({"environment": ["DOCKER_HOST=unix:///x"]})
    assert not scan_service({"volumes": ["./src:/app/src:ro"], "environment": {"A": "1"}})
