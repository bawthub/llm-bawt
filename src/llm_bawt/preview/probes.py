"""External isolation probes for the running preview profile (TASK-1802).

Run on the docker host after ``preview-up``; read-only. Three probes, each
printing PASS/FAIL lines, exit non-zero on any FAIL:

* ``inventory``: every preview container's mounts, env names, networks and
  ports, checked against the profile's rules (no checkout, credentials,
  sockets, published ports or non-preview networks; env names only).
* ``seed``: the seed row audit inside the preview DB: history/identity tables
  hold only the schema bootstrap's own rows, scrubbed columns NULL, actor columns ``SEED_ACTOR``, settings keys
  within the allowlist, and no private-tagged bot.
* ``network``: a reachability matrix from inside the preview network to the
  production DB, Redis, bridges and app; every target must be unreachable.
  Also proves (through ``pgq``) the preview role does not exist on the
  production cluster.

    python -m llm_bawt.preview.probes [inventory|seed|network ...]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from llm_bawt.preview import topology
from llm_bawt.preview.schema_init import row_digest_sql
from llm_bawt.preview.seed_allowlist import (
    BOT_SETTING_KEYS, COPY, GLOBAL_SETTING_KEYS, PRIVATE_BOTS_SQL, SCHEMA_ONLY, SEED_ACTOR,
    Actor, Null,
)

NETWORK = topology.NETWORK_NAMES["preview"]
PROBE_IMAGE = "busybox@sha256:73aaf090f3d85aa34ee199857f03fa3a95c8ede2ffd4cc2cdb5b94e566b11662"
DB_CONTAINER_FILTER = "com.docker.compose.service=preview-postgres"
ALLOWED_ENV_PREFIXES = ("LLM_BAWT_POSTGRES_", "POSTGRES_", "LLM_BAWT_REDIS_URL", "LLM_BAWT_PREVIEW_", "LLM_BAWT_SERVICE_")
IMAGE_ENV = frozenset({
    "PATH", "LANG", "GOSU_VERSION", "PG_MAJOR", "PG_VERSION", "PGDATA", "REDIS_VERSION",
    "REDIS_DOWNLOAD_URL", "REDIS_DOWNLOAD_SHA", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
    "PYTHON_VERSION", "PYTHON_SHA256", "GPG_KEY", "DEBIAN_FRONTEND", "UV_HTTP_TIMEOUT",
    "VIRTUAL_ENV", "HOME",
})
ALLOWED_MOUNT_DESTINATIONS = ("/var/lib/postgresql/data", "/data", "/seed", "/run/secrets/")
# Install-specific production surfaces outside Compose (the LAN database, the
# docker host, ...) come from the operator at run time, never from source:
# ``label=host:port`` entries separated by ``;``.
TARGETS_ENV = "LLM_BAWT_PREVIEW_PROBE_TARGETS"
# Ports prod services listen on without declaring them (e.g. the agent bridges).
PORTS_ENV = "LLM_BAWT_PREVIEW_PROBE_PORTS"
PROJECT_ENV = "COMPOSE_PROJECT_NAME"
DEFAULT_PROJECT = "llm-bawt"
# Generic: no route to the internet either.
INTERNET_TARGET = ("internet", "1.1.1.1", 443)
SERVICE_LABEL = "com.docker.compose.service"
PROJECT_LABEL = "com.docker.compose.project"
PREVIEW_NETWORK_NAMES = frozenset(topology.NETWORK_NAMES.values())


def _run(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), input=stdin, capture_output=True, text=True, check=False)


def _report(ok: bool, label: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {label}")
    return ok


def _project() -> str:
    return os.environ.get(PROJECT_ENV) or DEFAULT_PROJECT


def _project_containers() -> list[dict]:
    """Every container of the Compose project, running or not."""
    ids = _run("docker", "ps", "-aq", "--filter", f"label={PROJECT_LABEL}={_project()}").stdout.split()
    if not ids:
        return []
    return json.loads(_run("docker", "inspect", *ids).stdout)


def _service(c: dict) -> str:
    return (c["Config"].get("Labels") or {}).get(SERVICE_LABEL, "")


def _networks(c: dict) -> set[str]:
    return set((c["NetworkSettings"].get("Networks") or {}))


def parse_targets(value: str | None) -> list[tuple[str, str, int]]:
    """``label=host:port;...`` -> [(label, host, port)]; raises ValueError if malformed."""
    targets = []
    for entry in filter(None, (part.strip() for part in (value or "").split(";"))):
        label, sep, address = entry.partition("=")
        host, colon, port = address.rpartition(":")
        if not (sep and colon and label and host and port.isdigit()):
            raise ValueError(f"malformed probe target {entry!r}")
        targets.append((label, host, int(port)))
    return targets


def parse_ports(value: str | None) -> set[int]:
    """``8642,8681`` -> {8642, 8681}; raises ValueError if malformed."""
    parts = [part.strip() for part in (value or "").split(",") if part.strip()]
    if not all(part.isdigit() and 0 < int(part) < 65536 for part in parts):
        raise ValueError(f"malformed probe ports {value!r}")
    return {int(part) for part in parts}


def _declared_ports(c: dict) -> set[int]:
    keys = [*(c["Config"].get("ExposedPorts") or {}), *((c.get("HostConfig") or {}).get("PortBindings") or {})]
    return {int(k.split("/")[0]) for k in keys if k.endswith("/tcp")}


def prod_targets(containers: list[dict], extra_ports: set[int] = frozenset()) -> tuple[list[tuple[str, str, int]], list[str]]:
    """Every address of the project's non-preview containers x every known prod port.

    A container need not declare the port it listens on (the bridges EXPOSE
    nothing), so each address is probed on the union of every port any
    project container declares or binds plus ``extra_ports`` (operator
    config). Returns ``(targets, uncovered)``; ``uncovered`` names prod
    services that yielded no target at all, which the probe must fail.
    """
    prod = [c for c in containers if not topology.is_preview_service(_service(c))]
    ports = sorted(set(extra_ports).union(*(_declared_ports(c) for c in containers)))
    targets, uncovered = [], []
    for c in prod:
        service = _service(c)
        ips = [n.get("IPAddress") for n in (c["NetworkSettings"].get("Networks") or {}).values()]
        mine = [(f"prod {service}", ip, port) for ip in ips if ip for port in ports]
        if not mine:
            uncovered.append(service)
        targets.extend(mine)
    return targets, uncovered


def inventory_findings(containers: list[dict]) -> list[tuple[bool, str]]:
    """(ok, label) for every preview container and every prod container on a preview network.

    Preview containers are selected by service name, not by network, so a
    service that drifted off its networks is still inspected.
    """
    findings: list[tuple[bool, str]] = []
    preview = [c for c in containers if topology.is_preview_service(_service(c))]
    running = {_service(c) for c in preview if c["State"].get("Running")}
    missing = sorted(topology.REQUIRED_RUNNING - running)
    findings.append((not missing, f"required preview services running {'missing ' + str(missing) if missing else ''}"))
    for c in containers:
        service = _service(c)
        if topology.is_preview_service(service):
            continue
        allowed = {topology.NETWORK_NAMES[k] for k in topology.PROD_JOINERS.get(service, ())}
        stray = sorted((_networks(c) & PREVIEW_NETWORK_NAMES) - allowed)
        findings.append((not stray, f"{service}: off preview networks {stray or ''}"))
    for c in preview:
        findings.extend(_preview_container_findings(c))
    return findings


def _preview_container_findings(c: dict) -> list[tuple[bool, str]]:
    name = c["Name"].lstrip("/")
    expected = {topology.NETWORK_NAMES[k] for k in topology.networks_for(_service(c))}
    networks = _networks(c)
    findings = [(networks == expected, f"{name}: networks {sorted(networks)} == {sorted(expected)}")]
    ports = {k: v for k, v in (c["NetworkSettings"].get("Ports") or {}).items() if v}
    findings.append((not ports, f"{name}: no published ports {ports or ''}"))
    findings.append((not c["HostConfig"].get("Privileged"), f"{name}: unprivileged"))
    for m in c["Mounts"]:
        dest = m["Destination"]
        allowed = dest.startswith(ALLOWED_MOUNT_DESTINATIONS) and not m["Source"].endswith(".sock")
        if m["Type"] == "bind" and dest != "/seed" and not dest.startswith("/run/secrets/"):
            allowed = False
        if dest == "/seed":
            allowed &= not m.get("RW", True)
        findings.append((allowed, f"{name}: mount {m['Type']} -> {dest}"))
    env_names = sorted(e.split("=", 1)[0] for e in c["Config"]["Env"])
    bad = [n for n in env_names if n not in IMAGE_ENV and not n.startswith(ALLOWED_ENV_PREFIXES)]
    findings.append((not bad, f"{name}: env names {env_names}" + (f" unexpected {bad}" if bad else "")))
    inline = [n for n in env_names if "PASSWORD" in n and not n.endswith("_FILE")]
    findings.append((not inline, f"{name}: no inline password env {inline or ''}"))
    return findings


def inventory() -> bool:
    ok = True
    for passed, label in inventory_findings(_project_containers()):
        ok &= _report(passed, label)
    return ok


def seed_audit_queries() -> list[tuple[str, str]]:
    """(label, SQL returning a count that must be 0), derived from the allowlist."""
    # Schema-only tables hold exactly the rows the schema bootstrap wrote
    # (recorded by schema_init in preview_seed_meta.bootstrap_rows), never any
    # production row: the row digest must equal the recorded one ('' when the
    # bootstrap wrote none), so a replaced row fails as surely as an added one.
    checks = [(f"{t} bootstrap rows only",
               f"SELECT CASE WHEN ({row_digest_sql(t)}) = coalesce("
               f"(SELECT bootstrap_rows->>'{t}' FROM public.preview_seed_meta), '') THEN 0 ELSE 1 END")
              for t in sorted(SCHEMA_ONLY)]
    for table, rule in COPY.items():
        for column, treatment in rule.columns.items():
            if isinstance(treatment, Null):
                checks.append((f"{table}.{column} NULL",
                               f'SELECT count(*) FROM public.{table} WHERE "{column}" IS NOT NULL'))
            elif isinstance(treatment, Actor):
                checks.append((f"{table}.{column} scrubbed",
                               f"SELECT count(*) FROM public.{table} WHERE \"{column}\" IS NOT NULL "
                               f"AND \"{column}\" <> '{SEED_ACTOR}'"))
        if rule.bot_column:
            checks.append((f"{table} has no private bot",
                           f'SELECT count(*) FROM public.{table} WHERE "{rule.bot_column}" IN ({PRIVATE_BOTS_SQL})'))
    keys = lambda ks: ", ".join(f"'{k}'" for k in sorted(ks))  # noqa: E731
    checks.append(("runtime_settings keys allowlisted",
                   "SELECT count(*) FROM public.runtime_settings WHERE NOT ("
                   f"(scope_type = 'global' AND key IN ({keys(GLOBAL_SETTING_KEYS)})) OR "
                   f"(scope_type = 'bot' AND key IN ({keys(BOT_SETTING_KEYS)})))"))
    checks.append(("seed meta present", "SELECT 1 - count(*) FROM public.preview_seed_meta"))
    return checks


def seed() -> bool:
    ids = _run("docker", "ps", "-q", "--filter", f"label={DB_CONTAINER_FILTER}").stdout.split()
    if not _report(len(ids) == 1, "one preview-postgres container"):
        return False
    ok = True
    for label, sql in seed_audit_queries():
        # Local socket inside the container: no credential leaves it.
        out = _run("docker", "exec", ids[0], "psql", "-U", "preview", "-d", "preview", "-tAc", sql)
        ok &= _report(out.returncode == 0 and out.stdout.strip() == "0",
                      f"seed: {label}" + (f" ({out.stdout.strip() or out.stderr.strip()})" if out.stdout.strip() != "0" else ""))
    return ok


def network() -> bool:
    try:
        extra = parse_targets(os.environ.get(TARGETS_ENV))
        ports = parse_ports(os.environ.get(PORTS_ENV))
    except ValueError as exc:
        return _report(False, f"network: {exc}")
    ok = _report(bool(extra), f"network: {TARGETS_ENV} names the install's non-Compose prod surfaces")
    ok &= _report(bool(ports), f"network: {PORTS_ENV} names the undeclared prod ports")
    found, uncovered = prod_targets(_project_containers(), ports)
    ok &= _report(bool(found), f"network: production containers of project {_project()!r} found")
    ok &= _report(not uncovered, f"network: every prod container probed {uncovered or ''}")
    for label, host, port in [*found, *extra, INTERNET_TARGET]:
        # A throwaway container on the preview network: same reachability as
        # any preview service, without exec into one of them.
        probe = _run("docker", "run", "--rm", "--network", NETWORK, PROBE_IMAGE,
                     "nc", "-z", "-w", "3", host, str(port))
        ok &= _report(probe.returncode != 0, f"network: preview -/-> {label} {host}:{port}")
    roles = _run("pgq", "llm-bawt", "-tAc", "SELECT count(*) FROM pg_roles WHERE rolname = 'preview'")
    ok &= _report(roles.returncode == 0 and roles.stdout.strip() == "0",
                  "prod cluster has no preview role (zero grants)")
    return ok


PROBES = {"inventory": inventory, "seed": seed, "network": network}


def main(argv: list[str] | None = None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(PROBES)
    results = [PROBES[name]() for name in names]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
