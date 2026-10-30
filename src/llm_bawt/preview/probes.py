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
import subprocess
import sys

from llm_bawt.preview.seed_allowlist import (
    BOT_SETTING_KEYS, COPY, GLOBAL_SETTING_KEYS, PRIVATE_BOTS_SQL, SCHEMA_ONLY, SEED_ACTOR,
    Actor, Null,
)

NETWORK = "llm-bawt-preview"
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
# (label, host, port): production surfaces the preview must never reach.
TARGETS = (
    ("prod postgres", "10.0.2.32", 5432),
    ("prod redis", "llm-bawt-redis", 6379),
    ("prod app", "llm-bawt-app", 8642),
    ("prod mcp", "llm-bawt-app", 8001),
    ("claude bridge", "llm-bawt-claude-code-bridge", 8681),
    ("codex bridge", "llm-bawt-codex-bridge", 8682),
    ("echo host", "172.18.0.1", 22),
    ("internet", "1.1.1.1", 443),
)


def _run(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), input=stdin, capture_output=True, text=True, check=False)


def _report(ok: bool, label: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {label}")
    return ok


def _containers() -> list[dict]:
    ids = _run("docker", "ps", "-q", "--filter", "label=com.docker.compose.project",
               "--filter", f"network={NETWORK}").stdout.split()
    if not ids:
        return []
    return json.loads(_run("docker", "inspect", *ids).stdout)


def inventory() -> bool:
    containers = _containers()
    ok = _report(bool(containers), f"preview containers running on {NETWORK}")
    for c in containers:
        name = c["Name"].lstrip("/")
        networks = sorted(c["NetworkSettings"]["Networks"])
        ok &= _report(networks == [NETWORK], f"{name}: networks {networks}")
        ports = {k: v for k, v in (c["NetworkSettings"].get("Ports") or {}).items() if v}
        ok &= _report(not ports, f"{name}: no published ports {ports or ''}")
        ok &= _report(not c["HostConfig"].get("Privileged"), f"{name}: unprivileged")
        for m in c["Mounts"]:
            dest = m["Destination"]
            allowed = dest.startswith(ALLOWED_MOUNT_DESTINATIONS) and not m["Source"].endswith(".sock")
            if m["Type"] == "bind" and dest not in ("/seed",) and not dest.startswith("/run/secrets/"):
                allowed = False
            if dest == "/seed":
                allowed &= not m.get("RW", True)
            ok &= _report(allowed, f"{name}: mount {m['Type']} -> {dest}")
        env_names = sorted(e.split("=", 1)[0] for e in c["Config"]["Env"])
        bad = [n for n in env_names if n not in IMAGE_ENV and not n.startswith(ALLOWED_ENV_PREFIXES)]
        ok &= _report(not bad, f"{name}: env names {env_names}" + (f" unexpected {bad}" if bad else ""))
        inline = [n for n in env_names if "PASSWORD" in n and not n.endswith("_FILE")]
        ok &= _report(not inline, f"{name}: no inline password env {inline or ''}")
    return ok


def seed_audit_queries() -> list[tuple[str, str]]:
    """(label, SQL returning a count that must be 0), derived from the allowlist."""
    # Schema-only tables hold exactly the rows the schema bootstrap wrote
    # (recorded by schema_init in preview_seed_meta.bootstrap_rows), never any
    # production row. A table the bootstrap did not create counts as 0;
    # query_to_xml defers parsing so a missing relation is not an error.
    checks = [(f"{t} bootstrap rows only",
               f"SELECT coalesce((SELECT (xpath('/row/c/text()', query_to_xml("
               f"'SELECT count(*) AS c FROM public.{t}', false, true, '')))[1]::text::int "
               f"WHERE to_regclass('public.{t}') IS NOT NULL), 0)"
               f" - coalesce((SELECT (bootstrap_rows->>'{t}')::int FROM public.preview_seed_meta), 0)")
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
    ok = True
    for label, host, port in TARGETS:
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
