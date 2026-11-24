"""TASK-2590: the reader broker export must not break the app's same-image reconcile.

Runs the actual reconcile source classification (``source_plan``) on the app
row. The binding stand-in classifies exactly what the default app service
declares today, as an installed binding does; the reader overlay then adds one
mount and one env name, and nothing else is newly accepted.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from llm_bawt.ops.source_state import (ENV_NAME, PRODUCT_ENV, reserved_env_name, SourceStateError, SourceStateRefusals, source_plan,
                                       validate_source_spec)
from tests.ops_install_rows import SEEDS

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "/srv/example/llm-bawt"
READER_MOUNT = "/run/llm-bawt-reader-broker-token"
READER_ENV = "BRIDGE_BROKER_TOKEN_READER_EXPORT_FILE"


def _app(path):
    return yaml.safe_load((ROOT / path).read_text())["services"]["app"]


def _env(service):
    env = service.get("environment") or []
    if isinstance(env, dict):
        return {k: "" if v is None else str(v) for k, v in env.items()}
    # Conditional ``${A+A=}${A-...}`` entries resolve to names classified the
    # same way on both sides; only literal names matter for this delta.
    return {e.partition("=")[0]: e.partition("=")[2] for e in env if not e.startswith("${")}


def _targets(service):
    out = []
    for v in service.get("volumes") or []:
        if not isinstance(v, str):
            out.append(v["target"])
            continue
        # The source may hold ``${NAME:-default}``; the target is the first
        # ``:/``-led field after it (an optional mode follows).
        rest = v.rsplit("}", 1)[-1] if "}" in v else v
        out.append(rest.split(":")[1])
    return out


BASE = _app("docker-compose.yml")
OVERLAY = _app("docker-compose.reader-broker.yml")


def _binding(extra_mounts=(), extra_env=()):
    spec = json.loads(next(s for s in SEEDS if s["slug"] == "llm-bawt.reconcile-app-config")["command_script"])
    data = [t for t in _targets(BASE) if t not in spec["source_mounts"]]
    # An installed row predating this change: the seed's own new entry removed.
    installed = set(spec["data_mounts"]) - {READER_MOUNT}
    spec["data_mounts"] = sorted(installed | (set(data) - {READER_MOUNT}) | set(extra_mounts))
    spec["env_names"] = sorted({n for n in _env(BASE) if n not in PRODUCT_ENV and ENV_NAME.fullmatch(n)
                                and not reserved_env_name(n)} | set(extra_env))
    validate_source_spec(spec)
    return spec


def _codes(spec, service_env, targets):
    desired = {"mounts": {t: {"type": "bind", "read_only": False,
                              "source": f"{PROJECT}/{t.rsplit('/', 1)[-1]}" if t.startswith("/app/") and t in
                              spec["source_mounts"] else "/srv/data"} for t in targets},
               "env": service_env, "tmpfs": {}}
    try:
        source_plan(spec, {"project_dir": PROJECT, "compose_files": [f"{PROJECT}/docker-compose.yml"]},
                    {}, {}, desired)
    except SourceStateRefusals as caught:
        return {r.code: r.names for r in caught.refusals}
    except SourceStateError as caught:
        return {caught.code: caught.names}
    return {}


def _classify(spec, env, targets):
    codes = _codes(spec, env, targets)
    return codes.get("source_mount_unclassified", []), codes.get("source_env_unclassified", [])


def test_default_app_service_is_unchanged_and_stays_reconcilable():
    assert READER_ENV not in _env(BASE) and READER_MOUNT not in _targets(BASE)
    assert _classify(_binding(), _env(BASE), _targets(BASE)) == ([], [])


def test_enabled_overlay_needs_exactly_the_one_mount_and_env_name():
    env = {**_env(BASE), **_env(OVERLAY)}
    targets = _targets(BASE) + _targets(OVERLAY)
    assert _classify(_binding(), env, targets) == ([READER_MOUNT], [READER_ENV])
    assert _classify(_binding([READER_MOUNT], [READER_ENV]), env, targets) == ([], [])


def test_unrelated_unknown_mounts_and_env_are_still_refused():
    spec = _binding([READER_MOUNT], [READER_ENV])
    env = {**_env(BASE), **_env(OVERLAY), "SOMETHING_UNCLASSIFIED": "x"}
    targets = _targets(BASE) + _targets(OVERLAY) + ["/run/some-other-mount"]
    assert _classify(spec, env, targets) == (["/run/some-other-mount"], ["SOMETHING_UNCLASSIFIED"])
