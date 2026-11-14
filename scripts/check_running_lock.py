"""Fail when a running container's installed packages differ from uv.lock (TASK-1270).

Operator tooling, run on the Docker host as a window postflight after the app
image is recreated. It reads the container's installed distributions with
``docker exec <container> python`` and the committed ``uv.lock``; it changes
nothing. Exit 0 when every installed distribution is in the lock at the locked
version, 1 on any difference (report on stdout), 2 when it cannot read either side.

Lock packages that are not installed (other platforms, extras or groups) are
not a difference. The project itself is skipped: it is installed from source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

SCHEMA = "llm-bawt.running-lock-check/v1"
PROJECT = "llm-bawt"
_LIST = ("import importlib.metadata as m, json; "
         "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() if d.metadata['Name']}))")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_versions(text: str) -> dict[str, set[str]]:
    """Every locked version per package (a lock may hold more than one under markers)."""
    versions: dict[str, set[str]] = {}
    for package in tomllib.loads(text).get("package", []):
        if "version" in package:
            versions.setdefault(normalize(package["name"]), set()).add(package["version"])
    return versions


def compare(installed: dict[str, str], locked: dict[str, set[str]]) -> dict:
    mismatched, unlocked = {}, []
    for raw, version in sorted(installed.items()):
        name = normalize(raw)
        if name == PROJECT:
            continue
        if name not in locked:
            unlocked.append(f"{name}=={version}")
        elif version not in locked[name]:
            mismatched[name] = {"running": version, "lock": sorted(locked[name])}
    return {"schema": SCHEMA, "ok": not mismatched and not unlocked,
            "checked": len(installed), "mismatched": mismatched, "not_in_lock": unlocked}


def installed_in(container: str, python: str) -> dict[str, str]:
    out = subprocess.run(["docker", "exec", container, python, "-c", _LIST],
                         check=True, capture_output=True, text=True, timeout=60).stdout
    return json.loads(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--container", default="llm-bawt-app")
    parser.add_argument("--python", default="/app/.venv/bin/python")
    parser.add_argument("--lock", type=Path, required=True, help="uv.lock of the commit the image was built from")
    args = parser.parse_args(argv)
    try:
        locked = lock_versions(args.lock.read_text())
        installed = installed_in(args.container, args.python)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 2
    report = compare(installed, locked)
    report["container"] = args.container
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
