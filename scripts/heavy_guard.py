"""Refuse broad checks in agent bridges, before discovery or worker startup."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys


def in_bridge(env=None):
    env = os.environ if env is None else env
    return env.get("BAWT_AGENT_BRIDGE") == "1" or bool(env.get("BRIDGE_BROKER_TOKEN_FILE"))


def queue_message(command):
    return ("Full checks cannot run inside an agent bridge. Submit on the test host:\n"
            '  heavy-queue submit --cwd "$PWD" --image <test-image> '
            '--notify-url <service-url> --notify-bot <requester> -- '
            + shlex.join(command)
            + "\nFocused explicit test/source files still run here. "
            "The exhaustive private-PG suite stays on make gate (no Docker socket in queue jobs).")


def focused_pytest(args, root, repository=None):
    """An explicit file/sub-node selection, never root/directory discovery.

    Parse known pytest options so their values cannot masquerade as files.
    Unknown switches conservatively consume a following non-switch operand.
    """
    values = {"-k", "-m", "-c", "-o", "--override-ini", "--rootdir", "--confcutdir",
              "--basetemp", "--ignore", "--ignore-glob", "--deselect", "-n",
              "--numprocesses", "--dist", "--maxfail", "--junitxml", "--junit-xml",
              "-p", "--tb", "--capture", "--show-capture", "--durations",
              "--durations-min", "--timeout"}
    booleans = {"-q", "-qq", "-v", "-vv", "-x", "-s", "--collect-only", "--co",
                "--strict-markers", "--strict-config", "--disable-warnings",
                "--require-no-skips", "--require-pg-acceptance", "--lf", "--ff"}
    selected = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg.startswith("-"):
            if "=" not in arg and (arg in values or arg not in booleans):
                skip = True
            continue
        file = arg.split("::", 1)[0]
        if not file.endswith(".py") or any(c in file for c in "*?[]"):
            return False
        path = Path(file)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            return False
        selected.append(path.resolve())
    if not selected:
        return False
    inventory = {path.resolve() for path in ((repository or root) / "tests").rglob("test_*.py")}
    return not inventory or not inventory.issubset(selected)


def guard_pytest(args, root, repository=None):
    if in_bridge() and not focused_pytest(args, root, repository):
        import pytest
        raise pytest.UsageError(queue_message(["python", "-m", "pytest", *args]))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    mode = args.pop(0)
    if args[:1] == ["--"]:
        args.pop(0)
    focused = False
    if mode == "runner" and "scripts.test_suite" in args:
        index = args.index("scripts.test_suite") + 1
        focused = args[index:index + 1] in (["target"], ["sdk-probe"])
    if in_bridge() and not focused:
        print(queue_message(args), file=sys.stderr)
        return 2
    os.execvp(args[0], args)


if __name__ == "__main__":
    raise SystemExit(main())
