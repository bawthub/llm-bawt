"""Dry-run an operation's approval binding (TASK-1639).

Runs exactly what an approval request runs before it opens a card (the ops
service's ``prepare_invocation``: argument validation, the config/source bind,
the source-published check, the GPU voice guard and house checks) and prints
every refusal it reports. It creates no job and opens no approval, so it can
run before a cutover window instead of discovering refusals inside it::

    docker exec llm-bawt-app python -m llm_bawt.ops.bind_preflight \\
        llm-bawt.reconcile-app-config [--args '{"name": "value"}']

Exit status: 0 when the binding would be prepared, 1 on a refusal, 2 on usage.
"""
from __future__ import annotations

import argparse
import json
import sys

from .control_client import OpsControlClient
from .service import OpsDispatchError
from .source_state import REFUSAL_SEPARATOR


def run(operation: str, args: dict | None, prepare, out=sys.stdout) -> int:
    try:
        snapshot = prepare(operation, args)
    except OpsDispatchError as exc:
        refusals = [part for part in str(exc).split(REFUSAL_SEPARATOR) if part] or [exc.code]
        print(f"REFUSED {operation} ({exc.code}): {len(refusals)} refusal(s)", file=out)
        for refusal in refusals:
            print(f"  - {refusal}", file=out)
        return 1
    print(f"OK {operation}: the binding would be prepared; nothing was created", file=out)
    config = snapshot.get("config") if isinstance(snapshot, dict) else None
    if isinstance(config, dict) and config.get("drift") is not None:
        print(f"  drift: {', '.join(config['drift']) or 'none'}", file=out)
    if isinstance(snapshot, dict) and snapshot.get("gpu_target"):
        print(f"  GPU voice target: {snapshot['gpu_target']}", file=out)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bind_preflight", description=__doc__.split("\n\n")[0])
    parser.add_argument("operation", help="operation slug, e.g. llm-bawt.reconcile-app-config")
    parser.add_argument("--args", dest="args", default=None, help="operation arguments as a JSON object")
    ns = parser.parse_args(argv)
    try:
        args = None if ns.args is None else json.loads(ns.args)
    except ValueError:
        parser.error("--args must be JSON")
    if not (args is None or isinstance(args, dict)):
        parser.error("--args must be a JSON object")
    from ..utils.config import Config  # env_import: only when run as a command

    client = OpsControlClient(Config().OPS_CONTROL_URL)
    return run(ns.operation, args, client.prepare)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
