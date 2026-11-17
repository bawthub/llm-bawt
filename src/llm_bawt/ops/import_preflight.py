"""Import a bridge's entrypoints from the source about to load (TASK-2403). STANDARD LIBRARY ONLY.

A source-mounted service loads whatever is in its bind mount when it starts, so
a reconcile or restart that stops the running process and then finds the new
source cannot import leaves a crash loop that no restore can undo (the mount is
not part of the container). Before anything is stopped, this imports each
configured entrypoint, plus every module that entrypoint's file names in any
``import`` statement (including ones inside ``main()``), in a throwaway sealed
container from the same image with the same source bound read-only. Any
failure refuses the operation with the traceback; the running service is
untouched.

Spec field (opt-in per operation)::

    "import_preflight": {"source_mount": "/app/src", "modules": ["pkg.__main__", ...]}
"""
from __future__ import annotations

import re

if __package__:
    from .sealed_container import SealedContainer, SealedRunError, demux_streams
else:  # pragma: no cover - standalone worker image
    from sealed_container import SealedContainer, SealedRunError, demux_streams

ROLE = "import-preflight"
# Worker image label: an older worker would skip the field (restart) or reject it mid-job.
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.import-preflight"
MAX_OUTPUT_BYTES = 256 * 1024
TRACEBACK_CHARS = 2000
_MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")
_PATH_RE = re.compile(r"/[A-Za-z0-9._/-]{1,255}")

# Runs inside the sealed container with the candidate source first on sys.path.
PROBE = r"""
import ast, importlib, importlib.util, sys, traceback
failed = 0
GUARDS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}

def _name(n):
    return n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else None

def _guarded(handler):
    # A try whose handler absorbs a failed import: its body is optional.
    t = handler.type
    if t is None:
        return True
    return any(_name(x) in GUARDS for x in (t.elts if isinstance(t, ast.Tuple) else [t]))

def _collect(node, names, package):
    # Required imports only: skip `if TYPE_CHECKING:` bodies and import-guarded try bodies;
    # their else/except/finally branches still count.
    if isinstance(node, ast.If) and _name(node.test) == "TYPE_CHECKING":
        children = node.orelse
    elif isinstance(node, ast.Try) and any(_guarded(h) for h in node.handlers):
        children = [*node.handlers, *node.orelse, *node.finalbody]
    else:
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            names.add(importlib.util.resolve_name(base, package) if node.level else base)
        children = list(ast.iter_child_nodes(node))
    for child in children:
        _collect(child, names, package)

for entry in sys.argv[1:]:
    try:
        mod = importlib.import_module(entry)
        with open(mod.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        names = set()
        _collect(tree, names, mod.__package__)
        for name in sorted(names):
            importlib.import_module(name)
        print(f"ok {entry} ({len(names)} imports)")
    except BaseException:
        failed = 1
        print(f"FAILED {entry}", file=sys.stderr)
        traceback.print_exc()
sys.exit(failed)
"""


class ImportPreflightError(RuntimeError):
    pass


def validate(value) -> dict:
    if (not isinstance(value, dict) or set(value) != {"source_mount", "modules"}
            or not isinstance(value["source_mount"], str) or not _PATH_RE.fullmatch(value["source_mount"])
            or not isinstance(value["modules"], list) or not value["modules"]
            or not all(isinstance(m, str) and _MODULE_RE.fullmatch(m) for m in value["modules"])):
        raise ValueError("import_preflight must be {source_mount: absolute path, modules: [module names]}")
    return value


def mount_source(container: dict, destination: str) -> str | None:
    """Host path bound at ``destination`` in an inspected container."""
    for mount in container.get("Mounts") or []:
        if mount.get("Destination") == destination and mount.get("Type") == "bind":
            return mount.get("Source")
    return None


def run(api, config: dict, *, image: str, source_dir: str | None, tag: str) -> str:
    """Return a one-line success summary or raise :class:`ImportPreflightError`."""
    validate(config)
    if not source_dir or not _PATH_RE.fullmatch(source_dir):
        raise ImportPreflightError(f"import_preflight: no bind source at {config['source_mount']}")
    sealed = SealedContainer(api, image, role=ROLE, tag=tag, labels={})
    try:
        status, raw = sealed.run(entrypoint=["python", "-c", PROBE], cmd=list(config["modules"]),
                                 env=[f"PYTHONPATH={source_dir}", "PYTHONDONTWRITEBYTECODE=1"],
                                 directory=source_dir, max_bytes=MAX_OUTPUT_BYTES)
    except SealedRunError as exc:
        raise ImportPreflightError(f"import_preflight_unavailable ({exc.kind})") from None
    out, err = (part.decode("utf-8", "replace") for part in demux_streams(raw))
    if status != 0:
        tail = err.strip()[-TRACEBACK_CHARS:] or "no traceback"
        raise ImportPreflightError(f"import_preflight_failed (exit {status}):\n{tail}")
    return "import preflight ok: " + "; ".join(line for line in out.splitlines() if line.startswith("ok "))


__all__ = ["ImportPreflightError", "PROBE", "WORKER_CAPABILITY_LABEL", "mount_source", "run", "validate"]
