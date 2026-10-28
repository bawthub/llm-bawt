"""Remove ONE stopped Compose orphan container. STANDARD LIBRARY ONLY (TASK-1439 T2c).

A container stays on the daemon after its service leaves the Compose files, and
nothing in ops could delete it. This action deletes exactly one such container
and nothing else: no recreate, restart, network, volume or image call.

It refuses (fixed codes, known outcome, nothing touched) unless all of these hold,
checked at approval and again in the worker under the target lock right before
the DELETE:

* the target is the approved container (same Id), not an ops worker/renderer;
* it is stopped (``exited``, ``created`` or ``dead``);
* its Compose labels name the catalog row's project and a safe working dir;
* the WHOLE project, rendered from the row's current file set with every
  profile, has no service claiming it by service key, ``container_name`` or
  Compose's default ``<project>-<service>-<n>`` name. Any render failure refuses.

The allowlist is the catalog row's args enum; the project and its overlay files
are row data too, so no install's names live in source. Docker itself refuses
(409) a container started between the check and the un-forced DELETE.
Other Compose projects on the host cannot be enumerated: a different project
redeclaring the same ``container_name`` is outside this check, and the enum is
the operator's guard for it.
"""
from __future__ import annotations

import fcntl
import json
import posixpath
import re
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

if __package__:
    from .compose_render import (COMPOSE_FILE, MAX_COMPOSE_FILES, RENDER_LABEL, ComposeError, ComposeRenderer,
                                 OPS_JOB_LABEL, validate_interpolation_env, working_dir_valid)
    from .config_reconcile import CONTAINER_NAME_PATTERN, validate_renderer_image
    from .service_identity import PROJECT_LABEL, PROJECT_PATTERN, SERVICE_LABEL
else:  # standalone immutable worker
    from compose_render import (COMPOSE_FILE, MAX_COMPOSE_FILES, RENDER_LABEL, ComposeError, ComposeRenderer,
                                OPS_JOB_LABEL, validate_interpolation_env, working_dir_valid)
    from config_reconcile import CONTAINER_NAME_PATTERN, validate_renderer_image
    from service_identity import PROJECT_LABEL, PROJECT_PATTERN, SERVICE_LABEL

ACTION = "remove"
SCHEMA = "llm-bawt.ops.container-remove/v1"
# Worker image label proving it carries this module. An older worker would
# refuse the unknown action mid-job; the app refuses before approval instead.
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.container-remove"
REMOVABLE_STATES = frozenset({"exited", "created", "dead"})
_WORKING_DIR = "com.docker.compose.project.working_dir"
_ARG_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_SPEC_KEYS = frozenset({"action", "container_name_from_arg", "compose_project", "compose_config"})
BINDING_KEYS = frozenset({"container_id", "name", "compose_project", "project_dir", "compose_files", "status"})


class RemoveRefused(RuntimeError):
    """A refusal with a fixed code. Raised before the DELETE, or for a DELETE
    Docker refused, so the outcome is always known: nothing was removed."""

    outcome_known = True

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def validate_remove_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS:
        raise ValueError(f"remove spec requires exactly {sorted(_SPEC_KEYS)} (no force, volume or link flags)")
    if spec["action"] != ACTION:
        raise ValueError("remove spec action must be 'remove'")
    if not isinstance(spec["container_name_from_arg"], str) or not _ARG_NAME.fullmatch(spec["container_name_from_arg"]):
        raise ValueError("remove spec container_name_from_arg is invalid")
    if not isinstance(spec["compose_project"], str) or not re.fullmatch(PROJECT_PATTERN, spec["compose_project"]):
        raise ValueError("remove spec compose_project is invalid")
    config = spec["compose_config"]
    required = {"compose_files", "renderer_image"}
    if not isinstance(config, dict) or not required <= set(config) <= required | {"interpolation_env"}:
        raise ValueError("remove compose_config requires exactly compose_files and renderer_image"
                         " (and optional interpolation_env)")
    files = config["compose_files"]
    if (not isinstance(files, list) or not 1 <= len(files) <= MAX_COMPOSE_FILES or len(set(files)) != len(files)
            or not all(isinstance(f, str) and COMPOSE_FILE.fullmatch(f) for f in files)):
        raise ValueError("remove compose_config.compose_files must be 1..4 distinct Compose file basenames")
    validate_renderer_image(config)
    validate_interpolation_env(config)
    return spec


def validate_remove_schema(spec: dict, schema: dict) -> None:
    """The row's args enum IS the allowlist: required, closed, nonempty, names only."""
    arg = spec["container_name_from_arg"]
    prop = (schema.get("properties") or {}).get(arg) or {}
    enum = prop.get("enum")
    if (schema.get("additionalProperties") is not False or arg not in (schema.get("required") or [])
            or prop.get("type") != "string" or not isinstance(enum, list) or not enum
            or not all(isinstance(n, str) and re.fullmatch(CONTAINER_NAME_PATTERN, n) for n in enum)):
        raise ValueError("remove requires a required, closed string argument whose enum lists the "
                         "removable container names")


def target_name(spec: dict, args: dict) -> str:
    name = args.get(spec["container_name_from_arg"])
    if not isinstance(name, str) or not re.fullmatch(CONTAINER_NAME_PATTERN, name):
        raise RemoveRefused("args_invalid", "container name argument is invalid")
    return name


def _claims(project: str, name: str, label_service: str | None, services: dict) -> list[str]:
    claiming = []
    for key, service in services.items():
        if not isinstance(key, str) or not isinstance(service, dict):
            raise RemoveRefused("compose_unverifiable", "rendered services are malformed")
        explicit = service.get("container_name")
        if explicit is not None and not isinstance(explicit, str):
            raise RemoveRefused("compose_unverifiable", f"service {key} container_name is not a string")
        default = re.fullmatch(re.escape(project) + "[-_]" + re.escape(key) + r"[-_][0-9]+", name)
        if key == label_service or explicit == name or default:
            claiming.append(key)
    return sorted(claiming)


def removal_binding(spec: dict, name: str, container: dict, renderer: ComposeRenderer) -> dict:
    """Every refusal before a DELETE, read-only. Returns the approval binding."""
    labels = (container.get("Config") or {}).get("Labels") or {}
    if OPS_JOB_LABEL in labels or RENDER_LABEL in labels:
        raise RemoveRefused("ops_container", "ops workers and renderers are never removal targets")
    state = container.get("State") or {}
    status = state.get("Status")
    if state.get("Running") or state.get("Restarting") or state.get("Paused") or status not in REMOVABLE_STATES:
        raise RemoveRefused("target_running", f"container state is {status!r}; only a stopped container is removed")
    project, working_dir = labels.get(PROJECT_LABEL), labels.get(_WORKING_DIR)
    if not project or not working_dir:
        raise RemoveRefused("compose_labels_missing",
                            "no Compose project/working-dir labels; cannot prove no service declares it")
    if project != spec["compose_project"]:
        raise RemoveRefused("compose_project_mismatch", "the container's Compose project is not the row's")
    if not working_dir_valid(working_dir):
        raise RemoveRefused("compose_source_invalid", "working directory label")
    files = [posixpath.join(working_dir, f) for f in spec["compose_config"]["compose_files"]]
    try:
        rendered = renderer.project({"project_dir": working_dir, "compose_files": files}, project)
    except ComposeError as exc:  # fixed code and names only
        raise RemoveRefused("compose_unverifiable", exc.code) from None
    claiming = _claims(project, name, labels.get(SERVICE_LABEL), rendered["services"])
    if claiming:
        raise RemoveRefused("compose_declared", "still declared by service " + ", ".join(claiming))
    return {"container_id": container["Id"], "name": name, "compose_project": project,
            "project_dir": working_dir, "compose_files": files, "status": status}


def valid_remove_binding(snapshot: dict) -> bool:
    """A stored snapshot carries a removal binding exactly when its spec is a
    removal, and that binding names the approved args and the row's source."""
    binding = snapshot.get("remove")
    if snapshot["spec"].get("action") != ACTION:
        return binding is None
    if not isinstance(binding, dict) or set(binding) != BINDING_KEYS:
        return False
    try:
        spec = validate_remove_spec(snapshot["spec"])
        return (isinstance(binding["container_id"], str)
                and bool(re.fullmatch(r"[0-9a-f]{64}", binding["container_id"]))
                and binding["name"] == target_name(spec, snapshot["resolved_args"])
                and binding["compose_project"] == spec["compose_project"]
                and working_dir_valid(binding["project_dir"])
                and binding["compose_files"] == [posixpath.join(binding["project_dir"], f)
                                                 for f in spec["compose_config"]["compose_files"]]
                and binding["status"] in REMOVABLE_STATES)
    except (ValueError, KeyError, TypeError, RemoveRefused):
        return False


def approval_text(binding: dict, prompt: str | None) -> str:
    return (f"{prompt or 'Remove a stopped container?'}\n"
            f"Target: {binding['name']} ({binding['container_id'][:12]}, {binding['status']})\n"
            f"Not declared in Compose project {binding['compose_project']} at {binding['project_dir']}: "
            + ", ".join(posixpath.basename(f) for f in binding["compose_files"]) + "\n"
            "Deletes this one container only (volumes kept). Re-checked right before the delete; "
            "a started, replaced or re-declared container refuses.")


class ContainerRemover:
    def __init__(self, api, snapshot: dict, job_id: str, *, lock_root: Path, renderer=None):
        if not valid_remove_binding(snapshot):  # never trust the binding's shape
            raise RemoveRefused("args_invalid", "invalid removal approval binding")
        self.api = api
        self.spec = snapshot["spec"]
        self.binding = snapshot["remove"]
        self.job_id = job_id
        self.lock_root = lock_root
        self.renderer = renderer or ComposeRenderer(api, self.spec, tag="rm" + job_id[:12])

    @contextmanager
    def _target_lock(self):
        """The same per-name lock file deploys take, so a deploy and a removal
        of one name never interleave. Busy fails known with nothing changed."""
        self.lock_root.mkdir(mode=0o700, exist_ok=True)
        with (self.lock_root / f"{self.binding['name']}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RemoveRefused("target_busy", "another job holds this target") from None
            yield

    def run(self) -> str:
        name = self.binding["name"]
        with self._target_lock():
            try:
                container = self.api.request("GET", f"/containers/{quote(name, safe='')}/json")
            except RuntimeError as exc:
                if "HTTP 404" in str(exc):
                    raise RemoveRefused("not_found", "the container no longer exists") from None
                raise
            if container.get("Id") != self.binding["container_id"]:
                raise RemoveRefused("target_replaced", "the name now belongs to a different container")
            removal_binding(self.spec, name, container, self.renderer)
            # Handed to Docker: a lost reply may still have removed it.
            self.api.side_effect_started = True
            try:
                self.api.request("DELETE", f"/containers/{quote(container['Id'], safe='')}"
                                           "?force=false&v=false&link=false")
            except RuntimeError as exc:
                if "HTTP 409" in str(exc):
                    raise RemoveRefused("target_running", "Docker refused: the container is running") from None
                if "HTTP 404" in str(exc):
                    raise RemoveRefused("not_found", "the container was already gone") from None
                raise
        return json.dumps({"schema": SCHEMA, "job_id": self.job_id, "removed": container["Id"], "name": name},
                          sort_keys=True)
