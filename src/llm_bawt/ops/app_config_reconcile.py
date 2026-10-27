"""Same-image reconcile of a Compose-BUILT service (the llm-bawt app). STANDARD LIBRARY ONLY (TASK-1474).

The frontend reconcile (:mod:`config_reconcile`) recreates a digest-pinned
image. The app is built by Compose instead, so:

* the binding pins the running image ID and never re-resolves the Compose tag
  (``<project>-<service>``). A tag that now names another local build is shown
  on the approval card (``image_tag_moved``); it is never drift and never
  deployed. ``build`` is accepted and inert: a reconcile never builds or pulls;
* the per-kind :data:`BUILT_PROFILE` also models command/entrypoint, published
  ports, stdin_open/tty and container_name;
* ``depends_on`` is a precondition, not drift: right before the swap every
  dependency must be exactly one running (and, for ``service_healthy``,
  healthy) container of the same project. A dependency is only ever read;
* the compose files are the ones the target's own labels record (an install's
  overlay set is its own), validated and shown on the approval card;
* health is the app's own in-container health URL plus Docker health.

The target is the app that dispatched the job: recreating it cuts in-flight
chat turns and the MCP server. The worker is a separate container and the
ops-reconciler imports its receipt, so the outcome survives the app's restart.
"""
from __future__ import annotations

import json
from urllib.parse import urlencode

if __package__:
    from .compose_config import BUILT_PROFILE, evaluate, port_list
    from .compose_render import OPS_JOB_LABEL, SAFE_PATH, label_files_valid
    from .config_reconcile import (ConfigReconciler, check_compose_identity, current_config_hash,
                                   valid_hashes, validate_renderer_image, validate_target_fields)
    from .image_deploy import StepError
    from .source_state import (SOURCE_FIELDS, SourceChecker, SourceStateError, source_commit, source_plan,
                               valid_classification, valid_source_commit, validate_source_spec)
else:  # standalone immutable worker
    from compose_config import BUILT_PROFILE, evaluate, port_list
    from compose_render import OPS_JOB_LABEL, SAFE_PATH, label_files_valid
    from config_reconcile import (ConfigReconciler, check_compose_identity, current_config_hash,
                                  valid_hashes, validate_renderer_image, validate_target_fields)
    from image_deploy import StepError
    from source_state import (SOURCE_FIELDS, SourceChecker, SourceStateError, source_commit, source_plan,
                              valid_classification, valid_source_commit, validate_source_spec)

ACTION = "reconcile_app_config"
SCHEMA = "llm-bawt.ops.app-config-reconcile/v1"
# Worker image labels proving it carries this module and the source guard.
# Without both the op refuses: an older worker would recreate the app over
# unpublished source.
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.app-config-reconcile"
SOURCE_GUARD_LABEL = "llm-bawt.ops.app-source-guard"
_PATTERNS = {
    "container_name": r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}",
    "compose_project": r"[a-z0-9][a-z0-9_-]{0,62}",
    "compose_service": r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}",
    "health_url": r"http://127\.0\.0\.1:[1-9][0-9]{0,4}/[A-Za-z0-9._/-]{0,128}",
    "health_status": r"[A-Za-z0-9._-]{1,32}",
}
_SPEC_KEYS = frozenset({"action", "compose_config", "display_values", "stat_paths", "stop_grace_seconds",
                        "health_timeout_seconds", *SOURCE_FIELDS, *_PATTERNS})
_BINDING_KEYS = frozenset({"container_id", "image_id", "image_ref", "image_tag", "image_tag_moved", "source",
                           "current_config_hash", "desired_hash", "rendered_hash", "drift", "mounts",
                           "ports", "depends_on", "display", "source_commit", "classification"})
_CLASSES = ("source_mounts", "data_mounts", "opaque_env")


def validate_app_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS:
        raise ValueError("app config reconcile spec has missing or unsupported fields")
    if spec["action"] != ACTION:
        raise ValueError("not an app config reconcile spec")
    validate_target_fields(spec, _PATTERNS)
    config = spec["compose_config"]
    # No compose_files (the target's labels name them) and no image_variable
    # (Compose builds the image; nothing is interpolated).
    if not isinstance(config, dict) or set(config) != {"renderer_image"}:
        raise ValueError("app compose_config requires exactly renderer_image")
    validate_renderer_image(config)
    validate_source_spec(spec)
    return spec


def image_tag(spec: dict) -> str:
    """The tag Compose gives the image it builds for this service."""
    return f"{spec['compose_project']}-{spec['compose_service']}"


def app_config_binding(spec: dict, container: dict, image: dict, renderer, *,
                       image_tag_moved: bool, source_checker) -> tuple[dict, dict]:
    """``(public approval binding, private desired config)``; raises ValueError.

    ``image_tag_moved`` is resolved by the caller: the app at approval; the
    worker carries the approved value forward and never resolves the tag.
    ``source_checker`` proves the source mounts are exactly one commit
    (:mod:`source_state`); whether that commit is published is the app's
    check, made against the binding's ``source_commit``."""
    validate_app_spec(spec)
    check_compose_identity(spec, container)
    ref, tag = container["Config"].get("Image", ""), image_tag(spec)
    if image["Id"] != container["Image"] or ref not in (tag, image["Id"]):
        raise ValueError("app config target must run its own Compose-built image")
    result = evaluate(spec, container, image, renderer, ref, BUILT_PROFILE)
    desired = result["desired"]
    plan = source_plan(spec, result["source"], container, image, desired)
    head = source_checker.check(result["source"]["project_dir"], plan)
    binding = {"container_id": container["Id"], "image_id": image["Id"], "image_ref": ref,
               "image_tag": tag, "image_tag_moved": bool(image_tag_moved), "source": result["source"],
               "current_config_hash": current_config_hash(container, image),
               "desired_hash": result["desired_hash"], "rendered_hash": result["rendered_hash"],
               "drift": result["drift"], "mounts": result["mounts"], "ports": port_list(desired["ports"]),
               "depends_on": [f"{name}:{cond}" for name, cond in desired["depends_on"].items()],
               "display": result["display"], "source_commit": source_commit(spec, head),
               "classification": plan["classification"]}
    return binding, desired


def newly_trusted(classification: dict, previous) -> dict | None:
    """What this approval trusts that the last successful run did not:
    ``None`` on a first run (no earlier classification to compare)."""
    # A malformed record proves nothing was trusted before, so it reads as a
    # first run: the card then shows everything as new, never less.
    if not isinstance(previous, dict) or not all(
            isinstance(previous.get(key), list) and all(isinstance(x, str) for x in previous[key])
            for key in _CLASSES):
        return None
    return {key: sorted(set(classification[key]) - set(previous[key])) for key in _CLASSES}


def _valid_newly_trusted(value, classification: dict) -> bool:
    return value is None or (
        isinstance(value, dict) and set(value) == set(_CLASSES)
        and all(isinstance(value[k], list) and value[k] == sorted(set(value[k]))
                and set(value[k]) <= set(classification[k]) for k in _CLASSES))


def snapshot_approval_text(snapshot: dict, prefix: str | None) -> str:
    """The approval card for a prepared invocation snapshot."""
    return approval_text(snapshot["spec"], snapshot["config"], prefix, operation=snapshot["operation"],
                         newly_trusted=snapshot["newly_trusted"])


def valid_app_binding(snapshot: dict) -> bool:
    binding = snapshot.get("config")
    if (snapshot["resolved_args"] != {} or not isinstance(binding, dict) or set(binding) != _BINDING_KEYS
            or "newly_trusted" not in snapshot):
        return False
    try:
        spec = validate_app_spec(snapshot["spec"])
        source = binding["source"]
        lists = binding["drift"] + binding["mounts"] + binding["ports"] + binding["depends_on"]
        return (valid_hashes(binding)
                and binding["image_tag"] == image_tag(spec)
                and binding["image_ref"] in (binding["image_tag"], binding["image_id"])
                and type(binding["image_tag_moved"]) is bool
                and set(source) == {"project_dir", "compose_files"}
                and bool(SAFE_PATH.fullmatch(source["project_dir"]))
                and label_files_valid(source["project_dir"], source["compose_files"])
                and all(isinstance(x, str) for x in lists)
                and set(binding["display"]) == set(spec["display_values"])
                and valid_source_commit(spec, binding["source_commit"])
                and valid_classification(spec, binding["classification"])
                and _valid_newly_trusted(snapshot["newly_trusted"], binding["classification"]))
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def _names(values) -> str:
    return ", ".join(values) or "none"


def _one_line(value, limit: int = 64) -> str:
    """A self-reported catalog field as one bounded line. Catalog writes are
    unauthenticated, so a newline or other control character in, say, the
    writer's actor name must not be able to add an apparent card line."""
    text = " ".join("".join(c if c.isprintable() else " " for c in str(value)).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def trust_text(binding: dict, operation: dict, newly) -> str:
    """The card's trust block (TASK-1474 design B2). Catalog writes are not
    authenticated, so the approver is shown, in the form the worker executes,
    every mount and environment name this run trusts, and first of all what
    the last successful run did not trust."""
    classes = binding["classification"]
    if newly is None:
        head = "FIRST RUN: every mount and environment name below is newly trusted."
    elif any(newly.values()):
        head = ("NEWLY TRUSTED since the last successful run: "
                + "; ".join(f"{label} {_names(newly[key])}" for key, label in (
                    ("data_mounts", "data mounts"), ("opaque_env", "opaque env"),
                    ("source_mounts", "source mounts")) if newly[key]))
    else:
        head = "Nothing newly trusted since the last successful run."
    who = _one_line(operation.get("updated_by") or "") or "unknown"
    return (f"{head}\n"
            f"Catalog revision: v{_one_line(operation.get('version'))}, "
            f"updated {_one_line(operation.get('updated_at'))} by {who} (self-reported)\n"
            f"Source mounts (verified against the commit): {_names(classes['source_mounts'])}\n"
            f"Data mounts (unchecked, writable): {_names(classes['data_mounts'])}\n"
            f"Opaque env ({len(classes['opaque_env'])} names, values never shown or checked): "
            f"{_names(classes['opaque_env'])}\n")


def approval_text(spec: dict, binding: dict, prefix: str | None, *, operation: dict, newly_trusted) -> str:
    """Approval card body: names, ports, mounts and A1 enum values only."""
    drift = ", ".join(binding["drift"]) or "none (verifies only)"
    shown = ", ".join(f"{k}={v}" for k, v in sorted(binding["display"].items())) or "none"
    moved = (f"Note: tag {binding['image_tag']} now names a different local build. This keeps the RUNNING "
             "image; rebuilding and redeploying is separate.\n") if binding["image_tag_moved"] else ""
    commit = binding["source_commit"]
    return (f"{prefix or 'Authorize same-image app configuration reconcile?'}\n"
            f"{trust_text(binding, operation, newly_trusted)}"
            f"Target: {spec['container_name']} ({binding['container_id']})\n"
            f"Unchanged image: {binding['image_id']} (Compose tag {binding['image_tag']})\n{moved}"
            f"Source: github.com/{commit['repository']} {commit['branch']} @ {commit['head']} "
            "(published, clean)\n"
            f"Compose source: {', '.join(binding['source']['compose_files'])}\n"
            f"Drift (names only): {drift}\n"
            f"Ports after: {', '.join(binding['ports']) or 'none'}\n"
            f"Requires running: {', '.join(binding['depends_on']) or 'nothing'} (never restarted)\n"
            f"Mounts after: {'; '.join(binding['mounts']) or 'none'}\n"
            f"Shown values: {shown}\n"
            f"Fingerprints: current {binding['current_config_hash'][:16]}, desired {binding['desired_hash'][:16]}\n"
            "No build or pull. Environment values other than those shown are never displayed. "
            "Recreating this target cuts in-flight chat turns and MCP calls; the outcome is recorded "
            "by the ops reconciler. Restores the previous container if health verification fails. "
            "Any change to the container or the rendered configuration cancels execution.")


class AppConfigReconciler(ConfigReconciler):
    """The frontend reconcile's lock, swap, health and restore, for a built image."""

    SCHEMA = SCHEMA
    ACTION_NAME = "reconcile-app-config"
    DETAIL_KEYS = (*ConfigReconciler.DETAIL_KEYS, "ports", "depends_on", "image_tag_moved", "source_commit",
                   "classification")

    def __init__(self, *args, source_checker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._source_checker_override = source_checker
        self._source_check = None

    def _source_checker(self):
        """Sealed git source check (TASK-1474), injectable in tests."""
        return self._source_checker_override or SourceChecker(
            self.api, self.spec["compose_config"]["renderer_image"], tag=self.job_id[:12])

    def _valid_binding(self) -> bool:
        return valid_app_binding(self.snapshot)

    def _bind(self, old, image, binding):
        checker = self._source_checker()
        actual, desired = app_config_binding(self.spec, old, image, self._renderer(),
                                             image_tag_moved=binding["image_tag_moved"], source_checker=checker)
        if actual["source_commit"] != binding["source_commit"]:
            raise SourceStateError("source_moved_since_approval", detail="the checkout is not the approved commit")
        self._source_check = (checker, actual["source"]["project_dir"],
                              source_plan(self.spec, actual["source"], old, image, desired))
        return actual, desired

    def _release(self, binding):
        return None  # a local build carries no release identity

    def _swap_ref(self, binding):
        return binding["image_id"]  # never the tag: it may name another build

    def _before_swap(self):
        self._phase("dependencies")
        for service, condition in self._desired["depends_on"].items():
            self._check_dependency(service, condition)
        # The render and the dependency reads take seconds, so the source is
        # checked once more as the last read before the old container stops.
        # What remains is the window from here to the new container importing
        # its modules (seconds, under the target lock): a writer to the
        # checkout in that window is not caught. Source edits that land after
        # the swap are the next reconcile's or deploy's concern, as before.
        self._phase("source")
        checker, project_dir, plan = self._source_check
        try:
            head = checker.check(project_dir, plan)
        except SourceStateError as exc:
            self._fail(f"{exc}; nothing changed", known=True)
        if head != self.snapshot["config"]["source_commit"]["head"]:
            self._fail("source_moved_since_approval; nothing changed", known=True)

    def _check_dependency(self, service, condition):
        filters = {"label": [f"com.docker.compose.project={self.spec['compose_project']}",
                             f"com.docker.compose.service={service}"]}
        rows = self.api.request("GET", "/containers/json?" + urlencode({"all": "1", "filters": json.dumps(filters)}))
        rows = [row for row in rows or [] if OPS_JOB_LABEL not in (row.get("Labels") or {})]
        if len(rows) != 1:
            self._fail(f"dependency_unresolved: {service} matched {len(rows)} containers; nothing changed", known=True)
        state = self.api.request("GET", f"/containers/{rows[0]['Id']}/json").get("State") or {}
        if not state.get("Running"):
            self._fail(f"dependency_not_running: {service}; nothing changed", known=True)
        if condition == "service_healthy" and (state.get("Health") or {}).get("Status") != "healthy":
            self._fail(f"dependency_unhealthy: {service}; nothing changed", known=True)

    def _verify_release(self, container_id, expect):
        code, out = self._exec(container_id, ["curl", "-fsS", "--max-time", "10", self.spec["health_url"]])
        if code != 0:
            raise StepError(f"in-container health request exited {code}")
        try:
            payload = json.loads(out)
        except ValueError:
            raise StepError("in-container health response is not JSON") from None
        status = payload.get("status") if isinstance(payload, dict) else None
        if status != self.spec["health_status"]:
            # The response body is untrusted: name only the expected value, never what came back.
            raise StepError(f"health status did not match expected {self.spec['health_status']!r}")
        return {"status": status, "stat": self._stats(container_id)}


__all__ = ["ACTION", "SCHEMA", "SOURCE_GUARD_LABEL", "WORKER_CAPABILITY_LABEL", "AppConfigReconciler",
           "app_config_binding",
           "approval_text", "image_tag", "newly_trusted", "snapshot_approval_text", "trust_text",
           "valid_app_binding", "validate_app_spec"]
