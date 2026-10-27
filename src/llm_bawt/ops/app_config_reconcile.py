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
else:  # standalone immutable worker
    from compose_config import BUILT_PROFILE, evaluate, port_list
    from compose_render import OPS_JOB_LABEL, SAFE_PATH, label_files_valid
    from config_reconcile import (ConfigReconciler, check_compose_identity, current_config_hash,
                                  valid_hashes, validate_renderer_image, validate_target_fields)
    from image_deploy import StepError

ACTION = "reconcile_app_config"
SCHEMA = "llm-bawt.ops.app-config-reconcile/v1"
# Worker image label proving it carries this module. Without it the op refuses.
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.app-config-reconcile"
_PATTERNS = {
    "container_name": r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}",
    "compose_project": r"[a-z0-9][a-z0-9_-]{0,62}",
    "compose_service": r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}",
    "health_url": r"http://127\.0\.0\.1:[1-9][0-9]{0,4}/[A-Za-z0-9._/-]{0,128}",
    "health_status": r"[A-Za-z0-9._-]{1,32}",
}
_SPEC_KEYS = frozenset({"action", "compose_config", "display_values", "stat_paths", "stop_grace_seconds",
                        "health_timeout_seconds", *_PATTERNS})
_BINDING_KEYS = frozenset({"container_id", "image_id", "image_ref", "image_tag", "image_tag_moved", "source",
                           "current_config_hash", "desired_hash", "rendered_hash", "drift", "mounts",
                           "ports", "depends_on", "display"})


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
    return spec


def image_tag(spec: dict) -> str:
    """The tag Compose gives the image it builds for this service."""
    return f"{spec['compose_project']}-{spec['compose_service']}"


def app_config_binding(spec: dict, container: dict, image: dict, renderer, *,
                       image_tag_moved: bool) -> tuple[dict, dict]:
    """``(public approval binding, private desired config)``; raises ValueError.

    ``image_tag_moved`` is resolved by the caller: the app at approval; the
    worker carries the approved value forward and never resolves the tag."""
    validate_app_spec(spec)
    check_compose_identity(spec, container)
    ref, tag = container["Config"].get("Image", ""), image_tag(spec)
    if image["Id"] != container["Image"] or ref not in (tag, image["Id"]):
        raise ValueError("app config target must run its own Compose-built image")
    result = evaluate(spec, container, image, renderer, ref, BUILT_PROFILE)
    desired = result["desired"]
    binding = {"container_id": container["Id"], "image_id": image["Id"], "image_ref": ref,
               "image_tag": tag, "image_tag_moved": bool(image_tag_moved), "source": result["source"],
               "current_config_hash": current_config_hash(container, image),
               "desired_hash": result["desired_hash"], "rendered_hash": result["rendered_hash"],
               "drift": result["drift"], "mounts": result["mounts"], "ports": port_list(desired["ports"]),
               "depends_on": [f"{name}:{cond}" for name, cond in desired["depends_on"].items()],
               "display": result["display"]}
    return binding, desired


def valid_app_binding(snapshot: dict) -> bool:
    binding = snapshot.get("config")
    if snapshot["resolved_args"] != {} or not isinstance(binding, dict) or set(binding) != _BINDING_KEYS:
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
                and set(binding["display"]) == set(spec["display_values"]))
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def approval_text(spec: dict, binding: dict, prefix: str | None) -> str:
    """Approval card body: names, ports, mounts and A1 enum values only."""
    drift = ", ".join(binding["drift"]) or "none (verifies only)"
    shown = ", ".join(f"{k}={v}" for k, v in sorted(binding["display"].items())) or "none"
    moved = (f"Note: tag {binding['image_tag']} now names a different local build. This keeps the RUNNING "
             "image; rebuilding and redeploying is separate.\n") if binding["image_tag_moved"] else ""
    return (f"{prefix or 'Authorize same-image app configuration reconcile?'}\n"
            f"Target: {spec['container_name']} ({binding['container_id']})\n"
            f"Unchanged image: {binding['image_id']} (Compose tag {binding['image_tag']})\n{moved}"
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
    DETAIL_KEYS = (*ConfigReconciler.DETAIL_KEYS, "ports", "depends_on", "image_tag_moved")

    def _valid_binding(self) -> bool:
        return valid_app_binding(self.snapshot)

    def _bind(self, old, image, binding):
        return app_config_binding(self.spec, old, image, self._renderer(),
                                  image_tag_moved=binding["image_tag_moved"])

    def _release(self, binding):
        return None  # a local build carries no release identity

    def _swap_ref(self, binding):
        return binding["image_id"]  # never the tag: it may name another build

    def _before_swap(self):
        self._phase("dependencies")
        for service, condition in self._desired["depends_on"].items():
            self._check_dependency(service, condition)

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


__all__ = ["ACTION", "SCHEMA", "WORKER_CAPABILITY_LABEL", "AppConfigReconciler", "app_config_binding",
           "approval_text", "image_tag", "valid_app_binding", "validate_app_spec"]
