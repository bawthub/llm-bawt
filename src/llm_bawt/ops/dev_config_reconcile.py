"""Same-image reconcile of a Compose-built DEV service. STANDARD LIBRARY ONLY (TASK-1649).

Target: the BawtHub hot-reload frontend (Vite/HMR + Hono under ``pnpm dev``).
It is Compose-built like the app, but the app kind does not fit it:

* its source guard (:mod:`source_state`) proves the bind-mounted source is a
  published commit. A dev server serves its LIVE working tree by design and
  the HMR watcher reloads every edit without a recreate, so the recreate does
  not change which source runs and the guard would only refuse (dependency
  volumes nested under the source mount, the Node image's ``NODE_*`` runtime
  variables, a working tree that is dirty while bots edit it);
* it declares a memory cap (``mem_limit``/``memswap_limit``), which this kind
  models (:data:`DEV_PROFILE`) and the recreate applies.

So this kind binds the RUNNING image ID (never the tag, which a later build may
move), renders the target's own Compose service with :data:`DEV_PROFILE`,
checks ``depends_on`` as a precondition, waits for the declared Docker
healthcheck and verifies the target's in-container JSON health URL
(:class:`ServiceConfigReconciler`). It has no source check, so it is confined
to dev targets (Codex r1 Blocker), at approval and again in the worker:

* the target's OWN rendered Compose service must declare the label
  :data:`DEV_TARGET_LABEL` (``"1"``). The marker lives in the reviewed Compose
  files, which a catalog edit cannot change; a production service never
  carries it, so repointing a row's spec at this action refuses;
* the action is approval-only (``OpsService.APPROVAL_ONLY_ACTIONS``): no
  direct dispatch, and the card says ``DEV target`` and names the marker.
"""
from __future__ import annotations

import re

if __package__:
    from .app_config_reconcile import (HEALTH_PATTERNS, ServiceConfigReconciler, image_refs, image_tag,
                                       valid_image_tag, validate_health_value, validate_label_compose_config)
    from .compose_config import DEV_PROFILE, evaluate, port_list
    from .compose_render import SAFE_PATH, ComposeError, label_files_valid
    from .config_reconcile import (CONTAINER_NAME_PATTERN, check_compose_identity, current_config_hash,
                                   valid_hashes, validate_target_fields)
    from .service_identity import SPEC_PATTERNS
else:  # standalone immutable worker
    from app_config_reconcile import (HEALTH_PATTERNS, ServiceConfigReconciler, image_refs, image_tag,
                                      valid_image_tag, validate_health_value, validate_label_compose_config)
    from compose_config import DEV_PROFILE, evaluate, port_list
    from compose_render import SAFE_PATH, ComposeError, label_files_valid
    from config_reconcile import (CONTAINER_NAME_PATTERN, check_compose_identity, current_config_hash,
                                  valid_hashes, validate_target_fields)
    from service_identity import SPEC_PATTERNS

ACTION = "reconcile_dev_config"
SCHEMA = "llm-bawt.ops.dev-config-reconcile/v1"
# Worker image label proving it carries this module and the memory profile.
# Without it dispatch refuses first: an older worker rejects the action.
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.dev-config-reconcile"
# The Compose label a service declares to be a dev target this kind may
# recreate without a source check. A first-party contract name.
DEV_TARGET_LABEL = "llm-bawt.ops.dev-target"
_PATTERNS = {
    "container_name": CONTAINER_NAME_PATTERN,
    **SPEC_PATTERNS,
    **HEALTH_PATTERNS,
}
_SPEC_KEYS = frozenset({"action", "compose_config", "display_values", "stat_paths", "stop_grace_seconds",
                        "health_timeout_seconds", "health_value", *_PATTERNS})
_BINDING_KEYS = frozenset({"container_id", "image_id", "image_ref", "image_tag", "image_tag_moved", "source",
                           "current_config_hash", "desired_hash", "rendered_hash", "drift", "mounts", "tmpfs",
                           "ports", "depends_on", "memory", "display"})
_MEMORY_TEXT = re.compile(r"unlimited|[0-9]{1,16} bytes, swap (?:unlimited|[0-9]{1,16} bytes)")


def validate_dev_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS:
        raise ValueError("dev config reconcile spec has missing or unsupported fields")
    if spec["action"] != ACTION:
        raise ValueError("not a dev config reconcile spec")
    validate_target_fields(spec, _PATTERNS)
    validate_health_value(spec)
    validate_label_compose_config(spec)
    return spec


def memory_text(memory: dict) -> str:
    """The cap as the card shows it; 0 is no limit, a swap of -1 unlimited swap."""
    if not memory["Memory"]:
        return "unlimited"
    swap = "unlimited" if memory["MemorySwap"] == -1 else f"{memory['MemorySwap']} bytes"
    return f"{memory['Memory']} bytes, swap {swap}"


def dev_config_binding(spec: dict, container: dict, image: dict, renderer, *, tag_moved) -> tuple[dict, dict]:
    """``(public approval binding, private desired config)``; raises ValueError.

    ``tag_moved(tag) -> bool`` is resolved by the caller: the app inspects the
    tag at approval; the worker carries the approved value forward and never
    resolves it."""
    validate_dev_spec(spec)
    check_compose_identity(spec, container)
    ref = container["Config"].get("Image", "")
    if image["Id"] != container["Image"]:
        raise ValueError("dev config target must run its own Compose-built image")
    result = evaluate(spec, container, image, renderer, ref, DEV_PROFILE)
    desired = result["desired"]
    if desired["labels"].get(DEV_TARGET_LABEL) != "1":
        # The worker re-binds through here too, so execution enforces it as well.
        raise ComposeError("dev_target_unmarked", [DEV_TARGET_LABEL],
                           detail="the target's Compose service does not declare it a dev target")
    tag = image_tag(spec, desired)
    if ref != image["Id"] and ref not in image_refs(tag):
        raise ValueError("dev config target must run its own Compose-built image")
    binding = {"container_id": container["Id"], "image_id": image["Id"], "image_ref": ref,
               "image_tag": tag, "image_tag_moved": bool(tag_moved(tag)), "source": result["source"],
               "current_config_hash": current_config_hash(container, image),
               "desired_hash": result["desired_hash"], "rendered_hash": result["rendered_hash"],
               "drift": result["drift"], "mounts": result["mounts"],
               "tmpfs": sorted(desired.get("tmpfs", {})), "ports": port_list(desired["ports"]),
               "depends_on": [f"{name}:{cond}" for name, cond in desired["depends_on"].items()],
               "memory": memory_text(desired["memory"]), "display": result["display"]}
    return binding, desired


def valid_dev_binding(snapshot: dict) -> bool:
    binding = snapshot.get("config")
    if snapshot["resolved_args"] != {} or not isinstance(binding, dict) or set(binding) != _BINDING_KEYS:
        return False
    try:
        spec = validate_dev_spec(snapshot["spec"])
        source = binding["source"]
        lists = binding["drift"] + binding["mounts"] + binding["tmpfs"] + binding["ports"] + binding["depends_on"]
        return (valid_hashes(binding)
                and valid_image_tag(spec, binding["image_tag"])
                and (binding["image_ref"] == binding["image_id"]
                     or binding["image_ref"] in image_refs(binding["image_tag"]))
                and type(binding["image_tag_moved"]) is bool
                and set(source) == {"project_dir", "compose_files"}
                and bool(SAFE_PATH.fullmatch(source["project_dir"]))
                and label_files_valid(source["project_dir"], source["compose_files"])
                and all(isinstance(x, str) for x in lists)
                and all(re.fullmatch(r"/[^\s]*", x) for x in binding["tmpfs"])
                and isinstance(binding["memory"], str) and bool(_MEMORY_TEXT.fullmatch(binding["memory"]))
                and set(binding["display"]) == set(spec["display_values"]))
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def approval_text(spec: dict, binding: dict, prefix: str | None) -> str:
    """Approval card body: names, ports, mounts, the memory cap and A1 enum values only."""
    drift = ", ".join(binding["drift"]) or "none (verifies only)"
    shown = ", ".join(f"{k}={v}" for k, v in sorted(binding["display"].items())) or "none"
    moved = (f"Note: tag {binding['image_tag']} now names a different local image. This keeps the RUNNING "
             "image; rebuilding it is separate.\n") if binding["image_tag_moved"] else ""
    return (f"{prefix or 'Authorize same-image dev service configuration reconcile?'}\n"
            f"Target: {spec['container_name']} ({binding['container_id']})\n"
            f"Unchanged image: {binding['image_id']} (Compose tag {binding['image_tag']})\n{moved}"
            f"Compose source: {', '.join(binding['source']['compose_files'])}\n"
            f"Drift (names only): {drift}\n"
            f"Ports after: {', '.join(binding['ports']) or 'none'}\n"
            f"Requires running: {', '.join(binding['depends_on']) or 'nothing'} (never restarted)\n"
            f"Mounts after: {'; '.join(binding['mounts']) or 'none'}\n"
            f"Tmpfs after: {', '.join(binding['tmpfs']) or 'none'}\n"
            f"Memory after: {binding['memory']}\n"
            f"Shown values: {shown}\n"
            f"Fingerprints: current {binding['current_config_hash'][:16]}, desired {binding['desired_hash'][:16]}\n"
            f"DEV target (its Compose service declares {DEV_TARGET_LABEL}): it serves its live bind-mounted "
            "working tree, so no source check runs. "
            "No build or pull. Environment values other than those shown are never displayed. "
            "Recreating it drops open dev-server connections; pages reconnect once it is healthy. "
            "Restores the previous container if health verification fails. "
            "Any change to the container or the rendered configuration cancels execution.")


class DevConfigReconciler(ServiceConfigReconciler):
    """The service reconcile for a Compose-built dev image, swapped by running ID."""

    SCHEMA = SCHEMA
    ACTION_NAME = "reconcile-dev-config"
    DETAIL_KEYS = (*ServiceConfigReconciler.DETAIL_KEYS, "tmpfs", "memory")

    def _valid_binding(self) -> bool:
        return valid_dev_binding(self.snapshot)

    def _bind(self, old, image, binding):
        return dev_config_binding(self.spec, old, image, self._renderer(),
                                  tag_moved=lambda _tag: binding["image_tag_moved"])


__all__ = ["ACTION", "DEV_TARGET_LABEL", "SCHEMA", "WORKER_CAPABILITY_LABEL", "DevConfigReconciler",
           "approval_text", "dev_config_binding", "memory_text", "valid_dev_binding", "validate_dev_spec"]
