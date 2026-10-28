"""Same-image reconcile of a service that runs a public TAGGED image. STANDARD LIBRARY ONLY (TASK-1476).

Target: crawl4ai, whose Compose service names a moving tag
(``unclecode/crawl4ai:latest``) with no digest and no release labels, so
neither the frontend kind (digest-pinned, verified release) nor the app kind
(Compose-built, source guard) fits. This kind:

* binds the RUNNING image ID and swaps to it, never to the tag: a reconcile
  must not upgrade the service by re-resolving ``:latest``. A tag that now
  names another local image is shown on the approval card
  (``image_tag_moved``) and is never drift;
* renders the target's own Compose service (the files its labels record) with
  :data:`PINNED_PROCESS_PROFILE`: the pinned keys plus container_name,
  depends_on, entrypoint/command, published ports and tmpfs. The image's own
  HEALTHCHECK is inherited through the clone;
* checks ``depends_on`` as a precondition and verifies the target's own
  in-container health URL, as the app kind does (:class:`ServiceConfigReconciler`).
"""
from __future__ import annotations

import re

if __package__:
    from .app_config_reconcile import (HEALTH_PATTERNS, ServiceConfigReconciler, validate_health_value,
                                       validate_label_compose_config)
    from .compose_config import PINNED_PROCESS_PROFILE, evaluate, port_list
    from .compose_render import SAFE_PATH, label_files_valid
    from .config_reconcile import (CONTAINER_NAME_PATTERN, check_compose_identity, current_config_hash,
                                   valid_hashes, validate_target_fields)
    from .service_identity import SPEC_PATTERNS
else:  # standalone immutable worker
    from app_config_reconcile import (HEALTH_PATTERNS, ServiceConfigReconciler, validate_health_value,
                                      validate_label_compose_config)
    from compose_config import PINNED_PROCESS_PROFILE, evaluate, port_list
    from compose_render import SAFE_PATH, label_files_valid
    from config_reconcile import (CONTAINER_NAME_PATTERN, check_compose_identity, current_config_hash,
                                  valid_hashes, validate_target_fields)
    from service_identity import SPEC_PATTERNS

ACTION = "reconcile_tagged_config"
SCHEMA = "llm-bawt.ops.tagged-config-reconcile/v1"
# An explicit tag, never a digest: ``[registry[:port]/]repository:tag``.
IMAGE_TAG_PATTERN = (r"(?:[a-z0-9.-]+(?::[0-9]{1,5})?/)?[a-z0-9][a-z0-9._/-]{0,127}"
                     r":[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")
_PATTERNS = {
    "container_name": CONTAINER_NAME_PATTERN,
    **SPEC_PATTERNS,
    **HEALTH_PATTERNS,
    "image_tag": IMAGE_TAG_PATTERN,
}
_SPEC_KEYS = frozenset({"action", "compose_config", "display_values", "stat_paths", "stop_grace_seconds",
                        "health_timeout_seconds", "health_value", *_PATTERNS})
_BINDING_KEYS = frozenset({"container_id", "image_id", "image_ref", "image_tag", "image_tag_moved", "source",
                           "current_config_hash", "desired_hash", "rendered_hash", "drift", "mounts", "tmpfs",
                           "ports", "depends_on", "display"})


def validate_tagged_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS:
        raise ValueError("tagged config reconcile spec has missing or unsupported fields")
    if spec["action"] != ACTION:
        raise ValueError("not a tagged config reconcile spec")
    validate_target_fields(spec, _PATTERNS)
    if "/." in spec["image_tag"] or "//" in spec["image_tag"]:
        raise ValueError("config reconcile spec image_tag is invalid")
    validate_health_value(spec)
    validate_label_compose_config(spec)
    return spec


def tagged_config_binding(spec: dict, container: dict, image: dict, renderer, *, tag_moved) -> tuple[dict, dict]:
    """``(public approval binding, private desired config)``; raises ValueError.

    ``tag_moved(tag) -> bool`` is resolved by the caller: the app inspects the
    tag at approval; the worker carries the approved value forward and never
    resolves it."""
    validate_tagged_spec(spec)
    check_compose_identity(spec, container)
    tag = spec["image_tag"]
    ref = container["Config"].get("Image", "")
    # After this kind's first swap the container records the image ID, not the tag.
    if image["Id"] != container["Image"] or ref not in (tag, image["Id"]):
        raise ValueError("tagged config target must run its declared image tag")
    result = evaluate(spec, container, image, renderer, tag, PINNED_PROCESS_PROFILE)
    desired = result["desired"]
    binding = {"container_id": container["Id"], "image_id": image["Id"], "image_ref": ref,
               "image_tag": tag, "image_tag_moved": bool(tag_moved(tag)), "source": result["source"],
               "current_config_hash": current_config_hash(container, image),
               "desired_hash": result["desired_hash"], "rendered_hash": result["rendered_hash"],
               "drift": result["drift"], "mounts": result["mounts"],
               "tmpfs": sorted(desired.get("tmpfs", {})), "ports": port_list(desired["ports"]),
               "depends_on": [f"{name}:{cond}" for name, cond in desired["depends_on"].items()],
               "display": result["display"]}
    return binding, desired


def valid_tagged_binding(snapshot: dict) -> bool:
    binding = snapshot.get("config")
    if snapshot["resolved_args"] != {} or not isinstance(binding, dict) or set(binding) != _BINDING_KEYS:
        return False
    try:
        spec = validate_tagged_spec(snapshot["spec"])
        source = binding["source"]
        lists = binding["drift"] + binding["mounts"] + binding["tmpfs"] + binding["ports"] + binding["depends_on"]
        return (valid_hashes(binding)
                and binding["image_tag"] == spec["image_tag"]
                and binding["image_ref"] in (spec["image_tag"], binding["image_id"])
                and type(binding["image_tag_moved"]) is bool
                and set(source) == {"project_dir", "compose_files"}
                and bool(SAFE_PATH.fullmatch(source["project_dir"]))
                and label_files_valid(source["project_dir"], source["compose_files"])
                and all(isinstance(x, str) for x in lists)
                and all(re.fullmatch(r"/[^\s]*", x) for x in binding["tmpfs"])
                and set(binding["display"]) == set(spec["display_values"]))
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def approval_text(spec: dict, binding: dict, prefix: str | None) -> str:
    """Approval card body: names, ports, mounts and A1 enum values only."""
    drift = ", ".join(binding["drift"]) or "none (verifies only)"
    shown = ", ".join(f"{k}={v}" for k, v in sorted(binding["display"].items())) or "none"
    moved = (f"Note: tag {binding['image_tag']} now names a different local image. This keeps the RUNNING "
             "image; upgrading it is separate.\n") if binding["image_tag_moved"] else ""
    return (f"{prefix or 'Authorize same-image service configuration reconcile?'}\n"
            f"Target: {spec['container_name']} ({binding['container_id']})\n"
            f"Unchanged image: {binding['image_id']} (tag {binding['image_tag']})\n{moved}"
            f"Compose source: {', '.join(binding['source']['compose_files'])}\n"
            f"Drift (names only): {drift}\n"
            f"Ports after: {', '.join(binding['ports']) or 'none'}\n"
            f"Requires running: {', '.join(binding['depends_on']) or 'nothing'} (never restarted)\n"
            f"Mounts after: {'; '.join(binding['mounts']) or 'none'}\n"
            f"Tmpfs after: {', '.join(binding['tmpfs']) or 'none'}\n"
            f"Shown values: {shown}\n"
            f"Fingerprints: current {binding['current_config_hash'][:16]}, desired {binding['desired_hash'][:16]}\n"
            "No build or pull. Environment values other than those shown are never displayed. "
            "Restores the previous container if health verification fails. "
            "Any change to the container or the rendered configuration cancels execution.")


class TaggedConfigReconciler(ServiceConfigReconciler):
    """The service reconcile for a tagged public image, swapped by running ID."""

    SCHEMA = SCHEMA
    ACTION_NAME = "reconcile-tagged-config"
    DETAIL_KEYS = (*ServiceConfigReconciler.DETAIL_KEYS, "tmpfs")

    def _valid_binding(self) -> bool:
        return valid_tagged_binding(self.snapshot)

    def _bind(self, old, image, binding):
        return tagged_config_binding(self.spec, old, image, self._renderer(),
                                     tag_moved=lambda _tag: binding["image_tag_moved"])


__all__ = ["ACTION", "IMAGE_TAG_PATTERN", "SCHEMA", "TaggedConfigReconciler", "approval_text",
           "tagged_config_binding", "valid_tagged_binding", "validate_tagged_spec"]
