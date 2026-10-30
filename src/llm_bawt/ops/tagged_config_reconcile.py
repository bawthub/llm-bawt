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

TASK-1831 (redis: no HTTP endpoint, shared infrastructure) adds two optional
spec parts:

* ``health_probe: {"type": "docker_health"}`` in place of the health URL
  triple: the target's own Docker healthcheck is the probe, run by the
  image-deploy machinery (:mod:`health_probe`), never a second check;
* ``source_repository``/``source_branch``: the Compose source guard. The
  rendered Compose files come from the checkout, so a guarded row binds them to
  one commit (:func:`source_state.compose_source_plan`, the sealed git check
  the app kind runs, scoped to the Compose files and a clean tree) that the app
  proves is the published branch head, and re-checks before the swap.
"""
from __future__ import annotations

import re

if __package__:
    from .app_config_reconcile import (HEALTH_PATTERNS, SourceGuardedReconciler, validate_health_value,
                                       validate_label_compose_config)
    from .health_probe import DOCKER_HEALTH, validate_health_probe
    from .source_state import (compose_source_plan, source_commit, valid_source_commit,
                               validate_compose_source_spec)
    from .compose_config import PINNED_PROCESS_PROFILE, evaluate, port_list
    from .compose_render import SAFE_PATH, label_files_valid
    from .config_reconcile import (CONTAINER_NAME_PATTERN, check_compose_identity, current_config_hash,
                                   valid_hashes, validate_target_fields)
    from .service_identity import SPEC_PATTERNS
else:  # standalone immutable worker
    from app_config_reconcile import (HEALTH_PATTERNS, SourceGuardedReconciler, validate_health_value,
                                      validate_label_compose_config)
    from health_probe import DOCKER_HEALTH, validate_health_probe
    from source_state import (compose_source_plan, source_commit, valid_source_commit,
                              validate_compose_source_spec)
    from compose_config import PINNED_PROCESS_PROFILE, evaluate, port_list
    from compose_render import SAFE_PATH, label_files_valid
    from config_reconcile import (CONTAINER_NAME_PATTERN, check_compose_identity, current_config_hash,
                                  valid_hashes, validate_target_fields)
    from service_identity import SPEC_PATTERNS

ACTION = "reconcile_tagged_config"
# TASK-1831: a worker that runs a docker_health probe or the Compose source
# guard on this kind. Required only by a spec that uses either, so dispatch
# refuses before an older worker would refuse the spec shape mid-job.
TAGGED_GUARD_LABEL = "llm-bawt.ops.tagged-guard"
SCHEMA = "llm-bawt.ops.tagged-config-reconcile/v1"
# An explicit tag, never a digest: ``[registry[:port]/]repository:tag``.
IMAGE_TAG_PATTERN = (r"(?:[a-z0-9.-]+(?::[0-9]{1,5})?/)?[a-z0-9][a-z0-9._/-]{0,127}"
                     r":[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")
_PATTERNS = {
    "container_name": CONTAINER_NAME_PATTERN,
    **SPEC_PATTERNS,
    "image_tag": IMAGE_TAG_PATTERN,
}
_BASE_KEYS = frozenset({"action", "compose_config", "display_values", "stat_paths", "stop_grace_seconds",
                        "health_timeout_seconds", *_PATTERNS})
# Exactly one health part: the in-container URL triple or a docker_health probe.
_URL_HEALTH_KEYS = frozenset({"health_value", *HEALTH_PATTERNS})
_PROBE_HEALTH_KEYS = frozenset({"health_probe"})
_GUARD_KEYS = frozenset({"source_repository", "source_branch"})
_BINDING_KEYS = frozenset({"container_id", "image_id", "image_ref", "image_tag", "image_tag_moved", "source",
                           "current_config_hash", "desired_hash", "rendered_hash", "drift", "mounts", "tmpfs",
                           "ports", "depends_on", "display", "source_commit"})


def needs_tagged_guard_worker(spec: dict) -> bool:
    """Whether a (validated) tagged spec uses a TASK-1831 part."""
    return "health_probe" in spec or source_guarded(spec)


def source_guarded(spec: dict) -> bool:
    """Whether a (validated) tagged spec carries the Compose source guard."""
    return "source_repository" in spec


def validate_tagged_spec(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("tagged config reconcile spec has missing or unsupported fields")
    extra = set(spec) - _BASE_KEYS
    guard = extra & _GUARD_KEYS
    if (not _BASE_KEYS <= set(spec) or guard not in (set(), _GUARD_KEYS)
            or extra - guard not in (_URL_HEALTH_KEYS, _PROBE_HEALTH_KEYS)):
        raise ValueError("tagged config reconcile spec has missing or unsupported fields")
    if spec["action"] != ACTION:
        raise ValueError("not a tagged config reconcile spec")
    validate_target_fields(spec, {**_PATTERNS, **({} if "health_probe" in spec else HEALTH_PATTERNS)})
    if "/." in spec["image_tag"] or "//" in spec["image_tag"]:
        raise ValueError("config reconcile spec image_tag is invalid")
    if "health_probe" in spec:
        # A tagged image carries no release identity for an HTTP probe to echo.
        if validate_health_probe(spec["health_probe"])["type"] != DOCKER_HEALTH:
            raise ValueError("a tagged config reconcile health_probe must be docker_health")
    else:
        validate_health_value(spec)
    if guard:
        validate_compose_source_spec(spec)
    validate_label_compose_config(spec)
    return spec


def tagged_config_binding(spec: dict, container: dict, image: dict, renderer, *, tag_moved,
                          source_checker=None) -> tuple[dict, dict]:
    """``(public approval binding, private desired config)``; raises ValueError.

    ``tag_moved(tag) -> bool`` is resolved by the caller: the app inspects the
    tag at approval; the worker carries the approved value forward and never
    resolves it. A guarded spec requires ``source_checker``, which proves the
    Compose files and tree are exactly one commit; whether that commit is
    published is the app's check, made against ``source_commit``."""
    validate_tagged_spec(spec)
    if source_guarded(spec) and source_checker is None:
        raise ValueError("a source-guarded tagged reconcile needs the source check")
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
               "display": result["display"], "source_commit": None}
    if source_guarded(spec):
        head = source_checker.check(result["source"]["project_dir"], compose_source_plan(result["source"]))
        binding["source_commit"] = source_commit(spec, head)
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
                and set(binding["display"]) == set(spec["display_values"])
                and (valid_source_commit(spec, binding["source_commit"]) if source_guarded(spec)
                     else binding["source_commit"] is None))
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
            f"{_commit_line(binding['source_commit'])}"
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


def _commit_line(commit) -> str:
    if commit is None:
        return ""
    return (f"Compose source commit: {commit['head'][:12]} "
            f"(published {commit['repository']} {commit['branch']} head; clean tree required)\n")


class TaggedConfigReconciler(SourceGuardedReconciler):
    """The service reconcile for a tagged public image, swapped by running ID."""

    SCHEMA = SCHEMA
    ACTION_NAME = "reconcile-tagged-config"
    DETAIL_KEYS = (*SourceGuardedReconciler.DETAIL_KEYS, "tmpfs", "source_commit")

    def _valid_binding(self) -> bool:
        return valid_tagged_binding(self.snapshot)

    def _bind(self, old, image, binding):
        checker = self._source_checker() if source_guarded(self.spec) else None
        actual, desired = tagged_config_binding(self.spec, old, image, self._renderer(),
                                                tag_moved=lambda _tag: binding["image_tag_moved"],
                                                source_checker=checker)
        if checker is not None:
            self._hold_source(checker, actual, compose_source_plan(actual["source"]))
        return actual, desired

    def _before_swap(self):
        super()._before_swap()
        if source_guarded(self.spec):
            self._recheck_source()


__all__ = ["ACTION", "IMAGE_TAG_PATTERN", "SCHEMA", "TAGGED_GUARD_LABEL", "TaggedConfigReconciler", "approval_text",
           "needs_tagged_guard_worker", "source_guarded", "tagged_config_binding", "valid_tagged_binding", "validate_tagged_spec"]
