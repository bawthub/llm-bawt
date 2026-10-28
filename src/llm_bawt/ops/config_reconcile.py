"""Same-image reconcile of a Compose target to its declared config. STANDARD LIBRARY ONLY (TASK-1518).

Recreates the fixed target from its rendered Compose service at the SAME pinned
image: no build, no pull. Approval binds the live container's configuration
fingerprint (``current_config_hash``) and the rendered desired configuration
(``desired_hash``, values included but only hashed), plus drift NAMES, the mount
list and closed-enum display values (A1). The worker re-renders and recomputes
both under the per-target lock; any change refuses with nothing applied. The
swap, health, release-identity check and failure restore are
:class:`ImageDeployer`'s.
"""
from __future__ import annotations

import re
from urllib.parse import quote

if __package__:
    from .compose_config import apply_desired, canonical_hash, evaluate
    from .compose_render import COMPOSE_FILE, MAX_COMPOSE_FILES, SAFE_PATH, validate_interpolation_env
    from .image_deploy import ImageDeployer, clone_create_body, demux_stdout, release_from_labels
    from .service_identity import SPEC_PATTERNS, ComposeServiceRef
else:  # standalone immutable worker
    from compose_config import apply_desired, canonical_hash, evaluate
    from compose_render import COMPOSE_FILE, MAX_COMPOSE_FILES, SAFE_PATH, validate_interpolation_env
    from image_deploy import ImageDeployer, clone_create_body, demux_stdout, release_from_labels
    from service_identity import SPEC_PATTERNS, ComposeServiceRef

ACTION = "reconcile_frontend_config"
SCHEMA = "llm-bawt.ops.config-reconcile/v1"
# Worker image label proving it carries the Compose renderer, the config
# reconciler AND the deploy drift guard. Without it both ops refuse.
WORKER_CAPABILITY_LABEL = "llm-bawt.ops.config-reconcile"
CONTAINER_NAME_PATTERN = r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}"
_PATTERNS = {
    "container_name": CONTAINER_NAME_PATTERN,
    **SPEC_PATTERNS,
    "image_repository": r"ghcr\.io/[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*",
}
_RENDERER_IMAGE = r"(?:[a-z0-9.-]+(?::[0-9]{1,5})?/)?[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}"
_BINDING_KEYS = frozenset({"container_id", "image_id", "image_ref", "source", "current_config_hash",
                           "desired_hash", "rendered_hash", "drift", "mounts", "display", "release"})
_STAT = re.compile(r"[0-7]{3,4} [0-9]{1,10} [0-9]{1,10}")


def validate_compose_config(value) -> dict:
    """``compose_config`` shared by deploy_image and this op. Basenames only: the
    project directory comes from the target's own Compose labels (no host path
    in a catalog seed)."""
    required = {"compose_files", "renderer_image", "image_variable"}
    if not isinstance(value, dict) or not required <= set(value) <= required | {"interpolation_env"}:
        raise ValueError("compose_config requires exactly compose_files, renderer_image, image_variable"
                         " (and optional interpolation_env)")
    files = value["compose_files"]
    if (not isinstance(files, list) or not 1 <= len(files) <= MAX_COMPOSE_FILES or len(set(files)) != len(files)
            or not all(isinstance(f, str) and COMPOSE_FILE.fullmatch(f) for f in files)):
        raise ValueError("compose_config.compose_files must be 1..4 distinct Compose file basenames")
    validate_renderer_image(value)
    if not isinstance(value["image_variable"], str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", value["image_variable"]):
        raise ValueError("compose_config.image_variable is invalid")
    validate_interpolation_env(value)
    return value


def validate_renderer_image(value: dict) -> None:
    if not isinstance(value["renderer_image"], str) or not re.fullmatch(_RENDERER_IMAGE, value["renderer_image"]):
        raise ValueError("compose_config.renderer_image must be pinned by digest")


def validate_config_spec(spec: dict) -> dict:
    required = {"action", "compose_config", "display_values", "stat_paths", "stop_grace_seconds",
                "health_timeout_seconds", *_PATTERNS}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("config reconcile spec has missing or unsupported fields")
    if spec["action"] != ACTION:
        raise ValueError("not a config reconcile spec")
    validate_target_fields(spec, _PATTERNS)
    validate_compose_config(spec["compose_config"])
    return spec


def validate_target_fields(spec: dict, patterns: dict) -> None:
    """Fields every config-reconcile kind shares (TASK-1474)."""
    for key, pattern in patterns.items():
        if not isinstance(spec[key], str) or not re.fullmatch(pattern, spec[key]):
            raise ValueError(f"config reconcile spec {key} is invalid")
    shown = spec["display_values"]
    if (not isinstance(shown, dict) or len(shown) > 8
            or not all(isinstance(k, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", k)
                       and isinstance(v, list) and 1 <= len(v) <= 8
                       and all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,32}", x) for x in v)
                       for k, v in shown.items())):
        raise ValueError("display_values must map env names to closed enums of short tokens")
    paths = spec["stat_paths"]
    if (not isinstance(paths, list) or len(paths) > 4
            or not all(isinstance(p, str) and SAFE_PATH.fullmatch(p) and ".." not in p.split("/") for p in paths)):
        raise ValueError("stat_paths must be up to 4 absolute container paths")
    for key, low, high in (("stop_grace_seconds", 0, 60), ("health_timeout_seconds", 10, 180)):
        if type(spec[key]) is not int or not low <= spec[key] <= high:
            raise ValueError(f"config reconcile spec {key} is out of bounds")


def current_config_hash(container: dict, image: dict) -> str:
    """Fingerprint of everything a clone would carry forward (r2 B2)."""
    ref = container["Config"].get("Image", "")
    return canonical_hash(clone_create_body(container, image, image, ref, "binding"))


def check_compose_identity(spec: dict, container: dict) -> None:
    if (container["Name"] != "/" + spec["container_name"]
            or not ComposeServiceRef.from_spec(spec).matches(container["Config"].get("Labels"))):
        raise ValueError("config target Compose identity mismatch")


def valid_hashes(binding: dict) -> bool:
    return (bool(re.fullmatch(r"[0-9a-f]{64}", binding["container_id"]))
            and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", binding["image_id"]))
            and all(re.fullmatch(r"[0-9a-f]{64}", binding[k])
                    for k in ("current_config_hash", "desired_hash", "rendered_hash")))


def config_binding(spec: dict, container: dict, image: dict, renderer) -> tuple[dict, dict]:
    """``(public approval binding, private desired config)``; raises ValueError."""
    validate_config_spec(spec)
    config = container["Config"]
    check_compose_identity(spec, container)
    ref = config.get("Image", "")
    if (not re.fullmatch(re.escape(spec["image_repository"]) + r"@sha256:[0-9a-f]{64}", ref)
            or image["Id"] != container["Image"] or ref not in (image.get("RepoDigests") or [])):
        raise ValueError("config target must retain its locally available pinned image")
    release = release_from_labels(image["Config"].get("Labels"))
    if not release or not all(release.values()):
        raise ValueError("config target has no verifiable release identity")
    result = evaluate(spec, container, image, renderer, ref)
    binding = {"container_id": container["Id"], "image_id": image["Id"], "image_ref": ref,
               "source": result["source"], "current_config_hash": current_config_hash(container, image),
               "desired_hash": result["desired_hash"], "rendered_hash": result["rendered_hash"],
               "drift": result["drift"], "mounts": result["mounts"], "display": result["display"],
               "release": release}
    return binding, result["desired"]


def valid_config_binding(snapshot: dict) -> bool:
    binding = snapshot.get("config")
    if snapshot["spec"].get("action") != ACTION:
        return binding is None
    if snapshot["resolved_args"] != {} or not isinstance(binding, dict) or set(binding) != _BINDING_KEYS:
        return False
    try:
        spec = validate_config_spec(snapshot["spec"])
        source = binding["source"]
        return (valid_hashes(binding)
                and bool(re.fullmatch(re.escape(spec["image_repository"]) + r"@sha256:[0-9a-f]{64}", binding["image_ref"]))
                and set(source) == {"project_dir", "compose_files"}
                and source["compose_files"] == [f"{source['project_dir']}/{n}"
                                                for n in spec["compose_config"]["compose_files"]]
                and all(isinstance(x, str) for x in binding["drift"] + binding["mounts"])
                and set(binding["display"]) == set(spec["display_values"])
                and set(binding["release"]) == {"version", "source_sha", "workflow_run_id"}
                and all(isinstance(v, str) and v for v in binding["release"].values()))
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def approval_text(spec: dict, binding: dict, prefix: str | None) -> str:
    """Approval card body: names, mounts and A1 enum values only."""
    drift = ", ".join(binding["drift"]) or "none (verifies only)"
    shown = ", ".join(f"{k}={v}" for k, v in sorted(binding["display"].items())) or "none"
    return (f"{prefix or 'Authorize same-image configuration reconcile?'}\n"
            f"Target: {spec['container_name']} ({binding['container_id']})\n"
            f"Unchanged image: {binding['image_ref']}\n"
            f"Compose source: {', '.join(binding['source']['compose_files'])}\n"
            f"Drift (names only): {drift}\n"
            f"Mounts after: {'; '.join(binding['mounts']) or 'none'}\n"
            f"Shown values: {shown}\n"
            f"Fingerprints: current {binding['current_config_hash'][:16]}, desired {binding['desired_hash'][:16]}\n"
            "No build or pull. Environment values other than those shown are never displayed. "
            "Restores the previous container if health or release verification fails. "
            "Any change to the container or the rendered configuration cancels execution.")


class ConfigReconciler(ImageDeployer):
    """Use the existing target lock, swap, health and failure-restore machinery.

    Kind-specific steps are hooks (TASK-1474): ``_valid_binding``, ``_bind``,
    ``_release``, ``_swap_ref`` and ``_before_swap``."""

    SCHEMA = SCHEMA
    ACTION_NAME = "reconcile-config"
    DETAIL_KEYS = ("drift", "mounts", "display", "rendered_hash", "desired_hash", "current_config_hash")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detail.update(schema=self.SCHEMA, action=self.ACTION_NAME)
        self._desired = None

    def _valid_binding(self) -> bool:
        return valid_config_binding(self.snapshot)

    def _bind(self, old, image, binding):
        return config_binding(self.spec, old, image, self._renderer())

    def _release(self, binding):
        return binding["release"]

    def _swap_ref(self, binding):
        return binding["image_ref"]

    def _before_swap(self):
        """Last read-only check before production is touched."""

    def _perform(self):
        old, image = self._prepare()
        if self._verify_only(old):
            return
        self._before_swap()
        binding = self.snapshot["config"]
        self._replace(old, image, image, self._swap_ref(binding), self._release(binding))

    def _prepare(self):
        """Read-only: inspect, re-render and compare with the approved
        binding; require a running healthy target. ``(old, image)``. A
        multi-target plan (TASK-1476) runs this for every target before it
        touches any, and again as each target's last read before its swap."""
        if not self._valid_binding():
            self._fail("invalid config approval binding", known=True)
        binding = self.snapshot["config"]
        old, image = self._target(expected_image_id=binding["image_id"])
        self._phase("render")
        try:
            actual, self._desired = self._bind(old, image, binding)
        except ValueError as exc:  # includes ComposeError: fixed codes and names only
            self._fail(f"{exc}; nothing changed", known=True)
        if actual != binding:
            changed = sorted(k for k in binding if actual.get(k) != binding[k])
            self._fail(f"config_changed_since_approval ({', '.join(changed)}); nothing changed", known=True)
        state = old.get("State") or {}
        if not state.get("Running") or (state.get("Health") or {}).get("Status") != "healthy":
            self._fail("config reconcile requires a running healthy target", known=True)
        self.detail["config"] = {k: binding[k] for k in self.DETAIL_KEYS}
        return old, image

    def _verify_only(self, old) -> bool:
        """No drift: verify the running target in place and report True."""
        binding = self.snapshot["config"]
        if binding["drift"]:
            return False
        self.detail["health"] = self._verify_release(old["Id"], self._release(binding))
        self.detail["unchanged"] = True
        return True

    def _create_body(self, old, old_image, new_image, ref):
        body = super()._create_body(old, old_image, new_image, ref)
        return apply_desired(body, old, self._desired, container_name=self.name,
                             service=self.spec["compose_service"],
                             rendered_hash=self.snapshot["config"]["rendered_hash"], image=new_image)

    def _verify_release(self, container_id, expect):
        health = super()._verify_release(container_id, expect)
        health["stat"] = self._stats(container_id)
        return health

    def _stats(self, container_id):
        return {path: self._stat(container_id, path) for path in self.spec["stat_paths"]}

    def _exec(self, container_id, cmd):
        """``(exit code, stdout)`` of a fixed argv inside the target."""
        exec_id = self.api.request("POST", f"/containers/{quote(container_id, safe='')}/exec", body={
            "AttachStdout": True, "AttachStderr": True, "Cmd": cmd})["Id"]
        raw = self.api.request("POST", f"/exec/{exec_id}/start", body={"Detach": False, "Tty": False}, raw=True)
        return self.api.request("GET", f"/exec/{exec_id}/json").get("ExitCode"), demux_stdout(raw)

    def _stat(self, container_id, path):
        """Mode/uid/gid of a file via a fixed argv; never its contents."""
        code, out = self._exec(container_id, ["stat", "-c", "%a %u %g", path])
        if code != 0:
            return "missing"
        text = out.strip()
        return text if _STAT.fullmatch(text) else "unreadable"


__all__ = ["ACTION", "SCHEMA", "WORKER_CAPABILITY_LABEL", "ConfigReconciler", "approval_text",
           "check_compose_identity", "config_binding", "current_config_hash", "valid_config_binding",
           "valid_hashes", "validate_compose_config", "validate_config_spec", "validate_renderer_image",
           "validate_target_fields"]
