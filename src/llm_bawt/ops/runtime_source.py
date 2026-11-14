"""Approval/worker binding for source-activating restarts. STANDARD LIBRARY ONLY.

This is a dispatch fence, not an immutable deployment: edits after the final
check can still affect later lazy imports. It never claims otherwise.
"""
from __future__ import annotations

import posixpath
import re

if __package__:
    from .source_state import (
        MAX_CAPTURE_FILES, SHA, SourceChecker, SourceStateError, _clean_path, _within,
        source_commit, valid_source_commit, validate_compose_source_spec,
    )
else:
    from source_state import (
        MAX_CAPTURE_FILES, SHA, SourceChecker, SourceStateError, _clean_path, _within,
        source_commit, valid_source_commit, validate_compose_source_spec,
    )

WORKER_CAPABILITY_LABEL = "llm-bawt.ops.runtime-source"
EXTENDED_CAPABILITY_LABEL = "llm-bawt.ops.runtime-source-v2"
_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}")
_RELATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")
_HASH = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_FIELDS = {"project_dir", "checker_image", "paths", "mounts", "data_mounts"}
_SOURCE_FIELDS = {"source_repository", "source_branch"}


def extended_runtime_source(spec):
    guard = spec.get("runtime_source") or {}
    return "arg" in guard or bool(_SOURCE_FIELDS & set(guard))


def selected_runtime_spec(spec, args):
    """The same guard for a fixed target, or an explicitly classified arg value."""
    guard = spec.get("runtime_source")
    if guard is None:
        return None
    if "arg" in guard:
        value = args.get(guard["arg"])
        if not isinstance(value, str) or value not in guard["targets"]:
            raise SourceStateError("source_unverifiable", detail="restart target has no source classification")
        guard = guard["targets"][value]
    return {**spec, "runtime_source": guard} if guard is not None else None


def validate_runtime_source(spec):
    guard = spec.get("runtime_source")
    if guard is None:
        return
    if spec.get("action") != "restart" or not isinstance(guard, dict):
        raise ValueError("runtime_source is only supported on a restart")
    if "arg" in guard:
        if (set(guard) != {"arg", "targets"} or not isinstance(guard["arg"], str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", guard["arg"])
                or guard["arg"] not in (spec.get("container_name_from_arg"), spec.get("compose_service_from_arg"))
                or not isinstance(guard["targets"], dict) or not 1 <= len(guard["targets"]) <= 64
                or any(not isinstance(key, str) or not 1 <= len(key) <= 128 for key in guard["targets"])):
            raise ValueError("runtime_source arg map requires a named argument and bounded target map")
        for target in guard["targets"].values():
            if target is not None:
                _validate_guard(target, extended=True)
        return
    _validate_guard(guard)


def _validate_guard(guard, *, extended=False):
    if (not isinstance(guard, dict)
            or set(guard) not in (_FIELDS, _FIELDS | _SOURCE_FIELDS)
            or (extended and set(guard) != _FIELDS | _SOURCE_FIELDS)):
        raise ValueError("runtime_source requires project_dir/checker_image/paths/mounts/data_mounts on a restart")
    if _SOURCE_FIELDS & set(guard):
        validate_compose_source_spec(guard)
    if not _clean_path(guard["project_dir"]) or not _IMAGE.fullmatch(str(guard["checker_image"])):
        raise ValueError("runtime_source requires an absolute project directory and digest-pinned checker")
    paths, mounts = guard["paths"], guard["mounts"]
    if (not isinstance(paths, list) or not 1 <= len(paths) <= 32
            or any(not isinstance(p, str) or not _RELATIVE.fullmatch(p) or
                   any(part in ("", ".", "..", ".git") for part in p.split("/")) for p in paths)
            or len(set(paths)) != len(paths)):
        raise ValueError("runtime_source paths must be distinct plain repository-relative paths")
    if (not isinstance(mounts, dict) or not 1 <= len(mounts) <= 32
            or any(not _clean_path(dest) or rel not in paths for dest, rel in mounts.items())):
        raise ValueError("runtime_source mounts must map absolute destinations to checked paths")
    data = guard["data_mounts"]
    if (not isinstance(data, list) or len(data) > 64 or any(not _clean_path(p) for p in data)
            or len(set(data)) != len(data)
            or any(_within(d, s) or _within(s, d) for d in data for s in mounts)):
        raise ValueError("runtime_source data_mounts must be distinct and disjoint from source mounts")


def validate_runtime_args(spec, schema):
    """Cross-field catalog checks, following the existing quiesce arg-map pattern."""
    guard = spec.get("runtime_source")
    if guard is None:
        return
    props = schema.get("properties", {})
    required = schema.get("required", [])
    expected = props.get("expected_runtime_sha", {})
    if (expected.get("type") != "string"
            or (extended_runtime_source(spec) and expected.get("pattern") != "^[0-9a-f]{40}$")
            or ("arg" not in guard and "expected_runtime_sha" not in required)):
        raise ValueError("runtime_source requires a declared, required expected_runtime_sha string argument "
                         "(optional only for a classified arg map)")
    if extended_runtime_source(spec) and props.get("allow_dirty", {}).get("type") != "boolean":
        raise ValueError("extended runtime_source requires an explicit boolean allow_dirty argument")
    if "arg" in guard:
        selector = props.get(guard["arg"], {})
        choices = selector.get("enum")
        if (selector.get("type") != "string" or guard["arg"] not in required
                or not isinstance(choices, list) or any(not isinstance(v, str) for v in choices)
                or set(choices) != set(guard["targets"])):
            raise ValueError("runtime_source targets must exactly cover the required string enum")


def target_identity(spec, container):
    """Pin the actual source binds and container, never a mutable selector."""
    guard = spec["runtime_source"]
    actual = container.get("Mounts") or []
    destinations = guard["mounts"]
    allowed = {*destinations, *guard["data_mounts"]}
    if any(m.get("Destination") not in allowed for m in actual):
        raise SourceStateError("tree_changed", detail="unclassified runtime mount")
    for dest, rel in destinations.items():
        rows = [m for m in actual if m.get("Destination") == dest]
        if (len(rows) != 1 or rows[0].get("Type") != "bind"
                or rows[0].get("Source") != posixpath.join(guard["project_dir"], rel)):
            raise SourceStateError("tree_changed", detail="runtime source mount differs")
        if any(m.get("Destination") != dest and _within(m.get("Destination", ""), dest) for m in actual):
            raise SourceStateError("tree_changed", detail="runtime source mount is shadowed")
    cid, image = container.get("Id"), container.get("Image")
    if (not isinstance(cid, str) or not _HASH.fullmatch(cid) or not isinstance(image, str)
            or not _IMAGE_ID.fullmatch(image)):
        raise SourceStateError("source_unverifiable", detail="container identity unavailable")
    if _SOURCE_FIELDS <= set(guard):
        env = dict(item.partition("=")[::2] for item in (container.get("Config") or {}).get("Env") or [])
        if "PYTHONPATH" in env and any(
                not _clean_path(path)
                or any(_within(path, data) or _within(data, path) for data in guard["data_mounts"])
                for path in env["PYTHONPATH"].split(":")):
            raise SourceStateError("source_env_unsafe", ["PYTHONPATH"], detail="runtime imports may resolve through data")
    return {"container_id": cid, "image_id": image}


def source_revision(api, spec, container, revision, *, tag, allow_dirty=False):
    guard = spec["runtime_source"]
    env = dict(item.partition("=")[::2] for item in (container.get("Config") or {}).get("Env") or [])
    prefix = env.get("PYTHONPYCACHEPREFIX", "")
    bytecode_ok = _clean_path(prefix) and not any(
        _within(prefix, m.get("Destination", "/")) or _within(m.get("Destination", "/"), prefix)
        for m in container.get("Mounts") or [])
    try:
        return SourceChecker(api, guard["checker_image"], tag=tag).check_revision(
            guard["project_dir"], {"paths": guard["paths"], "compose_files": [],
                                   "bytecode_ok": bytecode_ok}, revision, allow_dirty=allow_dirty)
    except SourceStateError as exc:
        raise SourceStateError("tree_changed", exc.names, detail=f"runtime check refused: {exc.code}") from None


def bind_runtime_source(api, spec, args, container, *, tag):
    spec = selected_runtime_spec(spec, args)
    if spec is None:
        return None
    revision = args.get("expected_runtime_sha")
    if not isinstance(revision, str) or not SHA.fullmatch(revision):
        raise SourceStateError("source_unverifiable", detail="expected_runtime_sha must name the tested commit")
    extended = extended_runtime_source(spec)
    allow_dirty = args.get("allow_dirty", False) if extended else False
    if type(allow_dirty) is not bool:
        raise SourceStateError("source_unverifiable", detail="allow_dirty must be a boolean")
    binding = {**target_identity(spec, container),
               **source_revision(api, spec, container, revision, tag=tag, allow_dirty=allow_dirty)}
    if extended:
        binding.update(allow_dirty=allow_dirty, source_commit=source_commit(spec["runtime_source"], revision),
                       publication="unverified")
        binding.setdefault("dirty_paths", [])
    return binding


def valid_runtime_binding(snapshot):
    binding = snapshot.get("runtime_source")
    try:
        spec = selected_runtime_spec(snapshot["spec"], snapshot["resolved_args"])
    except (KeyError, TypeError, SourceStateError):
        return False
    if spec is None:
        return binding is None
    keys = {"container_id", "image_id", "expected_runtime_sha", "observed_head", "runtime_tree_hash"}
    if extended_runtime_source(spec):
        if not isinstance(binding, dict):
            return False
        keys |= {"allow_dirty", "source_commit", "publication", "dirty_paths"}
        allowed = binding.get("allow_dirty")
        if (type(allowed) is not bool or allowed != snapshot["resolved_args"].get("allow_dirty", False)
                or binding.get("publication") not in ("published", "unpublished")
                or (not allowed and binding["publication"] != "published")
                or not valid_source_commit(spec["runtime_source"], binding.get("source_commit"))
                or binding["source_commit"]["head"] != binding.get("expected_runtime_sha")):
            return False
        dirty = binding.get("dirty_paths")
        if (not isinstance(dirty, list) or len(dirty) > MAX_CAPTURE_FILES
                or any(not isinstance(path, str) or not path or path.startswith("/")
                       or any(part in ("", ".", "..") for part in path.split("/")) for path in dirty)
                or len(set(dirty)) != len(dirty) or (not allowed and dirty)):
            return False
        if allowed:
            keys.add("runtime_manifest")
            manifest = binding.get("runtime_manifest")
            if (not isinstance(manifest, dict) or len(manifest) > MAX_CAPTURE_FILES
                    or any(not isinstance(path, str) or not path or path.startswith("/")
                           or any(part in ("", ".", "..") for part in path.split("/"))
                           or not isinstance(value, list) or len(value) != 2
                           or value[0] not in ("100644", "100755")
                           or not isinstance(value[1], str) or not SHA.fullmatch(value[1])
                           for path, value in manifest.items())):
                return False
    return (isinstance(binding, dict) and set(binding) == keys
            and all(isinstance(binding[k], str) and SHA.fullmatch(binding[k])
                    for k in ("expected_runtime_sha", "observed_head"))
            and all(isinstance(binding[k], str) and _HASH.fullmatch(binding[k])
                    for k in ("container_id", "runtime_tree_hash"))
            and isinstance(binding["image_id"], str) and _IMAGE_ID.fullmatch(binding["image_id"])
            and binding["expected_runtime_sha"] == snapshot["resolved_args"].get("expected_runtime_sha"))


def verify_runtime_source(api, snapshot, container, *, tag):
    if not valid_runtime_binding(snapshot):
        raise SourceStateError("tree_changed", detail="runtime approval binding missing or invalid")
    approved = snapshot["runtime_source"]
    spec = selected_runtime_spec(snapshot["spec"], snapshot["resolved_args"])
    current = target_identity(spec, container)
    if any(current[k] != approved[k] for k in current):
        raise SourceStateError("tree_changed", detail="approved runtime container changed")
    result = source_revision(api, spec, container, approved["expected_runtime_sha"], tag=tag,
                             allow_dirty=approved.get("allow_dirty", False))
    if result["runtime_tree_hash"] != approved["runtime_tree_hash"]:
        raise SourceStateError("tree_changed", detail="approved runtime tree changed")
    return {**approved, **current, **result}
