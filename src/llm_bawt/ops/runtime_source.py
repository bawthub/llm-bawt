"""Approval/worker binding for source-activating restarts. STANDARD LIBRARY ONLY.

This is a dispatch fence, not an immutable deployment: edits after the final
check can still affect later lazy imports. It never claims otherwise.
"""
from __future__ import annotations

import posixpath
import re

if __package__:
    from .source_state import SHA, SourceChecker, SourceStateError, _clean_path, _within
else:
    from source_state import SHA, SourceChecker, SourceStateError, _clean_path, _within

WORKER_CAPABILITY_LABEL = "llm-bawt.ops.runtime-source"
_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}")
_RELATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")
_HASH = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def validate_runtime_source(spec):
    guard = spec.get("runtime_source")
    if guard is None:
        return
    if spec.get("action") != "restart" or not isinstance(guard, dict) or set(guard) != {
            "project_dir", "checker_image", "paths", "mounts", "data_mounts"}:
        raise ValueError("runtime_source requires project_dir/checker_image/paths/mounts/data_mounts on a restart")
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
    return {"container_id": cid, "image_id": image}


def source_revision(api, spec, container, revision, *, tag):
    guard = spec["runtime_source"]
    env = dict(item.partition("=")[::2] for item in (container.get("Config") or {}).get("Env") or [])
    prefix = env.get("PYTHONPYCACHEPREFIX", "")
    bytecode_ok = _clean_path(prefix) and not any(
        _within(prefix, m.get("Destination", "/")) or _within(m.get("Destination", "/"), prefix)
        for m in container.get("Mounts") or [])
    try:
        return SourceChecker(api, guard["checker_image"], tag=tag).check_revision(
            guard["project_dir"], {"paths": guard["paths"], "compose_files": [],
                                   "bytecode_ok": bytecode_ok}, revision)
    except SourceStateError as exc:
        raise SourceStateError("tree_changed", detail=f"runtime check refused: {exc.code}") from None


def bind_runtime_source(api, spec, args, container, *, tag):
    revision = args.get("expected_runtime_sha")
    if not isinstance(revision, str) or not SHA.fullmatch(revision):
        raise SourceStateError("source_unverifiable", detail="expected_runtime_sha must name the tested commit")
    return {**target_identity(spec, container),
            **source_revision(api, spec, container, revision, tag=tag)}


def valid_runtime_binding(snapshot):
    binding = snapshot.get("runtime_source")
    if "runtime_source" not in snapshot["spec"]:
        return binding is None
    return (isinstance(binding, dict) and set(binding) == {
                "container_id", "image_id", "expected_runtime_sha", "observed_head", "runtime_tree_hash"}
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
    current = target_identity(snapshot["spec"], container)
    if any(current[k] != approved[k] for k in current):
        raise SourceStateError("tree_changed", detail="approved runtime container changed")
    result = source_revision(api, snapshot["spec"], container, approved["expected_runtime_sha"], tag=tag)
    if result["runtime_tree_hash"] != approved["runtime_tree_hash"]:
        raise SourceStateError("tree_changed", detail="approved runtime tree changed")
    return {**current, **result}
