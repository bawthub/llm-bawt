"""Verify a staged, finite offline asset bundle. Never fetch, execute or attest it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path, PurePosixPath

HEX256 = re.compile(r"[0-9a-f]{64}\Z")
VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){2}\Z")
TOOLS = {"python", "node", "uv", "corepack", "pnpm"}


class Refused(ValueError):
    """The finite bundle could not be verified."""


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refused(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes) -> dict:
    value = json.loads(data, object_pairs_hook=_object)
    if not isinstance(value, dict):
        raise Refused("JSON object required")
    return value


def _file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise Refused("invalid relative path")
    parts = relative.split("/")
    if PurePosixPath(relative).is_absolute() or any(p in {"", ".", ".."} for p in parts):
        raise Refused("invalid relative path")
    current = root
    for index, part in enumerate(parts):
        current = current / part
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise Refused(f"symlink forbidden: {relative}")
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise Refused(f"directory required: {relative}")
        if index == len(parts) - 1 and not stat.S_ISREG(mode):
            raise Refused(f"regular file required: {relative}")
    return current


def verify_bundle(root: Path, manifest_data: bytes, expected_sha256: str, versions_data: bytes) -> dict:
    """Read only; caller must own/freeze staging and trust the manifest digest out of band.

    Declared versions are a comparison input, NOT proof that binaries ran or came
    from particular artifacts. No result here authorizes a job or constitutes a gate.
    """
    if not isinstance(expected_sha256, str) or not HEX256.fullmatch(expected_sha256):
        raise Refused("lowercase manifest SHA256 required")
    if hashlib.sha256(manifest_data).hexdigest() != expected_sha256:
        raise Refused("manifest SHA256 mismatch")
    manifest = _json(manifest_data)
    if set(manifest) != {"schema_version", "scope", "platform", "files", "tool_versions", "limitations"}:
        raise Refused("unexpected manifest fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise Refused("unsupported manifest schema")
    if manifest["scope"] != "finite-offline-assets-only" or manifest["platform"] != "linux-x86_64":
        raise Refused("unsupported manifest scope/platform")
    if not isinstance(manifest["limitations"], list) or not manifest["limitations"] or not all(
        isinstance(item, str) and item.strip() for item in manifest["limitations"]
    ):
        raise Refused("explicit limitations required")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise Refused("nonempty file inventory required")
    if root.is_symlink() or not root.is_dir():
        raise Refused("bundle root must be a real directory")
    root = root.resolve(strict=True)
    seen = set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes", "origin"}:
            raise Refused("invalid file entry")
        relative = entry["path"]
        if not isinstance(relative, str):
            raise Refused("invalid relative path")
        if relative in seen:
            raise Refused(f"duplicate bundle path: {relative}")
        seen.add(relative)
        if not isinstance(entry["sha256"], str) or not HEX256.fullmatch(entry["sha256"]):
            raise Refused("lowercase file SHA256 required")
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] <= 0:
            raise Refused("positive file size required")
        if not isinstance(entry["origin"], str) or not entry["origin"].strip():
            raise Refused("file origin required")
        path = _file(root, relative)
        with path.open("rb") as stream:
            if path.stat().st_size != entry["size_bytes"]:
                raise Refused(f"file size mismatch: {relative}")
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise Refused(f"file SHA256 mismatch: {relative}")
    requirements = manifest["tool_versions"]
    if not isinstance(requirements, dict) or set(requirements) != TOOLS:
        raise Refused("exact tool requirement set required")
    versions = _json(versions_data)
    if set(versions) != TOOLS:
        raise Refused("exact declared tool version set required")
    for tool, rule in requirements.items():
        if not isinstance(rule, dict) or set(rule) != {"match", "version", "basis"}:
            raise Refused("invalid tool requirement")
        if not isinstance(rule["basis"], str) or not rule["basis"].strip():
            raise Refused("tool requirement basis required")
        actual, wanted = versions[tool], rule["version"]
        if not isinstance(actual, str) or not VERSION.fullmatch(actual):
            raise Refused(f"exact numeric declared version required: {tool}")
        if rule["match"] == "exact" and isinstance(wanted, str) and VERSION.fullmatch(wanted):
            matches = actual == wanted
        elif tool == "python" and rule["match"] == "major_minor" and wanted == "3.12":
            matches = actual.startswith("3.12.")
        else:
            raise Refused(f"unsupported version rule: {tool}")
        if not matches:
            raise Refused(f"tool version mismatch: {tool}")
    return {
        "schema_version": 1,
        "status": "verified_finite_files_and_declared_versions",
        "manifest_sha256": expected_sha256,
        "file_count": len(files),
        "declared_tool_versions": versions,
        "attestation": False,
        "complete_offline_ci": False,
        "gate_receipt": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--declared-tool-versions", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_bundle(
            args.bundle, args.manifest.read_bytes(), args.manifest_sha256,
            args.declared_tool_versions.read_bytes(),
        )
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
