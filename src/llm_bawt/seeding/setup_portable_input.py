"""Strict input from an independently reviewed, exact-byte portable source."""
from __future__ import annotations

import json
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .bot_presets import PresetError

_DIGEST = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
_MAX_BYTES = 4 * 1024 * 1024
_TEXT_FIELDS = {"name", "summary", "system_prompt", "role", "icon", "color", "pick_if", "differs"}


@dataclass(frozen=True)
class PortableSetupInput:
    profiles: Mapping[str, dict]
    roles: Mapping[str, dict]


def is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def is_slug(value: object) -> bool:
    return isinstance(value, str) and _SLUG.fullmatch(value) is not None


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key")
        result[key] = value
    return result


def _entries(entries: object, *, role: bool) -> bool:
    if not isinstance(entries, dict):
        return False
    for slug, entry in entries.items():
        if not is_slug(slug) or not isinstance(entry, dict) or not is_digest(entry.get("revision")):
            return False
        if role:
            if (set(entry) != {"revision", "name", "description", "color", "prompt_key", "prompt_sha256"}
                    or not isinstance(entry["name"], str) or not isinstance(entry["description"], str)
                    or not (entry["color"] is None or isinstance(entry["color"], str))):
                return False
            key = entry["prompt_key"]
            if key is None:
                if entry["prompt_sha256"] is not None:
                    return False
            elif not (isinstance(key, str) and key.startswith("role.") and is_digest(entry["prompt_sha256"])):
                return False
        else:
            if set(entry) != {"revision", "roles", "preset"}:
                return False
            roles, preset = entry["roles"], entry["preset"]
            if (not isinstance(roles, list) or not all(is_slug(ref) for ref in roles)
                    or len(roles) != len(set(roles)) or not isinstance(preset, dict)
                    or not {"name", "summary", "system_prompt", "role"} <= preset.keys() <= _TEXT_FIELDS | {"recommended"}):
                return False
            for field, value in preset.items():
                if field == "recommended":
                    if not isinstance(value, bool):
                        return False
                elif not isinstance(value, str) and not (field == "color" and value is None):
                    return False
    return True


def load_portable_input(path: Path, approved_sha256: str) -> PortableSetupInput:
    if not is_digest(approved_sha256):
        raise PresetError("Portable input needs an exact approved SHA-256")
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise OSError("Not a regular file")
        with path.open("rb") as stream:
            payload = stream.read(_MAX_BYTES + 1)
    except OSError:
        raise PresetError("Approved portable input is unavailable") from None
    if len(payload) > _MAX_BYTES or sha256(payload).hexdigest() != approved_sha256:
        raise PresetError("Portable input does not match the approved bytes")
    try:
        data = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique)
        valid = (isinstance(data, dict) and set(data) == {"schema", "profiles", "roles"}
                 and type(data["schema"]) is int and data["schema"] == 1
                 and _entries(data["profiles"], role=False) and _entries(data["roles"], role=True))
    except (ValueError, RecursionError):
        valid = False
    if not valid:
        raise PresetError("Approved portable input has an invalid version or field schema")
    return PortableSetupInput(data["profiles"], data["roles"])
