"""Synced prompt defaults: live product prompt edits exported as shipped defaults.

``llm-bawt prompts sync-export`` (see :mod:`llm_bawt.prompt_sync`) writes one
``<key>.txt`` per product prompt whose live body differs from the shipped
default, under ``prompt_defaults/synced/``. ``manifest.json`` there registers
keys that have no Python definition yet (title, category, format, required
vars). Built-in definitions prefer a synced body over their Python constant, so
the synced file is the shipped default once it is published.

Kept free of other llm-bawt imports so ``prompt_definitions`` can use it.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SYNCED_DIR = Path(__file__).resolve().parent / "prompt_defaults" / "synced"
MANIFEST_NAME = "manifest.json"
BODY_SUFFIX = ".txt"


def body_sha256(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def body_path(key: str, root: Path | None = None) -> Path:
    if not key or "/" in key or "\\" in key or key.startswith("."):
        raise ValueError(f"invalid prompt key for a synced default: {key!r}")
    return (root or SYNCED_DIR) / f"{key}{BODY_SUFFIX}"


@lru_cache(maxsize=None)
def synced_body(key: str) -> str | None:
    """The synced default body for ``key``, or None when none is shipped."""
    try:
        path = body_path(key)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def load_manifest(root: Path | None = None) -> dict[str, dict[str, Any]]:
    path = (root or SYNCED_DIR) / MANIFEST_NAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", {}) if isinstance(data, dict) else {}
    return {str(k): dict(v) for k, v in prompts.items() if isinstance(v, dict)}


def write_manifest(entries: dict[str, dict[str, Any]], root: Path | None = None) -> None:
    path = (root or SYNCED_DIR) / MANIFEST_NAME
    payload = {
        "about": "Prompt defaults added by the prompt sync for keys with no Python definition.",
        "prompts": {key: entries[key] for key in sorted(entries)},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def clear_cache() -> None:
    synced_body.cache_clear()
