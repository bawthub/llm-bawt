"""Prompt sync: classify live prompt rows, report product drift, export it.

A live prompt row is either ``product`` (its body should ship as the default
for every install) or ``install`` (personal or deployment-specific; stays in
this database only). The class lives in the row's metadata under
:data:`SYNC_METADATA_KEY`; rows without one are classified by
:func:`classify`: personas and non-global scopes are ``install``, everything
else (including new edits) is ``product``.

*Drift* is a product row whose body differs from the shipped default, or a
product row with no shipped default at all. ``sync-export`` writes drift into
``prompt_defaults/synced/`` of a source checkout; that change then goes through
the normal review, gate and publish path. Tenants follow published defaults
through :meth:`PromptTemplateStore.seed_defaults`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .prompt_synced_defaults import (
    BODY_SUFFIX,
    body_path,
    body_sha256,
    load_manifest,
    write_manifest,
)

SYNC_METADATA_KEY = "sync"
PRODUCT = "product"
INSTALL = "install"
SYNC_CLASSES = (PRODUCT, INSTALL)
INSTALL_CATEGORIES = frozenset({"persona", "install"})
SEED_ACTOR = "system:seed"
SYNC_ACTOR = "system:sync"
#: Writers whose rows still carry a code default (rows predating the hash).
FOLLOWING_ACTORS = frozenset({SEED_ACTOR, SYNC_ACTOR})
#: Metadata key: sha256 of the code default a row was seeded or synced with.
DEFAULT_SHA_KEY = "default_sha256"


@dataclass(frozen=True)
class DriftItem:
    key: str
    kind: str  # "changed" (differs from shipped default) or "new" (no default)
    title: str
    category: str
    format: str
    required_vars: list[str]
    live_sha256: str
    default_sha256: str | None
    updated_by: str | None
    updated_at: str | None = None
    body: str | None = None


def _metadata(row: Any) -> dict[str, Any]:
    raw = getattr(row, "metadata_json", None)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def classify(row: Any) -> str:
    explicit = _metadata(row).get(SYNC_METADATA_KEY)
    if explicit in SYNC_CLASSES:
        return str(explicit)
    if getattr(row, "scope_type", "global") != "global":
        return INSTALL
    if (getattr(row, "category", None) or "").strip().lower() in INSTALL_CATEGORIES:
        return INSTALL
    return PRODUCT


def is_following(row: Any) -> bool:
    """True while a row still holds the code default it was seeded/synced with.

    Following rows are moved to new code defaults on upgrade, so they are
    never product drift even when their body is behind the shipped default.
    """
    tracked = _metadata(row).get(DEFAULT_SHA_KEY)
    if tracked:
        return tracked == body_sha256(row.body)
    return getattr(row, "updated_by", None) in FOLLOWING_ACTORS


def compute_drift(rows: Iterable[Any], definitions: dict[str, Any]) -> list[DriftItem]:
    """Product rows whose live body is not the shipped default."""
    items: list[DriftItem] = []
    for row in rows:
        if classify(row) != PRODUCT:
            continue
        definition = definitions.get(row.key)
        if definition is not None and is_following(row):
            continue
        definition = definitions.get(row.key)
        default = definition.default_body() if definition is not None else None
        if default is not None and default == row.body:
            continue
        try:
            required = json.loads(row.required_vars_json) if row.required_vars_json else []
        except (TypeError, ValueError):
            required = []
        items.append(
            DriftItem(
                key=row.key,
                kind="changed" if definition is not None else "new",
                title=row.title or (definition.title if definition else row.key),
                category=row.category or (definition.category if definition else "custom"),
                format=row.format or "plain_text",
                required_vars=[str(v) for v in required],
                live_sha256=body_sha256(row.body),
                default_sha256=body_sha256(default) if default is not None else None,
                updated_by=row.updated_by,
                updated_at=row.updated_at.isoformat() if getattr(row, "updated_at", None) else None,
                body=row.body,
            )
        )
    return sorted(items, key=lambda item: item.key)


def drift_report(
    rows: Iterable[Any], definitions: dict[str, Any], *, include_body: bool = False
) -> dict[str, Any]:
    rows = list(rows)
    drift = compute_drift(rows, definitions)
    counts = {PRODUCT: 0, INSTALL: 0}
    for row in rows:
        counts[classify(row)] += 1
    return {
        "in_sync": not drift,
        "drift_count": len(drift),
        "classified": counts,
        "drift": [_item_dict(item, include_body) for item in drift],
    }


def _item_dict(item: DriftItem, include_body: bool) -> dict[str, Any]:
    data = asdict(item)
    if not include_body:
        data.pop("body", None)
    return data


def install_terms_in(body: str, terms: Iterable[str]) -> list[str]:
    """Install-specific terms (whole words, case-sensitive) found in ``body``."""
    import re

    return sorted({t for t in terms if t and re.search(rf"(?<!\w){re.escape(t)}(?!\w)", body)})


def export_report(
    report: dict[str, Any], root: Path, install_terms: Iterable[str] = ()
) -> dict[str, Any]:
    """Write a drift report (with bodies) as synced default files under ``root``.

    ``root`` is ``src/llm_bawt/prompt_defaults/synced`` of a source checkout,
    never the running install's package directory. A body naming any
    ``install_terms`` (bot or user names, hosts) is held, not written: reword
    the live row or mark it ``install`` first.
    """
    terms = [str(t) for t in install_terms if str(t).strip()]
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(root)
    written: list[str] = []
    held: dict[str, list[str]] = {}
    manifest_changed = False
    for item in report.get("drift", []):
        body = item.get("body")
        if body is None:
            raise ValueError(f"drift item {item.get('key')!r} has no body; request include_body")
        key = str(item["key"])
        found = install_terms_in(body, terms)
        if found:
            held[key] = found
            continue
        body_path(key, root).write_text(body, encoding="utf-8")
        written.append(key)
        if item.get("kind") == "new":
            manifest[key] = {
                "title": item.get("title") or key,
                "category": item.get("category") or "custom",
                "format": item.get("format") or "plain_text",
                "required_vars": list(item.get("required_vars") or []),
            }
            manifest_changed = True
    if manifest_changed:
        write_manifest(manifest, root)
    return {"written": sorted(written), "count": len(written), "held": held,
            "root": str(root), "suffix": BODY_SUFFIX}


def _fetch_report(api: str) -> dict[str, Any]:
    # Stdlib only: the exporter runs from a source checkout without app deps.
    from urllib.request import urlopen

    with urlopen(f"{api.rstrip('/')}/v1/prompts/sync/drift?include_body=true", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_bot_terms(api: str) -> list[str]:
    """Bot slugs and display names from the live install (always install-specific)."""
    from urllib.request import urlopen

    with urlopen(f"{api.rstrip('/')}/v1/bots", timeout=30) as response:
        bots = json.loads(response.read().decode("utf-8")).get("data", [])
    return [str(v) for bot in bots for v in (bot.get("slug"), bot.get("name")) if v]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m llm_bawt.prompt_sync")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "export"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--api", default="http://app:8642", help="llm-bawt API base URL")
    sub.choices["export"].add_argument(
        "--root", required=True, type=Path,
        help="prompt_defaults/synced directory of the source checkout to write",
    )
    sub.choices["export"].add_argument(
        "--install-term", action="append", default=[],
        help="extra install-specific term (user name, host) that must not ship; repeatable."
             " PROMPT_SYNC_INSTALL_TERMS (comma-separated) adds more",
    )
    args = parser.parse_args(argv)
    report = _fetch_report(args.api)
    if args.command == "check":
        print(json.dumps({k: v for k, v in report.items() if k != "drift"} | {
            "keys": [item["key"] for item in report.get("drift", [])]}, indent=2))
        return 0 if report.get("in_sync") else 1
    import os

    terms = _fetch_bot_terms(args.api) + list(args.install_term) + [
        t.strip() for t in os.environ.get("PROMPT_SYNC_INSTALL_TERMS", "").split(",")]
    result = export_report(report, args.root, terms)
    print(json.dumps(result, indent=2))
    return 4 if result["held"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
