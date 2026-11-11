"""Prompt portability, default provenance and explicit approved fixture export.

``product`` means eligible for selected export, not an automatic publication or
permission to revoke a tenant override. Private scopes/personas are excluded
regardless of classification metadata. Managed-default lag is drift; intentional
tenant changes are overrides and possible promotion candidates, not failures.
Export writes only selected public fields to source, never the live database.
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
#: Server-owned intent. Body equality never cancels an explicit tenant save.
DEFAULT_MODE_KEY = "default_mode"
OVERRIDE = "override"


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
    scope_type: str
    scope_id: str
    sync_class: str
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
    # Portability exclusions are not overridable by a client-supplied sync flag.
    if (getattr(row, "scope_type", "global") != "global"
            or getattr(row, "scope_id", "*") != "*"):
        return INSTALL
    if (getattr(row, "category", None) or "").strip().lower() in INSTALL_CATEGORIES:
        return INSTALL
    if str(getattr(row, "key", "")).lower().startswith("persona."):
        return INSTALL
    explicit = _metadata(row).get(SYNC_METADATA_KEY)
    if explicit in SYNC_CLASSES:
        return str(explicit)
    return PRODUCT


def is_following(row: Any) -> bool:
    """True only with default-writer provenance and no explicit override intent.

    A hash round-tripped through an old editor does not prove tenant intent.
    Ambiguous or human-edited legacy rows remain overrides, even when their
    text happens to match a current or historical product default.
    """
    if (getattr(row, "scope_type", "global") != "global"
            or getattr(row, "scope_id", "*") != "*"):
        return False
    metadata = _metadata(row)
    if metadata.get(DEFAULT_MODE_KEY) == OVERRIDE:
        return False
    if getattr(row, "updated_by", None) not in FOLLOWING_ACTORS:
        return False
    tracked = metadata.get(DEFAULT_SHA_KEY)
    if tracked:
        return tracked == body_sha256(row.body)
    return True


def prompt_provenance(row: Any | None, definition: Any | None) -> dict[str, Any]:
    """Content-addressed product versions, separate from DB edit-history versions."""
    default = definition.default_body() if definition is not None else None
    default_version = body_sha256(default) if default is not None else None
    effective_version = body_sha256(row.body) if row is not None else default_version
    follows = row is None or (definition is not None and is_following(row))
    return {
        "provenance": "product_default" if follows else "tenant_override",
        "default_version": default_version,
        "effective_version": effective_version,
        "default_outdated": follows and effective_version != default_version,
    }


def compute_drift(rows: Iterable[Any], definitions: dict[str, Any]) -> list[DriftItem]:
    """Portable changed overrides available for explicit promotion, not failures."""
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
                scope_type=row.scope_type,
                scope_id=row.scope_id,
                sync_class=classify(row),
                updated_at=row.updated_at.isoformat() if getattr(row, "updated_at", None) else None,
                body=row.body,
            )
        )
    return sorted(items, key=lambda item: item.key)


def drift_report(
    rows: Iterable[Any], definitions: dict[str, Any], *, include_body: bool = False
) -> dict[str, Any]:
    rows = list(rows)
    candidates = compute_drift(rows, definitions)
    states = [
        {"key": row.key, "scope_type": row.scope_type, "scope_id": row.scope_id,
         **prompt_provenance(row, definitions.get(row.key))}
        for row in rows
    ]
    stale = [state for state in states if state["default_outdated"]]
    counts = {PRODUCT: 0, INSTALL: 0}
    for row in rows:
        counts[classify(row)] += 1
    return {
        "in_sync": not stale,
        "drift_count": len(stale),
        "classified": counts,
        "drift": stale,
        "overrides": [state for state in states if state["provenance"] == "tenant_override"],
        "promotion_candidates": [_item_dict(item, include_body) for item in candidates],
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
    report: dict[str, Any], root: Path, install_terms: Iterable[str] = (),
    *, approved_keys: Iterable[str],
) -> dict[str, Any]:
    """Write a drift report (with bodies) as synced default files under ``root``.

    ``root`` is ``src/llm_bawt/prompt_defaults/synced`` of a source checkout,
    never the running install's package directory. A body naming any
    ``install_terms`` (bot or user names, hosts) is held, not written: reword
    the live row or mark it ``install`` first.
    """
    from .prompt_promotion import validate_promotion_item

    approved = set(approved_keys)
    if not approved:
        raise ValueError("select at least one approved prompt key")
    terms = [str(t) for t in install_terms if str(t).strip()]
    manifest = load_manifest(root)
    written: list[str] = []
    held: dict[str, list[str]] = {}
    manifest_changed = False
    candidates = report.get("promotion_candidates", report.get("drift", []))
    selected = [item for item in candidates if item.get("key") in approved]
    missing = approved - {item["key"] for item in selected}
    if missing:
        raise ValueError(f"selected prompts missing from report: {', '.join(sorted(missing))}")
    if len({item["key"] for item in selected}) != len(selected):
        raise ValueError("duplicate selected prompt keys")
    validated: list[dict[str, Any]] = []
    for item in sorted(selected, key=lambda item: item["key"]):
        body = item.get("body")
        if body is None:
            raise ValueError(f"drift item {item.get('key')!r} has no body; request include_body")
        key = str(item["key"])
        found = validate_promotion_item(item)
        public_text = "\n".join(str(item.get(field) or "") for field in ("key", "title", "category", "body"))
        found += ["installation-specific term"] if install_terms_in(public_text, terms) else []
        if found:
            held[key] = found
            continue
        validated.append(item)
    if validated:
        root.mkdir(parents=True, exist_ok=True)
    for item in validated:
        key, body = item["key"], item["body"]
        body_path(key, root).write_text(body, encoding="utf-8")
        written.append(key)
        if item.get("kind") == "new" or key in manifest:
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

    with urlopen(f"{api.rstrip('/')}/v1/prompts/sync/drift?include_body=false", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_selected(api: str, keys: Iterable[str]) -> dict[str, Any]:
    """Read exactly the approved global rows; never enumerate bots/private scopes."""
    from types import SimpleNamespace
    from urllib.parse import quote
    from urllib.request import urlopen

    items = []
    for key in sorted(set(keys)):
        body_path(key)  # reject invalid keys before network access
        url = f"{api.rstrip('/')}/v1/prompts/{quote(key, safe='')}?scope_type=global&exact=true"
        with urlopen(url, timeout=30) as response:
            row = json.loads(response.read().decode("utf-8"))
        if row.get("key") != key:
            raise ValueError("selected prompt response key does not match request")
        row["sync_class"] = classify(SimpleNamespace(
            **{field: row.get(field) for field in ("key", "category", "scope_type", "scope_id")},
            metadata_json=json.dumps(row.get("metadata") or {}),
        ))
        # A manifest registration for a built-in is harmless and makes exported
        # metadata reviewable too. Existing Python definitions retain precedence.
        row["kind"] = "new"
        items.append(row)
    return {"promotion_candidates": items}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m llm_bawt.prompt_sync")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "export"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--api", default="http://app:8642", help="llm-bawt API base URL")
    sub.choices["export"].add_argument(
        "--key", action="append", required=True,
        help="approved portable prompt key to promote; repeat to explicitly allowlist more",
    )
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
    if args.command == "check":
        report = _fetch_report(args.api)
        print(json.dumps({
            "in_sync": report["in_sync"],
            "drift_count": report["drift_count"],
            "keys": [item["key"] for item in report.get("drift", [])],
            "override_count": len(report.get("overrides", [])),
            "promotion_candidates": [item["key"] for item in report.get("promotion_candidates", [])],
        }, indent=2))
        return 0 if report.get("in_sync") else 1
    import os

    terms = list(args.install_term) + [
        t.strip() for t in os.environ.get("PROMPT_SYNC_INSTALL_TERMS", "").split(",")]
    report = _fetch_selected(args.api, args.key)
    result = export_report(report, args.root, terms, approved_keys=args.key)
    print(json.dumps(result, indent=2))
    return 4 if result["held"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
