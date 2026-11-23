"""Bot presets from a terminal: ``python -m llm_bawt.seeding.preset_cli`` (TASK-2516).

The installer runs this inside the app container:

- ``list [--json]``: the lineup, the stored (or recommended) selection and
  what applying it would do on this install right now;
- ``select REF ...``: store a selection for the browser wizard to apply once a
  provider is connected (``none`` stores an empty one, ``current`` the one
  ``list`` showed);
- ``import FILE``: add or replace an optional pack (``-`` reads stdin);
- ``export --bot SLUG [--bot SLUG ...] [--pack ID] [--title T]``: show the
  exact text the named bots' pack would carry, with a review digest (exit 3);
  ``--approve DIGEST`` then writes that same pack, refusing if anything changed.
  With no ``--bot`` it lists the exportable bot IDs.

Export is per bot: the operator names each one. It reads only the travelling
fields of named, non-private bots in one read-only snapshot, refuses (rather
than redacts) a bot whose travelling fields contain an address, path or
secret-looking text, and still requires the operator to read the exact text, and records what it exported under ``exported`` (ignored
on import). Credentials, history, memories, endpoints, house and session
bindings and uploaded avatars never travel; only an emoji avatar does.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, TextIO

from datetime import UTC, datetime

from sqlalchemy import text

from ..bot_tags.vocabulary import LEGACY_PRIVATE_TAGS
from . import bot_presets as presets



def _config():
    from ..utils.config import Config

    return Config()


def _state(config, selection: list[str] | None = None) -> dict[str, Any]:
    from ..service.routes.setup_presets import load_state

    return load_state(config, selection)


def _plain(state: dict[str, Any], out: TextIO) -> None:
    selected = set(state["selection"])
    number = 0
    for pack in state["packs"]:
        out.write(f"\n  {pack['title']}\n")
        if pack.get("description"):
            out.write(f"    {pack['description']}\n")
        for preset in pack["presets"]:
            number += 1
            mark = "x" if preset["ref"] in selected else " "
            badge = "  (recommended)" if preset.get("recommended") else ""
            out.write(f"  {number:>2}) [{mark}] {preset['icon'] + ' ' if preset.get('icon') else ''}"
                      f"{preset['name']} - {preset['role']}{badge}\n")
            out.write(f"        {preset['summary']}\n")
            for label, key in (("Pick it if", "pick_if"), ("Differs", "differs")):
                if preset.get(key):
                    out.write(f"        {label}: {preset[key]}\n")
    actions = {row["ref"]: row for row in state["plan"]}
    blocked = [row for row in actions.values() if row["action"] == "blocked"]
    if blocked:
        out.write("\n  Bots are created after you connect a model provider in the browser setup.\n")


def _refs_from_numbers(state: dict[str, Any], words: list[str]) -> list[str]:
    """Accept refs or the 1-based numbers ``list`` printed."""
    ordered = [p["ref"] for pack in state["packs"] for p in pack["presets"]]
    refs: list[str] = []
    for word in words:
        for part in word.replace(",", " ").split():
            if part.isdigit():
                index = int(part) - 1
                if not 0 <= index < len(ordered):
                    raise presets.PresetError(f"no preset number {part}")
                refs.append(ordered[index])
            else:
                refs.append(part)
    return refs


# Private and protected bots never appear, in the same statement that reads
# the rest (no separate deny list that could go stale between two reads).
_EXPORTABLE = """
    FROM bot_profiles p
   WHERE p.include_in_global_search
     AND NOT EXISTS (SELECT 1 FROM bot_profile_tags t
                      WHERE t.bot_slug = lower(btrim(p.slug)) AND t.tag_slug = ANY(:private))
"""
_EXPORTABLE_SLUGS = text("SELECT lower(btrim(p.slug)) AS slug" + _EXPORTABLE + " ORDER BY 1")
_EXPORT_ROWS = text(
    "SELECT lower(btrim(p.slug)) AS slug, p.name, p.description, p.system_prompt, p.avatar, p.color"
    + _EXPORTABLE + " AND lower(btrim(p.slug)) = ANY(:slugs) ORDER BY 1")
# Role tags are this install's own data (the catalog's kind 'role'); a pack
# carries them with their catalog names, which become the preset's role label.
_EXPORT_TAGS = text(
    "SELECT t.bot_slug, t.tag_slug, c.name FROM bot_profile_tags t"
    " JOIN bot_tag_catalog c ON c.slug = t.tag_slug AND c.kind = 'role'"
    " WHERE t.bot_slug = ANY(:slugs) ORDER BY t.tag_slug")


def _read_snapshot(config, slugs: list[str] | None):
    """One read-only, repeatable-read transaction; no store or schema bootstrap."""
    from ..utils.db import build_engine_from_config

    engine = build_engine_from_config(config, application_name="llm-bawt-preset-export",
                                      pool_size=1, max_overflow=0)
    if engine is None:
        raise presets.PresetError("the bot database is not configured")
    private = [str(tag) for tag in LEGACY_PRIVATE_TAGS]
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ",
                                                postgresql_readonly=True) as conn:
            if slugs is None:
                return list(conn.execute(_EXPORTABLE_SLUGS, {"private": private}).scalars()), []
            rows = conn.execute(_EXPORT_ROWS, {"private": private, "slugs": slugs}).mappings().all()
            tags = conn.execute(_EXPORT_TAGS, {"slugs": slugs}).all()
            return rows, tags
    finally:
        engine.dispose()


def exportable_slugs(config) -> list[str]:
    return _read_snapshot(config, None)[0]


def _export(config, slugs: list[str], pack_id: str, title: str) -> dict[str, Any]:
    wanted = list(dict.fromkeys(s.strip().lower() for s in slugs if s.strip()))
    if not wanted:
        raise presets.PresetError("name the bots to export with --bot")
    rows, tag_rows = _read_snapshot(config, wanted)
    found = {row["slug"] for row in rows}
    missing = [s for s in wanted if s not in found]
    if missing:
        raise presets.PresetError(f"not found or private, so not exported: {', '.join(missing)}")
    tags: dict[str, list[tuple[str, str]]] = {}
    for bot, tag, label in tag_rows:
        tags.setdefault(bot, []).append((tag, label))
    items, problems = [], []
    for row in rows:
        slug = row["slug"]
        prompt = (row["system_prompt"] or "").strip()
        if not prompt:
            problems.append(f"{slug}: has no prompt")
            continue
        summary = (row["description"] or "").strip()[:400] or f"{row['name']}, as set up on the source install."
        roles = tags.get(slug, [])
        bot_tags = [tag for tag, _ in roles]
        avatar = (row["avatar"] or "").strip()
        color = (row["color"] or "").strip()
        item = {
            "key": slug, "slug": slug, "name": row["name"] or slug,
            "role": ((roles[0][1] or "").strip()[:80] if roles else "") or "Assistant",
            "summary": summary, "system_prompt": prompt,
            "icon": avatar if presets.portable_icon(avatar) else "",
            "color": color if presets.safe_color(color) else None, "tags": bot_tags,
        }
        # Every travelling text field, not just the prose.
        # Role tags are this install's data too; labelled by position so a hit
        # never echoes the tag itself.
        fields = {k: item[k] for k in ("key", "name", "role", "summary", "system_prompt", "icon", "color") if item[k]}
        fields.update({f"role tag {n}": tag for n, tag in enumerate(bot_tags, 1)})
        problems += [f"{slug}: {hit}" for hit in presets.nonportable_content(fields)]
        items.append(item)
    problems += [f"pack: {hit}" for hit in presets.nonportable_content({"pack": pack_id, "title": title})]
    if problems:
        raise presets.PresetError(
            "not exported; edit these fields or leave the bots out:\n  " + "\n  ".join(problems))
    document = {"pack": pack_id, "title": title, "presets": items,
                "exported": {"by": "preset_cli export", "at": datetime.now(UTC).isoformat(timespec="seconds"),
                             "bots": [i["slug"] for i in items]}}
    presets.parse_pack(document)  # same validation an import applies
    return document


def _visible(text: str) -> str:
    """Text as the pack stores it, with nothing hidden: each nonprinting
    character (control, format, separator other than a space) shows as
    \\u{XXXX} and a backslash as \\\\, so every stored character is on screen."""
    return "".join(
        "\\\\" if ch == "\\" else ch if ch.isprintable() else f"\\u{{{ord(ch):04X}}}" for ch in text)


def _review(document: dict[str, Any], digest: str, out: TextIO) -> None:
    """The exact text the pack would carry, for the operator to read before writing it."""
    out.write(f"Review before exporting. Pack '{_visible(document['pack'])}': {_visible(document['title'])}\n")
    out.write("Checks found no addresses or secret-looking text, but they can't recognise every\n"
              "private detail (a machine or person's name, a project). Read each field below.\n"
              "Characters that don't print are shown as \\u{XXXX}, and a backslash as \\\\.\n")
    for item in document["presets"]:
        out.write(f"\n--- {_visible(item['slug'])} ---\n")
        for key in ("name", "role", "summary", "icon", "color"):
            out.write(f"{key}: {_visible(item[key] or '')}\n")
        out.write(f"tags: {', '.join(_visible(t) for t in item['tags'])}\n")
        out.write("prompt:\n")
        # Split on newlines only; every other separator stays visible in its line.
        out.write("".join(f"  | {_visible(line)}\n" for line in item["system_prompt"].split("\n")))
    out.write(f"\nIf all of it can leave this install, write the pack with: --approve {digest}\n")


def main(argv: list[str] | None = None, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out = out or sys.stdout  # resolved per call so a replaced stdout is honoured
    err = err or sys.stderr
    parser = argparse.ArgumentParser(prog="preset_cli", description="Bot presets for this install.")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--json", action="store_true")
    select = sub.add_parser("select")
    select.add_argument("refs", nargs="+",
                        help="preset refs or list numbers; 'none' for no bots, 'current' keeps the shown one")
    importing = sub.add_parser("import")
    importing.add_argument("file")
    exporting = sub.add_parser("export")
    exporting.add_argument("--bot", action="append", default=[],
                           help="a bot ID to export; repeat for each bot (none lists the choices)")
    exporting.add_argument("--pack", default="mine")
    exporting.add_argument("--title", default="Bots from another install")
    exporting.add_argument("--approve", metavar="DIGEST",
                           help="write the pack shown by the review step (its digest must match)")
    args = parser.parse_args(argv)
    config = _config()
    try:
        if args.command == "list":
            state = _state(config)
            if args.json:
                out.write(json.dumps({k: v for k, v in state.items() if not k.startswith("_")}) + "\n")
            else:
                _plain(state, out)
        elif args.command == "select":
            from ..service.routes.setup_presets import save_selection

            current = _state(config)
            if args.refs == ["none"]:
                refs = []
            elif args.refs == ["current"]:
                refs = current["selection"]
            else:
                refs = _refs_from_numbers(current, args.refs)
            state = save_selection(config, refs)
            names = [row["name"] for row in state["plan"]]
            out.write(("Saved: " + ", ".join(names) if names else "Saved: no preset bots") + "\n")
        elif args.command == "import":
            from ..service.dependencies import get_runtime_settings_store

            source = sys.stdin if args.file == "-" else open(args.file, encoding="utf-8")
            with source:
                pack = presets.parse_pack(json.load(source))
            settings = get_runtime_settings_store(config)
            merged = presets.merge_pack(settings.get_value("global", "*", presets.PACKS_KEY), pack)
            settings.set_value("global", "*", presets.PACKS_KEY, merged)
            out.write(f"Imported {len(pack.presets)} preset(s) as pack '{pack.pack}'.\n")
        elif not args.bot:
            choices = exportable_slugs(config)
            print("Name each bot to export with --bot. Exportable: " + (", ".join(choices) or "none"),
                  file=sys.stderr)
            return 2
        else:
            document = _export(config, args.bot, args.pack, args.title)
            digest = presets.pack_digest(document)
            if args.approve is None:
                # The review never shares the stream the approved JSON is written to.
                _review(document, digest, err)
                return 3
            if args.approve != digest:
                raise presets.PresetError(
                    "the bots changed since that review; run the review again and check the new text")
            out.write(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    except (presets.PresetError, json.JSONDecodeError, OSError) as exc:
        print(f"preset_cli: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
