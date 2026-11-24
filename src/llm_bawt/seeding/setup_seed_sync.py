"""Build the generated Setup catalog from the live bot lineup (TASK-2516)."""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..prompt_sync import install_terms_in
from .bot_presets import GENERATED_PACK, PresetError, nonportable_content, parse_pack, portable_icon, safe_color
from .setup_live_source import LiveSetupSource
from .setup_role_dependencies import required_roles

# Allowed properties are data, not a roster. No source authorization, account,
# model, runtime, assistant history or tenant prompt layers enter this artifact.
_PRESET_FIELDS = (
    "key", "slug", "name", "role", "summary", "system_prompt", "icon", "color",
    "tags", "pick_if", "differs", "recommended",
)


@dataclass(frozen=True)
class SetupSeedSnapshot:
    """The source adapter must distinguish a complete empty set from a failed read.

    Eligible slugs are independent of successful projection: a held profile
    must not be represented as a deletion. Prompt keys are from the matching
    portable prompt candidate, never from unrestricted installation rows.
    """

    eligible_slugs: Collection[str]
    bots: Sequence[Mapping[str, object]]
    roles: Mapping[str, Mapping[str, object]]
    portable_prompt_keys: Collection[str]


_FALLBACK_SUMMARY = "General-purpose assistant."


def live_setup_snapshot(source: LiveSetupSource, portable_prompt_keys: Collection[str]) -> SetupSeedSnapshot:
    """Project the live lineup into preset fields; nothing is renamed or rewritten.

    A missing summary falls back to the bot's first role description. Avatars
    and colors travel only in their portable forms (an emoji, a hex or palette
    color); anything else is left out rather than copied.
    """
    bots = []
    for row in source.profiles:
        slug = str(row["slug"])
        roles = sorted(row["roles"])
        first = source.roles[roles[0]] if roles else {}
        avatar = str(row.get("avatar") or "").strip()
        color = str(row.get("color") or "").strip()
        summary = (str(row.get("description") or "").strip() or str(first.get("description") or "").strip()
                   or _FALLBACK_SUMMARY)
        if len(summary) > 400:
            summary = summary[:399].rstrip() + "\u2026"
        bots.append({
            "key": slug, "slug": slug,
            "name": str(row.get("name") or "").strip() or slug,
            "role": (str(first.get("name") or "").strip() or "Assistant")[:80],
            "summary": summary,
            "system_prompt": str(row.get("system_prompt") or "").strip(),
            "icon": avatar if portable_icon(avatar) else "",
            "color": color if safe_color(color) else None,
            "tags": roles,
        })
    definitions = {slug: {"slug": slug, **{f: role.get(f) for f in ("name", "description", "color", "prompt_key")}}
                   for slug, role in source.roles.items()}
    return SetupSeedSnapshot({str(row["slug"]) for row in source.profiles}, bots, definitions,
                             frozenset(portable_prompt_keys))


def render_catalog(snapshot: SetupSeedSnapshot, *, install_terms: Collection[str] = (),
                   held: dict[str, list[str]] | None = None) -> bytes:
    """Canonical candidate.

    A bot whose travelling text fails the scrub is left out and recorded in
    *held* (slug to the kinds of content found, never the matched text) so the
    sync reports it. Without a *held* collector any bot hold holds the whole
    catalog. Structural problems and role holds always hold the whole catalog.
    """
    items = []
    bot_holds: dict[str, list[str]] = {}
    for index, bot in enumerate(snapshot.bots, 1):
        item = {field: bot[field] for field in _PRESET_FIELDS if field in bot}
        if item.get("key") != item.get("slug") or not isinstance(item.get("slug"), str):
            raise PresetError(f"Setup preset {index} has no stable slug reference")
        if "recommended" in item and not isinstance(item["recommended"], bool):
            raise PresetError(f"Setup preset {index} has invalid recommendation metadata")
        strings = {k: v for k, v in item.items() if isinstance(v, str)}
        tags = item.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise PresetError(f"Setup preset {index} has invalid role references")
        # Assignments are a set in the source. Never publish an order-only diff.
        tags = sorted({tag.strip() for tag in tags if tag.strip()})
        item["tags"] = tags
        strings.update({f"role reference {i}": tag for i, tag in enumerate(tags, 1)})
        hits = nonportable_content(strings)
        hits.extend(f"{label} contains an installation-specific term" for label, value in strings.items()
                    if install_terms_in(value, install_terms))
        if not hits:
            try:
                parse_pack({"pack": GENERATED_PACK, "presets": [item]})
            except PresetError:
                # The validator's message can quote the record; report the kind only.
                hits.append("fields do not fit the preset schema (empty prompt, length or format)")
        if hits:
            if held is None:
                raise PresetError(f"Setup preset {index} is held: {'; '.join(hits)}")
            bot_holds[item["slug"]] = hits
            continue
        items.append(item)
    slugs = {item["slug"] for item in items}
    if (slugs | bot_holds.keys() != set(snapshot.eligible_slugs) or slugs & bot_holds.keys()
            or len(items) != len(slugs)):
        raise PresetError("Setup catalog held: projected roster does not match the complete eligible set")

    document = {"pack": GENERATED_PACK, "title": "Starter lineup", "presets": sorted(items, key=lambda i: i["slug"])}
    try:
        presets = parse_pack(document).presets if items else ()
    except PresetError:
        # Existing pack validation can include identifiers from an untrusted
        # record. Don't relay its raw message from an unattended sync.
        raise PresetError("Setup catalog held: projected presets do not match the portable schema") from None
    roles = required_roles(presets, snapshot.roles, snapshot.portable_prompt_keys, install_terms=install_terms)
    document["roles"] = [asdict(role) for role in roles]
    payload = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if held is not None:
        held.update(bot_holds)
    return payload


def stage_catalog(
    read_snapshot: Callable[[], SetupSeedSnapshot], target: Path, *, install_terms: Collection[str] = (),
    held: dict[str, list[str]] | None = None,
) -> bool:
    """Stage the catalog in the sync worktree only; return whether bytes changed.

    The reader's failure, a catalog hold or an invalid dependency leaves the
    previous candidate untouched. A complete, verified deletion is a normal
    update. Per-bot holds go to *held* (see :func:`render_catalog`).
    """
    payload = render_catalog(read_snapshot(), install_terms=install_terms, held=held)
    if target.exists() and target.read_bytes() == payload:
        return False
    with NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        scratch = Path(tmp.name)
        try:
            tmp.write(payload)
            tmp.close()
            os.replace(scratch, target)
        finally:
            scratch.unlink(missing_ok=True)
    return True
