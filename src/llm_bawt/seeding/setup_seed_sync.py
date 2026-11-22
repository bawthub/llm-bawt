"""Build the generated Setup catalog from a complete portable snapshot (TASK-2516)."""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..prompt_sync import install_terms_in
from .bot_presets import GENERATED_PACK, PresetError, nonportable_content, parse_pack
from .setup_portable_input import PortableSetupInput
from .setup_role_dependencies import required_roles
from .setup_structure import SetupSourceStructure

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


def approved_setup_snapshot(
    structure: SetupSourceStructure, approved: PortableSetupInput, prompt_revisions: Mapping[str, str],
) -> SetupSeedSnapshot:
    """Join the complete structural read to exact approved portable definitions."""
    if structure.profiles.keys() != structure.memberships.keys():
        raise PresetError("Setup source returned incomplete profile memberships")
    bots = []
    for index, (slug, revision) in enumerate(sorted(structure.profiles.items()), 1):
        entry = approved.profiles.get(slug)
        roles = sorted(structure.memberships[slug])
        if entry is None:
            raise PresetError(f"Setup preset {index} has no approved portable definition")
        if entry["revision"] != revision or sorted(entry["roles"]) != roles:
            raise PresetError(f"Setup preset {index} needs review for its changed revision or roles")
        bots.append({**entry["preset"], "key": slug, "slug": slug, "tags": roles})
    required = {role for roles in structure.memberships.values() for role in roles}
    if required != structure.roles.keys():
        raise PresetError("Setup source returned incomplete role revisions")
    definitions = {}
    for index, (slug, revision) in enumerate(sorted(structure.roles.items()), 1):
        entry = approved.roles.get(slug)
        if entry is None:
            raise PresetError(f"Setup role {index} has no approved portable definition")
        if entry["revision"] != revision:
            raise PresetError(f"Setup role {index} needs review for its changed revision")
        key = entry["prompt_key"]
        if key is not None and prompt_revisions.get(key) != entry["prompt_sha256"]:
            raise PresetError(f"Setup role {index} has no matching approved product prompt revision")
        definitions[slug] = {"slug": slug, **{field: entry[field] for field in ("name", "description", "color", "prompt_key")}}
    return SetupSeedSnapshot(structure.profiles.keys(), bots, definitions, frozenset(prompt_revisions))


def render_catalog(snapshot: SetupSeedSnapshot, *, install_terms: Collection[str] = ()) -> bytes:
    """Canonical candidate. Any hold prevents a partial catalog from replacing the prior one."""
    items = []
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
        if hits:
            raise PresetError(f"Setup preset {index} is held: {'; '.join(hits)}")
        items.append(item)
    slugs = {item["slug"] for item in items}
    if slugs != set(snapshot.eligible_slugs) or len(items) != len(slugs):
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
    return (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def stage_catalog(
    read_snapshot: Callable[[], SetupSeedSnapshot], target: Path, *, install_terms: Collection[str] = (),
) -> bool:
    """Stage the catalog in the sync worktree only; return whether bytes changed.

    The reader's failure, a hold or an invalid dependency leaves the previous
    candidate untouched. A complete, verified deletion is a normal update.
    """
    payload = render_catalog(read_snapshot(), install_terms=install_terms)
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
