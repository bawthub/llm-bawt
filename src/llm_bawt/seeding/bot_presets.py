"""Bot presets offered at install time (TASK-2516).

One catalog serves the terminal installer and the browser wizard:

- the built-in ``starter`` pack (``bot_presets.json``): owner-neutral agents,
  one job each;
- optional packs an operator imports (for example one exported from another
  installation with ``python -m llm_bawt.seeding.preset_cli export``). Packs
  are stored in runtime settings, never shipped as product defaults.

The installer only records a selection (it may run before any owner or
provider connection exists). Applying happens once a usable model endpoint
exists, through :func:`apply_presets`, which creates each selected bot with
the admin create-only writer: an existing slug is reported and left alone, so
a rerun never duplicates or overwrites a bot.

Packs carry identity, role tags and a starting prompt only. Credentials,
history, memories and deployment addresses are never part of a pack; the new
bot runs on this installation's own connections.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

SELECTION_KEY = "setup.bot_presets.selected"
PACKS_KEY = "setup.bot_presets.packs"
BUILTIN_PACK = "starter"

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_PACK = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_TEXT_LIMITS = {"name": 255, "icon": 16, "color": 64, "role": 80, "summary": 400,
                "pick_if": 300, "differs": 400, "title": 120, "description": 600}
_PROMPT_LIMIT = 40_000
_MAX_PRESETS = 64


class PresetError(ValueError):
    """A pack or selection that cannot be used; nothing was written."""


@dataclass(frozen=True)
class BotPreset:
    pack: str
    key: str
    slug: str
    name: str
    role: str
    summary: str
    system_prompt: str
    icon: str = ""
    color: str | None = None
    tags: tuple[str, ...] = ()
    pick_if: str = ""
    differs: str = ""
    recommended: bool = False
    provider_preference: tuple[str, ...] = ()
    description: str = ""

    @property
    def ref(self) -> str:
        return f"{self.pack}/{self.key}"

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["ref"] = self.ref
        data["tags"] = list(self.tags)
        data["provider_preference"] = list(self.provider_preference)
        return data


@dataclass(frozen=True)
class PresetPack:
    pack: str
    title: str
    description: str
    presets: tuple[BotPreset, ...] = field(default_factory=tuple)

    def public(self) -> dict[str, Any]:
        return {"pack": self.pack, "title": self.title, "description": self.description,
                "builtin": self.pack == BUILTIN_PACK,
                "presets": [p.public() for p in self.presets]}


def _text(raw: Mapping[str, Any], name: str, *, required: bool = False) -> str:
    value = raw.get(name, "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise PresetError(f"preset field '{name}' must be text")
    value = value.strip()
    if required and not value:
        raise PresetError(f"preset field '{name}' is required")
    limit = _TEXT_LIMITS.get(name)
    if limit and len(value) > limit:
        raise PresetError(f"preset field '{name}' is longer than {limit} characters")
    return value


def _strings(raw: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = raw.get(name) or []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PresetError(f"preset field '{name}' must be a list of text")
    return tuple(dict.fromkeys(v.strip() for v in value if v.strip()))


def parse_pack(raw: Any) -> PresetPack:
    """Validate one pack document. Unknown fields are ignored, never stored."""
    if not isinstance(raw, Mapping):
        raise PresetError("a preset pack must be a JSON object")
    pack = str(raw.get("pack") or "").strip().lower()
    if not _PACK.fullmatch(pack):
        raise PresetError("pack id must be 1-32 lowercase letters, digits, '-' or '_'")
    items = raw.get("presets")
    if not isinstance(items, list) or not items:
        raise PresetError(f"pack '{pack}' has no presets")
    if len(items) > _MAX_PRESETS:
        raise PresetError(f"pack '{pack}' has more than {_MAX_PRESETS} presets")
    presets: list[BotPreset] = []
    keys: set[str] = set()
    slugs: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise PresetError(f"pack '{pack}' has a preset that is not an object")
        key = _text(item, "key", required=True).lower()
        slug = (_text(item, "slug") or key).lower()
        if not _SLUG.fullmatch(key) or not _SLUG.fullmatch(slug):
            raise PresetError(f"pack '{pack}': invalid preset key or slug '{key}'")
        if key in keys or slug in slugs:
            raise PresetError(f"pack '{pack}': duplicate preset '{key}'")
        keys.add(key)
        slugs.add(slug)
        prompt = item.get("system_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise PresetError(f"pack '{pack}': preset '{key}' needs a system_prompt")
        if len(prompt) > _PROMPT_LIMIT:
            raise PresetError(f"pack '{pack}': preset '{key}' prompt is too long")
        icon = _text(item, "icon")
        if icon and not portable_icon(icon):
            raise PresetError(f"pack '{pack}': preset '{key}' icon must be an emoji")
        presets.append(BotPreset(
            pack=pack, key=key, slug=slug,
            name=_text(item, "name", required=True),
            role=_text(item, "role", required=True),
            summary=_text(item, "summary", required=True),
            system_prompt=prompt.strip(),
            icon=icon,
            color=_color(item, pack, key),
            tags=_strings(item, "tags"),
            pick_if=_text(item, "pick_if"),
            differs=_text(item, "differs"),
            recommended=bool(item.get("recommended", False)),
            provider_preference=tuple(v.lower() for v in _strings(item, "provider_preference")),
            description=_text(item, "summary"),
        ))
    return PresetPack(pack=pack, title=_text(raw, "title") or pack,
                      description=_text(raw, "description"),
                      presets=tuple(presets))


# Codepoints an emoji sequence is built from: pictographs, symbols, flags,
# keycaps, and the joiners/modifiers between them.
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0x2300, 0x23FF), (0x2B00, 0x2BFF),
    (0x2190, 0x21FF), (0x25A0, 0x25FF), (0x2900, 0x297F), (0x3030, 0x3030), (0x303D, 0x303D),
    (0x3297, 0x3297), (0x3299, 0x3299), (0x00A9, 0x00A9), (0x00AE, 0x00AE), (0x2122, 0x2122),
    (0x2139, 0x2139), (0x203C, 0x203C), (0x2049, 0x2049),
)
_EMOJI_GLUE = {0x200D, 0xFE0F, 0x20E3} | set(range(0xE0020, 0xE0080))
_SAFE_COLOR = re.compile(r"^(?:#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?|[a-z]{2,20})$")


def portable_icon(value: str) -> bool:
    """One short emoji sequence (never text, a URL, a path or an upload id)."""
    if not 0 < len(value) <= 8:
        return False
    pictographs = 0
    for ch in value:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _EMOJI_RANGES):
            pictographs += 1
        elif code not in _EMOJI_GLUE:
            return False
    return 0 < pictographs <= 4


def safe_color(value: str) -> bool:
    """A hex color or a plain palette name such as ``sky``."""
    return bool(_SAFE_COLOR.fullmatch(value))


def _color(item: Mapping[str, Any], pack: str, key: str) -> str | None:
    color = _text(item, "color")
    if color and not safe_color(color):
        raise PresetError(f"pack '{pack}': preset '{key}' color must be a hex color or a palette name")
    return color or None


# Content an exported pack must not carry. Matches are reported by kind only.
_NONPORTABLE = (
    ("a URL", re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.I)),
    ("an email address", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("an IP address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("a file path", re.compile(r"(?:^|[\s\"'(`=])(?:~|\.{1,2})?/[\w.-]+/[\w./-]*|\b[A-Za-z]:\\[\w\\. -]+")),
    ("a host:port", re.compile(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*:\d{2,5}\b", re.I)),
    ("a dotted host name", re.compile(r"\b[a-z][a-z0-9-]*(?:\.[a-z0-9-]+){2,}\b", re.I)),
    ("a host or domain name", re.compile(
        r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:lan|local|localdomain|home|internal|intranet|corp|"
        r"com|net|org|io|us|dev|app|ai|co|me|cloud)\b", re.I)),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("a provider token", re.compile(
        r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bxox[abprs]-[A-Za-z0-9-]{10,}"
        r"|\bAKIA[0-9A-Z]{16}\b|\bAIza[0-9A-Za-z_-]{30,}")),
    ("a credential assignment", re.compile(
        r"\b(?:password|passwd|secret|token|api[_-]?key|bearer)\b\s*[:=]\s*\S+", re.I)),
    ("a long key-like string", re.compile(r"(?=[A-Za-z0-9+/_=-]*\d)(?=[A-Za-z0-9+/_=-]*[A-Za-z])[A-Za-z0-9+/_=-]{32,}")),
)


def nonportable_content(fields: Mapping[str, str]) -> list[str]:
    """Field names and kinds of installation-specific or secret-looking content.

    Used before an export leaves the install. Conservative on purpose: a hit
    means the operator edits that prompt (or leaves the bot out), never that
    the text is redacted and shipped anyway.
    """
    hits = []
    for name, value in fields.items():
        for kind, pattern in _NONPORTABLE:
            if value and pattern.search(value):
                hits.append(f"{name} contains {kind}")
    return hits


def pack_digest(document: Mapping[str, Any]) -> str:
    """Digest of exactly what an export would write, minus its timestamp."""
    import hashlib

    body = {k: v for k, v in document.items() if k != "exported"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def builtin_pack() -> PresetPack:
    raw = json.loads(Path(__file__).with_name("bot_presets.json").read_text(encoding="utf-8"))
    return parse_pack(raw)


def stored_packs(raw: Any) -> list[PresetPack]:
    """Imported packs from the runtime setting value; a bad entry is skipped."""
    packs: list[PresetPack] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            pack = parse_pack(item)
        except PresetError:
            continue
        if pack.pack != BUILTIN_PACK:
            packs.append(pack)
    return packs


def all_packs(stored: Any) -> list[PresetPack]:
    return [builtin_pack(), *stored_packs(stored)]


def presets_by_ref(packs: Iterable[PresetPack]) -> dict[str, BotPreset]:
    return {p.ref: p for pack in packs for p in pack.presets}


def recommended_refs(packs: Iterable[PresetPack]) -> list[str]:
    return [p.ref for pack in packs if pack.pack == BUILTIN_PACK for p in pack.presets if p.recommended]


def normalize_selection(refs: Any, known: Mapping[str, BotPreset]) -> list[str]:
    """Known refs in catalog order; refuses unknown refs and two presets on one slug."""
    if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
        raise PresetError("selection must be a list of preset refs")
    wanted = {r.strip() for r in refs if r.strip()}
    unknown = sorted(wanted - set(known))
    if unknown:
        raise PresetError(f"unknown preset(s): {', '.join(unknown)}")
    ordered = [ref for ref in known if ref in wanted]
    slugs: dict[str, str] = {}
    for ref in ordered:
        other = slugs.setdefault(known[ref].slug, ref)
        if other != ref:
            raise PresetError(f"{other} and {ref} would both create bot '{known[ref].slug}'")
    return ordered


def merge_pack(stored: Any, pack: PresetPack) -> list[dict[str, Any]]:
    """Stored packs with *pack* added or replaced (by id); the built-in id is reserved."""
    if pack.pack == BUILTIN_PACK:
        raise PresetError(f"pack id '{BUILTIN_PACK}' is reserved for the built-in lineup")
    kept = [p for p in stored_packs(stored) if p.pack != pack.pack]
    return [_pack_document(p) for p in [*kept, pack]]


def _pack_document(pack: PresetPack) -> dict[str, Any]:
    presets = []
    for p in pack.presets:
        doc = p.public()
        for derived in ("pack", "ref", "description"):
            doc.pop(derived, None)
        presets.append(doc)
    return {"pack": pack.pack, "title": pack.title, "description": pack.description,
            "presets": presets}


# --- endpoints and requirements ---------------------------------------------------------

@dataclass(frozen=True)
class EndpointChoice:
    endpoint_id: int
    label: str
    vendor: str


def choose_endpoint(preset: BotPreset, usable: list[EndpointChoice]) -> EndpointChoice | None:
    """The first usable endpoint from the preset's preferred vendors, else any usable one."""
    for vendor in preset.provider_preference:
        for choice in usable:
            if choice.vendor == vendor:
                return choice
    return usable[0] if usable else None


def plan(selection: list[str], known: Mapping[str, BotPreset], usable: list[EndpointChoice],
         existing: set[str], known_tags: set[str] | None,
         endpoint_overrides: Mapping[str, int] | None = None) -> list[dict[str, Any]]:
    """What applying *selection* would do, per preset, without writing anything."""
    by_id = {c.endpoint_id: c for c in usable}
    rows: list[dict[str, Any]] = []
    for ref in selection:
        preset = known[ref]
        row: dict[str, Any] = {"ref": ref, "slug": preset.slug, "name": preset.name}
        override = (endpoint_overrides or {}).get(ref)
        choice = by_id.get(override) if override is not None else choose_endpoint(preset, usable)
        missing_tags = [] if known_tags is None else [t for t in preset.tags if t not in known_tags]
        if preset.slug in existing:
            row.update(action="skip", reason=f"A bot named '{preset.slug}' already exists; it is left unchanged.")
        elif override is not None and choice is None:
            row.update(action="blocked", reason="The chosen model is not usable by an agent on this install.")
        elif choice is None:
            wants = ", ".join(preset.provider_preference) or "any provider"
            row.update(action="blocked",
                       reason=f"No connected model can run this agent yet. Connect a provider ({wants}) first.")
        else:
            row.update(action="create", endpoint_id=choice.endpoint_id, endpoint=choice.label)
            if preset.provider_preference and choice.vendor != preset.provider_preference[0]:
                row["note"] = f"Prefers {preset.provider_preference[0]}; will use {choice.vendor}."
        if missing_tags:
            row["missing_tags"] = missing_tags
        rows.append(row)
    return rows
