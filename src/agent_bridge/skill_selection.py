"""Resolve a turn's skills: an explicit bundle, a managed selection, or ambient.

A managed selection (TASK-1309) is the skill list the app pinned for the
session (``llm_bawt.skills``); a bundle is the registry package path. They are
exclusive. With neither, the harness discovers its ambient skills unchanged.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .skill_registry import SkillRegistry, _name

#: Command field carrying a :class:`ManagedSelection`.
SELECTION_FIELD = "skill_selection"
#: Command field naming the skills whose bodies the turn's system prompt
#: carries (TASK-1691), comma-separated. Independent of any selection.
INJECTED_FIELD = "injected_skills"
#: The settings sources the Claude CLI loads when none are named. Passing
#: ``skills`` makes the SDK default them to user+project only, which would
#: silently drop local settings (hooks, permissions) for a managed bot.
CLAUDE_DEFAULT_SETTING_SOURCES = ("user", "project", "local")


class SkillSelectionError(ValueError):
    """A managed selection that cannot be applied; the turn fails closed."""


@dataclass(frozen=True)
class ManagedSelection:
    """The skills one managed session may use, by ambient ref (directory name).

    Built from the resolver's manifest (``SkillResolution.manifest``); its
    ``entries`` are the applied skills, available and injected alike.
    """

    refs: tuple[str, ...]
    manifest_sha: str

    @classmethod
    def from_manifest(cls, manifest: Mapping) -> ManagedSelection:
        from .skill_sources import _REF

        entries = manifest.get("entries") if isinstance(manifest, Mapping) else None
        sha = manifest.get("manifest_sha") if isinstance(manifest, Mapping) else None
        if not isinstance(entries, list) or not isinstance(sha, str) or not sha:
            raise SkillSelectionError("skill selection needs entries and a manifest_sha")
        refs = [entry.get("ref") if isinstance(entry, Mapping) else None for entry in entries]
        if not all(isinstance(ref, str) and _REF.fullmatch(ref) for ref in refs):
            raise SkillSelectionError("skill selection has an invalid ref")
        return cls(tuple(dict.fromkeys(refs)), sha)

    @classmethod
    def from_field(cls, raw: str | None) -> ManagedSelection | None:
        """Parse the command field; absent means unmanaged, malformed raises."""
        if raw is None or raw == "":
            return None
        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise SkillSelectionError("skill selection is not JSON") from exc
        return cls.from_manifest(data)

    def to_field(self) -> str:
        return json.dumps({"entries": [{"ref": ref} for ref in self.refs],
                           "manifest_sha": self.manifest_sha}, separators=(",", ":"))


def injected_field(refs) -> str:
    """The :data:`INJECTED_FIELD` value; an invalid ref raises."""
    from .skill_sources import _REF

    refs = list(refs)
    if not all(isinstance(ref, str) and _REF.fullmatch(ref) for ref in refs):
        raise SkillSelectionError("injected skills have an invalid ref")
    return ",".join(dict.fromkeys(refs))


def legacy_claude_skill_roots() -> tuple[Path, ...]:
    """Legacy CLI discovery roots (the bootstrap default links ~/.claude/skills)."""
    configured = os.environ.get("CLAUDE_CODE_SKILLS_PATH")
    return tuple(
        Path(value) for value in configured.split(os.pathsep) if value
    ) if configured else (Path.home() / ".claude" / "skills",)


def claude_skill_names(bundle: str | None,
                       selection: ManagedSelection | None = None) -> tuple[str, ...]:
    """Return the exact Skill selectors available to one Claude CLI run.

    Explicit bundles and managed selections already carry their authoritative
    selector list. Legacy/ambient runs discover the direct children of
    Claude's skill directory, matching the CLI's own ``<name>/SKILL.md``
    convention.
    """
    if bundle is not None or selection is not None:
        return tuple(claude_skill_options(bundle, selection).get("skills") or ())

    names: set[str] = set()
    for root in legacy_claude_skill_roots():
        try:
            for child in root.iterdir():
                if child.is_dir() and (child / "SKILL.md").is_file():
                    names.add(child.name)
        except OSError:
            continue
    return tuple(sorted(names))


def guard_resumed_bundle(bundle: str | None, *, thread: str | None, resume: str | None,
                         harness: str = 'claude') -> None:
    """Require a fresh conversation when its configured skill bundle changes."""
    import hashlib
    import json
    from .skill_registry import _atomic_json
    from .skill_packages import SkillPackageError

    # Opt-in means legacy turns have no registry dependency at all: no reads,
    # locks, directories or session-binding files when no bundle is selected.
    if bundle is None:
        return
    root = os.environ.get("AGENT_SKILL_REGISTRY")
    if not root or not thread:
        if resume:
            raise SkillPackageError("Start a new conversation before selecting a skill bundle")
        return
    registry = SkillRegistry(Path(root))
    state = registry.bundle(bundle)
    generation = None
    if harness == 'codex':
        from .skill_codex import codex_skill_env
        generation = codex_skill_env(bundle)['CODEX_HOME']
    digest = hashlib.sha256(json.dumps([state, generation], sort_keys=True).encode()).hexdigest()
    key = hashlib.sha256((harness + ':' + thread).encode()).hexdigest()
    path = registry.root / "session-bindings" / f"{key}.json"
    with registry._lock():
        if resume:
            if path.exists():
                old = json.loads(path.read_text())
                if old["bundle_digest"] != digest:
                    raise SkillPackageError("Skill bundle changed; start a new conversation to activate it")
            else:
                raise SkillPackageError("Start a new conversation before selecting a skill bundle")
        _atomic_json(path, {"bundle_digest": digest, "bundle": bundle})


def claude_skill_options(bundle: str | None, selection: ManagedSelection | None = None) -> dict:
    if selection is not None:
        if bundle is not None:
            raise SkillSelectionError("a skill bundle and a managed selection are exclusive")
        # ``skills`` filters the listing and allow-lists Skill(name); an empty
        # list hides every skill. Settings sources stay the CLI's own.
        return {"skills": list(selection.refs),
                "setting_sources": list(CLAUDE_DEFAULT_SETTING_SOURCES)}
    if bundle is None:
        return {}
    _name(bundle)
    root = os.environ.get("AGENT_SKILL_REGISTRY")
    if not root:
        raise ValueError("AGENT_SKILL_REGISTRY is required for skill_bundle selection")
    registry = SkillRegistry(Path(root))
    state = registry.bundle(bundle)
    paths = registry.bundle_paths(bundle)
    return {
        "plugins": [{"type": "local", "path": str(path)} for path in paths],
        "skills": [f"{receipt['name']}:{skill}"
                   for receipt in state['packages'] for skill in receipt['skills']],
        # Explicitly supplied skills must not opt into ambient user/project
        # discovery. The bridge still provides its MCP and hook settings.
        "setting_sources": [],
    }
