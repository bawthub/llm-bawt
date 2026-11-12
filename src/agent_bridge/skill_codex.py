"""Codex skill discovery and managed-selection overrides.

Codex has no skill allow-list, so a managed selection (TASK-1309) is applied
as ``codex -c`` overrides that disable every discovered skill it does not name.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from .skill_selection import ManagedSelection, SkillSelectionError


#: Marketplace ``source_type`` of a plugin built from a local checkout. Plugin
#: skills from any other marketplace are never selectable by an ambient ref.
LOCAL_MARKETPLACE = 'local'


def _skill_name(skill_dir: Path) -> str:
    from .skill_sources import _frontmatter_of

    name = _frontmatter_of(skill_dir / 'SKILL.md').get('name')
    return name.strip() if isinstance(name, str) and name.strip() else skill_dir.name


def _local_marketplaces(codex_home: Path) -> frozenset[str]:
    import tomllib

    try:
        config = tomllib.loads((codex_home / 'config.toml').read_text())
    except (OSError, ValueError):
        return frozenset()  # unreadable: no plugin skill is selectable
    marketplaces = config.get('marketplaces')
    if not isinstance(marketplaces, dict):
        return frozenset()
    return frozenset(name for name, table in marketplaces.items()
                     if isinstance(table, dict) and table.get('source_type') == LOCAL_MARKETPLACE)


def codex_skills(env: Mapping[str, str]) -> list[tuple[str, str, bool]]:
    """Every skill Codex discovers outside its bundled set: (name, ref, selectable).

    User skills (``$CODEX_HOME/skills``, ``$HOME/.agents/skills``) are named
    by their frontmatter name; plugin skills
    (``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/skills/<dir>``)
    are ``<plugin>:<name>`` and selectable only from a local marketplace.
    ``ref`` is the skill directory name, the selector a managed selection uses.
    """
    home = Path(env.get('HOME') or Path.home())
    codex_home = Path(env.get('CODEX_HOME') or home / '.codex')
    local = _local_marketplaces(codex_home)
    found: dict[str, tuple[str, str, bool]] = {}

    def add(name: str, ref: str, selectable: bool) -> None:
        if name not in found or (selectable and not found[name][2]):
            found[name] = (name, ref, selectable)

    for root in (codex_home / 'skills', home / '.agents' / 'skills'):
        for skill_dir in sorted(root.glob('*/')) if root.is_dir() else ():
            if not skill_dir.name.startswith('.') and (skill_dir / 'SKILL.md').is_file():
                add(_skill_name(skill_dir), skill_dir.name, True)
    for skill_dir in sorted((codex_home / 'plugins' / 'cache').glob('*/*/*/skills/*/')):
        if skill_dir.name.startswith('.') or not (skill_dir / 'SKILL.md').is_file():
            continue
        marketplace, plugin = skill_dir.parts[-5], skill_dir.parts[-4]
        add(f'{plugin}:{_skill_name(skill_dir)}', skill_dir.name, marketplace in local)
    return [found[name] for name in sorted(found)]


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def codex_config_overrides(selection: ManagedSelection | None, *,
                           env: Mapping[str, str] | None = None) -> list[str]:
    """``codex -c`` overrides that confine one run to a managed selection.

    Codex has no skill allow-list, so this turns the bundled skills off and
    disables every discovered skill the selection does not name (the
    deny-list complement). Bare ``key=value`` strings with TOML values; an
    unmanaged run (None) gets none. Each value is parsed back with tomllib,
    because Codex reads a value that is not valid TOML as a raw string.
    """
    import tomllib

    if selection is None:
        return []
    allowed = set(selection.refs)
    denied = [name for name, ref, selectable in codex_skills(os.environ if env is None else env)
              if not (selectable and ref in allowed)]
    overrides = {'skills.bundled.enabled': 'false'}
    if denied:
        overrides['skills.config'] = '[' + ','.join(
            f'{{name={_toml_string(name)},enabled=false}}' for name in denied) + ']'
    expected = {'bundled': {'enabled': False},
                **({'config': [{'name': n, 'enabled': False} for n in denied]} if denied else {})}
    out = [f'{key}={value}' for key, value in overrides.items()]
    try:
        parsed = tomllib.loads('\n'.join(out))
    except ValueError as exc:
        raise SkillSelectionError('Codex skill overrides are not valid TOML') from exc
    if parsed != {'skills': expected}:
        raise SkillSelectionError('Codex skill overrides do not round-trip')
    return out
