"""Native Codex preparation in a dedicated home, never on the chat hot path.

Auth/session files remain owned by the existing bridge. This isolates plugin
configuration, not filesystem access or repository-scoped skills.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
import shutil
import subprocess

from .skill_registry import SkillRegistry, _atomic_json
from .skill_packages import SkillPackageError
from .skill_selection import ManagedSelection, SkillSelectionError


def bundle_key(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def prepare_codex(registry: SkillRegistry, bundle: str, *, binary: str,
                  base_home: Path) -> dict:
    """Explicit admin action: native installation into a fresh isolated home.

    Does not mutate base_home, its config, native plugin caches or credentials.
    Generated homes retain the base configuration text and override only plugin
    and marketplace tables using a TOML-aware round trip.
    """
    import tomlkit

    state = registry.bundle(bundle)
    selection_key = bundle_key(state)
    config_path = base_home / 'config.toml'
    base_config = config_path.read_text() if config_path.exists() else ''
    config_digest = hashlib.sha256(base_config.encode()).hexdigest()
    key = hashlib.sha256((selection_key + str(base_home.resolve()) + config_digest).encode()).hexdigest()
    pointer = registry.root / 'codex-prepared' / f'{selection_key}.json'
    with registry._lock():
        homes = registry.root / 'codex'
        homes.mkdir(exist_ok=True)
        target = homes / key
        receipt_path = target / 'prepared.json'
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            _atomic_json(pointer, receipt)
            return receipt
        if target.exists():
            raise SkillPackageError('Incomplete Codex home; inspect before retrying')
        # Native manager stores absolute paths: build at the FINAL path, but
        # make it selectable only after every command succeeds.
        target.mkdir()
        try:
            codex_home = target / '.codex'
            codex_home.mkdir()
            config = tomlkit.parse(base_config)
            for section in ('plugins', 'marketplaces'):
                config.pop(section, None)
            (codex_home / 'config.toml').write_text(tomlkit.dumps(config))
            for name in ('auth.json', 'sessions'):
                if (base_home / name).exists():
                    (codex_home / name).symlink_to((base_home / name).resolve())
            marketplace = target / 'marketplace'
            entries = []
            for path, receipt in zip(registry.bundle_paths(bundle), state['packages']):
                destination = marketplace / 'plugins' / receipt['name']
                shutil.copytree(path, destination)
                entries.append({'name': receipt['name'],
                                'source': {'source': 'local', 'path': './plugins/' + receipt['name']},
                                'policy': {'installation': 'AVAILABLE', 'authentication': 'ON_INSTALL'},
                                'category': 'Productivity'})
            _atomic_json(marketplace / '.agents/plugins/marketplace.json',
                         {'name': 'bawt-managed', 'plugins': entries})
            env = dict(os.environ, HOME=str(target), CODEX_HOME=str(codex_home))
            def run(*args):
                result = subprocess.run([binary, 'plugin', *args, '--json'],
                                        env=env, cwd=target, text=True,
                                        capture_output=True, timeout=120, check=True)
                return json.loads(result.stdout)
            run('marketplace', 'add', str(marketplace))
            installed = [run('add', r['name'] + '@bawt-managed') for r in state['packages']]
            receipt = {'bundle_key': selection_key, 'generation': key,
                       'config_sha256': config_digest, 'bundle': bundle, 'home': str(target),
                       'codex_home': str(codex_home), 'installed': installed,
                       'base_home': str(base_home.resolve())}
            _atomic_json(receipt_path, receipt)
            _atomic_json(pointer, receipt)
            return receipt
        except BaseException:
            shutil.rmtree(target)
            raise


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


def codex_skill_env(bundle: str | None) -> dict | None:
    if bundle is None:
        return None
    root = os.environ.get('AGENT_SKILL_REGISTRY')
    if not root:
        raise SkillPackageError('AGENT_SKILL_REGISTRY is required for skill_bundle selection')
    registry = SkillRegistry(Path(root))
    key = bundle_key(registry.bundle(bundle))
    receipt_path = registry.root / 'codex-prepared' / f'{key}.json'
    if not receipt_path.is_file():
        raise SkillPackageError(f'Codex bundle {bundle!r} is not prepared; run prepare-codex first')
    receipt = json.loads(receipt_path.read_text())
    expected = registry.root / 'codex' / receipt['generation']
    if (Path(receipt['generation']).name != receipt['generation']
            or Path(receipt['home']) != expected
            or Path(receipt['codex_home']) != expected / '.codex'
            or not (expected / 'prepared.json').is_file()):
        raise SkillPackageError('Invalid prepared Codex home receipt')
    return dict(os.environ, HOME=receipt['home'], CODEX_HOME=receipt['codex_home'])
