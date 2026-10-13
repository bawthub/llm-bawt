"""Build a private compatibility package without editing the live skills repo.

This adapter preserves legacy root layout inside the artifact so ../ references
and shared scripts keep resolving. It is PRIVATE, never a public publish tool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

from .skill_packages import SkillPackageError, validate_package


# Product/codebase knowledge has one canonical owner in the public package.
# The personal package is an additive private overlay, never a second copy that
# can drift. tenant-tools is public-only but has no legacy private source entry.
PUBLIC_PACKAGE_SKILLS = frozenset({
    'agent-system',
    'bawthub',
    'frontend-design',
    'git-commits',
    'llm-bawt',
    'skill-creation',
    'skill-maintenance',
    'ui-ux-guidelines',
})
_PUBLIC_SKILL_LINK = re.compile(
    r"\[[^\]]*\]\((?:\.\./)+(?:" + "|".join(sorted(PUBLIC_PACKAGE_SKILLS))
    + r")/(?:[^/()]+\.md|(?:reference|references)/[^)]+\.md)(?:#[^)]+)?\)"
)
_PUBLIC_DOC_PATH = re.compile(
    r"(?:\.\./)+(?P<skill>" + "|".join(sorted(PUBLIC_PACKAGE_SKILLS))
    + r")/(?:[^/()\s`]+\.md|(?:reference|references)/[^\s`)]+\.md)(?:#[^\s`)]*)?"
)


def _replace_public_skill_links(text: str) -> str:
    """Replace links to excluded public trees with their qualified skill name."""
    def replacement(match: re.Match[str]) -> str:
        target = match.group(0)
        skill = next(name for name in PUBLIC_PACKAGE_SKILLS if f"/{name}/" in target)
        return f"`bawthub:{skill}` public skill"

    linked = _PUBLIC_SKILL_LINK.sub(replacement, text)
    # Private skills also cite public docs as inline code/plain paths. Those
    # paths cannot resolve after the public skill trees are excluded.
    return _PUBLIC_DOC_PATH.sub(
        lambda match: f"bawthub:{match.group('skill')} public skill", linked
    )


def export_personal(source: Path, destination: Path, *, version: str) -> dict:
    source = source.resolve(strict=True)
    if destination.exists():
        raise SkillPackageError('Destination already exists; exports never overwrite authoring trees')
    if destination.absolute().is_relative_to(source):
        raise SkillPackageError('Export must be outside the source checkout')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.personal-', dir=destination.parent))
    try:
        names = []
        for path in sorted(source.iterdir()):
            if (
                path.name not in PUBLIC_PACKAGE_SKILLS
                and path.is_dir()
                and (path / 'SKILL.md').is_file()
            ):
                # Reject links before copying so outside/private source ownership
                # is explicit and the resulting tree is stable and self-contained.
                if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):
                    raise SkillPackageError(f'Authoring skill contains symlinks: {path.name}')
                copied = stage / 'skills' / path.name
                shutil.copytree(path, copied, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                # The shared scripts directory stays at package root. Adjust only
                # references whose resolved source target is inside that tree.
                for document in copied.rglob('*.md'):
                    text = document.read_text()
                    depth = len(document.relative_to(copied).parts)
                    old_prefix = '../' * depth + 'scripts/'
                    new_prefix = '../' + old_prefix
                    document.write_text(_replace_public_skill_links(
                        text.replace(old_prefix, new_prefix)
                    ))
                # Legacy loaders tolerated unquoted one-line descriptions
                # containing ': '. Normalize only that known representation;
                # all other invalid metadata still fails validation.
                skill_file = copied / 'SKILL.md'
                text = skill_file.read_text()
                lines = text.splitlines()
                if lines and lines[0] == '---':
                    end = lines.index('---', 1)
                    for index in range(1, end):
                        if lines[index].startswith('description: '):
                            value = lines[index][13:]
                            if value and value[0] not in '\"\'|>':
                                lines[index] = 'description: ' + json.dumps(value, ensure_ascii=False)
                    skill_file.write_text('\n'.join(lines) + '\n')
                names.append(path.name)
        scripts = source / 'scripts'
        if scripts.is_dir():
            # Skills use ../scripts/...; keep this shared sibling resource tree.
            if any(p.is_symlink() for p in scripts.rglob('*')):
                raise SkillPackageError('Shared scripts contain symlinks')
            shutil.copytree(scripts, stage / 'skills' / 'scripts',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            # A resource directory is deliberately outside discovery, not a skill.
            shutil.move(stage / 'skills/scripts', stage / 'scripts')
            for document in (stage / 'scripts').rglob('*.md'):
                document.write_text(_replace_public_skill_links(document.read_text()))
        manifest = dict(name='personal', version=version,
                        description='Private operator skills; never distribute to tenants',
                        author={'name': 'Private operator'}, skills='./skills/')
        for vendor in ('claude', 'codex'):
            path = stage / f'.{vendor}-plugin/plugin.json'
            path.parent.mkdir()
            path.write_text(json.dumps(manifest, indent=2) + '\n')
        result = validate_package(stage)
        stage.rename(destination)
        return {'name': result.name, 'version': result.version, 'skills': names,
                'sha256': result.sha256, 'visibility': 'private', 'path': str(destination)}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    print(json.dumps(export_personal(args.source, args.destination, version=args.version), indent=2))


if __name__ == '__main__':
    main()
