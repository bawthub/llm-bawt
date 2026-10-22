
from pathlib import Path

import pytest

from agent_bridge.skill_export import PUBLIC_PACKAGE_SKILLS, export_personal
from agent_bridge.skill_packages import SkillPackageError, validate_package


def test_private_export_preserves_cross_skill_links_and_scripts(tmp_path):
    source = tmp_path / 'source'
    skill = source / 'one'
    skill.mkdir(parents=True)
    original = '---\nname: one\ndescription: Contains a colon: accepted by legacy\n---\nUse ../scripts/helper and ../two/SKILL.md\n'
    (skill / 'SKILL.md').write_text(original)
    (source / 'scripts').mkdir()
    (source / 'scripts/helper').write_text('helper')
    (source / 'scripts/README.md').write_text(
        'See [llm-bawt](../llm-bawt/SKILL.md) for product architecture.\n'
        'Also read `../llm-bawt/reference/ops-actions.md` and '
        '`../bawthub/references/prod-release.md`.\n'
    )
    target = tmp_path / 'package'
    result = export_personal(source, target, version='1.0.0')
    assert result['visibility'] == 'private'
    text = (target / 'skills/one/SKILL.md').read_text()
    assert '../../scripts/helper' in text
    assert '../two/SKILL.md' in text
    exported_readme = (target / 'scripts/README.md').read_text()
    assert '`bawthub:llm-bawt` public skill' in exported_readme
    assert '`bawthub:llm-bawt public skill`' in exported_readme
    assert '`bawthub:bawthub public skill`' in exported_readme
    assert '../llm-bawt/reference/' not in exported_readme
    assert (source / 'one/SKILL.md').read_text() == original
    assert validate_package(target).skills == ('one',)


def test_private_export_excludes_public_owned_skill_names(tmp_path):
    source = tmp_path / 'source'
    private = source / 'private-only'
    private.mkdir(parents=True)
    (private / 'SKILL.md').write_text(
        '---\nname: private-only\ndescription: Private overlay\n---\nprivate\n'
    )
    public = source / 'llm-bawt'
    public.mkdir()
    (public / 'SKILL.md').write_text(
        '---\nname: llm-bawt\ndescription: Duplicate public content\n---\nwrong owner\n'
    )

    result = export_personal(source, tmp_path / 'package', version='1.0.0')

    assert result['skills'] == ['private-only']
    assert not (tmp_path / 'package/skills/llm-bawt').exists()


def _write_skill(root: Path, name: str, body: str = '') -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: {name} skill\n---\n{body}')
    return skill


def _authoring_tree(root: Path) -> Path:
    """A synthetic authoring checkout shaped like the real one: every
    public-owned skill, private skills that cite public docs at several depths,
    a shared scripts tree, and directories that are not skills."""
    for name in PUBLIC_PACKAGE_SKILLS:
        _write_skill(root, name, 'public product knowledge\n')
    infra = _write_skill(
        root, 'local-infrastructure',
        'See [llm-bawt](../llm-bawt/SKILL.md) and '
        '[ops](../llm-bawt/reference/ops-actions.md).\n'
        'Read `../agent-system/task-conversation-architecture.md`.\n'
        'Run ../scripts/pgq.\n',
    )
    (infra / 'reference').mkdir()
    (infra / 'reference/hosts.md').write_text(
        'Release: ../../bawthub/references/prod-release.md\n'
        '[arch](../../agent-system/SKILL.md)\n'
    )
    _write_skill(root, 'game-modding', 'Prompt rules: ../bawthub/references/prompts.md\n')
    (root / 'scripts').mkdir()
    (root / 'scripts/pgq').write_text('#!/bin/sh\n')
    (root / 'scripts/README.md').write_text('See ../llm-bawt/reference/data-model.md\n')
    (root / 'backups').mkdir()  # no SKILL.md: not a skill
    (root / 'backups/notes.md').write_text('not exported\n')
    return root


def test_private_export_is_an_additive_overlay_of_the_authoring_tree(tmp_path):
    source = _authoring_tree(tmp_path / 'source')
    destination = tmp_path / 'personal'

    result = export_personal(source, destination, version='1.0.0')

    # Exactly the private skills: no public-owned name, no non-skill directory.
    assert result['skills'] == ['game-modding', 'local-infrastructure']
    assert validate_package(destination).skills == ('game-modding', 'local-infrastructure')
    assert not (destination / 'skills/backups').exists()
    # Every exported document is self-contained: no path into a public tree
    # survives, at any depth, linked or inline.
    documents = list(destination.rglob('*.md'))
    assert len(documents) == 4
    for document in documents:
        text = document.read_text()
        for public in PUBLIC_PACKAGE_SKILLS:
            assert f'../{public}/' not in text, (document, public)
    hosts = (destination / 'skills/local-infrastructure/reference/hosts.md').read_text()
    assert 'bawthub:bawthub public skill' in hosts
    assert '`bawthub:agent-system` public skill' in hosts
    # hosts.md only cites public docs (two levels up), so a partial rewrite
    # that leaves any dangling '../' prefix is a leak.
    assert '../' not in hosts, hosts
    # Shared scripts keep resolving from the moved skill location.
    infra = (destination / 'skills/local-infrastructure/SKILL.md').read_text()
    assert '../../scripts/pgq' in infra
    assert (destination / 'scripts/pgq').is_file()


def test_private_export_never_overwrites(tmp_path):
    with pytest.raises(SkillPackageError, match='already exists'):
        export_personal(tmp_path, tmp_path, version='1.0.0')


def test_private_export_rejects_symlink_resources(tmp_path):
    source = tmp_path / 'source'
    skill = source / 'one'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: one\ndescription: example\n---\n')
    (skill / 'outside').symlink_to(tmp_path / 'missing')
    with pytest.raises(SkillPackageError, match='symlinks'):
        export_personal(source, tmp_path / 'package', version='1.0.0')
    assert not (tmp_path / 'package').exists()
