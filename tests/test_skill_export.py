
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


def test_real_private_export_is_additive_overlay(tmp_path):
    source = next((path for path in (
        Path('/home/bridge/dev/agent-skills'),
        Path('/home/nick/dev/agent-skills'),
    ) if path.is_dir()), None)
    if source is None:
        pytest.skip('private authoring source is not mounted')

    destination = tmp_path / 'personal'
    result = export_personal(source, destination, version='1.0.0')

    assert 'local-infrastructure' in result['skills']
    assert not set(result['skills']).intersection({
        'agent-system', 'bawthub', 'frontend-design', 'git-commits',
        'llm-bawt', 'skill-creation', 'skill-maintenance', 'ui-ux-guidelines',
    })
    expected_private_skills = {
        path.name for path in source.iterdir()
        if path.is_dir() and (path / 'SKILL.md').is_file()
        and path.name not in PUBLIC_PACKAGE_SKILLS
    }
    assert set(result['skills']) == expected_private_skills
    for document in destination.rglob('*.md'):
        text = document.read_text()
        assert '../llm-bawt/reference/' not in text
        assert '../bawthub/references/' not in text
        assert '../agent-system/' not in text


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
