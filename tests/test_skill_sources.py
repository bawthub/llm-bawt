"""The read-only ambient skill source (TASK-1309): listing, bodies, tree, files."""
from __future__ import annotations

import os

import pytest

from agent_bridge.skill_sources import (
    MAX_VIEW_BYTES,
    AmbientSource,
    SkillSourceError,
    split_frontmatter,
)
from agent_bridge.tool_results import count_result_tokens


def _skill(root, name, body="# Body\nDo the thing.\n", meta="name: {n}\ndescription: about {n}"):
    path = root / name
    path.mkdir(parents=True)
    front = f"---\n{meta.format(n=name)}\n---\n" if meta is not None else ""
    (path / "SKILL.md").write_text(front + body)
    return path


def test_listing_reads_frontmatter_and_the_first_root_wins(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _skill(first, "shared", meta="name: from-first\ndescription: first copy")
    _skill(second, "shared", meta="name: from-second\ndescription: second copy")
    _skill(second, "only-second", meta=None)
    source = AmbientSource((first, second))

    listed = {e.ref: e for e in source.list()}
    assert sorted(listed) == ["only-second", "shared"]
    assert (listed["shared"].name, listed["shared"].description) == ("from-first", "first copy")
    assert listed["shared"].root == (first / "shared").resolve()
    assert (listed["only-second"].name, listed["only-second"].description) == ("only-second", "")


def test_listing_skips_escaping_links_odd_names_and_dirs_without_a_skill_file(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    _skill(root, "real")
    _skill(outside, "elsewhere")
    (root / "linked").symlink_to(outside / "elsewhere")
    (root / "inner-link").symlink_to(root / "real")
    _skill(root, ".hidden")
    _skill(root, "has space")
    (root / "empty").mkdir()
    refs = [e.ref for e in AmbientSource((root,)).list()]
    assert refs == ["inner-link", "real"]


def test_body_strips_frontmatter_counts_tokens_and_tracks_edits(tmp_path):
    root = tmp_path / "root"
    body = "# Body\n" + "Words of the skill. " * 50
    skill = _skill(root, "sized", body=body)
    source = AmbientSource((root,))

    read = source.read_body("sized")
    assert read.text == body and read.tokens == count_result_tokens(body)
    assert len(read.content_sha) == 64
    edited = body + "More.\n"
    (skill / "SKILL.md").write_text("---\nname: sized\n---\n" + edited)
    os.utime(skill / "SKILL.md", ns=(1, 2))  # a distinct mtime even on coarse clocks
    assert source.read_body("sized").text == edited
    assert source.read_body("missing") is None
    assert source.read_body("../root") is None


def test_malformed_frontmatter_reads_as_empty_metadata():
    assert split_frontmatter("---\n: [unclosed\n---\nbody") == ({}, "body")
    assert split_frontmatter("---\n- a list\n---\nbody") == ({}, "body")
    assert split_frontmatter("no front") == ({}, "no front")


def test_tree_lists_regular_files_inside_the_skill_only(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    skill = _skill(root, "treed")
    (skill / "reference").mkdir()
    (skill / "reference" / "deep.md").write_text("deep")
    (skill / ".git").mkdir()
    (skill / ".git" / "HEAD").write_text("x")
    (skill / ".secret").write_text("x")
    outside.mkdir()
    (outside / "leak.md").write_text("leak")
    (skill / "leak.md").symlink_to(outside / "leak.md")
    (skill / "linkdir").symlink_to(outside, target_is_directory=True)

    paths = [n.path for n in AmbientSource((root,)).tree("treed")]
    assert paths == ["SKILL.md", "reference/deep.md"]


@pytest.mark.parametrize("path", ["../other/SKILL.md", "/etc/passwd", "", "a\x00b", "leak.md"])
def test_read_file_refuses_paths_outside_the_skill(tmp_path, path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    _skill(root, "boxed")
    _skill(root, "other")
    outside.mkdir()
    (outside / "leak.md").write_text("leak")
    (root / "boxed" / "leak.md").symlink_to(outside / "leak.md")
    with pytest.raises(SkillSourceError) as caught:
        AmbientSource((root,)).read_file("boxed", path)
    assert caught.value.status == 400


def test_read_file_serves_text_and_refuses_big_binary_and_unknown(tmp_path):
    root = tmp_path / "root"
    skill = _skill(root, "files")
    (skill / "notes.md").write_text("héllo")
    (skill / "blob.bin").write_bytes(b"\x00\x01")
    (skill / "latin.txt").write_bytes(b"\xff\xfe")
    (skill / "big.md").write_text("x" * (MAX_VIEW_BYTES + 1))
    source = AmbientSource((root,))

    got = source.read_file("files", "notes.md")
    assert (got.path, got.text, got.size) == ("notes.md", "héllo", len("héllo".encode()))
    for path, status in [("blob.bin", 415), ("latin.txt", 415), ("big.md", 413), ("nope.md", 404)]:
        with pytest.raises(SkillSourceError) as caught:
            source.read_file("files", path)
        assert caught.value.status == status
    with pytest.raises(SkillSourceError) as caught:
        source.read_file("unknown", "SKILL.md")
    assert caught.value.status == 404


def test_provenance_outside_git_fails_soft(tmp_path):
    root = tmp_path / "root"
    _skill(root, "plain")
    found = AmbientSource((root,)).provenance("plain")
    assert found["source"] == "ambient"
    assert found["repo"] is None and found["commit"] is None
