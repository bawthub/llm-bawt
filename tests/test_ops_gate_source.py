"""Real fixed Git plumbing on scratch repositories; no runner or network."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess

import pytest

from llm_bawt.ops import gate_source as source
from llm_bawt.ops.gate_plan import GatePlan

pytestmark = pytest.mark.real_process(reason="fixed Git plumbing on disposable repositories, no network")


def git(repo: Path, *args, data=None) -> bytes:
    env = {"PATH": os.defpath, "HOME": "/nonexistent", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
    return subprocess.run(["git", "-C", str(repo), *args], input=data, env=env,
        capture_output=True, check=True).stdout


def commit(repo):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "fixture")
    return git(repo, "rev-parse", "HEAD").decode().strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "--template=", "-b", "main")
    (repo / "nested").mkdir()
    (repo / "nested/data.bin").write_bytes(b"\0binary\r\n\xff")
    (repo / "run.sh").write_bytes(b"#!/bin/sh\nexit 17\n")
    (repo / "run.sh").chmod(0o755)
    (repo / "tracked.txt").write_bytes(b"old\n")
    commit(repo)
    (repo / "tracked.txt").write_bytes(b"committed\n")
    commit(repo)
    return repo


@pytest.fixture
def staging(tmp_path):
    parent = tmp_path / "staging"
    parent.mkdir(mode=0o700)
    return parent


def plan(repo):
    return GatePlan("llm-bawt", git(repo, "rev-parse", "HEAD").decode().strip())


def fingerprint(repo):
    return {p.relative_to(repo).as_posix(): (stat.S_IMODE(p.lstat().st_mode),
        os.readlink(p) if p.is_symlink() else hashlib.sha256(p.read_bytes()).hexdigest())
        for p in repo.rglob("*") if p.is_symlink() or p.is_file()}


def test_exact_private_source_excludes_dirty_staged_untracked_and_inherited_metadata(repository, staging):
    selected = plan(repository)
    tree = git(repository, "rev-parse", "HEAD^{tree}").decode().strip()
    (repository / "tracked.txt").write_bytes(b"staged\n")
    git(repository, "add", "tracked.txt")
    (repository / "tracked.txt").write_bytes(b"dirty\n")
    (repository / "untracked-secret").write_text("fixture-untracked")
    git(repository, "config", "remote.origin.url", "https://fixture.invalid/private-remote")
    git(repository, "config", "credential.helper", "!exit 99")
    (repository / ".git/hooks").mkdir()
    (repository / ".git/hooks/post-checkout").write_text("fixture-host-hook")
    before = fingerprint(repository)
    with source.staged_gate_source(selected, repository=repository, staging_parent=staging) as staged:
        root = staged.directory
        assert staged.source_sha == selected.source_sha and staged.tree_sha == tree
        assert staged.file_count == 3 and staged.byte_count == len(b"committed\n\0binary\r\n\xff#!/bin/sh\nexit 17\n")
        assert (root / "tracked.txt").read_bytes() == b"committed\n"
        assert (root / "nested/data.bin").read_bytes() == b"\0binary\r\n\xff"
        assert stat.S_IMODE((root / "run.sh").stat().st_mode) == 0o755
        assert not (root / "untracked-secret").exists()
        assert not (root / ".git/hooks").exists()
        assert not (root / ".git/objects/info/alternates").exists()
        assert git(root, "remote", "-v") == b""
        assert b"fixture.invalid" not in (root / ".git/config").read_bytes()
        assert git(root, "status", "--porcelain") == b""
        assert git(root, "rev-parse", "--verify", "HEAD^{commit}") == (selected.source_sha + "\n").encode()
        assert git(root, "rev-list", "HEAD", "--count") == b"1\n"
        # The staged object database survives total loss of the scratch source.
        original = repository / ".git"
        hidden = repository / "saved-git"
        original.rename(hidden)
        try:
            assert git(root, "show", "HEAD:tracked.txt") == b"committed\n"
        finally:
            hidden.rename(original)
    assert fingerprint(repository) == before
    assert not root.exists() and list(staging.iterdir()) == []


def test_exports_the_requested_older_commit_not_head(repository, staging):
    older = git(repository, "rev-parse", "HEAD^").decode().strip()
    with source.staged_gate_source(GatePlan("llm-bawt", older), repository=repository, staging_parent=staging) as staged:
        assert staged.source_sha == older
        assert (staged.directory / "tracked.txt").read_bytes() == b"old\n"


def test_raw_bytes_ignore_checkout_filters_export_attributes_and_eol(repository, staging, tmp_path):
    marker = tmp_path / "filter-ran"
    (repository / ".gitattributes").write_text("tracked.txt filter=fixture text eol=crlf export-ignore\nrun.sh export-subst\n")
    selected = GatePlan("llm-bawt", commit(repository))
    git(repository, "config", "filter.fixture.smudge", f"touch {marker}")
    git(repository, "config", "filter.fixture.clean", f"touch {marker}")
    git(repository, "config", "filter.fixture.required", "true")
    git(repository, "config", "core.fsmonitor", f"touch {marker}")
    before = fingerprint(repository)
    with source.staged_gate_source(selected, repository=repository, staging_parent=staging) as staged:
        assert (staged.directory / "tracked.txt").read_bytes() == b"committed\n"
        assert (staged.directory / ".gitattributes").read_bytes() == (repository / ".gitattributes").read_bytes()
    assert not marker.exists() and fingerprint(repository) == before


def test_environment_and_replace_objects_cannot_redirect_staging(repository, staging, tmp_path, monkeypatch):
    selected = plan(repository)
    # Replacement points at the earlier commit; source identity must still be original.
    git(repository, "replace", selected.source_sha, git(repository, "rev-parse", "HEAD^").decode().strip())
    for key, value in {"GIT_DIR": str(tmp_path / "absent"), "GIT_WORK_TREE": str(tmp_path),
        "GIT_INDEX_FILE": str(tmp_path / "forbidden-index"), "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/absent",
        "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.bare", "GIT_CONFIG_VALUE_0": "true",
        "GIT_TEMPLATE_DIR": str(tmp_path), "PATH": "/absent", "LD_PRELOAD": "/absent"}.items():
        monkeypatch.setenv(key, value)
    with source.staged_gate_source(selected, repository=repository, staging_parent=staging) as staged:
        assert (staged.directory / "tracked.txt").read_bytes() == b"committed\n"
    assert not (tmp_path / "forbidden-index").exists()


@pytest.mark.parametrize("kind", ["symlink", "gitlink"])
def test_links_refuse_without_modifying_source_or_leaving_staging(repository, staging, kind, tmp_path):
    if kind == "symlink":
        (repository / "linked").symlink_to(tmp_path / "outside")
        selected = GatePlan("llm-bawt", commit(repository))
    else:
        git(repository, "update-index", "--add", "--cacheinfo", "160000", plan(repository).source_sha, "linked")
        git(repository, "commit", "-qm", "fixture gitlink")
        selected = plan(repository)
    before = fingerprint(repository)
    with pytest.raises(source.GateSourceError, match="unsupported links"), source.staged_gate_source(
            selected, repository=repository, staging_parent=staging):
        pytest.fail("unsupported source must not be yielded")
    assert fingerprint(repository) == before and list(staging.iterdir()) == []


@pytest.mark.parametrize("name", [b"..", b".git", b".GiT", b"/absolute", b"a\\b", b"bad\nname", b"x" * 256])
def test_malformed_tree_paths_refuse_before_materialization(repository, staging, name):
    blob = git(repository, "hash-object", "-w", "--stdin", data=b"fixture").decode().strip()
    tree = git(repository, "hash-object", "--literally", "-t", "tree", "-w", "--stdin",
        data=b"100644 " + name + b"\0" + bytes.fromhex(blob)).decode().strip()
    raw = f"tree {tree}\nauthor fixture <fixture@example.invalid> 1 +0000\ncommitter fixture <fixture@example.invalid> 1 +0000\n\nfixture\n".encode()
    sha = git(repository, "hash-object", "-t", "commit", "-w", "--stdin", data=raw).decode().strip()
    with pytest.raises(source.GateSourceError, match="path is unsafe"), source.staged_gate_source(
            GatePlan("llm-bawt", sha), repository=repository, staging_parent=staging):
        pytest.fail("unsafe source must not be yielded")
    assert list(staging.iterdir()) == []


@pytest.mark.parametrize("limit,value", [("_MAX_FILES", 1), ("_MAX_ENTRIES", 1),
    ("_MAX_BYTES", 1), ("_MAX_OBJECT", 1), ("_MAX_TREE_BYTES", 1), ("_MAX_DEPTH", 0)])
def test_resource_bounds_refuse_without_truncation_or_partial_stage(repository, staging, monkeypatch, limit, value):
    monkeypatch.setattr(source, limit, value)
    with pytest.raises(source.GateSourceError, match="bound"), source.staged_gate_source(
            plan(repository), repository=repository, staging_parent=staging):
        pytest.fail("bounded export must not truncate and yield")
    assert list(staging.iterdir()) == []


def test_promisor_repository_cannot_lazy_fetch_or_run_a_transport_helper(repository, staging, tmp_path):
    marker = tmp_path / "transport-ran"
    git(repository, "config", "core.repositoryformatversion", "1")
    git(repository, "config", "extensions.partialclone", "fixture-origin")
    git(repository, "config", "remote.fixture-origin.promisor", "true")
    git(repository, "config", "remote.fixture-origin.url", f"ext::touch {marker}")
    git(repository, "config", "protocol.ext.allow", "always")
    before = fingerprint(repository)
    with pytest.raises(source.GateSourceError), source.staged_gate_source(
            GatePlan("llm-bawt", "a" * 40), repository=repository, staging_parent=staging):
        pytest.fail("missing source must not fetch")
    assert not marker.exists() and fingerprint(repository) == before
    assert list(staging.iterdir()) == []


def test_unknown_commit_refuses_without_exposing_git_stderr(repository, staging):
    with pytest.raises(source.GateSourceError, match="fixed object plumbing"), source.staged_gate_source(
            GatePlan("llm-bawt", "a" * 40), repository=repository, staging_parent=staging):
        pytest.fail("missing source must not be yielded")
    assert list(staging.iterdir()) == []


@pytest.mark.parametrize("bad_parent", ["shared", "linked", "inside"])
def test_staging_location_must_be_private_real_and_outside_source(repository, staging, tmp_path, bad_parent):
    if bad_parent == "shared":
        staging.chmod(0o777)
    elif bad_parent == "linked":
        link = tmp_path / "link"
        link.symlink_to(staging, target_is_directory=True)
        staging = link
    else:
        staging = repository / "staging"
        staging.mkdir(mode=0o700)
    with pytest.raises(source.GateSourceError), source.staged_gate_source(
            plan(repository), repository=repository, staging_parent=staging):
        pytest.fail("unsafe staging location must not be yielded")


def test_context_removes_its_stage_when_consumer_raises(repository, staging):
    with pytest.raises(RuntimeError, match="fixture consumer"), source.staged_gate_source(
            plan(repository), repository=repository, staging_parent=staging) as staged:
        root = staged.directory
        raise RuntimeError("fixture consumer")
    assert not root.exists() and list(staging.iterdir()) == []


@pytest.mark.parametrize("change", ["bytes", "mode", "extra_file", "extra_directory", "missing", "link"])
def test_post_export_verifier_refuses_tampering_and_cleans_partial_stage(repository, staging, monkeypatch, change):
    original = source._verify

    def verify(root, *args):
        path = root / "tracked.txt"
        if change == "bytes":
            path.write_bytes(b"changed")
        elif change == "mode":
            path.chmod(0o755)
        elif change == "extra_file":
            (root / "extra").write_bytes(b"extra")
        elif change == "extra_directory":
            (root / "extra").mkdir()
        elif change == "missing":
            path.unlink()
        else:
            path.unlink()
            path.symlink_to(repository / "tracked.txt")
        return original(root, *args)

    monkeypatch.setattr(source, "_verify", verify)
    before = fingerprint(repository)
    with pytest.raises(source.GateSourceError), source.staged_gate_source(
            plan(repository), repository=repository, staging_parent=staging):
        pytest.fail("changed private source must not be yielded")
    assert list(staging.iterdir()) == [] and fingerprint(repository) == before


def test_empty_committed_directory_is_preserved(repository, staging):
    empty = git(repository, "hash-object", "-t", "tree", "-w", "--stdin", data=b"").decode().strip()
    tree = git(repository, "hash-object", "-t", "tree", "-w", "--stdin",
        data=b"40000 empty\0" + bytes.fromhex(empty)).decode().strip()
    raw = f"tree {tree}\nauthor fixture <fixture@example.invalid> 1 +0000\ncommitter fixture <fixture@example.invalid> 1 +0000\n\nfixture\n".encode()
    sha = git(repository, "hash-object", "-t", "commit", "-w", "--stdin", data=raw).decode().strip()
    with source.staged_gate_source(GatePlan("llm-bawt", sha), repository=repository, staging_parent=staging) as staged:
        assert (staged.directory / "empty").is_dir()
        assert staged.file_count == staged.byte_count == 0
        assert staged.tree_sha == tree
