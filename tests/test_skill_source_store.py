"""Skill source store: validation, pinning and atomic activation (TASK-2534)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from llm_bawt.skills.source_store import (
    GitSpec,
    Layer,
    SourceError,
    SourceStore,
    diff_skills,
    git_spec,
    scan_skills,
    validate_subdir,
)

REV_A = "a" * 40
REV_B = "b" * 40


def skill(root: Path, name: str, body: str = "body") -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n{body}\n")


class FakeFetcher:
    """Copies a prepared tree instead of running Git."""

    def __init__(self, tree: Path, revision: str):
        self.tree = tree
        self.revision = revision
        self.tokens: list[str | None] = []

    def fetch(self, spec, token, into):
        self.tokens.append(token)
        shutil.copytree(self.tree, into, symlinks=True)
        return self.revision

    def remote_revision(self, spec, token):
        return self.revision


def test_scan_accepts_skills_and_ignores_other_entries(tmp_path: Path) -> None:
    skill(tmp_path, "alpha")
    (tmp_path / "README.md").write_text("x")
    (tmp_path / ".github").mkdir()
    (tmp_path / "notes").mkdir()
    assert set(scan_skills(tmp_path)) == {"alpha"}


def test_scan_refuses_symlinks_and_bad_frontmatter(tmp_path: Path) -> None:
    skill(tmp_path, "alpha")
    (tmp_path / "alpha" / "leak").symlink_to("/etc/passwd")
    with pytest.raises(SourceError) as exc:
        scan_skills(tmp_path)
    assert exc.value.code == "symlink_refused"
    (tmp_path / "alpha" / "leak").unlink()
    (tmp_path / "alpha" / "SKILL.md").write_text("no frontmatter")
    with pytest.raises(SourceError) as exc:
        scan_skills(tmp_path)
    assert exc.value.code == "invalid_skill"


def test_scan_refuses_empty_tree(tmp_path: Path) -> None:
    with pytest.raises(SourceError) as exc:
        scan_skills(tmp_path)
    assert exc.value.code == "no_skills"


def test_diff_reports_added_changed_removed() -> None:
    assert diff_skills({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "9", "d": "4"}) == {
        "added": ["d"], "changed": ["b"], "removed": ["c"], "unchanged": ["a"],
    }


@pytest.mark.parametrize("value", ["../x", "/abs", "a/../b", ".hidden", "a\\b"])
def test_subdir_must_stay_inside_repo(value: str) -> None:
    with pytest.raises(SourceError):
        validate_subdir(value)


def test_git_spec_rejects_non_git_transports() -> None:
    for url in ("file:///etc", "ext::sh -c id", "https://user:pw@example.com/x.git"):
        with pytest.raises(SourceError):
            git_spec(url, None, None)
    spec = git_spec("git@github.com:example/skills.git", "v1.2.0", "plugins/x/skills/")
    assert spec.url == "https://github.com/example/skills.git"
    assert spec.ref == "v1.2.0" and spec.subdir == "plugins/x/skills" and spec.github


def test_stage_pins_revision_and_build_swaps_atomically(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    skill(repo / "pkg", "alpha", "v1")
    skill(repo / "pkg", "beta")
    bundle = tmp_path / "bundle"
    skill(bundle, "alpha", "bundled")
    skill(bundle, "gamma")
    store = SourceStore(tmp_path / "store", fallback=[bundle], fetcher=FakeFetcher(repo, REV_A))

    fetched = store.stage("main", GitSpec("https://example.com/r.git", subdir="pkg"), "tok")
    assert fetched.revision == REV_A and set(fetched.skills) == {"alpha", "beta"}
    assert store.fetcher.tokens == ["tok"]
    assert not any((tmp_path / "store" / "staging").iterdir())  # staging cleaned

    owners = store.build([Layer("main", store.revision_dir("main", REV_A))])
    assert owners == {"alpha": "main", "beta": "main", "gamma": "bundled"}
    current = store.current
    assert current.is_symlink()
    assert "v1" in (current / "alpha" / "SKILL.md").read_text()
    first = os.readlink(current)

    # A new revision builds a new generation; the old one stays for readers.
    skill(repo / "pkg", "alpha", "v2")
    store.fetcher = FakeFetcher(repo, REV_B)
    store.stage("main", GitSpec("https://example.com/r.git", subdir="pkg"), None)
    store.build([Layer("main", store.revision_dir("main", REV_B))])
    assert "v2" in (current / "alpha" / "SKILL.md").read_text()
    assert os.readlink(current) != first
    assert (tmp_path / "store" / first).is_dir()

    # Rollback is just a build from the previous pinned revision.
    store.build([Layer("main", store.revision_dir("main", REV_A))])
    assert "v1" in (current / "alpha" / "SKILL.md").read_text()


def test_failed_stage_leaves_active_tree_untouched(tmp_path: Path) -> None:
    good = tmp_path / "good"
    skill(good, "alpha", "good")
    store = SourceStore(tmp_path / "store", fetcher=FakeFetcher(good, REV_A))
    store.stage("main", GitSpec("https://example.com/r.git"), None)
    store.build([Layer("main", store.revision_dir("main", REV_A))])

    bad = tmp_path / "bad"
    skill(bad, "alpha", "bad")
    (bad / "alpha" / "link").symlink_to("../../x")
    store.fetcher = FakeFetcher(bad, REV_B)
    with pytest.raises(SourceError):
        store.stage("main", GitSpec("https://example.com/r.git"), None)
    assert "good" in (store.current / "alpha" / "SKILL.md").read_text()
    assert not store.revision_dir("main", REV_B).exists()


def test_missing_subdir_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    skill(repo, "alpha")
    store = SourceStore(tmp_path / "store", fetcher=FakeFetcher(repo, REV_A))
    with pytest.raises(SourceError) as exc:
        store.stage("main", GitSpec("https://example.com/r.git", subdir="nope"), None)
    assert exc.value.code == "invalid_subdir"


def test_subdir_through_a_linked_ancestor_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    skill(repo / "real" / "pkg", "alpha")
    (repo / "link").symlink_to("real")
    store = SourceStore(tmp_path / "store", fetcher=FakeFetcher(repo, REV_A))
    with pytest.raises(SourceError) as exc:
        store.stage("main", GitSpec("https://example.com/r.git", subdir="link/pkg"), None)
    assert exc.value.code == "symlink_refused"
    assert not store.revision_dir("main", REV_A).exists()


def test_disabled_store_is_unavailable() -> None:
    assert SourceStore(None).available() is False


def test_git_env_keeps_token_off_argv_and_disables_hooks() -> None:
    from llm_bawt.skills.source_store import GitFetcher

    env = GitFetcher._env("secret-token")
    pairs = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))}
    assert pairs["core.hooksPath"] == "/dev/null"
    assert pairs["protocol.allow"] == "never"
    assert "Authorization: Basic" in pairs["http.https://github.com/.extraHeader"]
    assert "GIT_CONFIG_COUNT" in GitFetcher._env(None)


def test_bridge_links_store_current_only_when_it_resolves(tmp_path: Path, monkeypatch) -> None:
    from agent_bridge.skill_selection import legacy_claude_skill_roots
    from claude_code_bridge.bootstrap import skills_link_target

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    current = tmp_path / "store" / "current"
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(bundle))
    monkeypatch.setenv("SKILL_SOURCES_CURRENT", str(current))
    assert skills_link_target(str(bundle), str(current)) == bundle  # volume not mounted
    assert legacy_claude_skill_roots() == (bundle,)
    current.parent.mkdir()
    current.symlink_to("generations/1")  # dangling: not built yet
    assert skills_link_target(str(bundle), str(current)) == bundle
    (current.parent / "generations" / "1").mkdir(parents=True)
    assert skills_link_target(str(bundle), str(current)) == current
    assert legacy_claude_skill_roots() == (current,)
    assert skills_link_target(str(bundle), None) == bundle  # old bridge setting unchanged


def _serve(directory: Path):
    import functools
    import http.server
    import threading

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _keypair():
    import base64

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return key, base64.b64encode(public).decode()


def _publish(site: Path, tree: Path, version: str, key=None, *, digest: str | None = None) -> None:
    """Write an immutable versioned tarball and its (optionally signed) manifest."""
    import base64
    import hashlib
    import json
    import tarfile

    site.mkdir(exist_ok=True)
    name = f"skills-{version}.tar.gz"
    with tarfile.open(site / name, "w:gz") as bundle:
        bundle.add(tree, arcname="bawthub")
    real = hashlib.sha256((site / name).read_bytes()).hexdigest()
    body = json.dumps({"version": version, "archive": name, "sha256": digest or real}).encode()
    (site / "manifest.json").write_bytes(body)
    if key is not None:
        (site / "manifest.json.sig").write_text(base64.b64encode(key.sign(body)).decode())


def test_archive_source_verifies_signature_and_hash(tmp_path: Path, monkeypatch) -> None:
    from llm_bawt.skills.source_store import ArchiveFetcher, archive_spec

    monkeypatch.setenv("SKILL_SOURCES_ALLOW_HTTP", "1")
    key, public = _keypair()
    tree = tmp_path / "pkg"
    skill(tree / "skills", "alpha", "released")
    site = tmp_path / "site"
    _publish(site, tree, "1.0.0", key)
    server, base = _serve(site)
    try:
        store = SourceStore(tmp_path / "store", archive_fetcher=ArchiveFetcher(timeout=10))
        spec = archive_spec(f"{base}/manifest.json", "bawthub/skills", [public])
        revision = store.archive_fetcher.remote_revision(spec, None)
        fetched = store.stage("bawthub", spec, None)
        assert fetched.revision == revision and set(fetched.skills) == {"alpha"}

        # Signed by someone else: refused before anything is downloaded.
        other, _ = _keypair()
        _publish(site, tree, "1.0.1", other)
        with pytest.raises(SourceError) as exc:
            store.stage("bawthub", spec, None)
        assert exc.value.code == "untrusted_manifest"

        # Unsigned: refused when keys are required.
        (site / "manifest.json.sig").unlink()
        with pytest.raises(SourceError):
            store.stage("bawthub", spec, None)

        # Correctly signed but the digest doesn't match the tarball.
        _publish(site, tree, "1.0.2", key, digest="0" * 64)
        with pytest.raises(SourceError) as exc:
            store.stage("bawthub", spec, None)
        assert exc.value.code == "hash_mismatch"

        # Signed releases name an immutable, versioned archive.
        _publish(site, tree, "latest", key)
        import base64
        import json
        body = json.dumps({"version": "2.0.0", "archive": "skills-latest.tar.gz", "sha256": "a" * 64}).encode()
        (site / "manifest.json").write_bytes(body)
        (site / "manifest.json.sig").write_text(base64.b64encode(key.sign(body)).decode())
        with pytest.raises(SourceError) as exc:
            store.stage("bawthub", spec, None)
        assert exc.value.code == "invalid_manifest"
    finally:
        server.shutdown()


def test_archive_refuses_links_and_plain_http(tmp_path: Path, monkeypatch) -> None:
    import io
    import tarfile

    from llm_bawt.skills.source_store import ArchiveFetcher, archive_spec

    with pytest.raises(SourceError):
        archive_spec("http://example.com/manifest.json", None)  # HTTPS only by default
    monkeypatch.setenv("SKILL_SOURCES_ALLOW_HTTP", "1")
    site = tmp_path / "site"
    site.mkdir()
    with tarfile.open(site / "skills.tar.gz", "w:gz") as bundle:
        info = tarfile.TarInfo("alpha/SKILL.md")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        bundle.addfile(info, io.BytesIO())
    import hashlib
    import json

    digest = hashlib.sha256((site / "skills.tar.gz").read_bytes()).hexdigest()
    (site / "manifest.json").write_text(json.dumps({"archive": "skills.tar.gz", "sha256": digest}))
    server, base = _serve(site)
    try:
        store = SourceStore(tmp_path / "store", archive_fetcher=ArchiveFetcher(timeout=10))
        with pytest.raises(SourceError) as exc:
            store.stage("bawthub", archive_spec(f"{base}/manifest.json", None), None)
        assert exc.value.code == "symlink_refused"
    finally:
        server.shutdown()


def test_replaced_archive_with_matching_checksum_is_refused_and_active_tree_kept(tmp_path: Path, monkeypatch) -> None:
    """An attacker swaps the tarball and rewrites the manifest's sha256 to match.

    The checksum then agrees with the archive, but the manifest no longer
    matches the publisher's signature, so nothing is staged and the active
    tree (and the bundle beneath it) stays exactly as it was.
    """
    import hashlib
    import json

    from llm_bawt.skills.source_store import ArchiveFetcher, archive_spec

    monkeypatch.setenv("SKILL_SOURCES_ALLOW_HTTP", "1")
    key, public = _keypair()
    tree = tmp_path / "pkg"
    skill(tree / "skills", "alpha", "genuine")
    bundle = tmp_path / "bundle"
    skill(bundle, "base", "bundled")
    site = tmp_path / "site"
    _publish(site, tree, "1.0.0", key)
    server, base = _serve(site)
    try:
        store = SourceStore(tmp_path / "store", fallback=[bundle], archive_fetcher=ArchiveFetcher(timeout=10))
        spec = archive_spec(f"{base}/manifest.json", "bawthub/skills", [public])
        good = store.stage("bawthub", spec, None)
        store.build([Layer("bawthub", store.revision_dir("bawthub", good.revision))])
        active = os.readlink(store.current)

        evil = tmp_path / "evil"
        skill(evil / "skills", "alpha", "malicious")
        import tarfile
        with tarfile.open(site / "skills-1.0.0.tar.gz", "w:gz") as bundle_file:
            bundle_file.add(evil, arcname="bawthub")
        manifest = json.loads((site / "manifest.json").read_text())
        manifest["sha256"] = hashlib.sha256((site / "skills-1.0.0.tar.gz").read_bytes()).hexdigest()
        (site / "manifest.json").write_text(json.dumps(manifest))  # old signature left in place

        with pytest.raises(SourceError) as exc:
            store.stage("bawthub", spec, None)
        assert exc.value.code == "untrusted_manifest"
        assert os.readlink(store.current) == active
        assert "genuine" in (store.current / "alpha" / "SKILL.md").read_text()
        assert "bundled" in (store.current / "base" / "SKILL.md").read_text()
        assert {p.name for p in (store.root / "sources" / "bawthub" / "revs").iterdir()} == {good.revision}
    finally:
        server.shutdown()


def test_trust_root_ships_with_the_app_and_is_not_env_configurable(monkeypatch) -> None:
    from llm_bawt.skills import source_config

    monkeypatch.setenv("SKILL_SOURCES_DEFAULT_KEYS", "attacker-key")
    keys = source_config.default_keys()
    assert "attacker-key" not in keys
    assert source_config.Path(source_config.__file__).with_name("publisher_keys.txt").is_file()


def test_archive_expansion_limit_stops_before_writing(tmp_path: Path, monkeypatch) -> None:
    from llm_bawt.skills import source_store
    from llm_bawt.skills.source_store import ArchiveFetcher, archive_spec

    monkeypatch.setenv("SKILL_SOURCES_ALLOW_HTTP", "1")
    key, public = _keypair()
    tree = tmp_path / "pkg"
    skill(tree / "skills", "alpha", "x" * 4096)
    skill(tree / "skills", "beta", "y" * 4096)
    site = tmp_path / "site"
    _publish(site, tree, "1.0.0", key)
    monkeypatch.setattr(source_store, "MAX_EXPANDED_BYTES", 5000)
    server, base = _serve(site)
    try:
        store = SourceStore(tmp_path / "store", archive_fetcher=ArchiveFetcher(timeout=10))
        spec = archive_spec(f"{base}/manifest.json", "bawthub/skills", [public])
        with pytest.raises(SourceError) as exc:
            store.stage("bawthub", spec, None)
        assert exc.value.code == "too_large"
        assert not (tmp_path / "store" / "sources" / "bawthub" / "revs").exists() or not any(
            (tmp_path / "store" / "sources" / "bawthub" / "revs").iterdir())
        assert not any((tmp_path / "store" / "staging").iterdir())
    finally:
        server.shutdown()


def test_lifespan_keeps_threading_global() -> None:
    """A local ``import threading`` in lifespan made the skill-source startup
    thread raise UnboundLocalError, so the built-in source never installed."""
    from llm_bawt.service import api

    assert "threading" not in api.lifespan.__code__.co_varnames
