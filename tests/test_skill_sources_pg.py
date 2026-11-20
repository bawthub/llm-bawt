"""Skill sources against a private PostgreSQL (TASK-2534).

Git is replaced by a fetcher that copies prepared trees, so these cover the
rows, collision rules, update/rollback/remove and failure isolation without
network access.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from llm_bawt.skills.source_store import GitSpec, SourceError, SourceStore
from llm_bawt.skills.sources import SkillSourceService
from tests.test_bot_tags_pg import rig  # noqa: F401

pytestmark = pytest.mark.integration

REV = {n: f"{n}" * 40 for n in "abcdef"}


def skill(root: Path, name: str, body: str = "body") -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n{body}\n")


class Remote:
    """Per-URL trees and revisions standing in for remote repositories."""

    def __init__(self):
        self.trees: dict[str, tuple[Path, str]] = {}
        self.fail: set[str] = set()
        self.tokens: list[str | None] = []

    def set(self, url: str, tree: Path, revision: str) -> None:
        self.trees[url] = (tree, revision)

    def remote_revision(self, spec: GitSpec, token):
        if spec.url in self.fail:
            raise SourceError(502, "fetch_failed", "Git could not read the source.")
        return self.trees[spec.url][1]

    def fetch(self, spec, token, into: Path):
        self.tokens.append(token)
        if spec.url in self.fail:
            raise SourceError(502, "fetch_failed", "Git could not read the source.")
        tree, revision = self.trees[spec.url]
        shutil.copytree(tree, into)
        return revision


URL_A = "https://github.com/example/skills-a.git"
URL_B = "https://example.com/team/skills-b.git"


@pytest.fixture
def world(rig, tmp_path):  # noqa: F811
    bundle = tmp_path / "bundle"
    skill(bundle, "base", "bundled")
    skill(bundle, "alpha", "bundled")
    remote = Remote()
    store = SourceStore(tmp_path / "store", fallback=[bundle], fetcher=remote)
    service = SkillSourceService(rig.engine, store, lambda spec: "minted-token")
    return service, remote, tmp_path


def body(service: SkillSourceService, name: str) -> str:
    return (service.store.current / name / "SKILL.md").read_text()


def test_fresh_store_serves_bundle(world) -> None:
    service, _, _ = world
    service.ensure_current()
    assert "bundled" in body(service, "base")
    assert service.list() == {"sources": [], "pending_removals": [], "bundled": ["alpha", "base"]}


def test_two_sources_update_independently_with_rollback(world) -> None:
    service, remote, tmp = world
    a1 = tmp / "a1"
    skill(a1, "alpha", "a-v1")
    remote.set(URL_A, a1, REV["a"])
    b1 = tmp / "b1"
    skill(b1 / "skills", "beta", "b-v1")
    remote.set(URL_B, b1, REV["b"])

    added = service.add(label="Team A", url=URL_A, ref=None, subdir=None, credential="github")
    assert added["changes"]["added"] == ["alpha"]
    assert remote.tokens == ["minted-token"]
    service.add(label="Team B", url=URL_B, ref="main", subdir="skills")
    assert "a-v1" in body(service, "alpha")  # a source shadows the bundle
    assert "b-v1" in body(service, "beta")
    assert "bundled" in body(service, "base")

    # A new revision of A: check shows the diff, update activates it, B unchanged.
    a2 = tmp / "a2"
    skill(a2, "alpha", "a-v2")
    skill(a2, "gamma")
    remote.set(URL_A, a2, REV["c"])
    checked = service.check("team-a")
    assert checked["changes"]["changed"] == ["alpha"] and checked["changes"]["added"] == ["gamma"]
    assert service.list()["sources"][0]["update_available"] is True
    assert "a-v1" in body(service, "alpha")  # check never activates
    # Update activates exactly the checked revision, not whatever is newest now.
    a3 = tmp / "a3"
    skill(a3, "alpha", "a-v3")
    remote.set(URL_A, a3, REV["d"])
    with pytest.raises(SourceError) as exc:
        service.update("team-a", revision=REV["d"])
    assert exc.value.code == "revision_not_staged"
    service.update("team-a", revision=checked["latest_revision"])
    assert "a-v2" in body(service, "alpha") and "b-v1" in body(service, "beta")

    rolled = service.rollback("team-a")
    assert rolled["changes"]["removed"] == ["gamma"]
    assert "a-v1" in body(service, "alpha")
    assert not (service.store.current / "gamma").exists()


def test_collisions_failures_and_removal_leave_others_untouched(world) -> None:
    service, remote, tmp = world
    a = tmp / "a"
    skill(a, "alpha", "from-a")
    remote.set(URL_A, a, REV["a"])
    service.add(label="A", url=URL_A, ref=None, subdir=None)

    clash = tmp / "clash"
    skill(clash, "alpha", "from-b")
    remote.set(URL_B, clash, REV["b"])
    with pytest.raises(SourceError) as exc:
        service.add(label="B", url=URL_B, ref=None, subdir=None)
    assert exc.value.status == 409 and exc.value.code == "name_collision"
    assert [s["id"] for s in service.list()["sources"]] == ["a"]

    # A failed refresh records the error and keeps the active revision.
    remote.fail.add(URL_A)
    with pytest.raises(SourceError):
        service.check("a")
    listed = service.list()["sources"][0]
    assert listed["active_revision"] == REV["a"] and listed["last_error"]
    assert "from-a" in body(service, "alpha")

    # Removing the source restores the bundled copy; the bundle is untouched.
    removed = service.remove("a")
    assert removed["skills"] == ["alpha"]
    assert "bundled" in body(service, "alpha") and "bundled" in body(service, "base")
    assert not (service.store.root / "sources" / "a").exists()


def test_unreadable_source_is_never_saved(world) -> None:
    service, remote, tmp = world
    empty = tmp / "empty"
    empty.mkdir()
    remote.set(URL_A, empty, REV["a"])
    with pytest.raises(SourceError) as exc:
        service.add(label="Empty", url=URL_A, ref=None, subdir=None)
    assert exc.value.code == "no_skills"
    assert service.list()["sources"] == []


class Archive(Remote):
    """The built-in source's published archive, keyed by manifest URL."""

    def fetch(self, spec, token, into: Path):
        assert spec.keys == ("publisher-key",)  # the built-in spec always carries its keys
        return super().fetch(spec, token, into)


DEFAULT_URL = "https://releases.example.com/skills/manifest.json"


def test_builtin_archive_source_installs_once_and_cannot_be_removed(world, tmp_path) -> None:
    service, remote, _ = world
    archive = Archive()
    service.store.archive_fetcher = archive
    release = tmp_path / "release"
    skill(release / "bawthub" / "skills", "product", "release-1")
    archive.set(DEFAULT_URL, release, REV["d"])

    with pytest.raises(SourceError) as exc:  # no trusted publisher key, nothing installed
        service.ensure_default(DEFAULT_URL, subdir="bawthub/skills")
    assert exc.value.code == "untrusted_manifest" and service.list()["sources"] == []
    service.default_keys = ("publisher-key",)

    installed = service.ensure_default(DEFAULT_URL, subdir="bawthub/skills")
    assert installed and installed["source"]["builtin"] and installed["source"]["kind"] == "archive"
    assert archive.tokens == [None]  # never a credential
    assert "release-1" in body(service, "product")
    assert service.ensure_default(DEFAULT_URL, subdir="bawthub/skills") is None  # idempotent

    # A user Git source sits beside it and updates independently.
    a = tmp_path / "a"
    skill(a, "alpha", "from-a")
    remote.set(URL_A, a, REV["a"])
    service.add(label="Team", url=URL_A, ref=None, subdir=None)
    release2 = tmp_path / "release2"
    skill(release2 / "bawthub" / "skills", "product", "release-2")
    archive.set(DEFAULT_URL, release2, REV["e"])
    builtin_id = installed["source"]["id"]
    assert service.check(builtin_id)["changes"]["changed"] == ["product"]
    service.update(builtin_id, revision=REV["e"])
    assert "release-2" in body(service, "product") and "from-a" in body(service, "alpha")
    service.rollback(builtin_id)
    assert "release-1" in body(service, "product")

    with pytest.raises(SourceError) as exc:
        service.remove(builtin_id)
    assert exc.value.code == "builtin_source"


def test_failed_commit_never_publishes_and_reconcile_repairs(world, monkeypatch) -> None:
    service, remote, tmp = world
    a1 = tmp / "a1"
    skill(a1, "alpha", "a-v1")
    remote.set(URL_A, a1, REV["a"])
    service.add(label="A", url=URL_A, ref=None, subdir=None)
    a2 = tmp / "a2"
    skill(a2, "alpha", "a-v2")
    remote.set(URL_A, a2, REV["b"])
    service.check("a")

    # The transaction fails: the tree, the pin and the staged revisions all stay.
    real_get = service._get

    def failing_get(conn, source_id, lock=False):
        row = real_get(conn, source_id, lock=lock)
        if lock:
            raise RuntimeError("db went away")
        return row

    monkeypatch.setattr(service, "_get", failing_get)
    with pytest.raises(RuntimeError):
        service.update("a", revision=REV["b"])
    monkeypatch.undo()
    assert "a-v1" in body(service, "alpha")
    assert service.list()["sources"][0]["active_revision"] == REV["a"]
    assert service.store.revision_dir("a", REV["b"]).is_dir()

    # Building the new tree fails before the swap (disk full): nothing takes
    # effect, now or later, without the owner retrying.
    real_prepare = service.store.prepare

    def full_disk(layers):
        if any(REV["b"] in str(layer.path) for layer in layers):
            raise OSError(28, "No space left on device")
        return real_prepare(layers)

    monkeypatch.setattr(service.store, "prepare", full_disk)
    with pytest.raises(OSError):
        service.update("a", revision=REV["b"])
    monkeypatch.undo()
    assert "a-v1" in body(service, "alpha")
    assert service.list()["sources"][0]["active_revision"] == REV["a"]
    service.ensure_current()  # a restart does not quietly apply it
    assert "a-v1" in body(service, "alpha")
    assert service.store.layers() == {"a": REV["a"]}

    # The commit is rejected: readers never saw B, even while the transaction was open.
    seen_during_tx: list[str] = []
    monkeypatch.setattr(service, "ensure_schema", lambda: None)  # the guard is per engine object
    monkeypatch.setattr(service, "engine", CommitFails(service.engine, commit=False,
                        during=lambda: seen_during_tx.append(body(service, "alpha"))))
    with pytest.raises(RuntimeError):
        service.update("a", revision=REV["b"])
    monkeypatch.undo()
    assert seen_during_tx and all("a-v1" in s for s in seen_during_tx)
    assert "a-v1" in body(service, "alpha")
    assert service.list()["sources"][0]["active_revision"] == REV["a"]
    assert service.store.revision_dir("a", REV["b"]).is_dir()

    # Saved but the swap fails: reported as pending, effective state stays A,
    # and the next reconcile applies the saved change.
    def swap_fails(gen):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(service.store, "activate", swap_fails)
    with pytest.raises(SourceError) as exc:
        service.update("a", revision=REV["b"])
    monkeypatch.undo()
    assert exc.value.code == "apply_pending"
    listed = service.list()["sources"][0]
    assert (listed["active_revision"], listed["effective_revision"], listed["pending"]) == (REV["b"], REV["a"], True)
    assert "a-v1" in body(service, "alpha")
    service.ensure_current()
    assert "a-v2" in body(service, "alpha")
    listed = service.list()["sources"][0]
    assert (listed["effective_revision"], listed["pending"]) == (REV["b"], False)


class CommitFails:
    """Engine stand-in whose ``begin`` runs the body, then loses the commit.

    ``commit=True`` commits and then raises, like a lost acknowledgment;
    ``commit=False`` rolls back and raises, like a rejected commit.
    """

    def __init__(self, engine, *, commit: bool, during=None):
        self._engine, self._commit, self._during = engine, commit, during

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def begin(self):
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            with self._engine.connect() as conn:
                tx = conn.begin()
                yield conn
                if self._during:
                    self._during()
                tx.commit() if self._commit else tx.rollback()
                raise RuntimeError("commit outcome lost")
        return ctx()


def test_add_keeps_files_when_commit_ack_is_lost(world, monkeypatch) -> None:
    service, remote, tmp = world
    a1 = tmp / "a1"
    skill(a1, "alpha", "a-v1")
    remote.set(URL_A, a1, REV["a"])
    service.ensure_current()  # schema and an initial tree, before commits start failing
    monkeypatch.setattr(service, "ensure_schema", lambda: None)  # the guard is per engine object
    monkeypatch.setattr(service, "engine", CommitFails(service.engine, commit=True))
    with pytest.raises(RuntimeError):
        service.add(label="A", url=URL_A, ref=None, subdir=None)
    monkeypatch.undo()
    assert service.store.revision_dir("a", REV["a"]).is_dir()
    listed = service.list()["sources"][0]
    assert (listed["active_revision"], listed["pending"]) == (REV["a"], True)
    service.ensure_current()
    assert "a-v1" in body(service, "alpha")


def test_cleanup_failure_after_swap_keeps_the_live_tree(world, monkeypatch) -> None:
    service, remote, tmp = world
    a1, a2 = tmp / "a1", tmp / "a2"
    skill(a1, "alpha", "a-v1")
    skill(a2, "alpha", "a-v2")
    remote.set(URL_A, a1, REV["a"])
    service.add(label="A", url=URL_A, ref=None, subdir=None)
    remote.set(URL_A, a2, REV["b"])
    service.check("a")

    def prune_fails(generations, keep):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(service.store, "_prune", prune_fails)
    service.update("a", revision=REV["b"])  # the swap is the commit point; cleanup is best effort
    monkeypatch.undo()
    assert "a-v2" in body(service, "alpha") and "bundled" in body(service, "base")
    listed = service.list()
    assert listed["sources"][0]["pending"] is False


def test_pending_removal_stays_visible(world, monkeypatch) -> None:
    service, remote, tmp = world
    a1 = tmp / "a1"
    skill(a1, "alpha", "a-v1")
    remote.set(URL_A, a1, REV["a"])
    service.add(label="A", url=URL_A, ref=None, subdir=None)

    def swap_fails(gen):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(service.store, "activate", swap_fails)
    with pytest.raises(SourceError) as exc:
        service.remove("a")
    monkeypatch.undo()
    assert exc.value.code == "apply_pending"
    listed = service.list()
    assert listed["sources"] == [] and listed["pending_removals"] == [{"id": "a", "skills": ["alpha"]}]
    service.ensure_current()
    assert service.list()["pending_removals"] == []


def test_missing_revision_files_refuse_a_build_that_would_drop_a_source(world) -> None:
    service, remote, tmp = world
    a1 = tmp / "a1"
    skill(a1, "alpha", "a-v1")
    remote.set(URL_A, a1, REV["a"])
    service.add(label="A", url=URL_A, ref=None, subdir=None)
    shutil.rmtree(service.store.revision_dir("a", REV["a"]))
    with pytest.raises(SourceError) as exc:
        service.rebuild()
    assert exc.value.code == "revision_missing"
    assert "a-v1" in body(service, "alpha")  # the last good tree stays live


def test_disabled_store_refuses_writes_and_creates_nothing(rig, tmp_path) -> None:  # noqa: F811
    service = SkillSourceService(rig.engine, SourceStore(None, fallback=[tmp_path]), lambda spec: None)
    service.ensure_current()
    with pytest.raises(SourceError) as exc:
        service.add(label="A", url=URL_A, ref=None, subdir=None)
    assert exc.value.status == 503
    assert list(tmp_path.iterdir()) == []
