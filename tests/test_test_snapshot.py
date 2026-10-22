"""Exact-bytes snapshot gate (TASK-1137): apply, pin, ownership, cleanup."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from scripts import test_snapshot as snap

SCRIPT = Path(snap.__file__).resolve()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def hermetic_git(monkeypatch):
    for key, value in {
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        # The test's own `git status` must not refresh the index it fingerprints.
        "GIT_OPTIONAL_LOCKS": "0",
    }.items():
        monkeypatch.setenv(key, value)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def repo(tmp_path, hermetic_git) -> Path:
    path = tmp_path / "llm-bawt"
    path.mkdir()
    git(path, "init", "-q", "-b", "master")
    (path / "app.py").write_text("a = 1\nx = 1\nz = 1\n")
    (path / "old.py").write_text("old = True\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    return path


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "snapshots"


def make_patch(repo: Path, tmp_path: Path, name: str, edit) -> Path:
    """Write a patch for ``edit`` against HEAD, leaving the tree as it was."""
    edit(repo)
    git(repo, "add", "-A")
    patch = tmp_path / name
    patch.write_text(git(repo, "diff", "--cached", "--binary", "HEAD"))
    git(repo, "reset", "-q", "--hard", "HEAD")
    git(repo, "clean", "-qfd")
    return patch


def check(source: str) -> str:
    return json.dumps([sys.executable, "-c", source])


SUITE_X2 = check("import pathlib,sys; sys.exit(0 if 'x = 2' in pathlib.Path('app.py').read_text() else 3)")


def run_cli(repo: Path, root: Path, *args: str, suite: str = SUITE_X2) -> tuple[int, dict]:
    code = snap.main(["--repo", str(repo), "--root", str(root), "--allow-container",
                      "--lock-wait", "0", "--suite-command", suite, *args])
    receipts = sorted((root / "runs").glob("*/receipt.json"), key=lambda p: p.stat().st_mtime)
    return code, json.loads(receipts[-1].read_text())


def worktrees(repo: Path) -> list[str]:
    return [line for line in git(repo, "worktree", "list", "--porcelain").splitlines()
            if line.startswith("worktree ")]


def assert_no_snapshot_left(repo: Path, root: Path) -> None:
    assert worktrees(repo) == [f"worktree {repo}"]
    metadata = repo / ".git" / "worktrees"
    assert not metadata.exists() or not any(metadata.iterdir())
    leftovers = list((root / "wt").iterdir()) if (root / "wt").exists() else []
    assert leftovers == []


def canonical_state(repo: Path) -> dict:
    files = {str(p.relative_to(repo)): sha(p.read_bytes())
             for p in sorted(repo.rglob("*")) if p.is_file() and ".git" not in p.parts}
    return {
        "index": sha((repo / ".git" / "index").read_bytes()),
        "status": git(repo, "status", "--porcelain=v2", "--untracked-files=all"),
        "head": git(repo, "rev-parse", "HEAD"),
        "refs": git(repo, "for-each-ref"),
        "stash": git(repo, "stash", "list"),
        "files": files,
    }


def x2(repo: Path) -> None:
    (repo / "app.py").write_text("a = 1\nx = 2\nz = 1\n")


def x2_and_new_file(repo: Path) -> None:
    x2(repo)
    (repo / "new.py").write_text("new = 1\n")


def test_receipt_diff_sha_equals_git_diff_of_the_committed_change(repo, root, tmp_path):
    patch = make_patch(repo, tmp_path, "approved.patch", x2_and_new_file)
    base = git(repo, "rev-parse", "HEAD").strip()

    code, receipt = run_cli(repo, root, "--patch", str(patch))

    assert (code, receipt["status"], receipt["mode"]) == (0, "passed", "patches")
    assert receipt["base"] == base
    assert receipt["files"] == ["M\tapp.py", "A\tnew.py"]
    assert receipt["patches"][0]["sha256"] == sha(patch.read_bytes())
    assert Path(receipt["patches"][0]["saved_as"]).read_bytes() == patch.read_bytes()
    # The pin reviewers use: sha256 of `git diff BASE COMMIT` once the change lands.
    git(repo, "apply", "--index", str(patch))
    git(repo, "commit", "-q", "-m", "change")
    landed = subprocess.run(["git", "diff", base, "HEAD"], cwd=repo, check=True,
                            capture_output=True).stdout
    assert receipt["diff_sha256"] == sha(landed)
    assert receipt["diff_sha256_short"] == sha(landed)[:16]
    assert_no_snapshot_left(repo, root)


def test_an_added_mutant_patch_turns_the_gate_red(repo, root, tmp_path):
    approved = make_patch(repo, tmp_path, "approved.patch", x2)
    git(repo, "apply", "--index", str(approved))
    (repo / "app.py").write_text("a = 1\nx = 3\nz = 1\n")
    mutant = tmp_path / "mutant.patch"  # relative to the approved bytes, as a reviewer writes it
    mutant.write_text(git(repo, "diff"))
    git(repo, "reset", "-q", "--hard", "HEAD")

    code, receipt = run_cli(repo, root, "--patch", str(approved), "--patch", str(mutant))

    assert (code, receipt["status"], receipt["suite_exit"]) == (1, "failed", 3)
    assert [Path(p["saved_as"]).name for p in receipt["patches"]] == ["01-approved.patch", "02-mutant.patch"]
    assert_no_snapshot_left(repo, root)


def test_a_rejected_hunk_fails_closed_without_running_the_suite(repo, root, tmp_path):
    patch = make_patch(repo, tmp_path, "stale.patch", x2)
    (repo / "app.py").write_text("a = 1\nx = 9\nz = 1\n")
    git(repo, "commit", "-qam", "moved on")
    ran = tmp_path / "suite-ran"

    code, receipt = run_cli(repo, root, "--patch", str(patch),
                            suite=check(f"open({str(ran)!r}, 'w')"))

    assert (code, receipt["status"]) == (1, "apply_failed")
    assert "does not apply" in receipt["error"]
    assert receipt["suite_exit"] is None and not ran.exists()
    assert_no_snapshot_left(repo, root)


def dirty_canonical(repo: Path) -> None:
    """Staged and unstaged hunks in one file, another bot's file, untracked, deleted."""
    (repo / "app.py").write_text("a = 2\nx = 1\nz = 1\n")
    git(repo, "add", "app.py")  # staged hunk
    (repo / "app.py").write_text("a = 2\nx = 1\nz = 2\n")  # plus an unstaged hunk
    (repo / "mine.py").write_text("mine = 1\n")
    (repo / "old.py").unlink()


def full_lines_provenance(tmp_path: Path, author: str, paths: dict[str, int]) -> Path:
    recon = {
        "dirty": [{"path": p, "change_count": n, "hunk_count": 1} for p, n in paths.items()],
        "groups": [{"bot_id": author, "files": [
            {"path": p, "confidence": "exact", "matched_changes": n} for p, n in paths.items()]}],
        "unattributed": [],
    }
    out = tmp_path / "provenance.json"
    out.write_text(json.dumps({"recon": recon}))
    return out


def test_canonical_tree_index_and_refs_are_byte_identical_across_runs(repo, root, tmp_path):
    dirty_canonical(repo)
    before = canonical_state(repo)
    patch = make_patch_from_copy(repo, tmp_path)
    provenance = full_lines_provenance(tmp_path, "snark", {"mine.py": 1, "old.py": 1})

    patches_code, _ = run_cli(repo, root, "--patch", str(patch),
                              suite=check("import sys; sys.exit(0)"))
    paths_code, paths_receipt = run_cli(
        repo, root, "--path", "mine.py", "--path", "old.py", "--author", "snark",
        "--provenance-file", str(provenance), suite=check("import sys; sys.exit(0)"))

    assert (patches_code, paths_code) == (0, 0)
    assert paths_receipt["files"] == ["A\tmine.py", "D\told.py"]
    assert canonical_state(repo) == before
    assert_no_snapshot_left(repo, root)


def make_patch_from_copy(repo: Path, tmp_path: Path) -> Path:
    """A patch built without touching the dirty canonical tree."""
    patch = tmp_path / "independent.patch"
    patch.write_text(
        "diff --git a/extra.py b/extra.py\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/extra.py\n@@ -0,0 +1 @@\n+extra = 1\n")
    return patch


def test_paths_carries_untracked_and_deleted_files_into_the_snapshot(repo, root, tmp_path):
    x2(repo)
    (repo / "new.py").write_text("new = 1\n")
    (repo / "old.py").unlink()
    provenance = full_lines_provenance(tmp_path, "snark", {"app.py": 2, "new.py": 1, "old.py": 1})
    suite = check("import pathlib,sys; p=pathlib.Path; "
                  "sys.exit(0 if p('new.py').exists() and not p('old.py').exists() "
                  "and 'x = 2' in p('app.py').read_text() else 4)")

    code, receipt = run_cli(repo, root, "--path", "app.py", "--path", "new.py", "--path", "old.py",
                            "--author", "Snark", "--provenance-file", str(provenance), suite=suite)

    assert (code, receipt["status"], receipt["mode"], receipt["author"]) == (0, "passed", "paths", "snark")
    assert receipt["files"] == ["M\tapp.py", "A\tnew.py", "D\told.py"]
    saved = Path(receipt["patches"][0]["saved_as"])
    assert saved.parent == Path(receipt["receipt"]).parent and saved.name == "01-paths.patch"
    assert set(receipt["ownership"]) == {"app.py", "new.py", "old.py"}


def test_paths_refuses_another_bots_lines_without_creating_a_snapshot(repo, root, tmp_path):
    x2(repo)
    recon = {"dirty": [{"path": "app.py", "change_count": 2}],
             "groups": [{"bot_id": "snark", "files": [{"path": "app.py", "confidence": "strong",
                                                         "matched_changes": 1}]},
                        {"bot_id": "al", "files": [{"path": "app.py", "confidence": "strong",
                                                      "matched_changes": 1}]}],
             "unattributed": []}
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"recon": recon}))

    code, receipt = run_cli(repo, root, "--path", "app.py", "--author", "snark",
                            "--provenance-file", str(provenance))

    assert (code, receipt["status"]) == (2, "refused")
    assert "attributed to al" in receipt["error"] and "Never clone" in receipt["error"]
    assert receipt["patches"] == [] and not (root / "wt").exists()


@pytest.mark.parametrize(("extra", "message"), [
    (["--base", "HEAD~1"], "BASE must be its HEAD"),
    (["--author", ""], "PATHS requires AUTHOR"),
])
def test_paths_preconditions(repo, root, extra, message):
    x2(repo)
    git(repo, "commit", "-qam", "second")
    (repo / "app.py").write_text("a = 1\nx = 5\nz = 1\n")
    code, receipt = run_cli(repo, root, "--path", "app.py", "--author", "snark", *extra)
    assert (code, receipt["status"]) == (2, "refused") and message in receipt["error"]


@pytest.mark.parametrize(("path", "message"), [
    ("../escape.py", "repo-relative"), ("/etc/passwd", "repo-relative"),
    ("pkg", "is a directory"), ("old.py", "unchanged vs HEAD"),
])
def test_paths_must_name_changed_files_inside_the_repo(repo, root, path, message):
    (repo / "pkg").mkdir()
    (repo / "pkg" / "m.py").write_text("m = 1\n")
    code, receipt = run_cli(repo, root, "--path", path, "--author", "snark")
    assert (code, receipt["status"]) == (2, "refused") and message in receipt["error"]


def test_snapshot_root_inside_the_repo_is_refused_before_writing_anything(repo):
    code = snap.main(["--repo", str(repo), "--root", str(repo / "snapshots"), "--patch", "x"])
    assert code == snap.EXIT_REFUSED and not (repo / "snapshots").exists()


def test_a_stale_snapshot_from_a_dead_process_is_swept(repo, root, tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = root / "wt" / "stale-run"
    stale.parent.mkdir(parents=True)
    git(repo, "worktree", "add", "--detach", "--lock", str(stale), "HEAD")
    (root / "wt" / "stale-run.owner.json").write_text(json.dumps(
        {"pid": dead.pid, "start": "gone", "repo": str(repo)}))
    patch = make_patch(repo, tmp_path, "approved.patch", x2)

    code, _ = run_cli(repo, root, "--patch", str(patch))

    assert code == 0
    assert not stale.exists()
    assert_no_snapshot_left(repo, root)


# ── Ownership proofs ───────────────────────────────────────────────────────

HEAD, MID, NOW = "h" * 64, "m" * 64, "n" * 64


def row(bot, before, after, ts, kind="modified", exact=True, path="app.py"):
    return snap.EditRow(bot_id=bot, path=path, change_kind=kind, before_sha256=before,
                        after_sha256=after, exact=exact, ts=ts)


def test_byte_chain_proves_ownership_across_several_author_turns():
    check = snap.OwnershipCheck("snark", [row("snark", HEAD, MID, 1), row("snark", MID, NOW, 2)], None)
    assert check.chain("app.py", HEAD, NOW) == (True, "byte chain: 2 recorded snark edit(s) from HEAD")


@pytest.mark.parametrize(("rows", "reason"), [
    ([row("snark", MID, NOW, 2)], "bytes before snark's edit were not produced"),
    ([row("snark", HEAD, MID, 1)], "current bytes were not produced"),
    ([row("snark", HEAD, NOW, 1), row("al", HEAD, "x" * 64, 2)], "also edited by al"),
    ([row("snark", HEAD, NOW, 1, exact=False)], "current bytes were not produced"),
    ([row("al", HEAD, NOW, 1)], "current bytes were not produced"),
])
def test_byte_chain_refuses_foreign_or_unexplained_bytes(rows, reason):
    ok, why = snap.OwnershipCheck("snark", rows, None).chain("app.py", HEAD, NOW)
    assert not ok and reason in why


def test_byte_chain_handles_added_and_deleted_files():
    check = snap.OwnershipCheck("snark", [row("snark", None, NOW, 1, kind="added", path="new.py"),
                                          row("snark", HEAD, None, 1, kind="deleted", path="old.py")],
                                None)
    assert check.chain("new.py", None, NOW)[0]
    assert check.chain("old.py", HEAD, None)[0]
    # An unknown "before" on a modified row is not "absent".
    assert not snap.OwnershipCheck("snark", [row("snark", None, NOW, 1)], None).chain("app.py", None, NOW)[0]


def recon_for(*groups, change_count=4, unattributed=()):
    return {"dirty": [{"path": "app.py", "change_count": change_count}],
            "groups": [{"bot_id": bot, "files": [{"path": "app.py", "confidence": conf,
                                                  "matched_changes": n}]} for bot, conf, n in groups],
            "unattributed": [{"path": p} for p in unattributed]}


@pytest.mark.parametrize(("recon", "ok", "reason"), [
    (recon_for(("snark", "strong", 3), ("snark", "partial", 1)), True, "4/4"),
    (recon_for(("snark", "exact", 3)), False, "3/4 dirty lines"),
    (recon_for(("snark", "exact", 3), ("al", "partial", 1)), False, "attributed to al"),
    (recon_for(("snark", "exact", 4), ("nova", "probable", 0)), False, "attributed to nova"),
    (recon_for(unattributed=("app.py",)), False, "no recorded edit explains"),
])
def test_line_coverage_requires_every_dirty_line_and_no_other_bot(recon, ok, reason):
    result = snap.OwnershipCheck("snark", None, recon).lines("app.py")
    assert result[0] is ok and reason in result[1]


def test_either_proof_suffices_and_a_refusal_names_both():
    lines_only = snap.OwnershipCheck("snark", [], recon_for(("snark", "exact", 4)))
    assert lines_only.verdict("app.py", HEAD, NOW) == (True, "recon: 4/4 dirty lines attributed to snark")
    neither = snap.OwnershipCheck("snark", [], recon_for(("snark", "exact", 1)))
    ok, why = neither.verdict("app.py", HEAD, NOW)
    assert not ok and "current bytes were not produced" in why and "1/4 dirty lines" in why


def test_provenance_rows_parse_and_normalize_paths():
    text = ('al,~/dev/llm-bawt/src/x.py,modified,aa,bb,f,10.5\n'
            'byte,~/dev/.byte-1233w/src/x.py,added,,cc,t,11\n')
    parsed = snap.rows_from_csv(text, "llm-bawt")
    assert parsed[0] == row("al", "aa", "bb", 10.5, path="src/x.py")
    assert parsed[1].path == "~/dev/.byte-1233w/src/x.py" and parsed[1].exact is False


# ── Interrupt (real process, real signals) ─────────────────────────────────


@pytest.mark.integration
@pytest.mark.offline_integration
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_an_interrupted_run_stops_the_suite_and_leaves_no_worktree(repo, root, tmp_path, sig):
    patch = make_patch(repo, tmp_path, "approved.patch", x2)
    started, child_pid = tmp_path / "started", tmp_path / "child.pid"
    suite = check(f"import os,time,pathlib; pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid())); "
                  f"pathlib.Path({str(started)!r}).touch(); time.sleep(60)")
    proc = subprocess.Popen([sys.executable, str(SCRIPT), "--repo", str(repo), "--root", str(root),
                             "--allow-container", "--lock-wait", "0", "--patch", str(patch),
                             "--suite-command", suite],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert started.exists(), "suite never started"

    proc.send_signal(sig)

    assert proc.wait(timeout=20) == snap.EXIT_INTERRUPTED
    receipt = json.loads(next((root / "runs").glob("*/receipt.json")).read_text())
    assert receipt["status"] == "interrupted"
    assert not Path(f"/proc/{child_pid.read_text()}").exists()
    assert_no_snapshot_left(repo, root)


@pytest.mark.integration
@pytest.mark.offline_integration
def test_a_second_interrupt_cannot_orphan_a_suite_that_ignores_sigint(repo, root, tmp_path):
    patch = make_patch(repo, tmp_path, "approved.patch", x2)
    started, child_pid = tmp_path / "started", tmp_path / "child.pid"
    suite = check("import os,signal,time,pathlib; signal.signal(signal.SIGINT, signal.SIG_IGN); "
                  f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid())); "
                  f"pathlib.Path({str(started)!r}).touch(); time.sleep(60)")
    proc = subprocess.Popen([sys.executable, str(SCRIPT), "--repo", str(repo), "--root", str(root),
                             "--allow-container", "--lock-wait", "0", "--patch", str(patch),
                             "--stop-grace", "1,1", "--suite-command", suite],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert started.exists(), "suite never started"

    proc.send_signal(signal.SIGINT)
    time.sleep(0.3)  # inside stop_group: the suite ignored SIGINT
    proc.send_signal(signal.SIGINT)

    assert proc.wait(timeout=20) == snap.EXIT_INTERRUPTED
    # The escalation finished (SIGTERM) before the second Ctrl-C was honoured.
    assert not Path(f"/proc/{child_pid.read_text()}").exists()
    assert_no_snapshot_left(repo, root)


def test_a_patch_that_changes_the_runner_is_flagged(repo, root, tmp_path):
    patch = make_patch(repo, tmp_path, "runner.patch",
                       lambda r: (x2(r), (r / "Makefile").write_text("all:\n")))
    code, receipt = run_cli(repo, root, "--patch", str(patch))
    assert code == 0 and receipt["runner_touched"] == ["Makefile"]
    unflagged = make_patch(repo, tmp_path, "plain.patch", x2)
    assert run_cli(repo, root, "--patch", str(unflagged))[1]["runner_touched"] == []
