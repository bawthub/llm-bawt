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
# The gate's contract is real git plumbing on a scratch repo; a fake would test nothing.
REAL_GIT = pytest.mark.real_process(reason="drives real git (and the runner) on a scratch repo")


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


@REAL_GIT
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


@REAL_GIT
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


@REAL_GIT
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


def head_bytes(repo: Path, path: str) -> bytes | None:
    shown = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=repo, capture_output=True)
    return shown.stdout if shown.returncode == 0 else None


def edit_row(repo: Path, author: str, path: str, ts: float = 1) -> str:
    """One recorded edit taking ``path`` from HEAD's bytes to its current bytes."""
    before = head_bytes(repo, path)
    after = (repo / path).read_bytes() if (repo / path).exists() else None
    kind = "added" if before is None else "deleted" if after is None else "modified"
    return (f"{author},{path},{kind},{sha(before) if before is not None else ''},"
            f"{sha(after) if after is not None else ''},f,{ts}\n")


def rows_provenance(repo: Path, tmp_path: Path, author: str, paths: list[str]) -> Path:
    out = tmp_path / "provenance.json"
    out.write_text(json.dumps({"rows": "".join(edit_row(repo, author, p) for p in paths)}))
    return out


@REAL_GIT
def test_canonical_tree_index_and_refs_are_byte_identical_across_runs(repo, root, tmp_path):
    dirty_canonical(repo)
    before = canonical_state(repo)
    patch = make_patch_from_copy(repo, tmp_path)
    provenance = rows_provenance(repo, tmp_path, "snark", ["mine.py", "old.py"])

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


@REAL_GIT
def test_paths_carries_untracked_and_deleted_files_into_the_snapshot(repo, root, tmp_path):
    x2(repo)
    (repo / "new.py").write_text("new = 1\n")
    (repo / "old.py").unlink()
    provenance = rows_provenance(repo, tmp_path, "snark", ["app.py", "new.py", "old.py"])
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


@REAL_GIT
def test_paths_refuses_another_bots_edit_without_creating_a_snapshot(repo, root, tmp_path):
    x2(repo)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"rows": edit_row(repo, "snark", "app.py", ts=1)
                                      + f"al,app.py,modified,{sha(b'x')},{sha(b'y')},f,2\n"}))

    code, receipt = run_cli(repo, root, "--path", "app.py", "--author", "snark",
                            "--provenance-file", str(provenance))

    assert (code, receipt["status"]) == (2, "refused")
    assert "also edited by al" in receipt["error"] and "Never clone" in receipt["error"]
    assert receipt["patches"] == [] and not (root / "wt").exists()


@REAL_GIT
def test_a_line_attribution_report_is_never_ownership_proof(repo, root, tmp_path):
    """Loopy r3 B1: recon credited AUTHOR with identical text in a different
    function. Only an exact byte chain proves ownership; a report is ignored."""
    before = "def first():\n    return False\n\ndef second():\n    return False\n"
    own_after = before.replace("return False", "return True", 1)
    foreign_after = "def first():\n    return False\n\ndef second():\n    return True\n"
    (repo / "app.py").write_text(before)
    git(repo, "commit", "-qam", "two similar functions")
    (repo / "app.py").write_text(foreign_after)  # an unrecorded edit, not snark's
    recon = {"dirty": [{"path": "app.py", "change_count": 2}], "unattributed": [],
             "groups": [{"bot_id": "snark", "files": [
                 {"path": "app.py", "confidence": "exact", "matched_changes": 2}]}]}
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "recon": recon,
        "rows": f"snark,app.py,modified,{sha(before.encode())},{sha(own_after.encode())},f,1\n"}))

    code, receipt = run_cli(repo, root, "--path", "app.py", "--author", "snark",
                            "--provenance-file", str(provenance), suite=check("pass"))

    assert (code, receipt["status"]) == (snap.EXIT_REFUSED, "refused"), receipt
    assert "current bytes were not produced by any recorded snark edit" in receipt["error"]


@REAL_GIT
def test_a_write_after_the_ownership_check_never_reaches_the_patch(repo, root, tmp_path, monkeypatch):
    """Loopy r3 B2: bytes written between the proof and the patch build stay out."""
    x2(repo)
    owned = (repo / "app.py").read_bytes()
    provenance = rows_provenance(repo, tmp_path, "snark", ["app.py"])
    real_build = snap.build_paths_patch

    def racing_build(*args, **kwargs):
        (repo / "app.py").write_bytes(owned + b"foreign = True\n")  # another writer
        return real_build(*args, **kwargs)

    monkeypatch.setattr(snap, "build_paths_patch", racing_build)
    code, receipt = run_cli(repo, root, "--path", "app.py", "--author", "snark",
                            "--provenance-file", str(provenance))

    pinned = Path(receipt["patches"][0]["saved_as"]).read_text()
    assert code == 0 and "+x = 2" in pinned and "foreign" not in pinned, receipt


@REAL_GIT
def test_paths_refuses_a_symlink(repo, root, tmp_path):
    (repo / "link.py").symlink_to("app.py")
    provenance = rows_provenance(repo, tmp_path, "snark", ["link.py"])
    code, receipt = run_cli(repo, root, "--path", "link.py", "--author", "snark",
                            "--provenance-file", str(provenance))
    assert (code, receipt["status"]) == (2, "refused") and "symlink" in receipt["error"]


@REAL_GIT
def test_paths_without_recorded_edits_is_refused(repo, root, tmp_path):
    x2(repo)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({}))
    code, receipt = run_cli(repo, root, "--path", "app.py", "--author", "snark",
                            "--provenance-file", str(provenance))
    assert (code, receipt["status"]) == (2, "refused") and "recorded edits" in receipt["error"]


@pytest.mark.parametrize(("extra", "message"), [
    (["--base", "HEAD~1"], "BASE must be its HEAD"),
    (["--author", ""], "PATHS requires AUTHOR"),
])
@REAL_GIT
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
@REAL_GIT
def test_paths_must_name_changed_files_inside_the_repo(repo, root, path, message):
    (repo / "pkg").mkdir()
    (repo / "pkg" / "m.py").write_text("m = 1\n")
    code, receipt = run_cli(repo, root, "--path", path, "--author", "snark")
    assert (code, receipt["status"]) == (2, "refused") and message in receipt["error"]


@REAL_GIT
def test_snapshot_root_inside_the_repo_is_refused_before_writing_anything(repo):
    code = snap.main(["--repo", str(repo), "--root", str(repo / "snapshots"), "--patch", "x"])
    assert code == snap.EXIT_REFUSED and not (repo / "snapshots").exists()


@REAL_GIT
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
    check = snap.OwnershipCheck("snark", [row("snark", HEAD, MID, 1), row("snark", MID, NOW, 2)])
    assert check.chain("app.py", HEAD, NOW) == (True, "byte chain: 2 recorded snark edit(s) from HEAD")


@pytest.mark.parametrize(("rows", "reason"), [
    ([row("snark", MID, NOW, 2)], "bytes before snark's edit were not produced"),
    ([row("snark", HEAD, MID, 1)], "current bytes were not produced"),
    ([row("snark", HEAD, NOW, 1), row("al", HEAD, "x" * 64, 2)], "also edited by al"),
    ([row("snark", HEAD, NOW, 1, exact=False)], "current bytes were not produced"),
    ([row("al", HEAD, NOW, 1)], "current bytes were not produced"),
])
def test_byte_chain_refuses_foreign_or_unexplained_bytes(rows, reason):
    ok, why = snap.OwnershipCheck("snark", rows).chain("app.py", HEAD, NOW)
    assert not ok and reason in why


def test_byte_chain_handles_added_and_deleted_files():
    check = snap.OwnershipCheck("snark", [row("snark", None, NOW, 1, kind="added", path="new.py"),
                                          row("snark", HEAD, None, 1, kind="deleted", path="old.py")])
    assert check.chain("new.py", None, NOW)[0]
    assert check.chain("old.py", HEAD, None)[0]
    # An unknown "before" on a modified row is not "absent".
    assert not snap.OwnershipCheck("snark", [row("snark", None, NOW, 1)]).chain("app.py", None, NOW)[0]


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


IGNORING_CHILD = ("import os,pathlib,signal,sys,time; "
                  "signal.signal(signal.SIGINT, signal.SIG_IGN); "
                  "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)")


def live(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def wait_for(path: Path, seconds: float = 10) -> int:
    deadline = time.monotonic() + seconds
    while not (path.exists() and path.read_text()) and time.monotonic() < deadline:
        time.sleep(0.01)
    return int(path.read_text())


@pytest.mark.integration
@pytest.mark.offline_integration
def test_stop_group_kills_a_descendant_that_outlives_its_leader(tmp_path):
    """Loopy r3 B3: the leader exits on SIGINT; its child ignores SIGINT/SIGTERM."""
    pid_file = tmp_path / "child.pid"
    leader = ("import signal,subprocess,sys,time; "
              "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
              f"subprocess.Popen([sys.executable, '-c', {IGNORING_CHILD!r}, {str(pid_file)!r}]); "
              "time.sleep(60)")
    proc = subprocess.Popen([sys.executable, "-c", leader], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        child = wait_for(pid_file)
        started = time.monotonic()
        snap.stop_group(proc, (0.2, 0.2))
        assert proc.poll() is not None
        assert not live(child), f"descendant {child} survived stop_group"
        assert not snap.group_alive(proc.pid)
        # timing-ok: real process group: stop_group's own waits plus 1s headroom; a stuck group never returns
        assert time.monotonic() - started < 0.2 + 0.2 + 5 + 1
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.integration
@pytest.mark.offline_integration
def test_a_finished_suite_leaves_no_descendant_behind(tmp_path):
    """A suite that exits 0 while a detached-stdout descendant lives on: run_suite
    still reports the suite's exit and stops the rest of the group."""
    pid_file = tmp_path / "child.pid"
    leader = ("import subprocess,sys; "
              f"subprocess.Popen([sys.executable, '-c', {IGNORING_CHILD!r}, {str(pid_file)!r}], "
              "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
              "import pathlib,time; p=pathlib.Path(sys.argv[1]); "
              "[time.sleep(0.01) for _ in range(1000) if not (p.exists() and p.read_text())]")
    receipt = snap.Receipt(run_id="t", repo=str(tmp_path))

    code = snap.run_suite([sys.executable, "-c", leader, str(pid_file)], tmp_path,
                          tmp_path / "log", 0, receipt, (0.2, 0.2))

    child = int(pid_file.read_text())
    assert code == 0
    assert not live(child), f"descendant {child} outlived the finished suite"


@REAL_GIT
def test_a_patch_that_changes_the_runner_is_flagged(repo, root, tmp_path):
    patch = make_patch(repo, tmp_path, "runner.patch",
                       lambda r: (x2(r), (r / "Makefile").write_text("all:\n")))
    code, receipt = run_cli(repo, root, "--patch", str(patch))
    assert code == 0 and receipt["runner_touched"] == ["Makefile"]
    unflagged = make_patch(repo, tmp_path, "plain.patch", x2)
    assert run_cli(repo, root, "--patch", str(unflagged))[1]["runner_touched"] == []


# -- SHIP gate receipt (TASK-1457) -------------------------------------------


def test_the_gate_suite_is_the_full_offline_suite_without_a_stress_step():
    # Repeated rounds only find timing flakes by chance; the SHIP gate is test-all at REV.
    assert snap.suite_command("gate", []) == ["make", "--no-print-directory", "test-all"]
    assert snap.suite_command("sweep", []) == ["make", "--no-print-directory", "test-sweep"]


def gate_receipt(repo: Path, root: Path, rev: str, capsys) -> tuple[int, str]:
    capsys.readouterr()
    code = snap.main(["--repo", str(repo), "--root", str(root), "--receipt-for", rev])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


@pytest.fixture
def tip(repo) -> str:
    """An unpushed commit on top of the repo's first one."""
    (repo / "app.py").write_text("a = 1\nx = 2\nz = 1\n")
    git(repo, "commit", "-qam", "tip")
    return git(repo, "rev-parse", "HEAD").strip()


@REAL_GIT
def test_a_ship_reads_the_newest_unpatched_gate_receipt_for_the_exact_tip(repo, root, tmp_path, capsys, tip):
    code, output = gate_receipt(repo, root, "HEAD", capsys)
    assert code == snap.EXIT_REFUSED and f"none for {tip}" in output

    code, receipt = run_cli(repo, root, "--suite", "gate", suite=check("pass"))
    assert code == snap.EXIT_PASSED
    assert (receipt["suite"], receipt["mode"], receipt["base"]) == ("gate", "base", tip)
    assert "stress_base" not in receipt
    code, output = gate_receipt(repo, root, "HEAD", capsys)
    assert code == snap.EXIT_PASSED and "PASSED" in output

    # A later failing gate at the same tip is the answer, not the earlier pass.
    code, _ = run_cli(repo, root, "--suite", "gate", suite=check("raise SystemExit(1)"))
    assert code == snap.EXIT_FAILED
    code, output = gate_receipt(repo, root, "HEAD", capsys)
    assert code == snap.EXIT_FAILED and "FAILED" in output

    # A PATCHES trial at that tip tests other bytes, so it never counts.
    patch = make_patch(repo, tmp_path, "z2.patch", lambda r: (r / "app.py").write_text("a = 1\nx = 2\nz = 2\n"))
    code, receipt = run_cli(repo, root, "--suite", "gate", "--patch", str(patch), suite=check("pass"))
    assert code == snap.EXIT_PASSED and receipt["mode"] != "base"
    code, _ = gate_receipt(repo, root, "HEAD", capsys)
    assert code == snap.EXIT_FAILED


@REAL_GIT
def test_a_receipt_from_the_stress_era_runner_still_reads(repo, root, capsys, tip):
    # Receipts written before the stress step was dropped carry a stress_base key.
    code, receipt = run_cli(repo, root, "--suite", "gate", suite=check("pass"))
    assert code == snap.EXIT_PASSED
    path = Path(receipt["receipt"])
    path.write_text(json.dumps({**json.loads(path.read_text()), "stress_base": tip}))
    code, output = gate_receipt(repo, root, tip, capsys)
    assert code == snap.EXIT_PASSED and "PASSED" in output


@REAL_GIT
def test_the_gate_no_longer_takes_a_stress_base(repo, root):
    with pytest.raises(SystemExit):
        snap.parse_args(["--repo", str(repo), "--root", str(root), "--suite", "gate", "--stress-base", "HEAD"])
