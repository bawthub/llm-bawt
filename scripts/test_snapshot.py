"""Gate exactly BASE + patches in a throwaway git worktree (TASK-1137).

The shared checkout always carries other bots' uncommitted work, so
``make test-all`` there tests a mix. This runs the same Make suite in a detached
worktree that holds only BASE plus the given patches, writes a receipt, and
removes the worktree on every exit path, including failure and interrupt.

    make test-snapshot PATHS="src/x.py tests/test_x.py" AUTHOR=<bot>
    make test-snapshot BASE=<sha> PATCHES="/abs/approved.patch /abs/mutant.patch"
    make test-snapshot PATHS=... AUTHOR=<bot> SUITE=target NODES="tests/test_x.py"

PATHS builds the patch from this checkout's working tree (BASE must be HEAD) and
refuses a path unless its uncommitted bytes provably belong to AUTHOR: either
they chain back to HEAD through only AUTHOR's recorded edits, or commit-recon
attributes every dirty line to AUTHOR and none to anyone else.

Run it on the test host (echo), never inside a bridge container. It reads the
canonical repository only: its working tree, index, HEAD, branches and stash
are never written. Git writes the snapshot's ``.git/worktrees/<id>`` metadata
and any new blob objects, and the metadata is removed with the worktree.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "llm-bawt-test-snapshot"
DEFAULT_LOCK_WAIT_S = 1800
STOP_GRACE_S = (30.0, 10.0)  # SIGINT → SIGTERM → SIGKILL
RUN_RETENTION_S = 14 * 24 * 3600
# A patch touching these changes the runner that judges it; the receipt says so.
RUNNER_FILES = ("Makefile", "pyproject.toml", "uv.lock", "scripts/test_suite.py",
                "scripts/test_snapshot.py", "tests/conftest.py", "tests/suite_manifest.py")
PROVENANCE_WINDOW_DAYS = 14
SUITE_TARGETS = {"offline": "test-all", "target": "test-target"}
LOCK_WAIT_ENV = "LLM_BAWT_TEST_LOCK_WAIT_SECONDS"
# Never inherit a caller's Make or git context into the snapshot.
SCRUBBED_ENV = ("MAKEFLAGS", "MAKELEVEL", "MFLAGS", "MAKEOVERRIDES", "GIT_DIR",
                "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_PREFIX")
EXIT_PASSED, EXIT_FAILED, EXIT_REFUSED, EXIT_INTERRUPTED = 0, 1, 2, 130
SUMMARY_RE = re.compile(r"\b\d+ (?:passed|failed|errors?|skipped|deselected)\b.* in [\d.]+s")
RUNNER_RE = re.compile(r"^(?:Offline coverage|Target selection|Test runner lock|Test runner):")


class Refused(Exception):
    """A precondition failed before any test ran."""


class Interrupted(BaseException):
    """SIGTERM or SIGHUP; handled like Ctrl-C so cleanup always runs."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV}
    # Read-only git commands must not refresh (rewrite) the canonical index.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.update(extra)
    return env


class Git:
    def __init__(self, cwd: Path, env: dict[str, str] | None = None) -> None:
        self.cwd = cwd
        self.env = env or clean_env()

    def run(self, *args: str, check: bool = True, env: dict[str, str] | None = None,
            ) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(["git", *args], cwd=self.cwd, env=env or self.env,
                                capture_output=True)
        if check and result.returncode:
            raise Refused(f"git {' '.join(args)} failed ({result.returncode}): "
                          f"{result.stderr.decode(errors='replace').strip()}")
        return result

    def out(self, *args: str, **kwargs: Any) -> str:
        return self.run(*args, **kwargs).stdout.decode().strip()

    def blob(self, rev: str, path: str) -> bytes | None:
        result = self.run("cat-file", "blob", f"{rev}:{path}", check=False)
        return result.stdout if result.returncode == 0 else None


# ── Receipt ────────────────────────────────────────────────────────────────


@dataclass
class PatchRecord:
    source: str
    saved_as: str
    sha256: str


@dataclass
class Receipt:
    run_id: str
    repo: str
    status: str = "error"
    exit_code: int = EXIT_FAILED
    mode: str = "base"
    author: str | None = None
    base: str | None = None
    canonical_head: str | None = None
    paths: list[str] = field(default_factory=list)
    patches: list[PatchRecord] = field(default_factory=list)
    ownership: dict[str, str] = field(default_factory=dict)
    diff_sha256: str | None = None
    diff_sha256_short: str | None = None
    files: list[str] = field(default_factory=list)
    runner_touched: list[str] = field(default_factory=list)
    suite: str = "offline"
    nodes: list[str] = field(default_factory=list)
    suite_exit: int | None = None
    runner: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    error: str | None = None
    log: str | None = None
    receipt: str | None = None
    worktree: str | None = None
    started_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
    duration_s: float | None = None

    def write(self, path: Path) -> None:
        self.receipt = str(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2) + "\n")
        tmp.replace(path)

    def summary(self) -> str:
        lines = [f"Snapshot receipt: {self.status.upper()} (exit {self.exit_code})",
                 f"  run:     {self.run_id}",
                 f"  base:    {self.base}  mode={self.mode}"
                 + (f" author={self.author}" if self.author else "")]
        for patch in self.patches:
            lines.append(f"  patch:   sha256:{patch.sha256[:16]}  {patch.saved_as}")
        if self.diff_sha256:
            lines.append(f"  diff:    sha256:{self.diff_sha256_short}  "
                         f"(git diff {self.base[:12] if self.base else '?'}; {len(self.files)} files)")
        if self.runner_touched:
            lines.append(f"  note:    the patch changes the test runner ({', '.join(self.runner_touched)}); "
                         "this gate ran the patched runner")
        lines += [f"  runner:  {line}" for line in self.runner]
        lines += [f"  pytest:  {line}" for line in self.summaries]
        if self.error:
            lines.append(f"  error:   {self.error}")
        lines += [f"  log:     {self.log}", f"  receipt: {self.receipt}"]
        return "\n".join(lines)


# ── Ownership (PATHS mode) ─────────────────────────────────────────────────

_UNKNOWN = object()


@dataclass(frozen=True)
class EditRow:
    """One turn's recorded before/after bytes for one file (turn_changed_files)."""

    bot_id: str
    path: str
    change_kind: str
    before_sha256: str | None
    after_sha256: str | None
    exact: bool  # False when binary or truncated: the sha is not of the real bytes
    ts: float

    def before_key(self) -> object:
        if not self.exact:
            return _UNKNOWN
        if self.before_sha256:
            return self.before_sha256
        return None if self.change_kind == "added" else _UNKNOWN

    def after_key(self) -> object:
        if not self.exact:
            return _UNKNOWN
        if self.after_sha256:
            return self.after_sha256
        return None if self.change_kind == "deleted" else _UNKNOWN


def relative_record_path(raw: str, repo_name: str) -> str:
    normalized = raw.replace("\\", "/")
    marker = f"/{repo_name}/"
    if marker in normalized:
        return normalized.rsplit(marker, 1)[1]
    stripped = normalized.removeprefix("./")
    return stripped.removeprefix(f"{repo_name}/")


class OwnershipCheck:
    """Decide whether every uncommitted byte of a path is AUTHOR's.

    Method A (byte chain): starting from the working-tree bytes, each step must be
    an AUTHOR edit whose recorded ``after`` equals the current bytes; its
    ``before`` becomes the next target, until HEAD's bytes are reached. No other
    bot may have touched the path since the first step.
    Method B (line coverage): commit-recon explains every dirty line with
    AUTHOR's recorded edits and none with anyone else's.
    Either proves ownership. Neither means refuse; the tool never guesses.
    """

    def __init__(self, author: str, rows: list[EditRow] | None, recon: dict | None) -> None:
        self.author = author
        self.rows = rows
        self.recon = recon

    def chain(self, path: str, head_sha: str | None, current_sha: str | None) -> tuple[bool, str]:
        if self.rows is None:
            return False, "no provenance rows (pgq unavailable)"
        rows = [row for row in self.rows if row.path == path]
        mine = sorted((row for row in rows if row.bot_id == self.author),
                      key=lambda row: row.ts, reverse=True)
        target: object = current_sha
        limit = float("inf")
        used: list[EditRow] = []
        while target != head_sha:
            step = next((row for row in mine if row.ts <= limit and row not in used
                         and row.after_key() == target), None)
            if step is None:
                where = "the current bytes" if not used else f"the bytes before {self.author}'s edit"
                return False, f"{where} were not produced by any recorded {self.author} edit"
            used.append(step)
            target, limit = step.before_key(), step.ts
            if len(used) > 200:
                return False, "edit chain too long"
        if not used:
            return False, "unchanged vs HEAD"
        start = min(row.ts for row in used)
        others = sorted({row.bot_id for row in rows if row.bot_id != self.author and row.ts >= start})
        if others:
            return False, f"also edited by {', '.join(others)} during {self.author}'s edits"
        return True, f"byte chain: {len(used)} recorded {self.author} edit(s) from HEAD"

    def lines(self, path: str) -> tuple[bool, str]:
        if self.recon is None:
            return False, "no commit-recon report"
        dirty = next((item for item in self.recon.get("dirty", []) if item.get("path") == path), None)
        if dirty is None:
            return False, "recon: not dirty"
        if any(item.get("path") == path for item in self.recon.get("unattributed", [])):
            return False, "recon: no recorded edit explains these changes"
        matched: dict[str, int] = defaultdict(int)
        foreign: set[str] = set()
        for group in self.recon.get("groups", []):
            bot = str(group.get("bot_id") or "unknown").lower()
            for item in group.get("files", []):
                if item.get("path") != path:
                    continue
                if item.get("confidence") == "probable" and bot != self.author:
                    foreign.add(bot)
                matched[bot] += int(item.get("matched_changes") or 0)
        foreign |= {bot for bot, count in matched.items() if bot != self.author and count}
        if foreign:
            return False, f"recon: lines attributed to {', '.join(sorted(foreign))}"
        total = int(dirty.get("change_count") or 0)
        if not total or matched[self.author] != total:
            return False, f"recon: {matched[self.author]}/{total} dirty lines attributed to {self.author}"
        return True, f"recon: {total}/{total} dirty lines attributed to {self.author}"

    def verdict(self, path: str, head_sha: str | None, current_sha: str | None) -> tuple[bool, str]:
        chain_ok, chain_why = self.chain(path, head_sha, current_sha)
        if chain_ok:
            return True, chain_why
        lines_ok, lines_why = self.lines(path)
        if lines_ok:
            return True, lines_why
        return False, f"{chain_why}; {lines_why}"


def find_tool(name: str, override: str | None) -> str | None:
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    for candidate in (Path.home() / "dev/agent-skills/scripts" / name,
                      Path("/opt/agent-skills/bin") / name):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def load_rows(pgq: str | None, repo_name: str) -> list[EditRow] | None:
    if not pgq or not re.fullmatch(r"[A-Za-z0-9._-]+", repo_name):
        return None
    sql = ("SELECT bot_id, path, change_kind, coalesce(before_sha256, ''), coalesce(after_sha256, ''), "
           "(\"binary\" OR truncated), extract(epoch from created_at) FROM turn_changed_files "
           f"WHERE repo_label = '{repo_name}' AND created_at >= now() - interval "
           f"'{PROVENANCE_WINDOW_DAYS} days'")
    try:
        result = subprocess.run([pgq, "llm-bawt", "--csv", "-t", "-c", sql], capture_output=True,
                                env=clean_env(), timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return rows_from_csv(result.stdout.decode(errors="replace"), repo_name)


def rows_from_csv(text: str, repo_name: str) -> list[EditRow]:
    rows = []
    for raw in csv.reader(io.StringIO(text)):
        if len(raw) != 7:
            continue
        bot, path, kind, before, after, inexact, ts = raw
        rows.append(EditRow(bot_id=(bot or "unknown").lower(),
                            path=relative_record_path(path, repo_name), change_kind=kind,
                            before_sha256=before or None, after_sha256=after or None,
                            exact=inexact.lower() not in {"t", "true", "1"}, ts=float(ts)))
    return rows


def load_recon(recon: str | None, repo: Path) -> dict | None:
    if not recon:
        return None
    try:
        result = subprocess.run([recon, "--repo", str(repo), "--json", "--no-repair",
                                 "--since", f"{PROVENANCE_WINDOW_DAYS}d"],
                                capture_output=True, env=clean_env(), timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


# ── Worktree lifecycle ─────────────────────────────────────────────────────


def process_start(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[19]


def owner_alive(owner: dict) -> bool:
    pid = owner.get("pid")
    return isinstance(pid, int) and process_start(pid) == owner.get("start")


def remove_worktree(repo: Path, worktree: Path) -> None:
    git = Git(repo)
    git.run("worktree", "remove", "--force", "--force", str(worktree), check=False)
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)
    # Drop only this worktree's metadata; a global prune would also drop other
    # people's worktrees whose paths this host can't see.
    common = Path(git.out("rev-parse", "--path-format=absolute", "--git-common-dir"))
    meta_root = common / "worktrees"
    if meta_root.is_dir():
        for meta in meta_root.iterdir():
            gitdir = meta / "gitdir"
            try:
                points_at = Path(gitdir.read_text().strip()).parent.resolve()
            except OSError:
                continue
            if points_at == worktree.resolve():
                shutil.rmtree(meta, ignore_errors=True)


def sweep_stale(root: Path) -> list[str]:
    """Remove snapshots whose owning process is gone (e.g. killed with SIGKILL)."""
    swept = []
    wt_root = root / "wt"
    if not wt_root.is_dir():
        return swept
    for owner_file in wt_root.glob("*.owner.json"):
        try:
            owner = json.loads(owner_file.read_text())
        except (OSError, json.JSONDecodeError):
            owner = {}
        if owner_alive(owner):
            continue
        worktree = wt_root / owner_file.name.removesuffix(".owner.json")
        repo = Path(owner.get("repo") or REPO)
        if repo.is_dir():
            remove_worktree(repo, worktree)
        elif worktree.exists():
            shutil.rmtree(worktree, ignore_errors=True)
        owner_file.unlink(missing_ok=True)
        swept.append(worktree.name)
    return swept


def prune_old_runs(root: Path, now: float | None = None) -> None:
    runs = root / "runs"
    if not runs.is_dir():
        return
    cutoff = (now or time.time()) - RUN_RETENTION_S
    for run_dir in runs.iterdir():
        if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff:
            shutil.rmtree(run_dir, ignore_errors=True)


@contextmanager
def snapshot_worktree(repo: Path, root: Path, run_id: str, base: str) -> Iterator[Path]:
    wt_root = root / "wt"
    wt_root.mkdir(parents=True, exist_ok=True)
    worktree = wt_root / run_id
    owner_file = wt_root / f"{run_id}.owner.json"
    # Atomic, so a concurrent sweep never reads a half-written owner as dead.
    partial = owner_file.with_suffix(".partial")
    partial.write_text(json.dumps({"pid": os.getpid(), "start": process_start(os.getpid()),
                                   "repo": str(repo), "worktree": str(worktree)}))
    partial.replace(owner_file)
    try:
        # --no-checkout skips the post-checkout hook's off-canonical warning;
        # --lock keeps any concurrent `git worktree prune` away from it.
        Git(repo).run("worktree", "add", "--detach", "--no-checkout", "--lock",
                      "--reason", f"test-snapshot pid {os.getpid()}", str(worktree), base)
        Git(worktree).run("read-tree", "--reset", "-u", "HEAD")
        yield worktree
    finally:
        with signals_blocked():
            remove_worktree(repo, worktree)
            owner_file.unlink(missing_ok=True)


@contextmanager
def signals_blocked() -> Iterator[None]:
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


# ── Patches ────────────────────────────────────────────────────────────────


def validate_paths(git: Git, paths: list[str]) -> list[str]:
    clean = []
    for raw in paths:
        path = PurePosixPath(raw)
        if path.is_absolute() or not raw or ".." in path.parts or str(path) in {".", ""}:
            raise Refused(f"PATHS entry {raw!r} must be a repo-relative file path")
        rel = str(path)
        if (git.cwd / rel).is_dir():
            raise Refused(f"PATHS entry {rel!r} is a directory; list files")
        status = git.out("status", "--porcelain=v1", "--untracked-files=all", "--", f":(literal){rel}")
        if not status:
            raise Refused(f"PATHS entry {rel!r} is unchanged vs HEAD (or ignored)")
        clean.append(rel)
    return sorted(set(clean))


def build_paths_patch(git: Git, base: str, paths: list[str], out: Path) -> None:
    """Diff BASE → working tree for exactly these paths, untracked files included,
    through a temporary index so the canonical index is never touched."""
    index = out.with_suffix(".index")
    env = clean_env(GIT_INDEX_FILE=str(index))
    try:
        git.run("read-tree", base, env=env)
        pathspecs = [f":(literal){path}" for path in paths]
        git.run("add", "-A", "--", *pathspecs, env=env)
        patch = git.run("diff", "--cached", "--binary", "--no-color", "--no-ext-diff",
                        base, "--", *pathspecs, env=env).stdout
    finally:
        index.unlink(missing_ok=True)
    if not patch:
        raise Refused("PATHS produced an empty patch")
    out.write_bytes(patch)


def pin_patches(sources: list[Path], run_dir: Path) -> list[PatchRecord]:
    records = []
    for number, source in enumerate(sources, 1):
        data = source.read_bytes()
        if not data.strip():
            raise Refused(f"patch {source} is empty")
        saved = run_dir / f"{number:02d}-{source.name}"
        if saved != source:
            saved.write_bytes(data)
        records.append(PatchRecord(source=str(source), saved_as=str(saved), sha256=sha256(data)))
    return records


# ── Suite ──────────────────────────────────────────────────────────────────


def suite_command(suite: str, nodes: list[str]) -> list[str]:
    command = ["make", "--no-print-directory", SUITE_TARGETS[suite]]
    if suite == "target":
        command.append("NODES=" + " ".join(nodes))
    return command


def run_suite(command: list[str], cwd: Path, log_path: Path, lock_wait: int,
              receipt: Receipt, stop_grace: tuple[float, float] = STOP_GRACE_S) -> int:
    env = clean_env(**{LOCK_WAIT_ENV: str(lock_wait)})
    echo = True
    with log_path.open("wb") as log:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, b""):
                log.write(line)
                log.flush()
                text = line.decode(errors="replace").rstrip()
                if RUNNER_RE.match(text):
                    receipt.runner.append(text)
                elif SUMMARY_RE.search(text):
                    receipt.summaries.append(text.strip("= "))
                if echo:
                    try:
                        sys.stdout.buffer.write(line)
                        sys.stdout.flush()
                    except (BrokenPipeError, OSError):
                        echo = False  # caller went away; finish the run and the receipt
            return proc.wait()
        except BaseException:
            # A second Ctrl-C must not abort the escalation and orphan a suite that
            # still holds the runner lock; it is delivered once the group is gone.
            with signals_blocked():
                stop_group(proc, stop_grace)
            raise


def stop_group(proc: subprocess.Popen, grace: tuple[float, float] = STOP_GRACE_S) -> None:
    """SIGINT first so pytest runs its teardown (e.g. private PostgreSQL), then escalate.
    Bounded: at most grace[0] + grace[1] + 5 seconds."""
    for sig, wait in ((signal.SIGINT, grace[0]), (signal.SIGTERM, grace[1]), (signal.SIGKILL, 5.0)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


# ── Main ───────────────────────────────────────────────────────────────────


def in_container() -> bool:
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("--patch", action="append", default=[], help="repeatable; applied in order")
    parser.add_argument("--path", action="append", default=[], help="repeatable; PATHS mode")
    parser.add_argument("--author", default="", help="bot id; required with --path")
    parser.add_argument("--suite", choices=sorted(SUITE_TARGETS), default="offline")
    parser.add_argument("--node", action="append", default=[], help="repeatable; suite=target")
    parser.add_argument("--lock-wait", type=int, default=DEFAULT_LOCK_WAIT_S)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--recon", help="commit-recon executable (default: discovered)")
    parser.add_argument("--pgq", help="pgq executable (default: discovered)")
    parser.add_argument("--provenance-file", type=Path,
                        help='offline/test input: {"recon": {...}, "rows": "<csv>"}')
    parser.add_argument("--allow-container", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--suite-command", help=argparse.SUPPRESS)  # tests: JSON argv
    parser.add_argument("--stop-grace", default=",".join(str(s) for s in STOP_GRACE_S),
                        help=argparse.SUPPRESS)  # tests: "SIGINT_s,SIGTERM_s"
    args = parser.parse_args(argv)
    try:
        sigint_s, sigterm_s = (float(part) for part in args.stop_grace.split(","))
    except ValueError:
        parser.error("--stop-grace must be two numbers, e.g. 30,10")
    args.stop_grace = (sigint_s, sigterm_s)
    args.patch = [p for p in args.patch if p]
    args.path = [p for p in args.path if p]
    args.node = [n for n in args.node if n]
    args.author = args.author.strip().lower()
    return args


def check_args(args: argparse.Namespace) -> None:
    if args.patch and args.path:
        raise Refused("use PATCHES or PATHS, not both")
    if args.path and not args.author:
        raise Refused("PATHS requires AUTHOR=<your bot id>")
    if args.suite == "target" and not args.node:
        raise Refused("SUITE=target requires NODES")
    if args.node and args.suite != "target":
        raise Refused("NODES requires SUITE=target")
    if not 0 <= args.lock_wait <= 24 * 3600:
        raise Refused("LOCK_WAIT must be 0-86400 seconds")
    if in_container() and not args.allow_container:
        raise Refused("run on the test host (ssh echo), never inside a bridge container")


def check_ownership(args: argparse.Namespace, git: Git, paths: list[str]) -> dict[str, str]:
    if args.provenance_file:
        payload = json.loads(args.provenance_file.read_text())
        recon = payload.get("recon")
        rows = rows_from_csv(payload["rows"], args.repo.name) if "rows" in payload else None
    else:
        recon = load_recon(find_tool("commit-recon", args.recon), args.repo)
        rows = load_rows(find_tool("pgq", args.pgq), args.repo.name)
    if recon is None and rows is None:
        raise Refused("PATHS needs edit provenance (pgq or commit-recon) and neither is available")
    check = OwnershipCheck(args.author, rows, recon)
    verdicts: dict[str, str] = {}
    refused = []
    for path in paths:
        head = git.blob("HEAD", path)
        current = (args.repo / path).read_bytes() if (args.repo / path).is_file() else None
        ok, why = check.verdict(path, sha256(head) if head is not None else None,
                                sha256(current) if current is not None else None)
        verdicts[path] = why
        if not ok:
            refused.append(f"  {path}: {why}")
    if refused:
        raise Refused(
            f"PATHS can't prove these changes are all {args.author}'s:\n" + "\n".join(refused)
            + "\nIf another bot has uncommitted changes in the same file, sequence it on the task "
            "through the coordinator. Never clone.")
    return verdicts


def execute(args: argparse.Namespace, receipt: Receipt, run_dir: Path) -> int:
    check_args(args)
    repo, root = args.repo, args.root
    git = Git(repo)
    receipt.base = git.out("rev-parse", "--verify", f"{args.base}^{{commit}}")
    receipt.canonical_head = git.out("rev-parse", "HEAD")
    receipt.suite, receipt.nodes, receipt.author = args.suite, args.node, args.author or None
    swept = sweep_stale(root)
    if swept:
        print(f"Snapshot: removed {len(swept)} stale snapshot(s): {', '.join(swept)}", flush=True)
    prune_old_runs(root)

    sources: list[Path] = []
    if args.path:
        receipt.mode = "paths"
        if receipt.base != receipt.canonical_head:
            raise Refused("PATHS diffs this checkout's working tree, so BASE must be its HEAD")
        receipt.paths = validate_paths(git, args.path)
        receipt.ownership = check_ownership(args, git, receipt.paths)
        generated = run_dir / "paths.patch"
        build_paths_patch(git, receipt.base, receipt.paths, generated)
        sources = [generated]
    elif args.patch:
        receipt.mode = "patches"
        sources = [Path(p).expanduser().resolve() for p in args.patch]
        missing = [str(p) for p in sources if not p.is_file()]
        if missing:
            raise Refused(f"patch file(s) not found: {', '.join(missing)}")
    receipt.patches = pin_patches(sources, run_dir)

    with snapshot_worktree(repo, root, receipt.run_id, receipt.base) as worktree:
        receipt.worktree = str(worktree)
        snap = Git(worktree)
        for patch in receipt.patches:
            applied = snap.run("apply", "--index", patch.saved_as, check=False)
            if applied.returncode:
                receipt.status = "apply_failed"
                receipt.error = (f"{Path(patch.saved_as).name} does not apply on "
                                 f"{receipt.base[:12]}: {applied.stderr.decode(errors='replace').strip()}")
                return EXIT_FAILED
        diff = snap.run("diff", "--cached", "--no-color", "--no-ext-diff", receipt.base).stdout
        receipt.diff_sha256 = sha256(diff)
        receipt.diff_sha256_short = receipt.diff_sha256[:16]
        receipt.files = snap.out("diff", "--cached", "--name-status", receipt.base).splitlines()
        changed = set(snap.out("diff", "--cached", "--name-only", receipt.base).splitlines())
        receipt.runner_touched = [path for path in RUNNER_FILES if path in changed]
        if snap.run("diff", "--quiet", check=False).returncode:
            raise RuntimeError("snapshot working tree differs from its index after apply")
        command = (json.loads(args.suite_command) if args.suite_command
                   else suite_command(args.suite, args.node))
        print(f"Snapshot: {receipt.base[:12]} + {len(receipt.patches)} patch(es), "
              f"diff sha256:{receipt.diff_sha256_short}; running {' '.join(command)}", flush=True)
        receipt.suite_exit = run_suite(command, worktree, Path(receipt.log or run_dir / "test.log"),
                                       args.lock_wait, receipt, args.stop_grace)
    if receipt.suite_exit == 0:
        receipt.status = "passed"
        return EXIT_PASSED
    receipt.status = "failed"
    receipt.error = f"suite exited {receipt.suite_exit}"
    return EXIT_FAILED


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.resolve()
    args.root = args.root.resolve()
    if args.root == args.repo or args.repo in args.root.parents:
        print(f"Snapshot refused: root {args.root} must be outside the repository", file=sys.stderr)
        return EXIT_REFUSED
    started = time.monotonic()
    tag = re.sub(r"[^a-z0-9-]", "", args.author)[:24] or "anon"
    run_id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{tag}-{secrets.token_hex(3)}"
    run_dir = args.root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = Receipt(run_id=run_id, repo=str(args.repo), log=str(run_dir / "test.log"))
    receipt_path = run_dir / "receipt.json"
    print(f"Snapshot: run {run_id}; receipt will be {receipt_path}", flush=True)

    def interrupt(signum: int, _frame: object) -> None:
        raise Interrupted(signal.Signals(signum).name)

    # Own SIGINT even if the parent ignored it (background jobs); keep a nohup'd SIGHUP ignored.
    handlers = {signal.SIGINT: signal.default_int_handler, signal.SIGTERM: interrupt}
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        handlers[signal.SIGHUP] = interrupt
    previous = {sig: signal.signal(sig, handler) for sig, handler in handlers.items()}
    try:
        receipt.exit_code = execute(args, receipt, run_dir)
    except Refused as exc:
        receipt.status, receipt.exit_code, receipt.error = "refused", EXIT_REFUSED, str(exc)
    except (KeyboardInterrupt, Interrupted) as exc:
        receipt.status, receipt.exit_code = "interrupted", EXIT_INTERRUPTED
        receipt.error = f"interrupted ({exc or 'SIGINT'})"
    except Exception as exc:  # noqa: BLE001 - every failure must still produce a receipt
        receipt.status, receipt.exit_code, receipt.error = "error", EXIT_FAILED, repr(exc)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        receipt.duration_s = round(time.monotonic() - started, 1)
        with signals_blocked():
            receipt.write(receipt_path)
    try:
        print(receipt.summary(), flush=True)
    except (BrokenPipeError, OSError):
        pass
    return receipt.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
