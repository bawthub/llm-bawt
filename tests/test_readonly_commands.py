"""TASK-2024: read-only tool calls do not make another turn's tree busy."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agent_bridge.readonly_commands import is_read_only_command, is_read_only_tool_call
from agent_bridge.turn_snapshots import TurnSnapshotObserver

READ_ONLY = [
    "git diff",
    "git status --short",
    "git -C /home/bridge/dev/.worktrees/x log --oneline -5 master..HEAD",
    "git --no-pager show HEAD:src/a.py",
    "cd /home/bridge/dev/.worktrees/x && git diff --stat && git status --short",
    "cat src/a.py | head -50",
    "rg -n 'def foo' src tests 2>/dev/null | head",
    "grep -rn TODO src >/dev/null 2>&1",
    "sed -n 1,80p src/a.py",
    "sed -n '10,$p;200p' a.py",
    "ls -la; wc -l src/*.py",
    "find . -name '*.py' -newer x",
    "git branch --show-current",
    "git worktree list --porcelain",
    "git stash list",
    "git config --get user.name",
    "sort -u names.txt",
    "bash -lc 'cd /repo && git diff --name-only'",
    "/bin/bash -lc 'rg foo | wc -l'",
    "git diff < /dev/null",
]

WRITERS = [
    "git commit -m x",
    "git checkout main",
    "git stash",
    "git stash pop",
    "git branch new-branch",
    "git config user.name x",
    "git config --get a --add b",
    "git -c core.pager=sh log",
    "git diff --output=patch.txt",
    "git log --output patch",
    "git grep -O foo",
    "cat a > b",
    "cat a >> b",
    "echo x >| b",
    "cat a 2> err.log",
    "echo hi &> out",
    "cat a | tee b",
    "cat a | sh",
    "rg foo | xargs rm",
    "sed -i s/a/b/ f",
    "sed -n 1p f -i",
    "sed -n '1w out' f",
    "sed 's/a/b/' f",
    "awk '{print > \"x\"}' f",
    "find . -delete",
    "find . -name x -exec rm {} ;",
    "sort -o out in",
    "sort -ro out in",
    "sort --output=out in",
    "tree -o out",
    "file -C -m magic",
    "rg --pre ./script foo",
    "echo $(rm x)",
    "echo `rm x`",
    "diff <(cat a) b",
    "cat a & rm b",
    "(cd x && rm y)",
    "cat <<EOF > f\nx\nEOF",
    "cat a\nrm b",
    "FOO=1 git diff",
    "/usr/bin/git diff",
    "python -c 'print(1)'",
    "bash script.sh",
    "bash -lc 'git diff > p'",
    "bash -lc \"bash -c 'git diff'\"",
    "npm test",
    "make gate",
    "unbalanced 'quote",
    "",
    "   ",
    "git",
    "git -C",
]


@pytest.mark.parametrize("command", READ_ONLY)
def test_read_only_commands(command: str) -> None:
    assert is_read_only_command(command), command


@pytest.mark.parametrize("command", WRITERS)
def test_writers_and_ambiguous_commands_still_register(command: str) -> None:
    assert not is_read_only_command(command), command


def test_tool_names() -> None:
    for name in ("Read", "Grep", "Glob", "LS"):
        assert is_read_only_tool_call(name, {"file_path": "/x"})
    for name in ("Edit", "Write", "MultiEdit", "NotebookEdit", "Task", "unknown"):
        assert not is_read_only_tool_call(name, {"file_path": "/x"})
    assert is_read_only_tool_call("Bash", {"command": "git diff"})
    assert is_read_only_tool_call("shell", {"command": ["bash", "-lc", "git status"]})
    assert not is_read_only_tool_call("shell", {"command": ["bash", "-lc", "git status > x"]})
    assert not is_read_only_tool_call("Bash", {"command": 5})
    assert not is_read_only_tool_call("Bash", None)


ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}


def git(cwd: Path, *args: str) -> None:
    subprocess.run(("git", "-C", str(cwd), *args), check=True, capture_output=True, env={**os.environ, **ENV})


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "a.txt").write_text("one\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    return root


def test_reviewer_reads_do_not_make_authors_tree_busy(repo: Path) -> None:
    author, reviewer = TurnSnapshotObserver(), TurnSnapshotObserver()
    author.observe_tool(request_id="ra", bot_id="nova", tool_name="Bash", tool_input={"command": "sed -i s/one/ONE/ a.txt"}, cwd=str(repo))
    reviewer.observe_tool(request_id="rb", bot_id="byte", tool_name="Bash", tool_input={"command": f"cd {repo} && git diff"}, cwd=str(repo))
    reviewer.observe_tool(request_id="rb", bot_id="byte", tool_name="Read", tool_input={"file_path": str(repo / "a.txt")}, cwd=str(repo))
    (repo / "a.txt").write_text("ONE\n")
    files_b, receipts_b = reviewer.finish("rb")
    files_a, receipts_a = author.finish("ra")
    assert files_b == [] and receipts_b == []
    assert [Path(f["path"]).name for f in files_a] == ["a.txt"]
    assert receipts_a[0]["status"] == "observed"


def test_reviewer_write_still_makes_both_ineligible(repo: Path) -> None:
    author, reviewer = TurnSnapshotObserver(), TurnSnapshotObserver()
    author.observe_tool(request_id="ra", bot_id="nova", tool_name="Bash", tool_input={"command": "sed -i s/one/ONE/ a.txt"}, cwd=str(repo))
    reviewer.observe_tool(request_id="rb", bot_id="byte", tool_name="Bash", tool_input={"command": "git diff"}, cwd=str(repo))
    reviewer.observe_tool(request_id="rb", bot_id="byte", tool_name="Bash", tool_input={"command": "echo x > b.txt"}, cwd=str(repo))
    (repo / "a.txt").write_text("ONE\n")
    (repo / "b.txt").write_text("x\n")
    _, receipts_b = reviewer.finish("rb")
    files_a, receipts_a = author.finish("ra")
    assert files_a == []
    assert receipts_a[0]["status"] == receipts_b[0]["status"] == "ineligible_shared_tree_busy"
