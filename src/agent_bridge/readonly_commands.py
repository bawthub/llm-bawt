"""Conservative read-only classification of tool calls (TASK-2024).

A turn that only *reads* a working tree (a reviewer's ``git diff``, ``cat``,
``rg``) cannot contaminate another turn's snapshot diff, so it should not
register a live marker there and make the owner's turn ``ineligible``.

This is an allowlist.  Anything not provably unable to write, including
anything this parser cannot fully understand, is classified as a possible
writer, which keeps the existing safe behavior (register; never misattribute).
"""

from __future__ import annotations

import re
import shlex
from typing import Any

# Claude Code tools that cannot modify files.
READ_ONLY_TOOLS = frozenset({"Read", "Grep", "Glob", "LS"})
SHELL_TOOLS = frozenset({"Bash", "shell"})

_SEPARATORS = frozenset({";", "&&", "||", "|"})
_SHELL_WRAPPERS = frozenset({"bash", "sh", "zsh", "/bin/bash", "/bin/sh", "/usr/bin/bash", "/bin/zsh", "/usr/bin/zsh"})
_SAFE_REDIRECT_TARGETS = frozenset({"/dev/null"})

# Commands with no file-writing mode, except for the options rejected below.
_PLAIN_READERS = frozenset({
    "cat", "head", "tail", "wc", "ls", "stat", "nl", "grep", "egrep", "fgrep",
    "pwd", "echo", "printf", "true", "false", "test", "[", "which", "type", "realpath",
    "readlink", "basename", "dirname", "du", "df", "diff", "cmp", "cut", "tr",
    "jq", "cd", "whoami", "id", "uname", "column", "md5sum",
    "sha1sum", "sha256sum", "rev", "tac",
})
# name -> (rejected long/word options, rejected short option letters).
_REJECTED_OPTIONS: dict[str, tuple[frozenset[str], str]] = {
    "rg": (frozenset({"--pre", "--pre-glob"}), ""),
    "sort": (frozenset({"--output", "--compress-program"}), "o"),
    # -R reruns tree with -o 00Tree.html in each directory.
    "tree": (frozenset(), "oR"),
    "file": (frozenset({"--compile"}), "C"),
    "find": (frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"}), ""),
}
# sed only as `sed -n <line-range>p [files]`: the script grammar can write (w, e).
_SED_PRINT_SCRIPT = re.compile(r"^(?:(?:\d+|\$)(?:,(?:\d+|\$))?p)(?:;(?:(?:\d+|\$)(?:,(?:\d+|\$))?p))*$")

_GIT_READ_SUBCOMMANDS = frozenset({
    "status", "diff", "log", "show", "blame", "rev-parse", "ls-files", "ls-tree",
    "cat-file", "describe", "shortlog", "grep", "merge-base", "rev-list",
    "for-each-ref", "name-rev", "count-objects", "check-ignore", "whatchanged",
    "show-ref", "diff-tree", "diff-files", "diff-index", "range-diff",
})
# Read-only only when the listed (sub)arguments are all that follows.
_GIT_LISTING = {
    "branch": frozenset({"-a", "-r", "-v", "-vv", "--list", "--show-current", "--all", "--remotes", "--no-color", "--color"}),
    "remote": frozenset({"-v", "--verbose", "show", "get-url"}),
    "stash": frozenset({"list", "show"}),
    "worktree": frozenset({"list", "--porcelain"}),
    "tag": frozenset({"-l", "--list", "-n"}),
    "reflog": frozenset({"show"}),
    "config": frozenset({"--get", "--get-all", "--list", "-l", "--show-origin", "--get-regexp"}),
}
_GIT_REJECTED_LONG = frozenset({"--output", "--output-directory", "--ext-diff", "--open-files-in-pager"})


def _rejected_option(arg: str, words: frozenset[str], letters: str) -> bool:
    """Exact word options, abbreviated long options, and short-option clusters."""
    if arg in words:
        return True
    if arg.startswith("--"):
        name = arg.split("=", 1)[0]
        # GNU getopt and git accept any unambiguous prefix (--out=x is --output).
        return len(name) > 2 and any(word.startswith(name) for word in words if word.startswith("--"))
    # A short-option cluster (-ro, -ofile) containing a rejected letter.
    return bool(letters) and arg.startswith("-") and len(arg) > 1 and any(letter in arg[1:] for letter in letters)


def is_read_only_tool_call(tool_name: str, tool_input: Any) -> bool:
    """True only when the call provably cannot write files."""
    if tool_name in READ_ONLY_TOOLS:
        return True
    if tool_name not in SHELL_TOOLS or not isinstance(tool_input, dict):
        return False
    command = tool_input.get("command")
    if isinstance(command, (list, tuple)):
        if not all(isinstance(part, str) for part in command):
            return False
        command = shlex.join(command)
    return isinstance(command, str) and is_read_only_command(command)


def is_read_only_command(command: str) -> bool:
    try:
        return _read_only(command, depth=0)
    except Exception:
        return False


def _tokens(command: str) -> list[str] | None:
    if not command.strip() or any(ch in command for ch in ("\n", "\r", "`", "\x00")):
        return None
    if "$(" in command or "<(" in command or ">(" in command:
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return None


def _read_only(command: str, *, depth: int) -> bool:
    tokens = _tokens(command)
    if not tokens:
        return False
    # `bash -lc '<script>'` (how Codex reports commands): classify the script.
    if tokens[0] in _SHELL_WRAPPERS:
        if depth > 0 or len(tokens) != 3 or tokens[1] not in ("-c", "-lc", "-cl"):
            return False
        return _read_only(tokens[2], depth=depth + 1)
    segment: list[str] = []
    for token in tokens + [";"]:
        if token in _SEPARATORS:
            if not segment or not _segment_read_only(segment):
                return False
            segment = []
        else:
            # Words and every other operator (&, (, <<, >, >&, ...) are judged per segment.
            segment.append(token)
    return True


def _segment_read_only(segment: list[str]) -> bool:
    words: list[str] = []
    index = 0
    while index < len(segment):
        token = segment[index]
        if token in (">", ">>", ">|", "&>", "&>>"):
            if index + 1 >= len(segment) or segment[index + 1] not in _SAFE_REDIRECT_TARGETS:
                return False
            index += 2
            continue
        if token == ">&":
            if index + 1 >= len(segment) or segment[index + 1] not in ("1", "2"):
                return False
            index += 2
            continue
        if token == "<":
            # Reading a file into stdin is harmless; the target must be a plain word.
            if index + 1 >= len(segment) or set(segment[index + 1]) <= set(";&|<>()"):
                return False
            index += 2
            continue
        if set(token) <= set(";&|<>()"):
            return False  # &, (, ), <<, <<<, <> and anything unrecognised
        words.append(token)
        index += 1
    if not words:
        return False
    name, args = words[0], words[1:]
    if "=" in name or name.startswith(("#", "-")) or "/" in name:
        return False  # env assignments, comments, explicit binary paths
    if name in _PLAIN_READERS:
        return True
    if name in _REJECTED_OPTIONS:
        words_rejected, letters = _REJECTED_OPTIONS[name]
        return not any(_rejected_option(arg, words_rejected, letters) for arg in args)
    if name == "sed":
        return _sed_read_only(args)
    if name == "git":
        return _git_read_only(args)
    return False


def _sed_read_only(args: list[str]) -> bool:
    if not args or args[0] != "-n" or len(args) < 2:
        return False
    if not _SED_PRINT_SCRIPT.match(args[1]):
        return False
    return not any(arg.startswith("-") for arg in args[2:])


def _git_read_only(args: list[str]) -> bool:
    index = 0
    while index < len(args) and args[index].startswith("-"):
        option = args[index]
        if option == "-C" and index + 1 < len(args):
            index += 2
        elif option in ("--no-pager", "-P", "--no-optional-locks"):
            index += 1
        else:
            return False  # -c, --git-dir, --work-tree, --exec-path, ...
    if index >= len(args):
        return False
    sub, rest = args[index], args[index + 1:]
    # -O (orderfile / open-in-pager), including inside clusters like -nO.
    if any(_rejected_option(arg, _GIT_REJECTED_LONG, "O") for arg in rest):
        return False
    if sub in _GIT_READ_SUBCOMMANDS:
        return True
    allowed = _GIT_LISTING.get(sub)
    if allowed is None:
        return False
    if sub == "reflog":  # bare `git reflog` is `git reflog show`
        return not rest or rest[0] in allowed
    if sub == "stash":  # bare `git stash` is a push
        return bool(rest) and rest[0] in allowed
    if sub == "worktree":
        return bool(rest) and all(arg in allowed for arg in rest)
    if sub == "remote":
        if not rest:
            return True
        if rest[0] in ("-v", "--verbose"):
            return len(rest) == 1
        return rest[0] in ("show", "get-url") and not any(arg in ("add", "set-url", "remove", "rm", "rename") for arg in rest[1:])
    if sub == "config":
        scopes = {"--show-origin", "--global", "--local", "--system", "--null", "-z", "--name-only"}
        return bool(rest) and rest[0] in allowed and all(not arg.startswith("-") or arg in scopes or arg in allowed for arg in rest[1:])
    return all(arg in allowed for arg in rest)
