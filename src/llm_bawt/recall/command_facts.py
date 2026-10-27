"""Structured facts projected from recorded shell commands (TASK-1557, ``tool_args``).

Opaque command text never crosses the recall boundary: it may carry
credentials or copied protected content that no pattern can recognize. A
recorded ``Bash`` command contributes only positively recognized facts built
from validated operands, here ``git clone <url> [<dest>]`` with the ``cd`` that
fixes its destination. Everything else contributes nothing (no text, no count).

Parsing is fail-closed and left to right, by a tokenizer that accepts only
what it can read exactly as the shell would. A word is a run of plain
characters, or one quoted string with nothing attached to it. A separator is
``&&``, ``;`` or a newline. Anything else ends the parse: ``#`` (a comment, or
a word the shell might read as one), a quote joined to other characters
(``'a'b`` is one shell word), substitution (``$``), backtick, backslash escape
or line continuation, unbalanced quote, pipe, redirect, heredoc, subshell,
background ``&`` or ``||``. Only segments completed *before* it are used; the
shell runs those first and nothing later changes their words. The segment it
interrupts and everything after are dropped.

A path is checked as the shell would resolve it: a ``cd`` target and a clone
destination must pass the path rules both as written and after joining the
working directory. After a directory change into a denied path (or one the
parser can't follow, naming one), a relative destination is refused too.

A fact records an *attempted* invocation, not that it ran or succeeded, and
says so in its rendered line.
"""
from __future__ import annotations

from dataclasses import dataclass
import posixpath
import re

# Paths are deliberately selected historical operands. No shell metacharacter,
# quote, space, percent-encoding, ``$`` or ``..`` segment can pass.
_PATH = re.compile(r"[A-Za-z0-9._/~+@-]{1,240}")
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}[A-Za-z0-9])?"
_HOST = rf"{_LABEL}(?:\.{_LABEL})*"
_REPO_PATH = r"[A-Za-z0-9._/~+-]{1,200}"
# https://host/path: no userinfo, port, query or fragment can match.
_HTTPS_URL = re.compile(rf"https://{_HOST}/({_REPO_PATH})")
# scp-like ssh: exactly the conventional git@host:owner/repo form.
_SSH_URL = re.compile(rf"git@{_HOST}:({_REPO_PATH})")
_BRANCH = re.compile(r"[A-Za-z0-9._/-]{1,100}")
_DEPTH = re.compile(r"[1-9][0-9]{0,5}")
# Credential stores and key material: a path naming one is never emitted.
_CREDENTIAL_PATH = re.compile(r"""(?ix)
    (?:^|/)\.env(?:rc|\.[A-Za-z0-9_.-]+)?(?:/|$)
  | (?:^|/)\.(?:pgpass|netrc|git-credentials|npmrc|pypirc|htpasswd)(?:/|$)
  | (?:^|/)\.(?:ssh|aws|gnupg)(?:/|$)
  | (?:^|/)\.docker/config\.json$ | (?:^|/)\.kube/config$
  | (?:^|/)auth\.json$ | (?:^|/)credentials[^/]*$ | (?:^|/)secrets?\.[^/]*$
  | (?:^|/)id_(?:rsa|dsa|ecdsa|ed25519)[^/]*$
  | \.(?:pem|key|p12|pfx|jks|keystore)$
  | ^/proc/
""")
# One token at the cursor. A plain run excludes every character with a shell
# meaning (quotes, ``#``, ``$``, backtick, backslash, operators, whitespace).
# ``\Z`` ends a word cleanly: a word must be followed by blank, a separator or
# the end, so a quote joined to anything matches nothing and ends the parse.
_TOKEN = re.compile(r"""
    (?P<blank>[ \t]+)
  | (?P<separator>&&|;|\n)
  | (?P<word>(?:'[^']*'|"[^"]*"|[^\s'"\#$`\\;&|<>()]+)(?=[ \t;\n]|&&|\Z))
""", re.VERBOSE)
# Characters that end the parse inside a quoted word, too.
_UNSAFE_IN_WORD = frozenset("$`\\\n")
# Builtins that change (or may change) the working directory: after one that is
# not a recognized ``cd <path>``, relative destinations are left unresolved.
_CWD_CHANGERS = frozenset({"cd", "pushd", "popd", "source", ".", "eval", "exec", "builtin", "command"})
_CLONE_SWITCHES = frozenset({"-q", "--quiet", "--single-branch"})
MAX_COMMAND_CHARS = 4000
MAX_FACTS = 4


@dataclass(frozen=True)
class CommandFact:
    op: str
    repo: str
    dest: str | None

    def render(self) -> str:
        line = f"attempted {self.op} repo={self.repo}"
        return f"{line} dest={self.dest}" if self.dest else line


def safe_path(value: str) -> bool:
    return (bool(_PATH.fullmatch(value)) and ".." not in value.split("/")
            and not _CREDENTIAL_PATH.search(value))


def safe_repo(value: str) -> bool:
    match = _HTTPS_URL.fullmatch(value) or _SSH_URL.fullmatch(value)
    return bool(match) and ".." not in match.group(1).split("/")


def _word(token: str) -> str | None:
    """A plain word, one quoted word unwrapped, or None to end the parse."""
    if any(char in _UNSAFE_IN_WORD for char in token):
        return None
    if token[0] in "'\"":
        inner = token[1:-1]
        # Neither quote kind inside: one plain string the shell reads verbatim.
        return None if set(inner) & set("'\"") else inner
    return token


def _segments(command: str) -> list[list[str]]:
    """Complete ``&&``/``;``/newline segments before the first unsafe token."""
    segments: list[list[str]] = []
    current: list[str] = []
    position = 0
    while position < len(command):
        match = _TOKEN.match(command, position)
        if match is None:
            # Anything the tokenizer can't read exactly: the interrupted
            # segment is dropped with the rest.
            return segments
        position = match.end()
        if match.lastgroup == "separator":
            if current:
                segments.append(current)
            current = []
        elif match.lastgroup == "word":
            word = _word(match.group())
            if word is None:
                return segments
            current.append(word)
    if current:
        segments.append(current)
    return segments


def _resolve(cwd: str | None, path: str) -> str | None:
    if path.startswith(("/", "~")):
        return path
    return posixpath.normpath(posixpath.join(cwd, path)) if cwd else None


def _safe_resolved(cwd: str | None, path: str, denied: bool) -> str | None:
    """``path`` as the shell resolves it, if both it and the result pass; else None.

    A relative path under an unknown directory stays as written, unless the
    shell may be in a denied directory (``denied``): then it is refused.
    """
    if not safe_path(path) or (denied and not path.startswith(("/", "~"))):
        return None
    resolved = _resolve(cwd, path) or path
    return resolved if safe_path(resolved) else None


def _clone(words: list[str], cwd: str | None, denied: bool) -> CommandFact | None:
    """``git clone [switches] <repo> [<dest>]`` with validated operands, else None."""
    index, operands = 2, []
    while index < len(words):
        word = words[index]
        if word in _CLONE_SWITCHES:
            index += 1
        elif word == "--depth" and index + 1 < len(words) and _DEPTH.fullmatch(words[index + 1]):
            index += 2
        elif word in {"--branch", "-b"} and index + 1 < len(words) and _BRANCH.fullmatch(words[index + 1]):
            index += 2
        elif word.startswith("-"):
            return None
        else:
            operands.append(word)
            index += 1
    if not 1 <= len(operands) <= 2 or not safe_repo(operands[0]):
        return None
    if len(operands) == 1:
        return CommandFact("clone", operands[0], None)
    dest = _safe_resolved(cwd, operands[1], denied)
    return CommandFact("clone", operands[0], dest) if dest is not None else None


def command_facts(command: object) -> list[CommandFact]:
    """Recognized facts from one recorded Bash ``command``; [] for anything else."""
    if not isinstance(command, str) or not command or len(command) > MAX_COMMAND_CHARS:
        return []
    facts: list[CommandFact] = []
    cwd: str | None = None  # the known absolute directory, or None when unknown
    denied = False  # the shell may be in a directory the path rules deny
    for words in _segments(command):
        if words[:2] == ["git", "clone"]:
            fact = _clone(words, cwd, denied)
            if fact is not None:
                facts.append(fact)
        elif words[0] == "cd" and len(words) == 2:
            if _safe_resolved(cwd, words[1], denied) is None:
                cwd, denied = None, True
            else:
                cwd = _resolve(cwd, words[1])
                denied = denied and cwd is None
        elif words[0] in _CWD_CHANGERS:
            # A change the parser can't follow; a denied operand may be the target.
            cwd = None
            denied = denied or not all(safe_path(word) for word in words[1:])
    return facts[:MAX_FACTS]
