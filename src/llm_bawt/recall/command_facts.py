"""Structured facts projected from recorded shell commands (TASK-1557, ``tool_args``).

Opaque command text never crosses the recall boundary: it may carry
credentials or copied protected content that no pattern can recognize. A
recorded ``Bash`` command contributes only positively recognized facts built
from validated operands, here ``git clone <url> [<dest>]`` with the ``cd`` that
fixes its destination. Everything else contributes nothing (no text, no count).

Parsing is fail-closed and left to right. The first token that is not a plain
word or a ``&&``/``;``/newline separator ends the parse: substitution
(``$``), backtick, backslash escape or line continuation, unbalanced quote,
pipe, redirect, heredoc, subshell, background ``&`` or ``||``. Only segments
completed *before* it are used; the shell runs those first and nothing later
changes their words. The segment it interrupts and everything after are dropped.

A fact records an *attempted* invocation, not that it ran or succeeded, and
says so in its rendered line.
"""
from __future__ import annotations

from dataclasses import dataclass
import posixpath
import re
import shlex

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
_PUNCTUATION = "();<>|&\n"
# Characters that end the parse inside any word, quoted or not.
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
    """A plain word, one fully quoted word unwrapped, or None to end the parse."""
    if any(char in _UNSAFE_IN_WORD for char in token):
        return None
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"" \
            and not set(token[1:-1]) & set("'\""):
        return token[1:-1]
    return token


def _segments(command: str) -> list[list[str]]:
    """Complete ``&&``/``;``/newline segments before the first unsafe token."""
    lexer = shlex.shlex(command, posix=False, punctuation_chars=_PUNCTUATION)
    lexer.whitespace = " \t"
    lexer.whitespace_split = True
    lexer.commenters = ""
    segments: list[list[str]] = []
    current: list[str] = []
    try:
        for token in lexer:
            if set(token) <= set(_PUNCTUATION):
                # Unquoted punctuation only; quoted "&&" is a word (non-posix keeps quotes).
                if token.replace("\n", "") not in {"", "&&", ";"}:
                    return segments
                if current:
                    segments.append(current)
                current = []
                continue
            word = _word(token)
            if word is None:
                return segments
            current.append(word)
    except ValueError:
        # Unbalanced quote: the interrupted segment is dropped with the rest.
        return segments
    if current:
        segments.append(current)
    return segments


def _resolve(cwd: str | None, path: str) -> str | None:
    if path.startswith(("/", "~")):
        return path
    return posixpath.normpath(posixpath.join(cwd, path)) if cwd else None


def _clone(words: list[str], cwd: str | None) -> CommandFact | None:
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
    if not safe_path(operands[1]):
        return None
    return CommandFact("clone", operands[0], _resolve(cwd, operands[1]) or operands[1])


def command_facts(command: object) -> list[CommandFact]:
    """Recognized facts from one recorded Bash ``command``; [] for anything else."""
    if not isinstance(command, str) or not command or len(command) > MAX_COMMAND_CHARS:
        return []
    facts: list[CommandFact] = []
    cwd: str | None = None
    for words in _segments(command):
        if words[0] == "cd":
            target = words[1] if len(words) == 2 else None
            cwd = _resolve(cwd, target) if target is not None and safe_path(target) else None
        elif words[:2] == ["git", "clone"]:
            fact = _clone(words, cwd)
            if fact is not None:
                facts.append(fact)
        elif words[0] in _CWD_CHANGERS:
            cwd = None
    return facts[:MAX_FACTS]
