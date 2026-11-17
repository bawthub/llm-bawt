"""Refuse secret values in env and compose files (TASK-2386).

Secrets reach containers only at startup: the master key and the database
password through the root-only Compose secrets volume, everything else
through the app's broker. So no env template, compose file or repository
``.env`` may hold one. This checker fails on:

* ``secret_literal``: a secret-shaped name (a ``KEY``/``TOKEN``/``SECRET``/
  ``PASSWORD``/``PASSWD``/``CREDENTIAL(S)`` name segment) assigned a
  non-empty literal, of any length;
* ``credential_url``: any value carrying ``scheme://user:password@``.

Pointers pass (``*_FILE``, ``*_PATH``, ``*_UID``, ``*_ENV``, ``*_MODE``), as do
empty values, pure Compose interpolation (``${NAME}``, ``$NAME``, with an
empty or interpolated default) and documented placeholders (``<...>``,
``changeme``, ``your-...``, ``xxx...``, a bare ``password``).

* ``env_backup``: a ``.env.bak*`` file exists at all. Never copy an env file.

Output is ``path:line NAME code``: never a value. Stdlib only, so the ops
worker and a bare host ``python3 -I`` can run it::

    python3 -I src/llm_bawt/secret_guard.py [ROOT ...]   # exit 1 on a finding
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_SECRET_NAME = re.compile(r"(^|_)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(_|$)")
_POINTER_SUFFIXES = ("_FILE", "_PATH", "_UID", "_ENV", "_MODE")
_GIT_CONFIG_NAME = re.compile(r"^GIT_CONFIG_KEY_\d+$")
#: Connection URLs: secret only through their credentials, so only the
#: ``credential_url`` check applies (a host-only URL is fine).
URL_NAMES = frozenset({"DATABASE_URL"})

#: ``NAME=value`` in an env file (optionally ``export``ed).
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")
#: Compose ``environment``: list form ``- NAME=value`` or map form ``NAME: value``.
_COMPOSE_LIST = re.compile(r"^\s*-\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)=(.*?)['\"]?\s*$")
_COMPOSE_MAP = re.compile(r"^\s*(?:-\s+)?(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1\s*:\s*(.*)$")
#: ``${NAME}``, ``${NAME:-default}``, ``${NAME:?msg}``, ``$NAME``.
_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-?+])((?:[^{}]|\{[^{}]*\})*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_CREDENTIAL_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://([^/\s:@'\"]+):([^/\s@'\"]+)@")
_PLACEHOLDER = re.compile(
    r"^(<[^<>]*>|changeme|change[-_]?me|your[-_].*|x{3,}|\*{3,}|\.\.\.|password|pass|secret|example)$", re.IGNORECASE)


def is_secret_shaped(name: str) -> bool:
    return (bool(_SECRET_NAME.search(name.upper())) and not name.upper().endswith(_POINTER_SUFFIXES)
            and not _GIT_CONFIG_NAME.match(name))


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    name: str
    code: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} {self.name} {self.code}"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    # An unquoted env value ends at an inline comment (``NAME= # note`` is empty).
    if value.startswith("#"):
        return ""
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def _is_literal(value: str) -> bool:
    """True when ``value`` carries anything but interpolation or a placeholder."""
    rest = value
    while True:
        match = _INTERPOLATION.search(rest)
        if not match:
            break
        operator, default = match.group(2), match.group(3)
        # ``${NAME:-literal}`` ships its default and ``${NAME:+literal}`` its
        # alternate; only ``${NAME:?message}`` carries no value.
        if operator in (":-", "-", ":+", "+") and default and _is_literal(default):
            return True
        rest = rest[:match.start()] + rest[match.end():]
    rest = rest.replace("$$", "").strip()
    return bool(rest) and not _PLACEHOLDER.match(rest)


def _credential_url(value: str) -> bool:
    for match in _CREDENTIAL_URL.finditer(value):
        if _is_literal(match.group(2)):
            return True
    return False


def _check(path: str, line: int, name: str, raw: str) -> list[Finding]:
    value = _unquote(raw)
    if _credential_url(value):
        return [Finding(path, line, name, "credential_url")]
    if name not in URL_NAMES and is_secret_shaped(name) and _is_literal(value):
        return [Finding(path, line, name, "secret_literal")]
    return []


def scan_env_text(text: str, path: str = "<env>") -> list[Finding]:
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if match:
            findings += _check(path, number, match.group(1), match.group(2))
    return findings


def _flow_items(body: str) -> list[str]:
    """Top-level comma-separated items of a YAML flow collection's inside."""
    items, depth, quote, current = [], 0, "", []
    for char in body:
        if quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        elif char == "," and depth == 0:
            items.append("".join(current))
            current = []
            continue
        current.append(char)
    items.append("".join(current))
    return [item.strip() for item in items if item.strip()]


def _strip_comment(value: str) -> str:
    """Cut a trailing `` # comment``, but never inside a quoted scalar."""
    quote = ""
    for index, char in enumerate(value):
        if quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].strip()
    return value.strip()


def _scan_flow(path: str, line: int, value: str, parent: str = "-") -> list[Finding]:
    """``environment: {A: x, B: y}`` / ``[A=x, "B=y"]``: check each item,
    recursing into nested collections (``services: {db: {environment: {...}}}``)."""
    value = _strip_comment(value)
    if len(value) < 2 or value[0] + value[-1] not in ("{}", "[]"):
        return []
    findings: list[Finding] = []
    for item in _flow_items(value[1:-1]):
        if item[:1] in "{[":
            findings += _scan_flow(path, line, item, parent)
            continue
        match = (re.fullmatch(r"(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1\s*:\s*(.*)", item, re.S)
                 or re.fullmatch(r"(['\"]?)([A-Za-z_][A-Za-z0-9_]*)=(.*?)\1", item, re.S))
        if match and match.group(3).strip()[:1] in ("{", "["):
            findings += _scan_flow(path, line, match.group(3), match.group(2))
        elif match:
            findings += _check(path, line, match.group(2), match.group(3))
        elif _credential_url(_unquote(item)):
            findings.append(Finding(path, line, parent, "credential_url"))
    return findings


def scan_compose_text(text: str, path: str = "<compose>") -> list[Finding]:
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        listed = _COMPOSE_LIST.match(line)
        mapped = None if listed else _COMPOSE_MAP.match(line)
        if listed:
            findings += _check(path, number, listed.group(1), listed.group(2))
        elif mapped and mapped.group(3).strip()[:1] in ("{", "["):
            findings += _scan_flow(path, number, mapped.group(3), mapped.group(2))
        elif mapped:
            findings += _check(path, number, mapped.group(2), mapped.group(3))
        elif stripped[:1] in ("{", "["):
            findings += _scan_flow(path, number, stripped)
        elif _credential_url(_unquote(stripped.lstrip("- "))):
            findings.append(Finding(path, number, "-", "credential_url"))
    return findings


def scan_root(root: Path) -> list[Finding]:
    """The compose files, env templates and ``.env`` directly under ``root``."""
    findings: list[Finding] = []
    for path in sorted(root.glob("docker-compose*.y*ml")) + sorted(root.glob("compose*.y*ml")):
        findings += scan_compose_text(path.read_text(encoding="utf-8", errors="replace"), str(path))
    for path in sorted(p for p in root.glob(".env*") if p.is_file()):
        if path.name.startswith(".env.bak"):
            findings.append(Finding(str(path), 0, "-", "env_backup"))
            continue
        findings += scan_env_text(path.read_text(encoding="utf-8", errors="replace"), str(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    roots = [Path(a) for a in (argv if argv is not None else sys.argv[1:])] or [Path(".")]
    findings = [f for root in roots for f in scan_root(root)]
    for finding in findings:
        print(finding)
    if findings:
        print(f"secret_guard: {len(findings)} secret value(s) in env/compose files; "
              "move them to the encrypted store or the startup injection", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
