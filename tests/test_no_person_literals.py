"""TASK-1383: product source names no person and bakes in no deployment infrastructure.

llm-bawt is built for general use, so ``src/`` (code, comments, docstrings and
bundled text files alike) holds no user's name, personal domain, private host
address or person's home directory. The deployment supplies those as data:
``LLM_BAWT_DEFAULT_USER``, runtime settings, the prompt registry and ``.env``.
Tests may use them as fixtures. ``ALLOWED`` pins any reasoned exception by file
and exact match; it is empty, and each entry added needs its reason.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# The DB holds the users; the names are listed here. Add a user's name when they are added.
PERSON_NAMES = ("Nick",)
PERSONAL_DOMAINS = ("ferreri.us", "zenoran.com")

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    *(("person's name", re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE)) for n in PERSON_NAMES),
    *(("personal domain", re.compile(rf"\b{re.escape(d)}\b", re.IGNORECASE)) for d in PERSONAL_DOMAINS),
    # A private-network host address. Network ranges (10.0.0.0/8) are generic and allowed.
    ("private IP", re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b(?!/\d)")),
    # A person's home directory. /home/bridge is the agent bridge containers' own user.
    ("home directory", re.compile(r"/home/(?!bridge\b|<)[a-z_][a-z0-9_-]*")),
)

# (path under src/, exact match) -> reason.
ALLOWED: dict[tuple[str, str], str] = {}

_TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".sh", ".j2", ".html", ".sql", ".cfg", ".ini"}


def _hits(rel: str, text: str) -> list[tuple[str, int, str, str]]:
    found = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for kind, pattern in PATTERNS:
            for match in pattern.finditer(line):
                found.append((rel, lineno, kind, match.group(0)))
    return found


def test_scanner_flags_names_domains_host_ips_and_home_dirs_only():
    text = "\n".join([
        "BASE = 'http://10.0.0.246:11434'",
        "# lives at /home/alice/dev",
        "NETS = ['10.0.0.0/8', '192.168.0.0/16']",
        "ROOT = '/home/bridge/dev'",
        "# mail nick@ferreri.us",
        "OTHER = '172.15.0.1'  # nickname is not a name",
    ])
    assert [(line, kind, match) for _, line, kind, match in _hits("x.py", text)] == [
        (1, "private IP", "10.0.0.246"),
        (2, "home directory", "/home/alice"),
        (5, "person's name", "nick"),
        (5, "personal domain", "ferreri.us"),
    ]


def test_src_has_no_person_or_deployment_infrastructure():
    unused = dict(ALLOWED)
    where = []
    for path in sorted(SRC.rglob("*")):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SRC).as_posix()
        for file, line, kind, match in _hits(rel, path.read_text(encoding="utf-8", errors="replace")):
            if (file, match) in unused:
                unused.pop((file, match))
            else:
                where.append(f"{file}:{line} {kind} {match!r}")
    assert where == [], "deployment-specific value in src/; read it from settings/data instead"
    assert list(unused) == [], "ALLOWED entries match nothing; delete them"


def test_compose_files_hold_no_deployment_hosts():
    # TASK-2030: shipped compose files take LAN names/addresses from .env.
    where = [f"{p.name}:{line} {kind} {match!r}"
             for p in sorted(ROOT.glob("docker-compose*.yml"))
             for _, line, kind, match in _hits(p.name, p.read_text(encoding="utf-8"))]
    assert where == [], "deployment-specific value in a compose file; interpolate it from .env"
