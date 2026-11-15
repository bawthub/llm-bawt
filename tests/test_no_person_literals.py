"""TASK-1383: product source names no person and bakes in no deployment infrastructure.

llm-bawt is built for general use, so ``src/`` (code, comments, docstrings and
bundled text files alike) holds no user's name, personal domain, private host
address or person's home directory. The deployment supplies those as data:
``LLM_BAWT_DEFAULT_USER``, runtime settings, the prompt registry and ``.env``.
Detector specimens use generic values. ``ALLOWED`` pins any reasoned exception by file
and exact match; it is empty, and each entry added needs its reason.
"""

from __future__ import annotations

import re
from pathlib import Path

from scripts.general_use_contract import load_policy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _patterns(names, domains) -> tuple[tuple[str, re.Pattern[str]], ...]:
    return (
        *(("person's name", re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE)) for n in names),
        *(("personal domain", re.compile(rf"\b{re.escape(d)}\b", re.IGNORECASE)) for d in domains),
        # A private-network host address. Network ranges (10.0.0.0/8) are generic and allowed.
        ("private IP", re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b(?!/\d)")),
        # A person's home directory. /home/bridge is the agent bridge containers' own user.
        ("home directory", re.compile(r"/home/(?!bridge\b|<)[a-z_][a-z0-9_-]*")),
    )


def _deployment_patterns():
    # The same untracked policy feeds the general-use gate; never copy its
    # installation-specific values into a test or silently skip a missing policy.
    policy = load_policy(ROOT)
    return _patterns(policy["operator_identities"], policy["personal_domains"])

# (path under src/, exact match) -> reason.
ALLOWED: dict[tuple[str, str], str] = {}

_TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".sh", ".j2", ".html", ".sql", ".cfg", ".ini"}


def _hits(rel: str, text: str, patterns) -> list[tuple[str, int, str, str]]:
    found = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for kind, pattern in patterns:
            for match in pattern.finditer(line):
                found.append((rel, lineno, kind, match.group(0)))
    return found


def test_scanner_flags_names_domains_host_ips_and_home_dirs_only():
    text = "\n".join([
        "BASE = 'http://10.0.0.246:11434'",
        "# lives at /home/alice/dev",
        "NETS = ['10.0.0.0/8', '192.168.0.0/16']",
        "ROOT = '/home/bridge/dev'",
        "# mail bob@example.com",
        "OTHER = '172.15.0.1'  # bobcat is not a name",
    ])
    patterns = _patterns(("Bob",), ("example.com", "example.org"))
    assert [(line, kind, match) for _, line, kind, match in _hits("x.py", text, patterns)] == [
        (1, "private IP", "10.0.0.246"),
        (2, "home directory", "/home/alice"),
        (5, "person's name", "bob"),
        (5, "personal domain", "example.com"),
    ]


def test_deployment_patterns_use_the_external_policy(monkeypatch):
    def policy(root):
        assert root == ROOT
        return {"operator_identities": ["Fixture Person"], "personal_domains": ["fixture.example"]}

    monkeypatch.setattr("tests.test_no_person_literals.load_policy", policy)
    hits = _hits("x.py", "fixture person at mail.fixture.example", _deployment_patterns())
    assert [(kind, match) for _, _, kind, match in hits] == [
        ("person's name", "fixture person"), ("personal domain", "fixture.example"),
    ]


def test_src_has_no_person_or_deployment_infrastructure():
    patterns = _deployment_patterns()
    unused = dict(ALLOWED)
    where = []
    for path in sorted(SRC.rglob("*")):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SRC).as_posix()
        for file, line, kind, match in _hits(rel, path.read_text(encoding="utf-8", errors="replace"), patterns):
            if (file, match) in unused:
                unused.pop((file, match))
            else:
                where.append(f"{file}:{line} {kind} {match!r}")
    assert where == [], "deployment-specific value in src/; read it from settings/data instead"
    assert list(unused) == [], "ALLOWED entries match nothing; delete them"


def test_compose_files_hold_no_deployment_hosts():
    # TASK-2030: shipped compose files take LAN names/addresses from .env.
    patterns = _deployment_patterns()
    where = [f"{p.name}:{line} {kind} {match!r}"
             for p in sorted(ROOT.glob("docker-compose*.yml"))
             for _, line, kind, match in _hits(p.name, p.read_text(encoding="utf-8"), patterns)]
    assert where == [], "deployment-specific value in a compose file; interpolate it from .env"
