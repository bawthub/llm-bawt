"""No new credential may be handed to a container through env (TASK-1476).

The only secrets outside the database are the bootstrap root
(``LLM_BAWT_SECRET_KEY``) and the database password. Every other credential
is connected in Settings and stored encrypted. This guard scans the tracked
compose files and env templates for secret-shaped names and fails on any name
that is neither bootstrap nor listed in :data:`PENDING_RETIREMENT`.

It is a ratchet. :data:`PENDING_RETIREMENT` lists names that still ship today,
each with the reason it remains. Removing one from the files makes the stale
check fail until its entry is deleted too, so the list can only shrink. Never
add to it to get a new name past the guard.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

#: Secret-shaped: a name segment that holds a credential.
_SECRET_NAME = re.compile(r"(^|_)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(_|$)")
#: About a secret, never the secret: a file or path, a uid, an env-var name, a mode.
_POINTER_SUFFIXES = ("_FILE", "_PATH", "_UID", "_ENV", "_MODE")
#: git's ``GIT_CONFIG_KEY_<n>`` holds a config key name (its value is ``GIT_CONFIG_VALUE_<n>``).
_GIT_CONFIG_NAME = re.compile(r"^GIT_CONFIG_KEY_\d+$")
_INTERPOLATION = re.compile(r"\$\{([A-Z][A-Z0-9_]*)")
_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=")

#: The bootstrap secrets that must stay in env.
BOOTSTRAP = frozenset({
    "LLM_BAWT_SECRET_KEY",
    "POSTGRES_PASSWORD",
    "LLM_BAWT_POSTGRES_PASSWORD",
    "PGPASSWORD",
})

#: Credentials that still ship through env, each with why. Shrink only.
PENDING_RETIREMENT: dict[str, str] = {
    # Readers are already DB-first. The env entry goes when the live record is
    # verified and the container is recreated (TASK-1518 window).
    "XAI_API_KEY": "TASK-1476: xai connection is DB-first; env is the legacy fallback",
    "LLM_BAWT_XAI_API_KEY": "TASK-1476: xai connection is DB-first; env is the legacy fallback",
    "OPENAI_API_KEY": "TASK-1476: openai-api connection is DB-first; env is the legacy fallback",
    "ANTHROPIC_API_KEY": "TASK-1476: anthropic-api connection is DB-first; env is the legacy fallback",
    "CLAUDE_CODE_OAUTH_TOKEN": "TASK-1476: the claude connection feeds the bridge; env is a dead fallback",
    "ZAI_API_KEY": "TASK-1476: zai connection is record-first; env applies only with no record",
    "KIMI_CODING_API_KEY": "TASK-1476: kimi-coding connection is record-first; env applies only with no record",
    "LLM_BAWT_HA_NATIVE_MCP_TOKEN": "TASK-1476: home-assistant connection is record-first",
    "LLM_BAWT_HA_MCP_AUTH_TOKEN": "TASK-1476: reserved HA MCP auth; folds into home-assistant",
    "LLM_BAWT_S3_ACCESS_KEY": "TASK-1476: object-store connection is record-first",
    "LLM_BAWT_S3_SECRET_KEY": "TASK-1476: object-store connection is record-first",
    # Still env-only; each needs a DB home first.
    "OPENCLAW_GATEWAY_TOKEN": "TASK-1476: openclaw-bridge reads it at boot; needs a broker read",
    "CRAWL4AI_API_TOKEN": "TASK-1476: the crawl4ai container needs it at boot",
    "LLM_BAWT_REDDIT_CLIENT_SECRET": "TASK-1476: Reddit search credentials have no connection yet",
    "LLM_BAWT_TALK_PROVISIONER_TOKEN": "TASK-1476: talk provisioner token has no connection yet",
    "HF_TOKEN": "TASK-1476: model downloads on GPU containers; not an app credential",
    "LOCAL_ANTHROPIC_API_KEY": "TASK-1476: optional placeholder for a LAN local-model passthrough",
}


def _tracked(pattern: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--", pattern], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if line]


def _compose_files() -> list[Path]:
    return _tracked("docker-compose*.yml")


def _env_templates() -> list[Path]:
    return [p for p in _tracked(".env*") if p.name != ".env"]


def _compose_names(path: Path) -> set[str]:
    text = path.read_text()
    names = set(_INTERPOLATION.findall(text))
    doc = yaml.safe_load(text) or {}
    for service in (doc.get("services") or {}).values():
        env = (service or {}).get("environment") or {}
        entries = env.keys() if isinstance(env, dict) else (str(e).split("=", 1)[0] for e in env)
        names.update(name.strip() for name in entries)
    return names


def _template_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text().splitlines():
        match = _ASSIGNMENT.match(line)  # commented lines are documentation
        if match:
            names.add(match.group(1))
    return names


def is_secret_shaped(name: str) -> bool:
    return (bool(_SECRET_NAME.search(name)) and not name.endswith(_POINTER_SUFFIXES)
            and not _GIT_CONFIG_NAME.match(name))


def _shipped_secret_names() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in _compose_files():
        for name in _compose_names(path):
            found.setdefault(name, set()).add(path.name)
    for path in _env_templates():
        for name in _template_names(path):
            found.setdefault(name, set()).add(path.name)
    return {name: files for name, files in found.items() if is_secret_shaped(name)}


def test_the_guard_scans_the_shipped_files():
    assert {p.name for p in _compose_files()} >= {"docker-compose.yml", "docker-compose.prod.yml"}
    assert {p.name for p in _env_templates()} >= {".env.docker", ".env.tenant.example"}


@pytest.mark.parametrize(("name", "shaped"), [
    ("LLM_BAWT_BRAVE_API_KEY", True),
    ("OPENCLAW_GATEWAY_TOKEN", True),
    ("LLM_BAWT_S3_SECRET_KEY", True),
    ("REDDIT_CLIENT_SECRET", True),
    ("POSTGRES_PASSWORD", True),
    ("BAWTHUB_SERVICE_TOKEN_FILE", False),
    ("BAWTHUB_SERVICE_TOKEN_EXPORT_UID", False),
    ("OPENCLAW_GATEWAY_TOKEN_ENV", False),
    ("CLAUDE_USAGE_CREDENTIALS_PATH", False),
    ("CLAUDE_USAGE_CREDENTIALS_MODE", False),
    ("GIT_CONFIG_KEY_0", False),
    ("GIT_CONFIG_KEY_TOKEN", True),
    ("ENABLE_TOOL_SEARCH", False),
    ("LLM_BAWT_SEARCH_PROVIDER", False),
    ("KEYBOARD_LAYOUT", False),
])
def test_secret_shape(name, shaped):
    assert is_secret_shaped(name) is shaped


def test_no_new_credential_ships_through_env():
    unexpected = {
        name: sorted(files) for name, files in _shipped_secret_names().items()
        if name not in BOOTSTRAP and name not in PENDING_RETIREMENT
    }
    assert not unexpected, (
        f"Credential-shaped env names outside the bootstrap set: {unexpected}. "
        "Store the credential as an encrypted connection (Settings) and read it "
        "record-first instead of adding it to compose or an env template."
    )


def test_the_retirement_list_only_shrinks():
    shipped = _shipped_secret_names()
    stale = sorted(name for name in PENDING_RETIREMENT if name not in shipped)
    assert not stale, f"No longer shipped; delete from PENDING_RETIREMENT: {stale}"
    assert not BOOTSTRAP & PENDING_RETIREMENT.keys()
