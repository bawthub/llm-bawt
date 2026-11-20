"""Where skill sources live and how private ones authenticate (TASK-2534).

``SKILL_SOURCES_ROOT`` is a writable directory outside the containers (a named
volume on tenants). Its ``current`` entry is the effective skill tree; set
``SKILL_SOURCES_CURRENT`` to it so the catalog and the bridges read the same
files, falling back to ``CLAUDE_CODE_SKILLS_PATH`` while it does not resolve. ``SKILL_SOURCES_FALLBACK`` (default ``/opt/bawthub/skills``, the image
bundle) fills in any skill no configured source provides and seeds a fresh
volume. Without ``SKILL_SOURCES_ROOT`` the feature reports 503 and skills keep
coming from ``CLAUDE_CODE_SKILLS_PATH`` as before.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .source_store import GitSpec, SourceStore
from .sources import SkillSourceService

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK = "/opt/bawthub/skills"
# Where the product exporter's archive keeps the skills (bawthub/skills/<name>).
DEFAULT_SUBDIR = "bawthub/skills"


def store_from_env() -> SourceStore:
    root = os.environ.get("SKILL_SOURCES_ROOT", "")
    fallback = os.environ.get("SKILL_SOURCES_FALLBACK", DEFAULT_FALLBACK)
    paths = [Path(p) for p in fallback.split(os.pathsep) if p]
    # Unset = disabled: available() is False and nothing is written.
    return SourceStore(Path(root) if root.strip() else None, fallback=paths)


def _github_token(config):
    def token_for(spec: GitSpec) -> str | None:
        from ..service.providers.github import GitHubAdapter, GitHubNotConnectedError
        from ..service.providers.registry import get_adapter

        adapter = get_adapter(config, "github")
        if not isinstance(adapter, GitHubAdapter):
            return None
        path = spec.url.removeprefix("https://github.com/").removesuffix(".git").strip("/")
        try:
            return adapter.git_credential_token(repository=path)
        except GitHubNotConnectedError:
            return None
        except Exception:  # noqa: BLE001 - never log upstream bodies or credentials
            logger.warning("skill source credential mint failed")
            return None

    return token_for


def install_default_source(service: SkillSourceService) -> None:
    """Install the built-in archive source named by ``SKILL_SOURCES_DEFAULT_URL``.

    The URL is a public manifest of reviewed BawtHub skill releases, signed by
    a key in the app's ``publisher_keys.txt``; tenants need no repository
    access. Unset means no built-in source (the image bundle still applies). Failures are logged and retried at the next start
    or by an owner from Bots → Skills.
    """
    url = os.environ.get("SKILL_SOURCES_DEFAULT_URL", "").strip()
    if not url:
        return
    subdir = os.environ.get("SKILL_SOURCES_DEFAULT_SUBDIR", DEFAULT_SUBDIR)
    try:
        if service.ensure_default(url, subdir=subdir):
            logger.info("Built-in skill source installed")
    except Exception as exc:  # noqa: BLE001 - never log upstream bodies
        logger.warning("Built-in skill source not installed (%s)", getattr(exc, "code", type(exc).__name__))


def default_keys() -> tuple[str, ...]:
    """The built-in source's trust root: ``publisher_keys.txt`` shipped in the app.

    Deliberately not configurable from the environment, so a deployment
    setting cannot swap the publisher. Without a key the built-in source is
    never installed.
    """
    try:
        raw = (Path(__file__).with_name("publisher_keys.txt")).read_text()
    except OSError:
        return ()
    return tuple(line.strip() for line in raw.splitlines() if line.strip() and not line.startswith("#"))


def source_service_for(engine, config) -> SkillSourceService:
    return SkillSourceService(engine, store_from_env(), _github_token(config), default_keys())
