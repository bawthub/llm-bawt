from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from agent_bridge import crawl4ai_token

# Codex requires plugin contents to live at
# ``<codex_home>/plugins/cache/<marketplace>/<plugin>/<version>/``.
# The version subdir is mandatory but unused — Codex doesn't track
# semantic versions for local plugins, so we always write into a fixed
# token. Changing this string forces a re-stage on next start.
_PLUGIN_CACHE_VERSION = "v1"
_DEFAULT_BAWTHUB_MCP_URL = "http://app:8001/mcp"
_DEFAULT_CRAWL4AI_MCP_URL = "http://crawl4ai:11235/mcp/sse"


class CodexMcpContextConfigError(RuntimeError):
    """Trusted MCP headers could not be hidden from model shell tools."""


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _safe_symlink(target: Path, link_path: Path) -> None:
    _remove_path(link_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target)


def _resolve_skill_target(
    *,
    skill_name: str,
    dev_root: Path,
    codex_home: Path,
    source_skill_entry: Path,
) -> Path | None:
    # A package's own content is authoritative. Name-based lookup exists only
    # for legacy mapping entries (empty directories), not as an override for
    # a bundled skill or an explicitly linked source from another repository.
    if source_skill_entry.is_symlink():
        resolved = source_skill_entry.resolve(strict=False)
        if (resolved / "SKILL.md").is_file():
            return resolved
        # A broken explicit source must not silently select unrelated content.
        return None

    if (source_skill_entry / "SKILL.md").is_file():
        return source_skill_entry

    repo_skill_dir = dev_root / "agent-skills" / skill_name
    if (repo_skill_dir / "SKILL.md").exists():
        return repo_skill_dir

    system_skill_dir = codex_home / "skills" / ".system" / skill_name
    if (system_skill_dir / "SKILL.md").exists():
        return system_skill_dir

    return None


def _read_json(path: Path, *, logger: logging.Logger) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read JSON at %s: %s", path, exc)
        return None


def _populate_plugin_cache(
    *,
    plugin_cache_root: Path,
    source_plugin_dir: Path,
    skill_targets: dict[str, Path],
) -> None:
    """Write the cache-layout copy of a plugin Codex actually loads.

    Layout produced (rooted at ``plugin_cache_root``):
      - ``.codex-plugin/`` — real copy of the source's manifest dir
      - ``skills/<name>`` — symlink to each resolved skill source dir
      - any other top-level files/dirs from the source plugin (e.g.
        ``.app.json``, ``assets/``) copied as real files so Codex's
        loader can find icon/manifest references inside the cache.
    """
    if plugin_cache_root.exists():
        shutil.rmtree(plugin_cache_root)
    plugin_cache_root.mkdir(parents=True)

    # Manifest must be a real file — Codex's loader reads
    # ``<plugin>/.codex-plugin/plugin.json`` and barfs on broken symlinks.
    src_manifest_dir = source_plugin_dir / ".codex-plugin"
    if src_manifest_dir.exists():
        dst_manifest_dir = plugin_cache_root / ".codex-plugin"
        shutil.copytree(src_manifest_dir, dst_manifest_dir, symlinks=False)

    # Copy any non-skill, non-manifest siblings (icons, .app.json, etc.).
    for child in source_plugin_dir.iterdir():
        if child.name in {".codex-plugin", "skills"}:
            continue
        target = plugin_cache_root / child.name
        if child.is_dir():
            shutil.copytree(child, target, symlinks=False)
        else:
            shutil.copy2(child, target, follow_symlinks=True)

    # Skills are symlinks so live edits to the source skill repo flow
    # through without restaging.
    if skill_targets:
        skills_dir = plugin_cache_root / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        for skill_name, skill_target in skill_targets.items():
            _safe_symlink(skill_target, skills_dir / skill_name)


def _ensure_config_toml_entries(
    *,
    config_path: Path,
    marketplace_name: str,
    marketplace_source: Path,
    plugin_names: list[str],
    logger: logging.Logger,
) -> None:
    """Append marketplace + plugin-enable blocks to ``config.toml`` if missing.

    We deliberately don't round-trip TOML: ``~/.codex/config.toml`` holds
    user-authored content (project trust levels, model defaults) we
    must not reformat. We just check whether each section header is
    already present and append the ones that aren't.
    """
    try:
        existing = config_path.read_text() if config_path.exists() else ""
    except OSError as exc:
        logger.warning("Could not read %s: %s", config_path, exc)
        return

    additions: list[str] = []

    marketplace_header = f"[marketplaces.{marketplace_name}]"
    if marketplace_header not in existing:
        additions.extend(
            [
                "",
                marketplace_header,
                'source_type = "local"',
                f'source = "{marketplace_source}"',
            ]
        )

    for plugin_name in plugin_names:
        plugin_header = f'[plugins."{plugin_name}@{marketplace_name}"]'
        if plugin_header not in existing:
            additions.extend(
                [
                    "",
                    plugin_header,
                    "enabled = true",
                ]
            )

    if not additions:
        return

    new_text = existing
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"
    new_text += "\n".join(additions) + "\n"

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(new_text)
    except OSError as exc:
        logger.warning("Could not update %s: %s", config_path, exc)
        return

    logger.info(
        "Updated %s: registered marketplace=%s and enabled plugins=%s",
        config_path,
        marketplace_name,
        plugin_names,
    )


def _upsert_mcp_server_config(
    *,
    config_path: Path,
    name: str,
    url: str,
    logger: logging.Logger,
    extra_lines: list[str] | None = None,
) -> None:
    """Ensure ``[mcp_servers.<name>]`` exists with the requested URL.

    Keep this as line-oriented surgery rather than TOML round-tripping so
    operator-authored Codex settings stay formatted exactly as written.

    Args:
        extra_lines: Additional TOML lines to include after the url line
            (e.g. ``['headers = { Authorization = "Bearer tok" }']``).
            Existing sections are updated by key without duplicating entries.
    """
    try:
        existing = config_path.read_text() if config_path.exists() else ""
    except OSError as exc:
        logger.warning("Could not read %s: %s", config_path, exc)
        return

    header = f"[mcp_servers.{name}]"
    url_line = f'url = "{url}"'

    if header not in existing:
        new_text = existing
        if new_text and not new_text.endswith("\n"):
            new_text += "\n"
        block = f"\n{header}\n{url_line}\n"
        if extra_lines:
            block += "\n".join(extra_lines) + "\n"
        new_text += block
    else:
        lines = existing.splitlines()
        start = next(
            (idx for idx, line in enumerate(lines) if line.strip() == header), None
        )
        if start is None:
            return

        end = len(lines)
        for idx in range(start + 1, len(lines)):
            stripped = lines[idx].strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                end = idx
                break

        # A server is either URL-based or stdio-based. Older bridge releases
        # registered Crawl4AI through ``mcp-remote`` (command/args/env); if we
        # merely add ``url`` to that section, Codex still classifies it as
        # stdio and rejects the mixed config before starting a turn.
        stdio_keys = {"command", "args", "cwd", "env"}
        section_body = [
            line
            for line in lines[start + 1 : end]
            if not (
                "=" in line
                and line.split("=", 1)[0].strip() in stdio_keys
            )
        ]
        lines[start + 1 : end] = section_body
        end = start + 1 + len(section_body)

        url_idx = next(
            (
                idx
                for idx in range(start + 1, end)
                if "=" in lines[idx]
                and lines[idx].split("=", 1)[0].strip() == "url"
            ),
            None,
        )
        if url_idx is not None:
            lines[url_idx] = url_line
        else:
            lines.insert(start + 1, url_line)
            end += 1

        for extra_line in extra_lines or []:
            extra_key = extra_line.split("=", 1)[0].strip()
            extra_idx = next(
                (
                    idx
                    for idx in range(start + 1, end)
                    if "=" in lines[idx]
                    and lines[idx].split("=", 1)[0].strip() == extra_key
                ),
                None,
            )
            if extra_idx is None:
                lines.insert(end, extra_line)
                end += 1
            else:
                lines[extra_idx] = extra_line

        new_text = "\n".join(lines)
        if existing.endswith("\n") or not new_text:
            new_text += "\n"

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(new_text)
    except OSError as exc:
        logger.warning("Could not update %s: %s", config_path, exc)
        return

    logger.info("Updated %s: MCP server %s -> %s", config_path, name, url)


def ensure_bawthub_mcp_config(
    *,
    logger: logging.Logger,
    codex_home: Path,
) -> None:
    url = os.getenv("CODEX_BAWTHUB_MCP_URL", _DEFAULT_BAWTHUB_MCP_URL)
    _upsert_mcp_server_config(
        config_path=codex_home / "config.toml",
        name="bawthub",
        url=url,
        logger=logger,
        extra_lines=[
            "env_http_headers = { "
            '"X-LLM-Bawt-Task-Turn-Context" = "LLM_BAWT_TASK_TURN_CONTEXT", '
            '"X-LLM-Bawt-MCP-Request-Context" = "LLM_BAWT_MCP_REQUEST_CONTEXT" '
            "}",
        ],
    )
    _ensure_mcp_context_shell_filter(
        config_path=codex_home / "config.toml",
        logger=logger,
    )


def _ensure_mcp_context_shell_filter(
    *, config_path: Path, logger: logging.Logger
) -> None:
    """Keep trusted MCP correlation out of model-invoked shell processes."""
    try:
        import tomlkit

        existing = config_path.read_text() if config_path.exists() else ""
        document = tomlkit.parse(existing)
        policy = document.get("shell_environment_policy")
        if policy is None:
            policy = tomlkit.table()
            document["shell_environment_policy"] = policy

        names = ["LLM_BAWT_TASK_TURN_CONTEXT", "LLM_BAWT_MCP_REQUEST_CONTEXT"]
        if "exclude" in policy or "include_only" in policy:
            excluded = list(policy.get("exclude") or [])
            for name in names:
                if name not in excluded:
                    excluded.append(name)
            policy["exclude"] = excluded
        else:
            filters = policy.get("filters")
            if filters is None:
                filters = tomlkit.inline_table()
                policy["filters"] = filters
            for name in names:
                filters[name] = "exclude"

        new_text = tomlkit.dumps(document)
        if new_text == existing:
            return
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(new_text)
    except Exception as exc:  # noqa: BLE001 - security boundary must fail closed
        logger.error(
            "Could not exclude trusted MCP context from Codex shell environment in %s: %s",
            config_path,
            exc,
        )
        raise CodexMcpContextConfigError(
            "trusted MCP context could not be excluded from shell tools"
        ) from exc


def ensure_crawl4ai_mcp_config(
    *,
    logger: logging.Logger,
    codex_home: Path,
) -> None:
    """Register the Crawl4AI native MCP server in Codex config."""
    url = os.getenv("CODEX_CRAWL4AI_MCP_URL", _DEFAULT_CRAWL4AI_MCP_URL)
    # Derived token file first, legacy env second (TASK-1476).
    token = crawl4ai_token.api_token()
    extra: list[str] = []
    if token:
        extra.append(f'headers = {{ Authorization = "Bearer {token}" }}')
    _upsert_mcp_server_config(
        config_path=codex_home / "config.toml",
        name="crawl4ai",
        url=url,
        logger=logger,
        extra_lines=extra or None,
    )


def install_repo_local_plugins(
    *,
    logger: logging.Logger,
    codex_home: Path,
    home: Path | None = None,
) -> None:
    """Stage repo-managed local plugins so Codex's loader can find them.

    Codex 0.128 won't load a "local" marketplace just because
    ``~/.agents/plugins/marketplace.json`` exists. The full setup is:

      1. ``[marketplaces.<name>]`` registered in ``~/.codex/config.toml``
         with ``source_type = "local"`` and a ``source`` path that has
         ``.agents/plugins/marketplace.json`` underneath.
      2. ``[plugins."<plugin>@<marketplace>"] enabled = true`` for each
         plugin we want active.
      3. Plugin contents installed at
         ``~/.codex/plugins/cache/<marketplace>/<plugin>/<version>/`` —
         the ``<version>`` subdir is required even though we don't track
         versions. ``.codex-plugin/plugin.json`` must be a real file.
         ``skills/<name>/`` may be symlinks into the source repo.

    The ``INSTALLED_BY_DEFAULT`` policy in marketplace.json is a no-op
    for local marketplaces in this Codex version — registering the
    marketplace alone does not populate the cache, so we do it here.

    The bridge runs from ``/home/bridge/dev``, so we stage from
    ``~/dev/agent-skills/codex`` (override via ``CODEX_LOCAL_PLUGINS_SRC``)
    and produce both the home-local layout (so ``codex plugin marketplace
    add`` and other tooling still work) and the cache-local layout (so
    the runtime loader actually finds the plugins).
    """

    if os.getenv("CODEX_LOCAL_PLUGINS_ENABLED", "1").lower() in {"0", "false", "no"}:
        logger.info(
            "Skipping local Codex plugin install: CODEX_LOCAL_PLUGINS_ENABLED=false"
        )
        ensure_bawthub_mcp_config(logger=logger, codex_home=codex_home)
        ensure_crawl4ai_mcp_config(logger=logger, codex_home=codex_home)
        return

    home = home or Path.home()
    dev_root = Path(os.getenv("CODEX_DEV_ROOT", "/home/bridge/dev"))
    source_root = Path(
        os.getenv("CODEX_LOCAL_PLUGINS_SRC", str(dev_root / "agent-skills" / "codex"))
    )
    source_marketplace = source_root / ".codex-plugin" / "marketplace.json"
    source_plugins_dir = source_root / "plugins"

    if not source_marketplace.exists() or not source_plugins_dir.is_dir():
        logger.info(
            "No repo-managed Codex plugin mapping found at %s; local plugin staging skipped",
            source_root,
        )
        ensure_bawthub_mcp_config(logger=logger, codex_home=codex_home)
        ensure_crawl4ai_mcp_config(logger=logger, codex_home=codex_home)
        return

    marketplace_data = _read_json(source_marketplace, logger=logger) or {}
    marketplace_name = marketplace_data.get("name")
    if not marketplace_name:
        logger.warning(
            "marketplace.json at %s is missing a 'name' field; cannot register marketplace",
            source_marketplace,
        )
        ensure_bawthub_mcp_config(logger=logger, codex_home=codex_home)
        ensure_crawl4ai_mcp_config(logger=logger, codex_home=codex_home)
        return

    # Home-local layout: a marketplace.json symlink + per-plugin scaffold
    # under ``~/plugins/<name>``. This is the path Codex CLI tools (e.g.
    # ``codex plugin marketplace add /home/bridge``) walk when validating
    # the marketplace `source`. Keep it intact so out-of-band ops still
    # see a coherent marketplace layout.
    agents_plugins_dir = home / ".agents" / "plugins"
    home_plugins_dir = home / "plugins"
    agents_plugins_dir.mkdir(parents=True, exist_ok=True)
    home_plugins_dir.mkdir(parents=True, exist_ok=True)

    _safe_symlink(source_marketplace, agents_plugins_dir / "marketplace.json")

    # Cache-local layout: the path Codex's runtime loader actually reads.
    cache_marketplace_root = codex_home / "plugins" / "cache" / marketplace_name
    cache_marketplace_root.mkdir(parents=True, exist_ok=True)

    installed_plugins: list[str] = []
    for source_plugin_dir in sorted(
        p for p in source_plugins_dir.iterdir() if p.is_dir()
    ):
        # ---- home-local scaffold (kept for tooling compatibility) ----
        target_plugin_dir = home_plugins_dir / source_plugin_dir.name
        _remove_path(target_plugin_dir)
        target_plugin_dir.mkdir(parents=True, exist_ok=True)

        for child in source_plugin_dir.iterdir():
            if child.name == "skills":
                continue
            _safe_symlink(child, target_plugin_dir / child.name)

        # Resolve skills once — used for both layouts.
        skill_targets: dict[str, Path] = {}
        source_skills_dir = source_plugin_dir / "skills"
        if source_skills_dir.is_dir():
            target_skills_dir = target_plugin_dir / "skills"
            target_skills_dir.mkdir(parents=True, exist_ok=True)
            for source_skill_entry in sorted(source_skills_dir.iterdir()):
                target = _resolve_skill_target(
                    skill_name=source_skill_entry.name,
                    dev_root=dev_root,
                    codex_home=codex_home,
                    source_skill_entry=source_skill_entry,
                )
                if target is None:
                    logger.warning(
                        "Skipping unresolved local skill '%s' from %s",
                        source_skill_entry.name,
                        source_skill_entry,
                    )
                    continue
                _safe_symlink(target, target_skills_dir / source_skill_entry.name)
                skill_targets[source_skill_entry.name] = target

        # ---- cache-local layout (what Codex actually loads) ----
        plugin_manifest = (
            _read_json(
                source_plugin_dir / ".codex-plugin" / "plugin.json", logger=logger
            )
            or {}
        )
        plugin_name = plugin_manifest.get("name") or source_plugin_dir.name
        plugin_cache_root = cache_marketplace_root / plugin_name / _PLUGIN_CACHE_VERSION
        _populate_plugin_cache(
            plugin_cache_root=plugin_cache_root,
            source_plugin_dir=source_plugin_dir,
            skill_targets=skill_targets,
        )

        installed_plugins.append(plugin_name)

    if not installed_plugins:
        logger.info(
            "No plugins discovered under %s; nothing to register in config.toml",
            source_plugins_dir,
        )
        ensure_bawthub_mcp_config(logger=logger, codex_home=codex_home)
        ensure_crawl4ai_mcp_config(logger=logger, codex_home=codex_home)
        return

    _ensure_config_toml_entries(
        config_path=codex_home / "config.toml",
        marketplace_name=marketplace_name,
        marketplace_source=home,
        plugin_names=installed_plugins,
        logger=logger,
    )

    logger.info(
        "Staged repo-managed Codex plugins into %s and %s for marketplace=%s: %s",
        home,
        cache_marketplace_root,
        marketplace_name,
        ", ".join(installed_plugins),
    )

    ensure_bawthub_mcp_config(logger=logger, codex_home=codex_home)
    ensure_crawl4ai_mcp_config(logger=logger, codex_home=codex_home)
