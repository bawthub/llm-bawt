"""Bot editor workflows for bot profiles and global runtime settings."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

# Custom representer so multiline strings use block scalar style (|)
def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

_BlockDumper = type("_BlockDumper", (yaml.SafeDumper,), {})
_BlockDumper.add_representer(str, _str_representer)
from rich.prompt import Prompt  # noqa: E402

from llm_bawt.utils.config import Config, RuntimeTunables  # noqa: E402

console = Console()


def _get_service_client(config: Config):
    """Build a service client for the bot editor."""
    from llm_bawt.service.client import get_service_client
    return get_service_client(config)


def _notify_service_reload(config: Config) -> None:
    """Tell the running service to reload bots. Best-effort, silent on failure."""
    import urllib.request
    import urllib.error

    host = getattr(config, "SERVICE_HOST", "127.0.0.1")
    port = int(getattr(config, "SERVICE_PORT", 8642))
    url = f"http://{host}:{port}/v1/admin/reload-bots"
    try:
        req = urllib.request.Request(url, method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=3):
            console.print("[dim]Service notified — bot config reloaded.[/dim]")
    except (urllib.error.URLError, OSError):
        pass  # Service not running, that's fine


def _build_settings_yaml(settings: dict) -> str:
    """Render the settings dict as YAML with help comments above each key."""
    lines = ["settings:"]
    for key, value in settings.items():
        # Look up help from RuntimeTunables field descriptions
        config_field = key.upper()
        field_info = RuntimeTunables.model_fields.get(config_field)
        if field_info and field_info.description:
            lines.append(f"  # {field_info.description}")
        # Let PyYAML format the value correctly
        rendered = yaml.safe_dump({key: value}, default_flow_style=False).strip()
        lines.append(f"  {rendered}")
    lines.append("")
    return "\n".join(lines)


def _runtime_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for field_name, field_info in RuntimeTunables.model_fields.items():
        key = field_name.lower()
        defaults[key] = field_info.get_default(call_default_factory=True)
    return defaults


def _open_yaml_editor(initial_text: str, suffix: str) -> dict[str, Any] | None:
    with tempfile.NamedTemporaryFile(
        mode="w+",
        suffix=suffix,
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp_path = tmp.name
        tmp.write(initial_text)

    editor = os.environ.get("EDITOR", "vi")
    editor_cmd = [editor]
    editor_base = os.path.basename(editor).lower()
    if editor_base in ("code", "cursor", "code-insiders"):
        editor_cmd.append("--wait")

    try:
        subprocess.run([*editor_cmd, tmp_path], check=True)
        parsed = yaml.safe_load(Path(tmp_path).read_text(encoding="utf-8")) or {}
        if not isinstance(parsed, dict):
            return None
        return parsed
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def edit_bot_yaml(config: Config, bot_slug: str) -> bool:
    """Edit one bot in a temp YAML file, then save personality/settings to DB."""
    from llm_bawt.bots import (
        get_bot,
        get_bot_settings_template,
        get_raw_bot_data,
    )

    slug = (bot_slug or "").strip().lower()
    if not slug:
        console.print("[red]Bot slug is required.[/red]")
        return False

    bot = get_bot(slug)
    raw = get_raw_bot_data(slug)
    if not bot or raw is None:
        console.print(f"[red]Unknown bot: {slug}[/red]")
        return False

    template = get_bot_settings_template()

    client = _get_service_client(config)
    if not client or not client.is_available():
        console.print("[red]Service not available. Start with: llm-service[/red]")
        return False

    resp = client.get_settings(scope_type="bot", scope_id=slug)
    db_bot_settings: dict[str, Any] = {}
    if resp:
        db_bot_settings = {item["key"]: item["value"] for item in resp.get("settings", [])}

    # Build settings: template < YAML < DB (DB wins)
    effective_settings = {**template, **(raw.get("settings") or {}), **db_bot_settings}

    # Build editable dict with only non-empty values
    editable_fields = {
        "slug": slug,
        "name": bot.name,
        "color": bot.color,
        "description": bot.description,
        "system_prompt": bot.system_prompt,
        "requires_memory": bot.requires_memory,
        "voice_optimized": bot.voice_optimized,
        "tts_mode": bot.tts_mode,
        "include_summaries": bot.include_summaries,
        "include_in_global_search": bot.include_in_global_search,
        "uses_tools": bot.uses_tools,
        "uses_search": bot.uses_search,
        "uses_home_assistant": bot.uses_home_assistant,
    }

    # Only include optional fields if they have values
    if bot.default_model:
        editable_fields["default_model"] = bot.default_model

    nextcloud_config = raw.get("nextcloud")
    if nextcloud_config:
        editable_fields["nextcloud"] = nextcloud_config

    yaml_text = yaml.dump(editable_fields, Dumper=_BlockDumper, sort_keys=False, allow_unicode=True)
    yaml_text += _build_settings_yaml(effective_settings)

    try:
        parsed = _open_yaml_editor(yaml_text, suffix=f".bot-{slug}.yaml")
    except Exception as e:
        console.print(f"[red]Edit failed:[/red] {e}")
        return False

    if parsed is None:
        console.print("[red]Invalid YAML format (expected mapping).[/red]")
        return False
    if parsed.get("slug", slug) != slug:
        console.print("[red]Slug mismatch in edited file; slug cannot be changed here.[/red]")
        return False
    required_str = ("name", "description", "system_prompt")
    for key in required_str:
        if not isinstance(parsed.get(key), str):
            console.print(f"[red]Invalid '{key}': expected string.[/red]")
            return False
    if "color" in parsed and parsed.get("color") is not None and not isinstance(parsed.get("color"), str):
        console.print("[red]Invalid 'color': expected string or null.[/red]")
        return False
    required_bool = (
        "requires_memory",
        "voice_optimized",
        "tts_mode",
        "include_summaries",
        "include_in_global_search",
        "uses_tools",
        "uses_search",
        "uses_home_assistant",
    )
    for key in required_bool:
        if key in parsed and not isinstance(parsed.get(key), bool):
            console.print(f"[red]Invalid '{key}': expected boolean.[/red]")
            return False
    if "settings" in parsed and not isinstance(parsed.get("settings"), dict):
        console.print("[red]Invalid 'settings': expected mapping.[/red]")
        return False

    # Split settings from personality — both persist to DB
    new_settings = parsed.pop("settings", {}) or {}
    out = parsed.copy()
    out.pop("slug", None)

    edited_color = out.pop("color", None)

    if out == editable_fields and new_settings == effective_settings:
        console.print("[dim]No changes detected.[/dim]")
        return True

    if sys.stdin.isatty():
        confirm = Prompt.ask("Apply these bot config changes?", choices=["y", "n"], default="y")
        if confirm != "y":
            console.print("[yellow]Cancelled.[/yellow]")
            return False

    profile_payload = {
        "name": out.get("name", bot.name),
        "description": out.get("description", bot.description),
        "system_prompt": out.get("system_prompt", bot.system_prompt),
        "requires_memory": out.get("requires_memory", bot.requires_memory),
        "voice_optimized": out.get("voice_optimized", bot.voice_optimized),
        "tts_mode": out.get("tts_mode", bot.tts_mode),
        "include_summaries": out.get("include_summaries", bot.include_summaries),
        "include_in_global_search": out.get("include_in_global_search", bot.include_in_global_search),
        "uses_tools": out.get("uses_tools", bot.uses_tools),
        "uses_search": out.get("uses_search", bot.uses_search),
        "uses_home_assistant": out.get("uses_home_assistant", bot.uses_home_assistant),
        "default_model": out.get("default_model", bot.default_model),
        "color": edited_color,
        "default_voice": out.get("default_voice"),
        "nextcloud_config": out.get("nextcloud", raw.get("nextcloud")),
        "bot_type": raw.get("bot_type", bot.bot_type),
        "agent_backend": bot.agent_backend,
        "agent_backend_config": raw.get("agent_backend_config", bot.agent_backend_config),
    }

    client = _get_service_client(config)
    if client and client.is_available():
        client.upsert_bot_profile(slug, profile_payload)
        console.print(f"[green]Personality saved:[/green] {slug} -> service")

        settings_changed = 0
        for key, value in new_settings.items():
            old_val = db_bot_settings.get(key)
            template_val = template.get(key)
            if old_val != value or (old_val is None and value != template_val):
                client.set_setting("bot", slug, key, value)
                settings_changed += 1
        for key in db_bot_settings:
            if key not in new_settings:
                client.delete_setting("bot", key, scope_id=slug)
                settings_changed += 1
        if settings_changed:
            console.print(f"[green]Settings saved:[/green] {settings_changed} key(s) -> service (bot/{slug})")
        else:
            console.print("[dim]Settings unchanged.[/dim]")

    from llm_bawt.bots import invalidate_bots_cache

    invalidate_bots_cache()
    _notify_service_reload(config)

    return True


def edit_global_settings(config: Config) -> bool:
    """Edit global runtime settings in DB (scope=global/*)."""
    defaults = _runtime_defaults()
    current_db: dict[str, Any] = {}

    client = _get_service_client(config)
    if not client or not client.is_available():
        console.print("[red]Service not available. Start with: llm-service[/red]")
        return False

    resp = client.get_settings(scope_type="global")
    if resp:
        current_db = {item["key"]: item["value"] for item in resp.get("settings", [])}

    effective = {**defaults, **current_db}

    header = "# Edit global runtime settings.\n# Values are saved to runtime_settings scope global/*.\n"
    yaml_text = header + _build_settings_yaml(effective)

    try:
        parsed = _open_yaml_editor(yaml_text, suffix=".global-settings.yaml")
    except Exception as e:
        console.print(f"[red]Edit failed:[/red] {e}")
        return False

    if parsed is None:
        console.print("[red]Invalid YAML format (expected mapping).[/red]")
        return False

    new_settings = parsed.get("settings")
    if not isinstance(new_settings, dict):
        console.print("[red]Invalid 'settings': expected mapping.[/red]")
        return False

    if new_settings == effective:
        console.print("[dim]No changes detected.[/dim]")
        return True

    if sys.stdin.isatty():
        confirm = Prompt.ask("Apply these global settings changes?", choices=["y", "n"], default="y")
        if confirm != "y":
            console.print("[yellow]Cancelled.[/yellow]")
            return False

    changed = 0

    for key, value in new_settings.items():
        old_val = current_db.get(key)
        default_val = defaults.get(key)
        if old_val == value:
            continue
        if old_val is None and value == default_val:
            continue
        client.set_setting("global", None, key, value)
        changed += 1

    for key in current_db:
        if key not in new_settings:
            client.delete_setting("global", key)
            changed += 1

    if changed:
        console.print(f"[green]Global settings saved:[/green] {changed} key(s) -> service (global/*)")
    else:
        console.print("[dim]Global settings unchanged.[/dim]")

    return True
