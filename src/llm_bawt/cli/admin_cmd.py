"""CLI admin command handlers (TASK-554).

Extracted verbatim from ``cli/app.py``: user profile display/setup, runtime
settings CRUD + bootstrap, and the bots->DB migration. ``app`` re-imports these
names so the ``cli.__init__`` facade (``show_users``, ``show_user_profile``,
``run_user_profile_setup``, ``ensure_user_profile``) is unchanged.
"""

import argparse
import json
import logging
import sys

from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from llm_bawt.utils.config import Config, set_config_value

from ._common import console, get_service_client


def show_user_profile(config: Config, user_id: str):
    """Display user profile via service."""
    client = get_service_client(config)
    if not client or not client.is_available():
        console.print("[yellow]Service not available. Start with: llm-service[/yellow]")
        return

    data = client.get_profile(user_id, entity_type="user")
    if not data:
        console.print(f"[yellow]No profile found for user '{user_id}'.[/yellow]")
        return

    console.print(Panel.fit(f"[bold cyan]User Profile: {user_id}[/bold cyan]", border_style="cyan"))
    console.print()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("[bold]Identity[/bold]", "")
    table.add_row("  Display Name", data.get("display_name") or "[dim]not set[/dim]")
    table.add_row("  Description", data.get("description") or "[dim]not set[/dim]")

    by_category: dict[str, list] = {}
    for attr in data.get("attributes", []):
        cat = attr.get("category", "other")
        by_category.setdefault(cat, []).append(attr)

    category_names = {"preference": "Preferences", "fact": "Facts", "interest": "Interests", "communication": "Communication", "context": "Context"}
    for category, attrs in sorted(by_category.items()):
        table.add_row("", "")
        table.add_row(f"[bold]{category_names.get(category, category.title())}[/bold]", "")
        for attr in sorted(attrs, key=lambda a: a.get("key", "")):
            value_str = str(attr.get("value", ""))
            if len(value_str) > 60:
                value_str = value_str[:57] + "..."
            conf = attr.get("confidence", 1.0)
            conf_str = f" [dim]({conf:.0%})[/dim]" if conf < 1.0 else ""
            table.add_row(f"  {attr.get('key', '?')}", f"{value_str}{conf_str}")

    console.print(table)
    console.print()
    console.print("[dim]Add attributes: llm --user-profile-set category.key=value[/dim]")
    console.print()


def show_users(config: Config):
    """Display all user profiles via service."""
    client = get_service_client(config)
    if not client or not client.is_available():
        console.print("[yellow]Service not available. Start with: llm-service[/yellow]")
        return

    data = client.list_profiles(entity_type="user")
    profiles_list = (data or {}).get("profiles", []) if data else []

    if not profiles_list:
        console.print("[yellow]No user profiles found.[/yellow]")
        console.print("[dim]Create one with: llm --user-profile-setup[/dim]")
        return

    console.print(Panel.fit("[bold cyan]User Profiles[/bold cyan]", border_style="cyan"))
    console.print()
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("User ID", style="cyan")
    table.add_column("Display Name")
    table.add_column("Description")

    for p in profiles_list:
        desc = p.get("description") or ""
        table.add_row(
            p.get("entity_id", "?"),
            p.get("display_name") or "[dim]-[/dim]",
            (desc[:50] + "..." if len(desc) > 50 else desc) or "[dim]-[/dim]",
        )

    console.print(table)
    console.print()
    console.print("[dim]Use --user <id> to select a user profile[/dim]")
    console.print()


def _parse_runtime_setting_value(raw: str):
    """Parse runtime setting value from CLI string input."""
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text


def show_runtime_settings(config: Config, scope: str, bot_id: str | None) -> bool:
    """Display runtime settings from DB for global or bot scope via service."""
    if scope == "global":
        scope_id = "*"
        title = "Runtime Settings (global)"
    else:
        target_bot = bot_id or config.DEFAULT_BOT
        scope_id = target_bot
        title = f"Runtime Settings (bot: {target_bot})"

    client = get_service_client(config)
    if not client or not client.is_available():
        console.print("[yellow]Service not available. Start with: llm-service[/yellow]")
        return False

    resp = client.get_settings(scope_type=scope, scope_id=scope_id if scope != "global" else None)
    if resp is None:
        console.print("[red]Failed to load settings from service.[/red]")
        return False

    items = resp.get("settings", [])
    if not items:
        console.print("[yellow]No runtime settings found for this scope.[/yellow]")
        console.print("[dim]Tip: run `llm --settings-bootstrap` to seed DB settings from current bot/template config.[/dim]")
        return True

    table = Table(title=title)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for item in sorted(items, key=lambda i: i.get("key", "")):
        table.add_row(item.get("key", "?"), repr(item.get("value")))
    console.print(table)
    return True


def set_runtime_setting(config: Config, scope: str, bot_id: str | None, key: str, raw_value: str) -> bool:
    """Set runtime setting via service."""
    scope_id = "*" if scope == "global" else (bot_id or config.DEFAULT_BOT)
    value = _parse_runtime_setting_value(raw_value)

    client = get_service_client(config)
    if not client or not client.is_available():
        console.print("[yellow]Service not available. Start with: llm-service[/yellow]")
        return False

    resp = client.set_setting(scope, scope_id if scope != "global" else None, key, value)
    if resp and resp.get("success"):
        console.print(f"[green]Set runtime setting[/green] {scope}:{scope_id} {key}={value!r}")
        return True
    console.print("[red]Failed to set runtime setting via service.[/red]")
    return False


def delete_runtime_setting(config: Config, scope: str, bot_id: str | None, key: str) -> bool:
    """Delete runtime setting via service."""
    scope_id = "*" if scope == "global" else (bot_id or config.DEFAULT_BOT)

    client = get_service_client(config)
    if not client or not client.is_available():
        console.print("[yellow]Service not available. Start with: llm-service[/yellow]")
        return False

    resp = client.delete_setting(scope, key, scope_id=scope_id if scope != "global" else None)
    if resp and resp.get("success"):
        if resp.get("deleted"):
            console.print(f"[green]Deleted runtime setting[/green] {scope}:{scope_id} {key}")
        else:
            console.print("[yellow]No matching runtime setting found.[/yellow]")
        return True
    console.print("[red]Failed to delete runtime setting via service.[/red]")
    return False


def bootstrap_runtime_settings(config: Config, bot_id: str | None, overwrite: bool = False) -> bool:
    """Seed DB runtime settings from current config/bot effective settings via service."""
    from llm_bawt.bots import BotManager

    bot_manager = BotManager(config)
    bots = bot_manager.list_bots()
    if bot_id:
        target = bot_manager.get_bot(bot_id)
        if not target:
            console.print(f"[red]Unknown bot: {bot_id}[/red]")
            return False
        bots = [target]

    # TASK-615: seed CANONICAL keys only, sourced from the setting registry
    # (not env fields — the legacy context/summarization env fields are
    # deleted). Seeding the old keys here would resurrect the exact rows the
    # TASK-614 migration removes.
    from llm_bawt.setting_definitions import setting_default

    global_map = {
        "history_tokens": setting_default("history_tokens", 12000),
        "max_output_tokens": setting_default("max_output_tokens", 4096),
        "history_reload_ttl_seconds": config.HISTORY_RELOAD_TTL_SECONDS,
        "summary_count": setting_default("summary_count", 5),
        "compact_context": setting_default("compact_context", True),
        "memory_n_results": config.MEMORY_N_RESULTS,
        "memory_min_relevance": config.MEMORY_MIN_RELEVANCE,
        "memory_dedup_similarity": config.MEMORY_DEDUP_SIMILARITY,
        "profile_maintenance_interval_minutes": config.PROFILE_MAINTENANCE_INTERVAL_MINUTES,
        "history_summarization_interval_minutes": config.HISTORY_SUMMARIZATION_INTERVAL_MINUTES,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
    }

    client = get_service_client(config)
    if not client or not client.is_available():
        console.print("[yellow]Service not available. Start with: llm-service[/yellow]")
        return False

    # Fetch current global settings to check for existing keys
    existing_resp = client.get_settings(scope_type="global")
    if existing_resp is None and not overwrite:
        # Can't tell what exists, so we can't honour overwrite=False.
        console.print("[red]Could not read existing settings; nothing was written.[/red]")
        return False
    global_existing_keys = set()
    if existing_resp:
        for item in existing_resp.get("settings", []):
            global_existing_keys.add(item.get("key"))

    batch_items: list[dict] = []
    set_count = 0
    skipped_count = 0

    for key, value in global_map.items():
        if key in global_existing_keys and not overwrite:
            skipped_count += 1
            continue
        batch_items.append({"scope_type": "global", "scope_id": "*", "key": key, "value": value})
        set_count += 1

    for bot in bots:
        bot_resp = client.get_settings(scope_type="bot", scope_id=bot.slug)
        if bot_resp is None and not overwrite:
            console.print(f"[red]Could not read settings for {bot.slug}; nothing was written.[/red]")
            return False
        bot_existing_keys = set()
        if bot_resp:
            for item in bot_resp.get("settings", []):
                bot_existing_keys.add(item.get("key"))
        for key, value in (bot.settings or {}).items():
            if key in bot_existing_keys and not overwrite:
                skipped_count += 1
                continue
            batch_items.append({"scope_type": "bot", "scope_id": bot.slug, "key": key, "value": value})
            set_count += 1

    if batch_items and client.batch_set_settings(batch_items) is None:
        console.print("[red]Saving settings failed (or its outcome is unknown).[/red]")
        return False

    scope_label = f"bot '{bot_id}'" if bot_id else "all bots"
    console.print(
        f"[green]Bootstrapped runtime settings[/green] ({scope_label}): "
        f"set={set_count}, skipped={skipped_count}, overwrite={overwrite}"
    )
    return True


def migrate_bots_to_db(config: Config) -> bool:
    """Migrate YAML bot personalities to DB (deprecated — migration already complete)."""
    console.print("[green]Bot migration is no longer needed — all config is DB-only.[/green]")
    return True


def run_user_profile_setup(config: Config, user_id: str) -> bool:
    """Run interactive user profile setup wizard via service."""
    client = get_service_client(config)
    if not client or not client.is_available():
        console.print("[yellow]Service not available. Start with: llm-service[/yellow]")
        return False

    console.print(Panel.fit("[bold cyan]User Profile Setup[/bold cyan]", border_style="cyan"))
    console.print()
    console.print(f"[dim]Setting up profile for user: {user_id}[/dim]")
    console.print("[dim]Press Enter to skip any field.[/dim]")
    console.print()

    try:
        # Fetch existing profile for defaults
        existing_name = ""
        existing_attrs: dict[str, str] = {}
        data = client.get_profile(user_id, entity_type="user")
        if data:
            existing_name = data.get("display_name") or ""
            for attr in data.get("attributes", []):
                existing_attrs[attr.get("key", "")] = attr.get("value", "")

        name = Prompt.ask("What's your name?", default=existing_name or existing_attrs.get("name", ""))
        occupation = Prompt.ask("What do you do? (occupation)", default=existing_attrs.get("occupation", ""))
        console.print()

        failed = [
            key
            for key, value in (("name", name), ("occupation", occupation))
            if value and client.upsert_profile_attribute("user", user_id, "fact", key, value) is None
        ]
        if failed:
            console.print(f"[red]Not saved (failed or unknown): {', '.join(failed)}[/red]")
            return False
        console.print(f"[green]✓ User profile saved for '{user_id}'[/green]")
        console.print()
        return True

    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Setup cancelled.[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]Error during setup: {e}[/red]")
        return False


def ensure_user_profile(config: Config, user_id: str) -> bool:
    """Ensure user profile exists via service, prompting for setup if needed.

    Returns True if profile exists or setup succeeded, False if setup was cancelled.
    """
    try:
        client = get_service_client(config)
        if client and client.is_available():
            data = client.get_profile(user_id, entity_type="user")
            if data and data.get("display_name"):
                return True
            # No profile or no name — run setup
            console.print()
            console.print("[yellow]Welcome! Let's set up your user profile.[/yellow]")
            console.print()
            return run_user_profile_setup(config, user_id)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not ensure user profile via service: {e}")
    return True  # Don't block on profile errors


def ensure_default_user(config: Config, args: argparse.Namespace) -> None:
    """Prompt for and persist DEFAULT_USER if missing and no --user is provided."""
    if getattr(args, "user", None):
        return
    default_user = getattr(config, "DEFAULT_USER", None)
    if default_user and str(default_user).strip():
        return
    if not sys.stdin.isatty():
        console.print("[bold red]DEFAULT_USER is required to continue.[/bold red]")
        console.print("[dim]Set LLM_BAWT_DEFAULT_USER or pass --user.[/dim]")
        sys.exit(1)
    console.print("[yellow]No default user configured.[/yellow]")
    user_id = Prompt.ask("Enter default user id", default="default").strip()
    if not user_id:
        console.print("[red]No user id provided. Aborting.[/red]")
        sys.exit(1)
    if not set_config_value("DEFAULT_USER", user_id, config):
        console.print("[red]Failed to save DEFAULT_USER to config.[/red]")
        sys.exit(1)
    config.DEFAULT_USER = user_id
    console.print(f"[green]Default user set to '{user_id}'.[/green]")


