"""Interactive .env configuration walkthrough for the llm-bawt CLI client.

Walks through the settings the client actually uses, pre-filling with
current values or defaults.  Server-side / DB-persisted settings are
intentionally omitted — those live in Docker or in runtime settings.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple
from urllib.parse import quote_plus

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

if TYPE_CHECKING:
    from llm_bawt.utils.config import Config


# ---------------------------------------------------------------------------
# Setting descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Setting:
    """One configurable .env setting."""

    key: str                       # Config field name (e.g. "DEFAULT_MODEL_ALIAS")
    description: str               # Human-readable one-liner
    secret: bool = False           # Mask input/display
    env_only: bool = False         # Not a Config field — raw env var (e.g. OPENAI_API_KEY)


@dataclass(frozen=True)
class SettingGroup:
    """A logical group of related settings."""

    title: str
    settings: List[Setting]
    note: str = ""                 # Optional note printed before prompts


# ---------------------------------------------------------------------------
# Which settings matter for the client
# ---------------------------------------------------------------------------

CLIENT_SETTING_GROUPS: List[SettingGroup] = [
    SettingGroup(
        title="Core",
        settings=[
            Setting("DEFAULT_MODEL_ALIAS", "Optional default model alias (leave blank for service-selected model)"),
            Setting("DEFAULT_BOT", "Default bot personality (a bot slug, e.g. helper)"),
            Setting("DEFAULT_USER", "Default user profile name"),
            Setting("USE_SERVICE", "Route queries through the background service by default"),
        ],
    ),
    SettingGroup(
        title="Service connection",
        note="The CLI client connects to the llm-bawt service (typically Docker).",
        settings=[
            Setting("SERVICE_HOST", "Service hostname or IP"),
            Setting("SERVICE_PORT", "Service port"),
        ],
    ),
    SettingGroup(
        title="PostgreSQL",
        note="Memory backend — needed for persistent memory / history.",
        settings=[
            Setting("POSTGRES_HOST", "Database hostname"),
            Setting("POSTGRES_PORT", "Database port"),
            Setting("POSTGRES_USER", "Database username"),
            Setting("POSTGRES_PASSWORD", "Database password", secret=True),
            Setting("POSTGRES_DATABASE", "Database name"),
        ],
    ),
    SettingGroup(
        title="Web search",
        settings=[
            Setting("SEARCH_PROVIDER", "Search backend: duckduckgo, tavily, brave, or reddit"),
            Setting("TAVILY_API_KEY", "Tavily API key (optional)", secret=True),
            Setting("BRAVE_API_KEY", "Brave Search API key (optional)", secret=True),
            Setting("REDDIT_CLIENT_ID", "Reddit OAuth client ID (optional)", secret=True),
            Setting("REDDIT_CLIENT_SECRET", "Reddit OAuth client secret (optional)", secret=True),
            Setting("REDDIT_USER_AGENT", "Reddit API user agent (optional)"),
        ],
    ),
    SettingGroup(
        title="Ollama",
        note="Only needed if you have Ollama-type models defined.",
        settings=[
            Setting("OLLAMA_URL", "Ollama server URL"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_value(config: "Config", setting: Setting) -> str:
    """Return the current effective value for a setting as a string."""
    if setting.env_only:
        return os.environ.get(setting.key, "")
    val = getattr(config, setting.key, None)
    if val is None:
        return ""
    return str(val)


def _default_value(config: "Config", setting: Setting) -> str:
    """Return the schema default for a Config field, or '' for env-only."""
    if setting.env_only:
        return ""
    field_info = config.model_fields.get(setting.key)
    if field_info is None:
        return ""
    default = field_info.default
    if default is None:
        return ""
    return str(default)


def _display_value(value: str, secret: bool) -> str:
    """Format a value for display, masking secrets."""
    if not value:
        return "[dim]not set[/dim]"
    if secret:
        if len(value) <= 8:
            return "****"
        return value[:4] + "****" + value[-4:]
    return value


def _env_var_name(setting: Setting, env_prefix: str) -> str:
    """Full environment variable name for a setting."""
    if setting.env_only:
        return setting.key  # already the raw env var name
    return f"{env_prefix}{setting.key}"


def _save_value(
    config: "Config",
    setting: Setting,
    value: str,
    dotenv_path: Path,
    env_prefix: str,
) -> bool:
    """Persist a value to the .env file and update the live config/env."""
    from llm_bawt.utils.env import set_env_value

    env_var = _env_var_name(setting, env_prefix)
    if not set_env_value(dotenv_path, env_var, value):
        return False

    # Update live state
    os.environ[env_var] = value
    if not setting.env_only:
        # Coerce booleans / ints back to the right type
        field_info = config.model_fields.get(setting.key)
        if field_info and field_info.annotation is bool:
            setattr(config, setting.key, value.lower() in ("true", "1", "yes"))
        elif field_info and field_info.annotation is int:
            try:
                setattr(config, setting.key, int(value))
            except ValueError:
                setattr(config, setting.key, value)
        else:
            setattr(config, setting.key, value)
    return True


# ---------------------------------------------------------------------------
# Main walkthrough
# ---------------------------------------------------------------------------

def run_config_setup(config: "Config", console: Console) -> int:
    """Walk the user through client-relevant .env settings.

    Returns 0 on success.
    """
    dotenv_path = Path(str(config.model_config.get("env_file", "~/.config/llm-bawt/.env")))
    env_prefix: str = str(config.model_config.get("env_prefix", "LLM_BAWT_"))
    dotenv_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold]llm-bawt client configuration[/bold]\n"
        f"[dim]Config file: {dotenv_path}[/dim]\n"
        "[dim]Press Enter to keep current value · Ctrl-C to abort[/dim]",
        border_style="cyan",
    ))

    changed = 0

    for group in CLIENT_SETTING_GROUPS:
        console.print(f"\n[bold magenta]── {group.title} ──[/bold magenta]")
        if group.note:
            console.print(f"[dim]{group.note}[/dim]")

        for setting in group.settings:
            current = _current_value(config, setting)
            default = _default_value(config, setting)
            prefill = current or default

            if setting.key == "DEFAULT_MODEL_ALIAS" and not config.defined_models.get("models"):
                console.print("  [dim]No local models.yaml aliases found; leave blank to let service/bot choose.[/dim]")

            # Build prompt label
            env_var = _env_var_name(setting, env_prefix)
            display = _display_value(current, setting.secret)
            console.print(
                f"  [cyan]{env_var}[/cyan]  {setting.description}  "
                f"\\[{display}]"
            )

            try:
                prompt_kwargs: dict[str, object] = {
                    "password": setting.secret,
                    "show_default": bool(prefill) and not setting.secret,
                }
                if prefill:
                    prompt_kwargs["default"] = prefill
                new_value = Prompt.ask("  →", **prompt_kwargs)
            except KeyboardInterrupt:
                console.print("\n[yellow]Aborted.[/yellow]")
                return 1

            new_value = new_value.strip()

            # Skip if unchanged
            if new_value == current:
                continue

            if _save_value(config, setting, new_value, dotenv_path, env_prefix):
                changed += 1
            else:
                console.print(f"  [red]Failed to save {env_var}[/red]")

    # Summary
    console.print()
    if changed:
        console.print(
            f"[green]✓ {changed} setting{'s' if changed != 1 else ''} "
            f"saved to {dotenv_path}[/green]"
        )
    else:
        console.print("[dim]No changes made.[/dim]")

    # Quick connectivity checks
    console.print("\n[bold magenta]── Connectivity ──[/bold magenta]")
    _run_connectivity_checks(config, console)

    return 0


# ---------------------------------------------------------------------------
# Connectivity test helpers
# ---------------------------------------------------------------------------

def _http_get_json(url: str, headers: dict[str, str] | None = None, timeout: float = 5.0) -> Tuple[bool, str]:
    """Fire a GET request and return (ok, status_message)."""
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return True, f"{resp.status} {resp.reason}"
            return False, f"{resp.status} {resp.reason}"
    except urllib.error.HTTPError as e:
        return False, f"{e.code} {e.reason}"
    except Exception as e:
        return False, str(e)


def _test_ollama(url: str) -> Tuple[bool, str]:
    clean = url.rstrip("/")
    return _http_get_json(f"{clean}/api/tags")


def _test_postgres(config: "Config") -> Tuple[bool, str]:
    try:
        from sqlalchemy import create_engine, text
    except Exception as e:
        return False, f"sqlalchemy unavailable: {e}"

    password = getattr(config, "POSTGRES_PASSWORD", "")
    if not password:
        return False, "missing password"

    host = getattr(config, "POSTGRES_HOST", "localhost")
    port = int(getattr(config, "POSTGRES_PORT", 5432))
    user = getattr(config, "POSTGRES_USER", "llm_bawt")
    database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
    encoded_password = quote_plus(password)
    url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"

    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "connected"
    except Exception as e:
        return False, str(e)


def _test_service(config: "Config") -> Tuple[bool, str]:
    try:
        from llm_bawt.service import ServiceClient
    except Exception as e:
        return False, f"service client unavailable: {e}"

    http_url = f"http://{config.SERVICE_HOST}:{config.SERVICE_PORT}"
    try:
        client = ServiceClient(http_url=http_url)
        if not client.is_available(force_check=True):
            return False, "not reachable"
        status = client.get_status(silent=True)
        if getattr(status, "available", False):
            return True, "available"
        return False, "unhealthy"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Post-setup connectivity checks
# ---------------------------------------------------------------------------

def _run_connectivity_checks(config: "Config", console: Console) -> None:
    """Run lightweight connectivity checks after setup."""
    # Service
    if config.USE_SERVICE:
        ok, msg = _test_service(config)
        label = f"http://{config.SERVICE_HOST}:{config.SERVICE_PORT}"
        _print_check(console, "Service", label, ok, msg)

    # PostgreSQL
    if config.POSTGRES_PASSWORD:
        ok, msg = _test_postgres(config)
        label = f"{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DATABASE}"
        _print_check(console, "PostgreSQL", label, ok, msg)
    else:
        console.print("  [dim]PostgreSQL: skipped (no password set)[/dim]")

    # Ollama (only if models reference it)
    models = config.defined_models.get("models", {})
    has_ollama = any(info.get("type") == "ollama" for info in models.values())
    if has_ollama:
        ok, msg = _test_ollama(config.OLLAMA_URL)
        _print_check(console, "Ollama", config.OLLAMA_URL, ok, msg)


def _print_check(console: Console, name: str, label: str, ok: bool, msg: str) -> None:
    if ok:
        console.print(f"  [green]✓ {name}: {label} ({msg})[/green]")
    else:
        console.print(f"  [red]✗ {name}: {label} ({msg})[/red]")
