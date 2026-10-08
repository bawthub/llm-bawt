"""
Manages AI model configurations for the llm-bawt tool.

This module provides functionality to list, update, and delete model aliases
stored in the models configuration file (typically `~/.config/llm-bawt/models.yaml`).

External Usage:
    - Use `list_models(config)` to display available model aliases.
    - Use `delete_model(alias, config)` to remove a specific model alias.
    - Use `update_models_interactive(config, provider=None)` to fetch the latest models
      from providers (OpenAI, Ollama) and interactively add new ones.

The `ModelManager` class can be used directly for more granular control over
model configurations.
"""

import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytz
import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.rule import Rule

from llm_bawt.utils.config import (
    PROVIDER_CODEX,
    PROVIDER_GGUF,
    PROVIDER_GROK,
    PROVIDER_HF,
    PROVIDER_AGENT_BACKEND,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_UNKNOWN,
)
from .utils.config import Config


def is_service_mode_enabled(config: Config) -> bool:
    """
    Check if service mode is enabled via environment variable or config.
    
    This is the SINGLE source of truth for determining if service mode should be used.
    All CLI commands should use this function to check service mode status.
    
    Returns:
        True if USE_SERVICE is enabled via LLM_BAWT_USE_SERVICE env var or config.USE_SERVICE
    """
    use_service_env = os.getenv("LLM_BAWT_USE_SERVICE", "").lower() in ("true", "1", "yes")
    use_service_config = getattr(config, "USE_SERVICE", False)
    return use_service_env or use_service_config

console = Console()

# Reference pricing + context windows for ChatGPT/Codex models, keyed by the
# backend slug (the bit after any ``openai_chatgpt/`` proxy prefix). The codex
# /models endpoint carries context_window but NO pricing (it's a subscription
# surface), so rates are OpenAI's published per-1M-token API prices — used only
# as an ESTIMATE on the cost pill (always rendered with a "*"). cache_read is
# OpenAI's 90%-off cached-input rate; cache_write is set only for gpt-5.6+
# (billed 1.25x input) — older models omit it and fall back to the input rate.
# Injected onto a model row at add-time (see routes/models.py) when the user
# didn't supply their own, for both the native codex bridge and the claude-code
# ``openai_chatgpt/`` proxy path. Keep in sync when OpenAI revises pricing.
CODEX_MODEL_PRICING: dict[str, dict] = {
    "gpt-5.6-sol": {
        "context_window": 372000,
        "pricing": {"input": 5.0, "output": 30.0, "cache_read": 0.50, "cache_write": 6.25},
    },
    "gpt-5.6-terra": {
        "context_window": 372000,
        "pricing": {"input": 2.5, "output": 15.0, "cache_read": 0.25, "cache_write": 3.125},
    },
    "gpt-5.6-luna": {
        "context_window": 372000,
        "pricing": {"input": 1.0, "output": 6.0, "cache_read": 0.10, "cache_write": 1.25},
    },
    "gpt-5.5": {
        "context_window": 272000,
        "pricing": {"input": 5.0, "output": 30.0, "cache_read": 0.50},
    },
    "gpt-5.4": {
        "context_window": 272000,
        "pricing": {"input": 2.5, "output": 15.0, "cache_read": 0.25},
    },
}


def codex_model_spec(model_id: str | None) -> dict | None:
    """Return the reference spec (pricing + context_window) for a codex model.

    Accepts a bare slug (``gpt-5.6-sol``) or a proxy-prefixed id
    (``openai_chatgpt/gpt-5.6-sol``); the prefix is stripped before lookup.
    Returns None for unknown/unpriced models (e.g. gpt-5.4-mini).
    """
    if not model_id:
        return None
    slug = model_id.split("/")[-1]
    return CODEX_MODEL_PRICING.get(slug)


def normalize_for_match(text: str) -> str:
    """Normalize text for fuzzy matching by lowercasing and removing special characters."""
    return re.sub(r'[^a-z0-9]', '', text.lower())


def _provider_key_for_entry(model_info: dict[str, Any]) -> str:
    """Return the public provider grouping key for a model entry."""
    model_type = str(model_info.get("type") or PROVIDER_UNKNOWN).lower()
    if model_type == PROVIDER_AGENT_BACKEND and str(model_info.get("backend") or "").lower() == PROVIDER_CODEX:
        return PROVIDER_CODEX
    return model_type

class _PathWrapper:
    """Wraps a pathlib.Path to delegate file operations while allowing dynamic attributes."""
    __slots__ = ('_path', '__dict__')
    def __init__(self, path: Path):
        self._path = path
    @property
    def parent(self):
        return _PathWrapper(self._path.parent)
    def __getattr__(self, name):
        return getattr(self._path, name)
    def __fspath__(self):  # support os.PathLike
        return str(self._path)
    def __str__(self):
        return str(self._path)
    def __repr__(self):
        return repr(self._path)


class ModelManager:
    """Manages models defined in the configuration file."""

    def __init__(self, config: Config, local_only: bool = False, service_client=None):
        self.config = config
        self.local_only = local_only
        self.service_client = service_client
        self._changed_aliases: set = set()  # Track aliases modified this session
        raw_path = Path(self.config.MODELS_CONFIG_PATH)
        self.config_path = _PathWrapper(raw_path)
        self.models_data: Dict[str, Any] = {}
        self.load_config()
        if not local_only:
            catalog = self.config.ensure_model_catalog()
            if catalog is not None:
                self.models_data = {"models": catalog.compatibility_mapping()}

    def load_config(self) -> bool:
        if not self.config_path.is_file():
            self.models_data = {"models": {}}
            return True
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f) or {}
            if not isinstance(loaded_data, dict) or any(
                not isinstance(k, str) or not k.isidentifier()
                for k in loaded_data.keys() if k != 'models'
            ):
                raise Exception(f"Invalid format in {self.config_path}")
            if 'models' not in loaded_data or not isinstance(loaded_data.get('models'), dict):
                console.print(f"[yellow]Warning:[/yellow] Invalid format in {self.config_path}. Resetting 'models' dictionary.")
                loaded_data['models'] = {}
            self.models_data = loaded_data
            return True
        except Exception as e:
            console.print(f"[bold red]Error loading config {self.config_path}:[/bold red] {e}")
            self.models_data = {"models": {}}
            return False

    def save_config(self, added: int = 0, updated: int = 0, deleted: int = 0) -> bool:
        if self.service_client:
            return self._save_via_service(added, updated, deleted)
        console.print(f"\nSaving configuration to {self.config_path}...")
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            defined_models = self.models_data.get('models', {})
            sorted_models = dict(sorted(
                defined_models.items(),
                key=lambda item: (_provider_key_for_entry(item[1]), item[0])
            ))
            self.models_data['models'] = sorted_models
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.models_data, f, sort_keys=False, indent=2, default_flow_style=False, allow_unicode=True)
            messages = []
            if added > 0:
                messages.append(f"{added} added")
            if updated > 0:
                messages.append(f"{updated} updated")
            if deleted > 0:
                messages.append(f"{deleted} deleted")
            if messages:
                console.print(f"[bold green]Successfully processed config: {', '.join(messages)}.[/bold green]")
            else:
                console.print(f"[green]Config saved to {self.config_path}.[/green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error saving configuration to {self.config_path}:[/bold red] {e}")
            return False

    def _save_via_service(self, added: int = 0, updated: int = 0, deleted: int = 0) -> bool:
        """Push changed model definitions to the service instead of writing YAML."""
        models = self.models_data.get('models', {})
        ok = 0
        failed = 0
        for alias in self._changed_aliases:
            entry = models.get(alias)
            if entry is None:
                continue
            resp = self.service_client.upsert_catalog_model(alias, entry)
            if resp:
                ok += 1
            else:
                console.print(f"[red]Failed to push '{alias}' to service.[/red]")
                failed += 1
        self._changed_aliases.clear()

        if failed:
            console.print(f"[red]Service update: {ok} succeeded, {failed} failed.[/red]")
            return False

        messages = []
        if added > 0:
            messages.append(f"{added} added")
        if updated > 0:
            messages.append(f"{updated} updated")
        if deleted > 0:
            messages.append(f"{deleted} deleted")
        if messages:
            console.print(f"[bold green]Successfully processed via service: {', '.join(messages)}.[/bold green]")
        else:
            console.print("[green]Models synced to service.[/green]")
        return True

    def list_available_models(self):
        """Display locally-defined models from models.yaml / DB (local mode)."""
        defined_models = self.models_data.get("models", {})
        available_aliases = set(self.config.get_model_options())

        console.print(f"[bold green]Local Models[/bold green]  [dim]{self.config_path}[/dim]")
        console.print()

        if not defined_models:
            console.print("  [yellow]No models defined in the configuration file.[/yellow]")
            console.print()
            return

        models_by_type = defaultdict(list)
        for alias, info in defined_models.items():
            models_by_type[_provider_key_for_entry(info)].append(alias)

        for mtype in sorted(models_by_type):
            console.rule(f"[bold magenta]{mtype.upper()}[/bold magenta]")
            for alias in sorted(models_by_type[mtype]):
                info = defined_models[alias]
                details = self._format_model_details(mtype, info)
                marker = "[green]✓[/green]" if alias in available_aliases else "[red]✗[/red]"
                note = self._get_dependency_note(mtype) if alias not in available_aliases else ""
                console.print(f"  {marker} [bold][bright_blue]{alias}[/bright_blue][/bold]: {details}{note}")
            console.print()

        console.rule(style="#777777")
        console.print("[green]✓[/green] = Dependencies met | [red]✗[/red] = Dependencies missing")
        if self.config.DEFAULT_MODEL_ALIAS:
            status = "[green]✓[/green]" if self.config.DEFAULT_MODEL_ALIAS in available_aliases else "[red]✗[/red]"
            console.print(f"Default alias: {status} [cyan]{self.config.DEFAULT_MODEL_ALIAS}[/cyan]")

    def resolve_model_alias(self, requested_alias: Optional[str]) -> Optional[str]:
        defined = self.models_data.get("models", {})
        available_set = set(self.config.get_model_options())
        if not requested_alias:
            default = self.config.DEFAULT_MODEL_ALIAS
            if default and default in available_set:
                return default
            console.print("[bold red]Error:[/bold red] No model specified or default unavailable.")
            self.list_available_models()
            return None
        norm = normalize_for_match(requested_alias)
        if requested_alias in available_set:
            return requested_alias
        if requested_alias in defined:
            console.print(f"[bold red]Error:[/bold red] '{requested_alias}' defined but unavailable.")
            return None
        matches = [a for a in self.config.get_model_options() if norm in normalize_for_match(a)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return self._prompt_for_match(requested_alias, matches)
        console.print(f"[bold red]Error:[/bold red] Alias '{requested_alias}' not found.")
        self.list_available_models()
        return None

    def delete_model_alias(self, alias: str) -> bool:
        console.print(Rule(f"[bold yellow]Deleting Model Alias: {alias}[/bold yellow]"))
        models = self.models_data.get('models', {})
        if alias not in models:
            console.print(f"[bold red]Error:[/bold red] Alias '{alias}' not found.")
            return False
        info = models[alias]
        confirm = Confirm.ask(f"Delete '{alias}' (type: {info.get('type')})?", default=False)
        if not confirm:
            console.print("[info]Deletion cancelled.[/info]")
            return True
        del models[alias]

        if self.service_client:
            resp = self.service_client.delete_catalog_model(alias)
            if resp:
                console.print(f"[bold green]Deleted '{alias}' from service.[/bold green]")
                return True
            else:
                console.print(f"[red]Failed to delete '{alias}' from service.[/red]")
                return False

        # Local mode is deliberately file-only. Database catalog management is
        # available through the normalized service API / Models admin UI.
        return self.save_config(deleted=1)

    def update_models(self, provider_type: Optional[str] = None):
        if self.service_client:
            console.print("[dim]Mode: service[/dim]")
        else:
            console.print("[dim]Mode: local (models.yaml)[/dim]")
        targets = [provider_type] if provider_type else [PROVIDER_OPENAI, PROVIDER_GROK, PROVIDER_CODEX, PROVIDER_OLLAMA]
        success = True # Track overall success
        for prov in targets:
            console.print(Rule(f"Updating models for provider: {prov}"))
            try:
                prov_success = self._update_provider_models(prov)
                if not prov_success:
                    success = False # Mark failure if any provider fails
            except Exception as e:
                console.print(f"[bold red]Error updating {prov}:[/bold red] {e}")
                success = False # Mark failure on exception
        return success

    def _update_provider_models(self, provider_type: str):
        success, api_models = self._fetch_api_models(provider_type)
        if not success:
            console.print(f"[yellow]Skipping {provider_type}, failed to fetch API models.[/yellow]")
            return False

        local_models = self.models_data.get('models', {})
        existing_map = {
            m["model_id"]: alias
            for alias, m in local_models.items()
            if self._entry_matches_provider(m, provider_type) and m.get("model_id")
        }
        updated_count = self._update_existing_descriptions(provider_type, api_models, existing_map)
        new_models = [m for m in api_models if m['id'] not in existing_map]

        added_count = 0
        if new_models:
            if provider_type == PROVIDER_OLLAMA:
                console.print(f"[info]Detected {len(new_models)} new Ollama models. Adding automatically...[/info]")
                selected = new_models
                added_count = self._prepare_new_model_entries(provider_type, selected)
            else:
                selected = self._prompt_for_new_models(new_models, provider_type)
                added_count = self._prepare_new_model_entries(provider_type, selected)

        if updated_count or added_count:
            save_ok = self.save_config(added=added_count, updated=updated_count)
            if save_ok:
                if provider_type == PROVIDER_OLLAMA:
                    self.config.force_ollama_check()
            return save_ok
        else:
            console.print("[green]Configuration is up-to-date.[/green]")
            return True

    def _fetch_api_models(self, provider_type: str) -> Tuple[bool, List[Dict[str, Any]]]:
        if provider_type == PROVIDER_OPENAI:
            return fetch_openai_api_models()
        if provider_type == PROVIDER_GROK:
            api_key = (
                getattr(self.config, 'XAI_API_KEY', '')
                or os.getenv('XAI_API_KEY', '')
                or os.getenv('LLM_BAWT_XAI_API_KEY', '')
            )
            if not api_key:
                console.print("[bold red]Error:[/bold red] XAI_API_KEY not set. Set LLM_BAWT_XAI_API_KEY in .env or XAI_API_KEY env var.")
                return False, []
            return fetch_grok_api_models(api_key)
        if provider_type == PROVIDER_OLLAMA:
            ollama_url = getattr(self.config, 'OLLAMA_URL', None)
            if not ollama_url:
                console.print("[bold red]Error:[/bold red] OLLAMA_URL not set in config.")
                return False, []
            return fetch_ollama_api_models(ollama_url)
        if provider_type == PROVIDER_CODEX:
            return fetch_codex_models()
        console.print(f"[yellow]Provider '{provider_type}' not supported for update.[/yellow]")
        return False, []

    @staticmethod
    def _entry_matches_provider(model_info: dict[str, Any], provider_type: str) -> bool:
        if provider_type == PROVIDER_CODEX:
            return (
                str(model_info.get("type") or "").lower() == PROVIDER_AGENT_BACKEND
                and str(model_info.get("backend") or "").lower() == PROVIDER_CODEX
            )
        return str(model_info.get("type") or "").lower() == provider_type

    def _update_existing_descriptions(self, provider_type: str, api_models: List[Dict[str, Any]], existing_map: Dict[str, str]) -> int:
        """Update auto-generated timestamp descriptions for existing models.

        Only overwrites descriptions that match the auto-generated pattern
        (contain 'Created:' or 'Modified:').  Custom descriptions are left alone.
        """
        tz = pytz.timezone('US/Eastern')
        count = 0
        for m in api_models:
            mid = m['id']
            if mid in existing_map:
                alias = existing_map[mid]
                entry = self.models_data['models'][alias]
                if provider_type == PROVIDER_CODEX:
                    expected = m.get("summary", mid)
                    current_desc = entry.get("description", "")
                    if current_desc != expected:
                        entry["description"] = expected
                        self._changed_aliases.add(alias)
                        count += 1
                        console.print(f"Updated description for alias '{alias}'.")
                    continue
                ts = self._format_model_timestamp_str(provider_type, m, tz)
                if not ts:
                    continue
                label = 'Created' if provider_type in (PROVIDER_OPENAI, PROVIDER_GROK) else 'Modified'
                expected = f"{mid} ({label}: {ts})"
                current_desc = entry.get('description', '')
                # Skip models with custom (non-auto-generated) descriptions
                if current_desc and f'{label}:' not in current_desc:
                    continue
                if current_desc != expected:
                    entry['description'] = expected
                    self._changed_aliases.add(alias)
                    count += 1
                    console.print(f"Updated description for alias '{alias}'.")
        return count

    def _prompt_for_new_models(
        self,
        new_models: List[Dict[str, Any]],
        provider_type: str = PROVIDER_OPENAI,
    ) -> List[Dict[str, Any]]:
        tz = pytz.timezone('US/Eastern')
        console.print("New models detected:")
        new_models.sort(key=lambda m: m['id'])
        choices_map = {str(idx + 1): model for idx, model in enumerate(new_models)}

        for idx, m in enumerate(new_models, start=1):
            model_id_colored = f"[bright_blue]{m['id']}[/bright_blue]"
            if provider_type == PROVIDER_CODEX:
                summary = m.get("summary", "")
                console.print(f"  [cyan]{idx}[/cyan]: {model_id_colored} - {summary}")
            else:
                ts = ModelManager._format_model_timestamp_str(None, provider_type, m, tz)
                console.print(f"  [cyan]{idx}[/cyan]: {model_id_colored} ({ts})")
        sep = ',' if provider_type in (PROVIDER_OPENAI, PROVIDER_CODEX) else ' '
        prompt = (
            f"Enter numbers to add ({'comma' if sep == ',' else 'space'}-separated), or press Enter to skip"
        )
        resp = Prompt.ask(prompt, default="")
        if not resp.strip():
            return []
        parts = resp.split(sep)
        selected: List[Dict[str, Any]] = []
        for p in parts:
            i = p.strip()
            if i.isdigit() and i in choices_map:
                selected.append(choices_map[i])
        return selected

    def _prepare_new_model_entries(self, provider_type: str, selected_models: List[Dict[str, Any]]) -> int:
        tz = pytz.timezone('US/Eastern')
        existing = set(self.models_data['models'].keys())
        count = 0
        for m in selected_models:
            default_alias = self._generate_alias(provider_type, m, existing)
            alias = Prompt.ask(f"Alias for [bright_blue]{m['id']}[/bright_blue]", default=default_alias).strip()
            if not alias:
                alias = default_alias
            if alias in existing:
                console.print(f"[yellow]Alias '{alias}' already exists, using '{default_alias}'.[/yellow]")
                alias = default_alias
            existing.add(alias)
            if provider_type == PROVIDER_CODEX:
                entry = {
                    "type": PROVIDER_AGENT_BACKEND,
                    "backend": PROVIDER_CODEX,
                    "model_id": m["id"],
                    "description": m.get("summary", m["id"]),
                    "tool_support": "none",
                }
            else:
                ts = self._format_model_timestamp_str(provider_type, m, tz)
                label = 'Created' if provider_type in (PROVIDER_OPENAI, PROVIDER_GROK) else 'Modified'
                desc = f"{m['id']} ({label}: {ts})"
                entry = {'type': provider_type, 'model_id': m['id'], 'description': desc}
            self.models_data['models'][alias] = entry
            self._changed_aliases.add(alias)
            count += 1
            console.print(f"Added new alias '{alias}'.")
        return count

    def _generate_alias(self, provider_type: str, model_info: Dict[str, Any], existing_aliases: set) -> str:
        base = model_info['id'].lower().replace('/', '-').replace(':', '-')
        alias = base
        i = 1
        while alias in existing_aliases:
            alias = f"{base}{i}"
            i += 1
        return alias

    def _format_model_timestamp_str(self, provider_type: str, model_info: Dict[str, Any], tz: Any) -> Optional[str]:
        dt = model_info.get('created') if provider_type in (PROVIDER_OPENAI, PROVIDER_GROK) else model_info.get('modified_at')
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            local = dt.astimezone(tz)
            return local.strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            return None

    def _format_model_details(self, model_type: str, model_info: Dict[str, Any]) -> str:
        parts = []
        mid = None

        if model_type == PROVIDER_GGUF:
            repo = model_info.get('repo_id')
            filename = model_info.get('filename')
            desc = model_info.get('description')
            chat_format = model_info.get('chat_format')
            # Avoid duplicating repo info if already in description
            if repo and (not desc or repo not in desc):
                parts.append(f"Repo: {repo}")
            if chat_format:
                parts.append(f"Format: {chat_format}")
            if desc:
                parts.append(desc)
            # Add file size if the file exists
            if repo and filename:
                file_size = self._get_gguf_file_size(repo, filename)
                if file_size:
                    parts.append(f"[dim]{file_size}[/dim]")

        elif model_type == PROVIDER_CODEX:
            mid = model_info.get("model_id")
            desc = model_info.get("description")
            if mid:
                parts.append(f"ID: [bright_blue]{mid}[/bright_blue]")
            if desc:
                parts.append(desc)

        elif model_type in (PROVIDER_HF, PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_GROK):
            mid = model_info.get('model_id')
            desc = model_info.get('description')
            formatted_id_str = ""
            raw_id_str = ""
            processed_desc = desc  # Work with a modifiable copy
            if processed_desc:
                provider_prefix = f"{model_type.capitalize()} "
                if processed_desc.lower().startswith(provider_prefix.lower()):
                    processed_desc = processed_desc[len(provider_prefix):].strip()
            if mid:
                formatted_id_str = f"[bright_blue]{mid}[/bright_blue]"
                raw_id_str = mid # Keep raw ID for matching
            if processed_desc and raw_id_str and raw_id_str in processed_desc:
                escaped_raw_id = re.escape(raw_id_str)
                final_desc = re.sub(rf'\\b{escaped_raw_id}\\b', formatted_id_str, processed_desc, count=1)
                parts = [final_desc]
            else:
                if mid:
                    parts.append(f"ID: {formatted_id_str}")
                if processed_desc: # Use potentially prefix-removed description
                    parts.append(processed_desc)

        elif model_type == PROVIDER_AGENT_BACKEND:
            backend = model_info.get("backend", "")
            bot_config = model_info.get("bot_config", {})
            session_key = bot_config.get("session_key")
            model_id = model_info.get("model_id")
            if backend:
                parts.append(f"Backend: [bright_blue]{backend}[/bright_blue]")
            if model_id:
                parts.append(f"Model: [bright_blue]{model_id}[/bright_blue]")
            if session_key:
                parts.append(f"Session: [bright_blue]{session_key}[/bright_blue]")

        return ", ".join(parts) or "No details"

    def _get_gguf_file_size(self, repo_id: str, filename: str) -> Optional[str]:
        """Get the file size of a GGUF model file if it exists locally."""
        from pathlib import Path
        
        # Check in the model cache directory
        file_path = Path(self.config.MODEL_CACHE_DIR) / repo_id / filename
        if not file_path.exists():
            return None
        
        try:
            size_bytes = file_path.stat().st_size
            return self._format_size(size_bytes)
        except Exception:
            return None
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable size (e.g., 14.5G, 3.2M)."""
        size = float(size_bytes)
        for unit in ['B', 'K', 'M', 'G', 'T']:
            if abs(size) < 1024:
                if unit == 'B':
                    return f"{int(size)}{unit}"
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}P"

    @staticmethod
    def _get_dependency_note(model_type: str) -> str:
        """Return a note about missing dependencies for the given model type."""
        if model_type == PROVIDER_HF:
            return " (Requires: [dim]llm-bawt[huggingface][/dim])"
        if model_type == PROVIDER_GGUF:
            return " (Requires: [dim]llama-cpp-python, huggingface-hub[/dim])"
        return ''

    def _prompt_for_match(self, requested_alias: str, matches: List[str]) -> Optional[str]:
        console.print(f"Multiple matches for '{requested_alias}':")
        choices = {str(i+1): m for i, m in enumerate(matches)}
        for k, m in choices.items():
            console.print(f"  [cyan]{k}[/cyan]: {m}")
        choice = Prompt.ask("Enter number", choices=list(choices.keys()), default="1")
        return choices.get(choice)

    @staticmethod
    def _infer_provider_type(alias: str) -> str:
        """Infer provider type for a missing model alias."""
        normalized = alias.strip().lower()
        if normalized.startswith("grok"):
            return PROVIDER_GROK
        if normalized.startswith("gpt") or normalized.startswith(("o1", "o3", "o4")):
            return PROVIDER_OPENAI
        return PROVIDER_OPENAI

    def set_context_window(self, alias: str, context_window: int) -> bool:
        """Set or create a model's context_window.

        In service mode, upserts via the service API.
        In local mode, writes to models.yaml.
        If the alias is missing, creates a minimal model entry.
        """
        if context_window <= 0:
            console.print("[bold red]Error:[/bold red] context_window must be a positive integer.")
            return False

        models = self.models_data.setdefault("models", {})
        added = 0
        updated = 0

        entry = models.get(alias)
        if entry is None:
            provider_type = self._infer_provider_type(alias)
            entry = {
                "type": provider_type,
                "model_id": alias,
                "description": (
                    f"{provider_type.capitalize()} model (added via --set-context-window override)"
                ),
            }
            models[alias] = entry
            added = 1

        entry["context_window"] = int(context_window)
        self._changed_aliases.add(alias)
        if not added:
            updated = 1

        if self.save_config(added=added, updated=updated):
            console.print(
                f"[green]Set context_window={context_window} for model alias '{alias}'.[/green]"
            )
            return True
        return False


def fetch_ollama_api_models(ollama_url: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """Fetches model list from Ollama API."""
    import requests # Import lazily
    models_details = []
    start = time.time()
    endpoint = f"{ollama_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data.get('models'), list):
            for d in data['models']:
                mid = d.get('name')
                mat = d.get('modified_at')
                mod_dt = None
                if mat:
                    try:
                        mod_dt = _parse_iso(mat)
                    except Exception:
                        console.print(f"[yellow]Warning:[/] Could not parse timestamp {mat}")
                models_details.append({'id': mid, 'modified_at': mod_dt})
        console.print(f"⏱️ Ollama query took {(time.time()-start)*1000:.0f}ms, found {len(models_details)} models")
        return True, models_details
    except Exception as e:
        console.print(f"[bold red]Ollama API error:[/bold red] {e}")
        return False, []

def fetch_openai_api_models(api_key: str | None = None) -> Tuple[bool, List[Dict[str, Any]]]:
    """Fetches model list from OpenAI API."""
    from openai import OpenAI  # Import lazily
    details = []
    start = time.time()
    try:
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception as e:
        console.print(f"[bold red]OpenAI init error:[/bold red] {e}")
        return False, []
    try:
        resp = client.models.list()
        for m in resp.data:
            if 'gpt' in m.id or m.owned_by in ('openai', 'system'):
                ts = getattr(m, 'created', None)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
                details.append({'id': m.id, 'created': dt})
        console.print(f"⏱️ OpenAI query took {(time.time()-start)*1000:.0f}ms, found {len(details)} models")
        return True, details
    except Exception as e:
        console.print(f"[bold red]OpenAI fetch error:[/bold red] {e}")
        return False, []


def fetch_grok_api_models(api_key: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """Fetches model list from xAI API (Grok models)."""
    from openai import OpenAI
    details = []
    start = time.time()
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        resp = client.models.list()
        for m in resp.data:
            # Skip image/video generation models — not usable as chat models
            if 'imagine' in m.id or 'video' in m.id:
                continue
            ts = getattr(m, 'created', None)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
            details.append({'id': m.id, 'created': dt})
        console.print(f"⏱️ xAI query took {(time.time()-start)*1000:.0f}ms, found {len(details)} models")
        return True, details
    except Exception as e:
        console.print(f"[bold red]xAI API error:[/bold red] {e}")
        return False, []


def fetch_codex_models() -> Tuple[bool, List[Dict[str, Any]]]:
    """Return the live Codex catalog for the authenticated ChatGPT plan.

    Discovery is authoritative: bridge, authentication, upstream, and empty-list
    failures return ``False`` so callers surface an error instead of presenting a
    stale or misleading model list.
    """
    url = os.getenv(
        "CODEX_BRIDGE_MODELS_URL", "http://codex-bridge:8682/models"
    )
    try:
        import httpx

        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        models = [
            {
                "id": item["id"],
                "summary": item.get("description", ""),
                **(
                    {"context_length": item["context_window"]}
                    if isinstance(item.get("context_window"), int)
                    else {}
                ),
            }
            for item in resp.json().get("models", [])
            if isinstance(item, dict) and item.get("id")
        ]
        if not models:
            console.print("[bold red]Codex model discovery returned no models[/bold red]")
            return False, []
        return True, models
    except Exception as e:  # bridge down, auth expiry, network, invalid payload
        console.print(f"[bold red]Codex model discovery failed:[/bold red] {e}")
        return False, []


def fetch_anthropic_api_models(
    api_key: str | None = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Fetch Claude model list from the Anthropic API.

    Used to populate the Claude Code bridge's model picker — the bridge
    has no model fallback, so users need to see a real, live list of SDK
    IDs to choose from. Requires ``ANTHROPIC_API_KEY`` (the Claude Code
    subscription OAuth token does NOT authenticate /v1/models).
    """
    api_key = api_key or (
        os.getenv("LLM_BAWT_ANTHROPIC_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or ""
    )
    if not api_key:
        console.print("[yellow]Anthropic discovery skipped: no ANTHROPIC_API_KEY[/yellow]")
        return False, []

    try:
        from anthropic import Anthropic  # lazy import
    except Exception as e:
        console.print(f"[bold red]Anthropic SDK import error:[/bold red] {e}")
        return False, []

    details: List[Dict[str, Any]] = []
    start = time.time()
    try:
        client = Anthropic(api_key=api_key)
        # SDK exposes models.list() which paginates. Cap at a couple pages
        # in case Anthropic ever adds many — 100 is plenty for the picker.
        page = client.models.list(limit=100)
        for m in getattr(page, "data", []) or []:
            mid = getattr(m, "id", None)
            if not mid:
                continue
            display = getattr(m, "display_name", "") or ""
            created_at = getattr(m, "created_at", None)
            entry: Dict[str, Any] = {
                "id": mid,
                "description": display,
                "created": created_at,
            }
            # The API reports the real input window; keep it so the Add Model
            # dialog never asks the user to type a number we already have.
            max_input = getattr(m, "max_input_tokens", None)
            if isinstance(max_input, int) and not isinstance(max_input, bool) and max_input > 0:
                entry["context_length"] = max_input
            details.append(entry)
        console.print(
            f"⏱️ Anthropic query took {(time.time()-start)*1000:.0f}ms, "
            f"found {len(details)} models"
        )
        return True, details
    except Exception as e:
        console.print(f"[bold red]Anthropic API error:[/bold red] {e}")
        return False, []


def list_models(config: Config, service_mode: bool | None = None):
    """CLI wrapper: list models.

    In service mode, queries the running service for its authoritative model
    catalog. In local mode, displays only models from models.yaml on disk.
    """
    if service_mode is None:
        service_mode = is_service_mode_enabled(config)

    if service_mode:
        _list_models_from_service(config)
    else:
        try:
            manager = ModelManager(config, local_only=True)
        except Exception:
            manager = ModelManager.__new__(ModelManager)
            manager.config = config
            manager.local_only = True
        manager.list_available_models()


def _infer_type(alias: str) -> str:
    """Best-effort type inference from a model alias when the service doesn't report type."""
    low = alias.lower()
    if "codex" in low:
        return PROVIDER_CODEX
    if low.startswith("grok"):
        return "grok"
    if low.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if low.startswith("openclaw"):
        return "openclaw"
    if low == "openclaw":
        return "agent_backend"
    return "other"


def _list_models_from_service(config: Config):
    """Display model catalog from the running service."""
    from llm_bawt.service.client import get_service_client

    client = get_service_client(config)
    if not client or not client.is_available(force_check=True):
        service_url = getattr(client, "http_url", None) if client else None
        console.print("[yellow]Service not reachable.[/yellow]")
        if service_url:
            console.print(f"[dim]Expected at: {service_url}[/dim]")
        return

    service_models = client.list_models() or []
    if not service_models:
        console.print("[yellow]No models reported by service.[/yellow]")
        return

    service_url = client.http_url or ""
    console.print(f"[bold cyan]Service Models[/bold cyan]  [dim]{service_url}[/dim]")
    console.print()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for model in service_models:
        mtype = (model.get("type") or _infer_type(model["id"])).lower()
        grouped[mtype].append(model)

    provider_order = [
        "openai", "codex", "grok", "openclaw", "ollama", "vllm",
        "gguf", "huggingface", "agent_backend", "other",
    ]
    present = [p for p in provider_order if grouped.get(p)]
    present.extend(sorted(p for p in grouped if p not in present))

    for provider in present:
        console.rule(f"[bold magenta]{provider.upper()}[/bold magenta]")
        for model in sorted(grouped[provider], key=lambda m: m["id"]):
            alias = model["id"]
            parts: list[str] = []
            if model.get("model_id"):
                parts.append(f"[bright_blue]{model['model_id']}[/bright_blue]")
            if model.get("description"):
                parts.append(model["description"])
            detail = ", ".join(parts) if parts else ""
            console.print(f"  [bold]{alias}[/bold]  {detail}")
        console.print()

def delete_model(alias: str, config: Config, service_client=None) -> bool:
    manager = ModelManager(config, service_client=service_client)
    # Get model info before deletion from config
    model_info = manager.models_data.get('models', {}).get(alias)

    delete_ok = manager.delete_model_alias(alias)
    if delete_ok:
        # Avoid probing Ollama during unrelated operations.
        # Ollama connectivity/model checks happen only when explicitly refreshing Ollama models.
        
        # If the model is a GGUF model, check if we should delete the model files as well
        if model_info and model_info.get('type') == PROVIDER_GGUF:
            repo_id = model_info.get('repo_id')
            if repo_id:
                from pathlib import Path
                import shutil
                
                # Get the entire repo directory path
                repo_dir = Path(config.MODEL_CACHE_DIR) / repo_id
                
                if repo_dir.exists():
                    console.print(f"\nFound model directory at: [cyan]{repo_dir}[/cyan]")
                    confirm = Confirm.ask("Do you want to delete the entire model directory?", default=False)
                    if confirm:
                        try:
                            # Delete the entire directory tree
                            shutil.rmtree(repo_dir)
                            console.print(f"[green]Successfully deleted model directory: {repo_dir}[/green]")
                        except Exception as e:
                            console.print(f"[bold red]Error deleting model directory:[/bold red] {e}")
    
    return delete_ok

def update_models_interactive(config: Config, provider: Optional[str] = None, service_client=None):
    return ModelManager(config, service_client=service_client).update_models(provider)


def set_model_context_window(alias: str, context_window: int, config: Config, service_client=None) -> bool:
    """CLI wrapper: set per-model context window override."""
    return ModelManager(config, service_client=service_client).set_context_window(alias=alias, context_window=context_window)


def _parse_iso(ts_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except Exception:
        return None
