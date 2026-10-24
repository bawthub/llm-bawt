import argparse
import json
import subprocess
import traceback
import sys  # Import sys for exit codes
from rich.console import Console
from rich.markup import escape
from rich.text import Text
from rich.prompt import Prompt
from llm_bawt.utils.config import Config
from llm_bawt.utils.config import set_config_value
from llm_bawt.utils.input_handler import MultilineInputHandler
from llm_bawt.core import LLMBawt
from llm_bawt.model_manager import (
    ModelManager,
    delete_model,
    list_models,
    set_model_context_window,
    update_models_interactive,
)
from llm_bawt.gguf_handler import handle_add_gguf
from llm_bawt.bot_types import format_bot_type
from llm_bawt.cli.bot_create import handle_add_chat_bot
from llm_bawt.cli.vllm_handler import handle_add_vllm
from llm_bawt.cli.openclaw_handler import handle_add_openclaw
from llm_bawt.bots import BotManager
from llm_bawt.utils.streaming import render_streaming_response, render_complete_response
from llm_bawt.shared.logging import LogConfig
import logging
from pathlib import Path
from rich.table import Table
from rich.panel import Panel
from typing import Iterator

# CLI command handlers were split out of this module (TASK-554). They are
# re-imported here so this module stays the facade: the cli.__init__ re-exports
# (show_status, show_bots, show_users, ...) and main()/run_app() references all
# resolve unchanged.
from ._common import console, get_service_client, _is_service_mode
from .tool_render import (
    _format_tool_call,
    _summarize_tool_args,
    _truncate_middle,
    _tool_badge,
    _format_tool_preview,
    _render_tool_event,
)
from .display_cmd import (
    query_via_service,
    inspect_effective_config,
    show_status,
    show_bots,
    _show_bots_from_service,
    _show_bots_from_local,
    _render_job_status_from_service,
    show_job_status,
    trigger_job,
)
from .admin_cmd import (
    show_user_profile,
    show_users,
    _parse_runtime_setting_value,
    show_runtime_settings,
    set_runtime_setting,
    delete_runtime_setting,
    bootstrap_runtime_settings,
    migrate_bots_to_db,
    run_user_profile_setup,
    ensure_user_profile,
    ensure_default_user,
)


def parse_arguments(config_obj: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query LLM models from the command line using model aliases defined in models.yaml")
    parser.add_argument("-m","--model",type=str,default=None,help=f"Model alias defined in {config_obj.MODELS_CONFIG_PATH}. Supports partial matching. (Default: bot's default or {config_obj.DEFAULT_MODEL_ALIAS or 'None'})")
    parser.add_argument("--list-models",action="store_true",help="List available model aliases defined in the configuration file and exit.")
    parser.add_argument("--add-gguf",type=str,metavar="REPO_ID",help="(Deprecated: use --add-model gguf) Add a GGUF model from a Hugging Face repo ID.")
    parser.add_argument("--add-model",type=str,choices=['ollama', 'openai', 'grok', 'codex', 'gguf', 'vllm', 'openclaw'],metavar="TYPE",help="Add models: 'ollama' (refresh from server), 'openai' (query API), 'grok' (query xAI API), 'codex' (install Codex agent model aliases), 'gguf' (add from HuggingFace repo), 'vllm' (add vLLM model from HuggingFace), 'openclaw' (deprecated: use --add-bot openclaw)")
    parser.add_argument("--add-bot", type=str, choices=["chat", "openclaw"], metavar="TYPE", help="Add bots: 'chat' (create a chat bot profile), 'openclaw' (create an OpenClaw agent bot)")
    parser.add_argument("--delete-model",type=str,metavar="ALIAS",help="Delete the specified model alias from the configuration file after confirmation.")
    parser.add_argument(
        "--set-context-window",
        nargs=2,
        metavar=("ALIAS", "TOKENS"),
        help="Set per-model context window in models.yaml. Creates missing aliases (e.g., grok-* from --list-models).",
    )
    parser.add_argument("--config-set", nargs=2, metavar=("KEY", "VALUE"), help="Set a configuration value (e.g., DEFAULT_MODEL_ALIAS) in the .env file.")
    parser.add_argument("--list-config", action="store_true", help="List the current effective configuration settings.")
    parser.add_argument("--setup", action="store_true", help="Walk through client .env settings interactively (pre-filled with current values).")
    parser.add_argument("--settings-scope", choices=["bot", "global"], default="bot", help="Scope for runtime settings operations (default: bot)")
    parser.add_argument("--settings-list", action="store_true", help="List runtime settings from DB for selected scope")
    parser.add_argument("--settings-set", nargs=2, metavar=("KEY", "VALUE"), help="Set runtime setting in DB for selected scope")
    parser.add_argument("--settings-delete", metavar="KEY", help="Delete runtime setting from DB for selected scope")
    parser.add_argument("--settings-bootstrap", action="store_true", help="Seed DB runtime settings from current bot/template/env config")
    parser.add_argument("--settings-bootstrap-overwrite", action="store_true", help="When bootstrapping, overwrite existing DB keys")
    parser.add_argument("--settings-edit", action="store_true", help="Edit global runtime settings in your editor and save to DB")
    parser.add_argument("--bot-edit", metavar="SLUG", help="Edit a single bot config in your editor and save to DB")
    parser.add_argument("--migrate-bots", action="store_true", help="Migrate merged YAML bots to bot_profiles and runtime_settings DB")
    parser.add_argument("question", nargs="*", help="Your question for the LLM model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-dh", "--delete-history", action="store_true", help="Clear chat history")
    parser.add_argument("-ph", "--print-history", nargs="?", const=-1, type=int, default=None,help="Print chat history (optional: number of recent pairs)")
    parser.add_argument("-c", "--command", help="Execute command and add output to question")
    parser.add_argument("--plain", action="store_true", help="Use plain text output")
    parser.add_argument("--no-stream", action="store_true", default=False, help="Disable streaming output")
    parser.add_argument("--local", action="store_true", help="Use local filesystem for history instead of database")
    parser.add_argument("--service", action="store_true", help="Route queries through the background service (if running)")
    parser.add_argument("-b", "--bot", type=str, default=None, help="Bot slug to use. Use --list-bots to see all.")
    parser.add_argument("--list-bots", action="store_true", help="List available bots and exit")
    parser.add_argument("--status", action="store_true", help="Show memory system status and configuration")
    parser.add_argument("--job-status", action="store_true", help="Show scheduled background jobs and recent run history")
    parser.add_argument("--run-job", type=str, metavar="TYPE", help="Trigger a scheduled job immediately (profile_maintenance, memory_consolidation, memory_decay)")
    parser.add_argument("--user", type=str, default=None, help="User profile to use (creates if not exists). Defaults to config.DEFAULT_USER")
    parser.add_argument("--list-users", action="store_true", help="List all user profiles")
    parser.add_argument("--user-profile", action="store_true", help="Show current user profile")
    parser.add_argument("--user-profile-set", metavar="FIELD=VALUE", help="Set a user profile field (e.g., name=\"Nick\")")
    parser.add_argument("--user-profile-setup", action="store_true", help="Run user profile setup wizard")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging output.")
    return parser.parse_args()

def main():
    try:
        # --- Minimal Pre-parsing for flags affecting config loading/display --- 
        # We need to parse --verbose, --plain early, but also handle config paths if specified.
        # Using a separate parser for this minimal set.
        prelim_parser = argparse.ArgumentParser(add_help=False)
        prelim_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        prelim_parser.add_argument("--plain", action="store_true", help="Use plain text output")
        prelim_args, remaining_argv = prelim_parser.parse_known_args()

        # --- Instantiate Config --- 
        config_obj = Config()

    except Exception as e:
        # Catch potential Pydantic validation errors or file issues during Config init
        console.print(f"[bold red]Error initializing configuration:[/bold red] {e}")
        sys.exit(1)
    
    # Parse full arguments now that config is loaded
    args = parse_arguments(config_obj)
    config_obj.VERBOSE = args.verbose
    config_obj.DEBUG = args.debug
    config_obj.PLAIN_OUTPUT = args.plain
    config_obj.NO_STREAM = args.no_stream
    config_obj.INTERACTIVE_MODE = not args.question and not args.command  # Set interactive mode flag

    # Configure logging via centralized LogConfig
    LogConfig.configure(verbose=args.verbose, debug=args.debug)

    # Interactive config setup walkthrough
    if getattr(args, 'setup', False):
        from llm_bawt.cli.config_setup import run_config_setup
        sys.exit(run_config_setup(config_obj, console))

    # Handle setting config values, guard missing attribute in stubbed args
    if getattr(args, 'config_set', None):
        key, value = args.config_set
        success = set_config_value(key, value, config_obj)
        if success:
            console.print(f"[green]Configuration '{key}' set to '{value}' in {config_obj.model_config.get('env_file', 'unknown')}.[/green]")
            sys.exit(0)
        else:
            console.print(f"[bold red]Failed to set configuration '{key}'.[/bold red]")
            sys.exit(1)
    elif getattr(args, "set_context_window", None):
        alias, token_value = args.set_context_window
        try:
            context_window = int(token_value)
        except ValueError:
            console.print(f"[bold red]Invalid token count:[/bold red] '{token_value}' is not an integer.")
            sys.exit(1)

        service_mode = _is_service_mode(args, config_obj)
        svc_client = None
        if service_mode:
            svc_client = get_service_client(config_obj)
            if not svc_client or not svc_client.is_available():
                console.print("[bold red]Service not available.[/bold red] Start with: [yellow]llm-service[/yellow]")
                console.print("[dim]Or use --local to write directly to models.yaml.[/dim]")
                sys.exit(1)

        success = set_model_context_window(alias, context_window, config_obj, service_client=svc_client)
        sys.exit(0 if success else 1)
    # Handle listing config values
    elif getattr(args, 'list_config', False):
        env_file_path = config_obj.model_config.get('env_file', 'unknown')
        console.print(f"[bold magenta]Current Configuration Settings[/bold magenta]")
        console.print(f"[dim]Config file: {env_file_path}[/dim]\n")
        
        # Define which settings are important and should be set by the user
        # Format: field_name -> (is_secret, is_required, description)
        important_settings = {
            'OPENAI_API_KEY': (True, False, "Required for OpenAI/compatible APIs"),
            'DEFAULT_MODEL_ALIAS': (False, False, "Default model to use"),
            'DEFAULT_BOT': (False, False, "Default bot personality"),
            'DEFAULT_USER': (False, False, "Default user profile"),
            'POSTGRES_PASSWORD': (True, True, "Required for memory features"),
            'POSTGRES_HOST': (False, False, "PostgreSQL server hostname"),
            'OLLAMA_URL': (False, False, "Ollama server URL"),
        }
        
        # Check for missing important settings
        missing_settings = []
        for field_name, (is_secret, is_required, desc) in important_settings.items():
            value = getattr(config_obj, field_name, None)
            if not value or (isinstance(value, str) and not value.strip()):
                if is_required:
                    missing_settings.append((field_name, desc))
        
        if missing_settings:
            console.print("[yellow]⚠ Missing recommended settings:[/yellow]")
            for field_name, desc in missing_settings:
                env_var = f"LLM_BAWT_{field_name}"
                console.print(f"  [red]✗[/red] {env_var} - {desc}")
            console.print()
        
        # Show all settings grouped by category
        exclude_fields = {'defined_models', 'available_ollama_models', 'ollama_checked', 'model_config', 'SYSTEM_MESSAGE'}
        secret_fields = {'OPENAI_API_KEY', 'POSTGRES_PASSWORD', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'}
        
        console.print("[bold]All Settings:[/bold]")
        for field_name, field_info in sorted(config_obj.model_fields.items()):
            if field_name not in exclude_fields:
                current_value = getattr(config_obj, field_name)
                if isinstance(current_value, Path):
                    current_value_str = str(current_value)
                elif field_name in secret_fields and current_value:
                    # Mask secrets
                    current_value_str = "[green]****[/green] (set)"
                elif field_name in secret_fields:
                    current_value_str = "[dim]not set[/dim]"
                elif current_value is None or (isinstance(current_value, str) and not current_value.strip()):
                    current_value_str = "[dim]not set[/dim]"
                else:
                    current_value_str = repr(current_value)

                console.print(f"  [cyan]{field_name}[/cyan]: {current_value_str}")
        
        console.print(f"\n  [dim]SYSTEM_MESSAGE: (set by bot profile in database)[/dim]")
        console.print(f"\n[dim]To set a value: llm --config-set KEY value[/dim]")
        console.print(f"[dim]Or edit: {env_file_path}[/dim]")
        sys.exit(0)
    elif getattr(args, "settings_list", False):
        ok = show_runtime_settings(config_obj, scope=args.settings_scope, bot_id=args.bot)
        sys.exit(0 if ok else 1)
    elif getattr(args, "settings_set", None):
        key, value = args.settings_set
        ok = set_runtime_setting(
            config_obj,
            scope=args.settings_scope,
            bot_id=args.bot,
            key=key,
            raw_value=value,
        )
        sys.exit(0 if ok else 1)
    elif getattr(args, "settings_delete", None):
        ok = delete_runtime_setting(
            config_obj,
            scope=args.settings_scope,
            bot_id=args.bot,
            key=args.settings_delete,
        )
        sys.exit(0 if ok else 1)
    elif getattr(args, "settings_bootstrap", False):
        ok = bootstrap_runtime_settings(
            config_obj,
            bot_id=args.bot if args.settings_scope == "bot" else None,
            overwrite=bool(getattr(args, "settings_bootstrap_overwrite", False)),
        )
        sys.exit(0 if ok else 1)
    elif getattr(args, "settings_edit", False):
        from llm_bawt.cli.bot_editor import edit_global_settings

        ok = edit_global_settings(config_obj)
        sys.exit(0 if ok else 1)
    elif getattr(args, "bot_edit", None):
        from llm_bawt.cli.bot_editor import edit_bot_yaml
        ok = edit_bot_yaml(config_obj, args.bot_edit)
        sys.exit(0 if ok else 1)
    elif getattr(args, "migrate_bots", False):
        ok = migrate_bots_to_db(config_obj)
        sys.exit(0 if ok else 1)

    elif getattr(args, 'status', False):
        show_status(config_obj, args)
        sys.exit(0)

    elif getattr(args, 'job_status', False):
        show_job_status(config_obj)
        sys.exit(0)

    elif getattr(args, 'run_job', None):
        trigger_job(config_obj, args.run_job)
        sys.exit(0)

    elif getattr(args, 'list_bots', False):
        show_bots(config_obj, service_mode=_is_service_mode(args, config_obj))
        sys.exit(0)

    elif getattr(args, 'inspect_config', None):
        inspect_effective_config(config_obj, args)
        sys.exit(0)

    elif getattr(args, 'list_users', False):
        show_users(config_obj)
        sys.exit(0)

    elif getattr(args, 'user_profile', False):
        ensure_default_user(config_obj, args)
        user_id = args.user if args.user else config_obj.DEFAULT_USER
        show_user_profile(config_obj, user_id)
        sys.exit(0)
    
    elif getattr(args, 'user_profile_setup', False):
        ensure_default_user(config_obj, args)
        user_id = args.user if args.user else config_obj.DEFAULT_USER
        success = run_user_profile_setup(config_obj, user_id)
        sys.exit(0 if success else 1)
    
    elif getattr(args, 'user_profile_set', None):
        ensure_default_user(config_obj, args)
        try:
            field, value = args.user_profile_set.split("=", 1)
            field = field.strip()
            value = value.strip().strip('"').strip("'")
            user_id = args.user if args.user else config_obj.DEFAULT_USER

            # Parse field: category.key or just key
            valid_categories = {"preference", "fact", "interest", "communication", "context"}
            if "." in field:
                category_str, key = field.split(".", 1)
                if category_str.lower() not in valid_categories:
                    console.print(f"[red]Invalid category: {category_str}[/red]")
                    console.print("[dim]Valid categories: preference, fact, interest, communication, context[/dim]")
                    sys.exit(1)
                category_str = category_str.lower()
            else:
                key = field
                category_str = "fact"

            client = get_service_client(config_obj)
            if not client or not client.is_available():
                console.print("[yellow]Service not available. Start with: llm-service[/yellow]")
                sys.exit(1)

            client.upsert_profile_attribute("user", user_id, category_str, key, value)
            console.print(f"[green]Set {category_str}.{key} = {value}[/green]")
        except ValueError:
            console.print("[red]Use format: --user-profile-set category.key=value[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(0)

    elif args.list_models:
        list_models(config_obj, service_mode=_is_service_mode(args, config_obj))
        sys.exit(0)
        return
    if args.delete_model:
        service_mode = _is_service_mode(args, config_obj)
        svc_client = None
        if service_mode:
            svc_client = get_service_client(config_obj)
            if not svc_client or not svc_client.is_available():
                console.print("[bold red]Service not available.[/bold red] Start with: [yellow]llm-service[/yellow]")
                console.print("[dim]Or use --local to write directly to models.yaml.[/dim]")
                sys.exit(1)
        success = False
        try:
            success = delete_model(args.delete_model, config_obj, service_client=svc_client)
            if success:
                console.print(f"[green]Model alias '{args.delete_model}' deleted successfully.[/green]")
        except Exception as e:
            console.print(f"[bold red]Error during delete model operation:[/bold red] {e}")
            if config_obj.VERBOSE:
                traceback.print_exc()
            success = False
        sys.exit(0 if success else 1)
        return
    if args.add_bot:
        success = False
        try:
            if args.add_bot == "chat":
                success = handle_add_chat_bot(config_obj)
            elif args.add_bot == "openclaw":
                success = handle_add_openclaw(config_obj)
            if success:
                console.print(f"[green]Bot add for '{args.add_bot}' completed.[/green]")
            else:
                console.print(f"[red]Bot add for '{args.add_bot}' failed or was cancelled.[/red]")
        except KeyboardInterrupt:
            console.print("[bold red]Bot add cancelled.[/bold red]")
            success = False
        except Exception as e:
            console.print(f"[bold red]Error during bot add:[/bold red] {e}")
            if config_obj.VERBOSE:
                traceback.print_exc()
            success = False
        sys.exit(0 if success else 1)
        return
    if args.add_model:
        service_mode = _is_service_mode(args, config_obj)
        svc_client = None
        if service_mode:
            svc_client = get_service_client(config_obj)
            if not svc_client or not svc_client.is_available():
                console.print("[bold red]Service not available.[/bold red] Start with: [yellow]llm-service[/yellow]")
                console.print("[dim]Or use --local to write directly to models.yaml.[/dim]")
                sys.exit(1)
        success = False
        try:
            if args.add_model in ('openai', 'grok', 'ollama', 'codex'):
                success = update_models_interactive(config_obj, provider=args.add_model, service_client=svc_client)
            elif args.add_model == 'gguf':
                # Prompt for HuggingFace repo ID
                repo_id = Prompt.ask("Enter HuggingFace repo ID (e.g., TheBloke/Llama-2-7B-GGUF)")
                if repo_id and repo_id.strip():
                    success = handle_add_gguf(repo_id.strip(), config_obj)
                else:
                    console.print("[red]No repo ID provided. Cancelled.[/red]")
                    success = False
            elif args.add_model == 'vllm':
                success = handle_add_vllm(config_obj)
            elif args.add_model == 'openclaw':
                console.print("[yellow]`--add-model openclaw` is deprecated. Use `--add-bot openclaw`.[/yellow]")
                success = handle_add_openclaw(config_obj)
            if success:
                console.print(f"[green]Model add for '{args.add_model}' completed.[/green]")
            else:
                console.print(f"[red]Model add for '{args.add_model}' failed or was cancelled.[/red]")
        except KeyboardInterrupt:
            console.print("[bold red]Model add cancelled.[/bold red]")
            success = False # Ensure failure on interrupt
        except Exception as e:
            console.print(f"[bold red]Error during model add:[/bold red] {e}")
            if config_obj.VERBOSE:
                traceback.print_exc()
            success = False
        sys.exit(0 if success else 1)
        return

    ensure_default_user(config_obj, args)
    
    # Check if this is a history-only operation (doesn't need model)
    history_only = (args.delete_history or args.print_history is not None) and not args.question and not args.command
    
    # Check if using service mode (don't validate model locally - let service handle it)
    use_service = _is_service_mode(args, config_obj)
    
    # Determine which bot will be used (needed to get bot's default model)
    bot_manager = BotManager(config_obj, local_only=getattr(args, 'local', False))
    if args.bot:
        target_bot = bot_manager.get_bot(args.bot)
        if not target_bot:
            if use_service:
                # Service mode: don't validate bot locally, let the service resolve it.
                # Create a minimal placeholder so the rest of the CLI flow has something.
                from llm_bawt.bots import Bot
                target_bot = Bot(slug=args.bot.lower().strip(), name=args.bot, description="", system_prompt="")
            else:
                console.print(f"[bold red]Unknown bot: {args.bot}[/bold red]. Use --list-bots to see available bots.")
                sys.exit(1)
    elif args.local:
        target_bot = bot_manager.get_default_bot(local_mode=True)
    else:
        target_bot = bot_manager.get_bot(config_obj.DEFAULT_BOT) or bot_manager.get_default_bot()

    # For history-only operations, skip model resolution entirely
    if history_only:
        resolved_alias = None
    elif use_service:
        # Service mode: the service is authoritative for model resolution.
        # Only pass the user's explicit --model (or None to let service decide).
        resolved_alias = args.model or None
    else:
        # Local mode: resolve model fully using bot defaults and config
        selection = bot_manager.select_model(args.model, bot_slug=target_bot.slug, local_mode=args.local)
        effective_model = selection.alias

        # Auto-switch to service for non-OpenAI model types
        model_def = (
            config_obj.resolve_model(
                effective_model,
                harness=getattr(target_bot, "harness", None),
                default={},
            )
            if effective_model
            else {}
        )
        effective_model_type = model_def.get("type")
        if effective_model_type and effective_model_type != "openai":
            service_client = get_service_client(config_obj)
            if service_client and service_client.is_available(force_check=True):
                use_service = True
                if config_obj.VERBOSE:
                    console.print(f"[dim]Detected {effective_model_type} model; using service mode[/dim]")
                # In service mode, pass only the explicit user override
                resolved_alias = args.model or None
            else:
                console.print(
                    "[bold red]Service not available for local model. Start the service or pass --service when it is running.[/bold red]"
                )
                sys.exit(1)
        elif target_bot.agent_backend:
            resolved_alias = None  # Agent backend, no LLM model needed
        else:
            # Validate model is available locally
            model_manager = ModelManager(config_obj)
            resolved_alias = model_manager.resolve_model_alias(effective_model)
            if not resolved_alias:
                sys.exit(1)
                return
    
    try:
        run_app(args, config_obj, resolved_alias)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during application execution:[/bold red] {e}")
        if config_obj.VERBOSE:
             traceback.print_exc()
        sys.exit(1)
    else:
        sys.exit(0)

def _query_agent_backend(prompt: str, bot, plaintext_output: bool):
    """Dispatch a query to an external agent backend (e.g. OpenClaw via SSH).

    Called when a bot has ``agent_backend`` set and the service is not in use.
    """
    import asyncio
    from llm_bawt.agent_backends import get_backend
    from rich.style import Style

    backend = get_backend(bot.agent_backend)
    if not backend:
        console.print(f"[bold red]Agent backend '{bot.agent_backend}' not found.[/bold red]")
        return

    # Canonical model injection (mirrors ServiceLLMBawt._init_bot): resolve
    # the bot's default_model through the catalog and pass the SDK model_id
    # to the backend. agent_backend_config.model is migrated away.
    backend_config = dict(bot.agent_backend_config or {})
    try:
        from llm_bawt.bot_types import agent_backend_for_model_def
        from llm_bawt.utils.config import Config
        _cfg = Config()
        default_alias = getattr(bot, "default_model", None)
        if default_alias:
            model_def = _cfg.resolve_model(
                default_alias,
                harness=getattr(bot, "harness", None),
                default={},
            )
            if (
                agent_backend_for_model_def(model_def) == bot.agent_backend
                and model_def.get("model_id")
            ):
                backend_config["model"] = model_def["model_id"]
    except Exception:
        pass  # fall through; backend surfaces a clear error if model is required

    try:
        response = asyncio.run(backend.chat(prompt, backend_config))
    except Exception as e:
        console.print(f"[bold red]Agent backend error:[/bold red] {e}")
        return

    if not response:
        console.print("[yellow]Agent returned empty response.[/yellow]")
        return

    if plaintext_output:
        print(response)
    else:
        from rich.markdown import Markdown
        from rich.panel import Panel

        bot_color = getattr(bot, "color", None) or "green"
        try:
            Style.parse(bot_color)
        except Exception:
            bot_color = "green"

        title = f"[bold {bot_color}]{bot.name}[/bold {bot_color}] [dim][{bot.agent_backend}][/dim]"
        render_complete_response(
            response=response,
            console=console,
            panel_title=title,
            panel_border_style=bot_color,
        )


def run_app(args: argparse.Namespace, config_obj: Config, resolved_alias: str):
    # Determine which bot to use
    bot_manager = BotManager(config_obj, local_only=getattr(args, 'local', False))
    
    # Determine if service is likely in play (for bot validation)
    service_mode = not args.local and (getattr(args, 'service', False) or config_obj.USE_SERVICE)

    # Handle --local with explicit --bot (conflicting intent warning)
    if args.local and args.bot:
        bot = bot_manager.get_bot(args.bot)
        if not bot:
            console.print(f"[bold red]Unknown bot: {args.bot}[/bold red]. Use --list-bots to see available bots.")
            sys.exit(1)
        # Only warn if the bot requires memory (e.g., nova)
        if bot.requires_memory and config_obj.VERBOSE:
            console.print(f"[dim]Note: Using --local with --bot {args.bot}. The bot will run without database memory.[/dim]")
        bot_id = bot.slug
    elif args.bot:
        # Explicit bot selection
        bot = bot_manager.get_bot(args.bot)
        if not bot:
            if service_mode:
                # Service mode: let the service resolve the bot
                from llm_bawt.bots import Bot
                bot = Bot(slug=args.bot.lower().strip(), name=args.bot, description="", system_prompt="")
            else:
                console.print(f"[bold red]Unknown bot: {args.bot}[/bold red]. Use --list-bots to see available bots.")
                sys.exit(1)
        bot_id = bot.slug
    elif args.local:
        # Local mode defaults to Spark
        bot = bot_manager.get_default_bot(local_mode=True)
        bot_id = bot.slug
    else:
        # Default bot from config
        bot = bot_manager.get_bot(config_obj.DEFAULT_BOT)
        if not bot:
            default_slug = (config_obj.DEFAULT_BOT or "").lower().strip()
            if service_mode and default_slug:
                # No local bot registry (e.g. remote CLI with no DB creds): keep
                # the configured slug and let the service resolve the profile,
                # instead of silently coercing to a generic local "assistant".
                from llm_bawt.bots import Bot
                bot = Bot(slug=default_slug, name=config_obj.DEFAULT_BOT, description="", system_prompt="")
            else:
                bot = bot_manager.get_default_bot()
        bot_id = bot.slug
    
    # Use config default if --user not specified
    user_id = args.user if args.user else config_obj.DEFAULT_USER
    
    # Determine service mode: --local flag disables, otherwise check config/flags
    service_explicitly_set = not args.local and (getattr(args, 'service', False) or config_obj.USE_SERVICE)
    use_tools = getattr(bot, 'uses_tools', False)
    use_service = service_explicitly_set

    # Agent backend bots still go through service for history/memory logging.
    # Only if service is unavailable will they fall back to direct dispatch.
    has_agent_backend = bool(getattr(bot, 'agent_backend', None))
    if has_agent_backend and not use_service and not args.local:
        use_service = True
        if config_obj.VERBOSE:
            console.print(f"[dim]Bot '{bot.name}' uses agent backend; enabling service mode[/dim]")

    # Auto-enable service mode when bot uses tools and not in local mode
    if use_tools and not use_service and not args.local:
        use_service = True
        if config_obj.VERBOSE:
            console.print(f"[dim]Bot '{bot.name}' uses tools; enabling service mode[/dim]")

    # Check service availability; fallback rules differ based on how service mode was activated
    if use_service:
        service_client = get_service_client(config_obj)
        if not service_client or not service_client.is_available(force_check=True):
            # Agent backend bots can fall back to direct dispatch
            if has_agent_backend:
                use_service = False
                if config_obj.VERBOSE:
                    console.print("[dim]Service unavailable; agent backend will run direct (no history/memory).[/dim]")
            # Service mode was explicitly configured (env var / --service flag) or tools require it
            elif service_explicitly_set or use_tools:
                console.print("[bold red]Service not available.[/bold red] Start with: [yellow]llm-service[/yellow]")
                console.print("[dim]Or use --local for direct API calls without memory/tools.[/dim]")
                sys.exit(1)
            # Service mode was auto-detected but not explicitly required — fall back to direct OpenAI
            else:
                use_service = False
                if config_obj.VERBOSE:
                    console.print("[dim]Service unavailable; falling back to direct API mode (no memory/tools).[/dim]")

    llm_bawt = None
    
    # For history operations, we can use a lightweight path that doesn't require model init
    history_only = (args.delete_history or args.print_history is not None) and not args.question and not args.command
    
    if history_only:
        if not use_service:
            console.print("[yellow]History operations require the service. Start with: llm-service[/yellow]")
            console.print("[dim]Or use --local with a query for direct API calls without history.[/dim]")
            sys.exit(1)

        service_client = get_service_client(config_obj)
        if not service_client or not service_client.is_available():
            console.print("[bold red]Service not available.[/bold red] Start with: [yellow]llm-service[/yellow]")
            sys.exit(1)

        from llm_bawt.clients.base import StubClient
        from llm_bawt.models.message import Message
        from llm_bawt.utils.history import HistoryManager

        if args.delete_history:
            clear_resp = service_client.clear_history(bot_id=bot_id)
            if clear_resp and clear_resp.get("success"):
                console.print("[green]Chat history cleared.[/green]")
            else:
                detail = service_client.last_error or "service did not return success"
                console.print(f"[yellow]Could not clear history via service: {detail}[/yellow]")

        if args.print_history is not None:
            pair_limit = args.print_history
            if pair_limit == -1:
                messages = service_client.get_all_history(bot_id=bot_id)
                history_resp = {"messages": messages or []} if messages is not None else None
            else:
                message_limit = max(1, pair_limit * 2)
                history_resp = service_client.get_history(bot_id=bot_id, limit=message_limit)

            if history_resp is None:
                detail = service_client.last_error or "service unavailable"
                console.print(f"[yellow]Could not load history via service: {detail}[/yellow]")
            else:
                stub_client = StubClient(config=config_obj, bot_name=bot.name)
                history_manager = HistoryManager(
                    client=stub_client,
                    config=config_obj,
                    db_backend=None,
                    bot_id=bot_id,
                )
                messages = history_resp.get("messages", [])
                history_manager.messages = [
                    Message(
                        role=str(msg.get("role", "")),
                        content=str(msg.get("content", "")),
                        timestamp=float(msg.get("timestamp") or 0.0),
                        db_id=str(msg.get("id") or ""),
                    )
                    for msg in messages
                ]
                history_manager.print_history(pair_limit)

                message_ids = [str(msg.get("id")) for msg in messages if msg.get("id")]
                if message_ids:
                    tool_events = service_client.get_tool_call_events(
                        bot_id=bot_id,
                        message_ids=message_ids,
                        limit=200,
                    )
                    if tool_events and tool_events.get("events"):
                        console.print("[bold]Tool Activity:[/bold]")
                        for event in tool_events.get("events", []):
                            msg_id = str(event.get("message_id", ""))[:8]
                            tools = ", ".join(
                                str(tc.get("name", "tool"))
                                for tc in (event.get("tool_calls") or [])
                                if isinstance(tc, dict)
                            )
                            if tools:
                                console.print(f"[dim]· {msg_id}: {tools}[/dim]")
        
        console.print()
        return
    
    # For query operations, we need full LLMBawt when not using service
    # Agent backend bots skip LLMBawt entirely — they dispatch to external agents
    if not use_service and not bot.agent_backend:
        try:
            llm_bawt = LLMBawt(
                resolved_model_alias=resolved_alias,
                config=config_obj,
                local_mode=args.local,
                bot_id=bot_id,
                user_id=user_id,
                verbose=args.verbose,
                debug=args.debug,
            )
        except (ImportError, FileNotFoundError, ValueError, Exception) as e:
             console.print(f"[bold red]Failed to initialize LLM client for '{resolved_alias}':[/bold red] {e}")
             if config_obj.VERBOSE:
                 traceback.print_exc()
             sys.exit(1)
    console.print("")
    command_output_str = ""
    if args.command:
        if config_obj.VERBOSE:
            console.print(f"Executing command: [yellow]{args.command}[/yellow]", highlight=False)
        try:
            result = subprocess.run(args.command,shell=True,capture_output=True,text=True,check=False)
            output = result.stdout.strip()
            error = result.stderr.strip()
            command_prefix = f"Command `[cyan]{args.command}[/cyan]` executed.\n"
            command_output_str += command_prefix
            if output:
                command_output_str += f"\nOutput:\n```\n{output}\n```\n"
            if error:
                command_output_str += f"\nError Output:\n```\n{error}\n```\n"
            if result.returncode != 0:
                status_msg = f"\n(Command exited with status {result.returncode})"
                console.print(f"[yellow]Warning: Command exited with status {result.returncode}[/yellow]", highlight=False)
                command_output_str += status_msg
            command_output_str += "\n---\n"
        except Exception as e:
            error_msg = f"Error executing command '{args.command}': {e}"
            console.print(f"[bold red]{error_msg}[/bold red]", highlight=False)
            command_output_str += f"{error_msg}\n\n---\n"
        console.print()
    stream_flag = not config_obj.NO_STREAM
    plaintext_flag = config_obj.PLAIN_OUTPUT
    # use_tools was already set above and factored into use_service decision
    
    def do_query(prompt: str):
        """Execute query via service, agent backend, or local client."""
        nonlocal llm_bawt
        if bot.agent_backend and not use_service:
            _query_agent_backend(prompt, bot, plaintext_flag)
            return
        if use_service:
            if query_via_service(
                prompt,
                model=resolved_alias,
                bot_id=bot_id,
                user_id=user_id,
                plaintext_output=plaintext_flag,
                stream=stream_flag,
                config=config_obj,
            ):
                return
            # Service call failed
            service_client = get_service_client(config_obj)
            error_detail = ""
            if service_client and hasattr(service_client, 'last_error') and service_client.last_error:
                error_detail = f"\n[dim]Error: {service_client.last_error}[/dim]"
            console.print(
                f"[bold red]Service call failed for model '{resolved_alias}' (bot '{bot_id}').[/bold red]{error_detail}"
            )
        elif llm_bawt:
            llm_bawt.query(prompt, plaintext_output=plaintext_flag, stream=stream_flag)

    if args.question:
        question_text = command_output_str + " ".join(args.question)
        do_query(question_text.strip())
    elif command_output_str:
        if config_obj.VERBOSE:
            console.print("Command output captured, querying LLM with it...", highlight=False)
        do_query(command_output_str.strip())
    elif not sys.stdin.isatty():
        # Handle piped input - read once and exit
        piped_input = sys.stdin.read().strip()
        if piped_input:
            do_query(piped_input)
    else:
        console.print("[bold green]Entering interactive mode. Type 'exit' or 'quit' to leave.[/bold green]", highlight=False)
        console.print("[bold green]Type '>' at the beginning of a line for multiline input mode (end with Ctrl+D or Ctrl+Z).[/bold green]", highlight=False)
        handler_console = console
        if llm_bawt and hasattr(llm_bawt.client, 'console') and llm_bawt.client.console:
            handler_console = llm_bawt.client.console
        input_handler = MultilineInputHandler(console=handler_console)
        while True:
            try:
                prompt_text, is_multiline = input_handler.get_input("Enter your question:")
                if prompt_text is None or prompt_text.strip().lower() in ["exit", "quit"]:
                    console.print("[bold red]Exiting interactive mode.[/bold red]", highlight=False)
                    console.print()
                    break
                if not prompt_text.strip():
                    console.print("[dim]Empty input received. Asking again...[/dim]")
                    continue
                console.print()
                if prompt_text.strip():
                    do_query(prompt_text)
                console.print()
            except (KeyboardInterrupt, EOFError):
                console.print("[bold red]Exiting interactive mode.[/bold red]", highlight=False)
                console.print()
                break
    console.print()
