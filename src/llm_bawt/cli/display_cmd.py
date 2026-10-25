"""CLI display/query command handlers (TASK-554).

Extracted verbatim from ``cli/app.py``: service-mode query streaming, effective
config inspection, status/job/bots displays. ``app`` re-imports these names so
the ``cli.__init__`` facade (``show_status``, ``show_bots``, ...) is unchanged.
"""

import argparse
import json
import logging
import sys
import traceback
from typing import Iterator

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from llm_bawt.bot_types import format_bot_type
from llm_bawt.bots import BotManager
from llm_bawt.core import LLMBawt
from llm_bawt.service.delivery_reply import is_accepted_receipt
from llm_bawt.utils.config import Config
from llm_bawt.utils.streaming import render_complete_response, render_streaming_response

from ._common import console, get_service_client
from .tool_render import _render_tool_event


def query_via_service(
    prompt: str,
    model: str | None,
    bot_id: str | None,
    user_id: str | None,
    plaintext_output: bool,
    stream: bool,
    config: Config | None = None,
) -> bool:
    """
    Query via the background service if available.

    Returns True if query was handled by service, False if service unavailable.
    """
    from rich.markdown import Markdown
    from rich.align import Align
    from rich.style import Style

    if config is None:
        config = Config()
    client = get_service_client(config)
    if not client or not client.is_available():
        return False

    messages = [{"role": "user", "content": prompt}]

    # Get bot name for display
    bot_manager = BotManager(Config())
    bot = bot_manager.get_bot(bot_id) if bot_id else bot_manager.get_default_bot()
    bot_name = bot.name if bot else (bot_id or "Assistant")

    # Determine panel style from bot config color with safe fallback.
    configured_color = (getattr(bot, "color", None) if bot else None) or "green"
    try:
        Style.parse(configured_color)
        panel_color = configured_color
    except Exception:
        panel_color = "green"

    title_style, border_style = panel_color, panel_color

    def resolve_accepted_receipt(receipt: dict) -> str | None:
        reference = receipt.get("delivery_id") or receipt.get("turn_id") or "unknown"
        if receipt.get("object") == "chat.completion.in_flight":
            console.print(f"[yellow]Turn {escape(str(reference))} was already accepted; check its result in chat.[/yellow]")
            return None
        try:
            reply = client.wait_for_queued_reply(receipt, bot_id=bot_id, user_id=user_id)
        except Exception as exc:
            console.print(f"[yellow]Request {escape(str(reference))} was accepted; its reply could not be verified yet: {escape(str(exc))}[/yellow]")
            return None
        if reply is None:
            console.print(f"[yellow]Request {escape(str(reference))} was accepted; its reply is still pending.[/yellow]")
            return None
        if reply.get("status") != "DELIVERED":
            console.print(f"[yellow]Request {escape(str(reference))} ended with status {escape(str(reply.get('status')))}.[/yellow]")
            return None
        answer = reply.get("response_text")
        if not isinstance(answer, str):
            console.print(f"[yellow]Request {escape(str(reference))} finished without a verified text reply.[/yellow]")
            return None
        return answer

    def format_panel_title(actual_model: str | None) -> str:
        """Format panel title with actual model used."""
        # Use \[ to escape opening bracket so Rich doesn't interpret [model] as markup
        # Use dim style for model and service parts to emphasize bot name
        model_part = f" [dim]\\[{actual_model}][/dim]" if actual_model else ""
        return f"[bold {title_style}]{bot_name}[/bold {title_style}]{model_part} [dim](service)[/dim]"

    try:
        if stream:
            # Stream response using shared streaming utilities
            raw_iterator = client.chat_completion(
                messages=messages,
                model=model,
                bot_id=bot_id,
                user_id=user_id,
                stream=True,
            )

            # If service returned None (failed), fall back to local
            if raw_iterator is None:
                return False

            # Extract model from first metadata chunk, then yield content
            actual_model = model  # Default to requested model
            got_metadata = False  # Track if service responded with model info
            accepted_receipt = False

            def content_iterator() -> Iterator[str]:
                nonlocal actual_model, got_metadata, accepted_receipt
                for item in raw_iterator:
                    if is_accepted_receipt(item):
                        accepted_receipt = True
                        answer = resolve_accepted_receipt(item)
                        if answer:
                            yield answer
                    elif isinstance(item, dict) and "stream_error" in item:
                        # The service admitted the turn and then reported its
                        # failure; falling back would run the prompt twice.
                        accepted_receipt = True
                        console.print(f"[bold red]{item['stream_error']}[/bold red]")
                    elif isinstance(item, dict) and "warnings" in item:
                        # Service warnings (e.g. model fallback)
                        for warning in item["warnings"]:
                            console.print(f"[yellow]{warning}[/yellow]")
                        if item.get("model"):
                            actual_model = item["model"]
                            got_metadata = True
                    elif isinstance(item, dict) and "tool_call" in item:
                        tool_name = item.get("tool_call") or "tool"
                        tool_args = item.get("tool_args", {}) or {}
                        _render_tool_event(
                            console,
                            tool_name=tool_name,
                            tool_args=tool_args,
                            plaintext_output=plaintext_output,
                        )
                        if item.get("model"):
                            actual_model = item["model"]
                            got_metadata = True
                    elif isinstance(item, dict) and "tool_result" in item:
                        result_text = item.get("result_text")
                        if result_text:
                            tool_name = item.get("tool_result") or "tool"
                            _render_tool_event(
                                console,
                                tool_name=tool_name,
                                result_text=str(result_text),
                                plaintext_output=plaintext_output,
                            )
                        if item.get("model"):
                            actual_model = item["model"]
                            got_metadata = True
                    elif isinstance(item, dict) and "model" in item:
                        # Metadata chunk with actual model - service is responding
                        actual_model = item["model"]
                        got_metadata = True
                    elif isinstance(item, str):
                        yield item

            # We need to peek at the first item to get the model before rendering
            content_iter = content_iterator()
            first_content = None
            try:
                first_content = next(content_iter)
            except StopIteration:
                if accepted_receipt:
                    # Accepted (or admitted-then-failed) work is never an
                    # unavailable-service fallback.
                    return True
                if got_metadata:
                    # Service responded (we got the model) but produced no content.
                    # This typically means a server-side error (e.g. tool execution failed).
                    # Show error instead of silently falling back to a different model.
                    error_detail = ""
                    if client.last_error:
                        error_detail = f" {client.last_error}"
                    console.print(f"[bold red]Service returned empty response.[/bold red]{error_detail}")
                    console.print("[dim]Check service logs for details.[/dim]")
                    return True  # Handled (don't fall back to different model)
                # Service didn't respond at all - allow fallback
                return False

            # Now we have the actual_model set, create panel title
            panel_title = format_panel_title(actual_model)

            # Create a new iterator that includes the first content
            def full_content_iterator() -> Iterator[str]:
                if first_content:
                    yield first_content
                yield from content_iter

            result = render_streaming_response(
                stream_iterator=full_content_iterator(),
                console=console,
                panel_title=panel_title,
                panel_border_style=border_style,
                plaintext_output=plaintext_output,
            )
            # If streaming returned no content (service error), fall back to local
            if not result:
                return accepted_receipt
            return True
        else:
            # Non-streaming: get full response to extract actual model
            response = client.chat_completion(
                messages=messages,
                model=model,
                bot_id=bot_id,
                user_id=user_id,
                stream=False,
            )
            if is_accepted_receipt(response):
                answer = resolve_accepted_receipt(response)
                if answer:
                    if plaintext_output:
                        print(answer)
                    else:
                        render_complete_response(
                            response=answer, console=console,
                            panel_title=format_panel_title(model),
                            panel_border_style=border_style,
                        )
                return True
            if response and "choices" in response:
                content = response["choices"][0].get("message", {}).get("content")
                actual_model = response.get("model", model)  # Use actual model from response
                panel_title = format_panel_title(actual_model)

                if content:
                    if plaintext_output:
                        print(content)
                    else:
                        # Use shared render function for split display
                        render_complete_response(
                            response=content,
                            console=console,
                            panel_title=panel_title,
                            panel_border_style=border_style,
                        )
                    return True
    except Exception as e:
        # Store error in service client for later display
        service_client = get_service_client(config)
        if service_client:
            service_client.last_error = str(e)
        logging.warning(f"Service query failed: {e}")
        if config.VERBOSE:
            import traceback
            traceback.print_exc()

    return False


def inspect_effective_config(config: Config, args: argparse.Namespace) -> None:
    """Print the effective-config inspection for a bot (TASK-487).

    Agent-backend bots are never built as local LLMBawt instances (they dispatch
    to external agents), so this queries the service endpoint that constructs the
    bot the same way a live turn does. Requires a running service.
    """
    import json as _json
    import urllib.parse
    import urllib.request

    bot_id = (getattr(args, "inspect_config", "") or "").strip().lower()
    user_id = (getattr(args, "user", "") or config.DEFAULT_USER or "").strip()

    service_url = getattr(config, "SERVICE_URL", None)
    if not service_url and hasattr(config, "SERVICE_PORT"):
        host = getattr(config, "SERVICE_HOST", "localhost") or "localhost"
        service_url = f"http://{host}:{config.SERVICE_PORT}"
    if not service_url:
        console.print("[bold red]No service URL configured[/bold red] — the inspector needs a running service.")
        sys.exit(1)

    params = urllib.parse.urlencode({"bot": bot_id, "user": user_id})
    url = f"{service_url.rstrip('/')}/v1/config/effective?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        console.print(f"[bold red]Inspection failed:[/bold red] {e}")
        sys.exit(1)

    console.print_json(_json.dumps(data))


def show_status(config: Config, args: argparse.Namespace | None = None):
    """Display overall system status including dependencies, bots, memory, and configuration."""
    from llm_bawt.core.status import collect_system_status

    def make_table() -> Table:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Val")
        return table

    def _format_uptime(seconds: float) -> str:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    # ── Collect all data via shared module ────────────────────────
    bot_slug = getattr(args, "bot", None) if args else None
    explicit_model = getattr(args, "model", None) if args else None
    user_id = getattr(args, "user", None) if args else None

    is_local = getattr(args, "local", False) if args else False
    status = collect_system_status(
        config,
        bot_slug=bot_slug,
        model_alias=explicit_model,
        user_id=user_id,
        local_only=is_local,
    )

    s_cfg = status.config
    s_svc = status.service
    s_mcp = status.mcp
    s_model = status.model
    s_mem = status.memory
    s_dep = status.dependencies

    # ── Header panel ─────────────────────────────────────────────
    if s_cfg.mode == "service":
        if s_svc.available and s_svc.healthy:
            mode_text = "[green]✓ Service Mode[/green]"
        elif s_svc.available:
            mode_text = "[yellow]⚠ Service Mode[/yellow] [dim](unhealthy)[/dim]"
        else:
            mode_text = "[red]✗ Service Mode[/red] [dim](not reachable)[/dim]"
    else:
        mode_text = "[cyan]Direct Mode[/cyan]"

    env_label = "[dim](docker)[/dim]" if s_cfg.environment == "docker" else "[dim](local)[/dim]"

    header_parts = [f"[dim]v{s_cfg.version}[/dim]", mode_text]
    if s_cfg.service_url:
        header_parts.append(f"[dim]{s_cfg.service_url}[/dim]")
    header_parts.append(env_label)
    header_line = "  ".join(header_parts)
    console.print(Panel(header_line, title="[bold magenta]llm-bawt[/bold magenta]", border_style="grey39", padding=(0, 2)))
    console.print()

    # ── Config + Service (side-by-side) ─────────────────────────
    main_table = Table(show_header=False, box=None, padding=(0, 2))
    main_table.add_column("Key", style="cyan", no_wrap=True)
    main_table.add_column("Val")
    main_table.add_column("Key2", style="cyan", no_wrap=True)
    main_table.add_column("Val2")

    bot_display = f"[bold cyan]{s_cfg.bot_name}[/bold cyan] ({s_cfg.bot_slug})"

    # Model display
    if s_cfg.model_alias and s_model:
        type_suffix = f" [dim]{s_model.type}[/dim]" if s_model.type else ""
        source_suffix = f" [dim]({s_cfg.model_source})[/dim]" if s_cfg.model_source else ""
        model_display = f"[green]{s_cfg.model_alias}[/green]{type_suffix}{source_suffix}"
    elif s_cfg.model_alias:
        source_suffix = f" [dim]({s_cfg.model_source})[/dim]" if s_cfg.model_source else ""
        model_display = f"[green]{s_cfg.model_alias}[/green]{source_suffix}"
    else:
        model_display = "[dim]not set[/dim]"

    # Service status values
    if s_svc.available and s_svc.healthy:
        uptime_str = _format_uptime(s_svc.uptime_seconds) if s_svc.uptime_seconds else ""
        svc_status = "[green]✓ Up[/green]"
        if uptime_str:
            svc_status += f" [dim]({uptime_str})[/dim]"
        svc_loaded = f"[green]{s_svc.current_model}[/green]" if s_svc.current_model else "[dim]not loaded[/dim]"
        svc_default = f"[cyan]{s_svc.default_model}[/cyan]" if s_svc.default_model else "[dim]not set[/dim]"
        svc_tasks = f"{s_svc.tasks_processed} / {s_svc.tasks_pending} pending"
    elif s_svc.available:
        svc_status = "[yellow]⚠ Unhealthy[/yellow]"
        svc_loaded = "[dim]—[/dim]"
        svc_default = "[dim]—[/dim]"
        svc_tasks = "[dim]—[/dim]"
    elif s_cfg.mode == "service":
        svc_status = "[red]✗ Not reachable[/red]"
        svc_loaded = "[dim]—[/dim]"
        svc_default = "[dim]—[/dim]"
        svc_tasks = "[dim]—[/dim]"
    else:
        svc_status = "[dim]○ Not running[/dim]"
        svc_loaded = "[dim]—[/dim]"
        svc_default = "[dim]—[/dim]"
        svc_tasks = "[dim]—[/dim]"

    # Scheduler / models catalog
    scheduler_display = "[green]Enabled[/green]" if s_cfg.scheduler_enabled else "[dim]Disabled[/dim]"
    scheduler_display += f" [dim]({s_cfg.scheduler_interval}s)[/dim]"

    models_catalog = f"{s_cfg.models_defined} defined"
    if s_cfg.models_service is not None:
        models_catalog += f" [dim]({s_cfg.models_service} service)[/dim]"

    # Bots list
    bot_parts = []
    for b in s_cfg.all_bots:
        if b.is_default:
            bot_parts.append(f"[cyan]{b.slug}*[/cyan]")
        else:
            bot_parts.append(b.slug)
    bots_display = " ".join(bot_parts)

    # ── Config / Service (side-by-side) ──────────────────────────
    # Left: config settings | Right: service runtime info
    # Rows are balanced — no blank right-side cells.
    main_table.add_row("Bot", bot_display, "LLM", svc_status)
    main_table.add_row("Model", model_display, "Default", svc_default)
    main_table.add_row("Source", f"[dim]{s_cfg.model_source or 'unknown'}[/dim]", "Loaded", svc_loaded)
    main_table.add_row("User", f"{s_cfg.user_id}" if s_cfg.user_id else "[dim]not set[/dim]", "Tasks", svc_tasks)
    main_table.add_row("Bots", bots_display, "Bind", f"[dim]{s_cfg.bind_host}[/dim]")
    main_table.add_row("Models", models_catalog, "Scheduler", scheduler_display)

    console.print(Panel(main_table, title="[bold]Config[/bold] / [bold]Service[/bold]", border_style="grey39"))
    console.print()

    # ── MCP + HA (side-by-side) ───────────────────────────────────
    mcp_table = make_table()
    if s_mcp.mode == "server":
        mcp_table.add_row("Mode", "[green]Server[/green]")
        if s_mcp.status == "up":
            mcp_table.add_row("Status", "[green]✓ Up[/green]")
        elif s_mcp.status == "error":
            mcp_table.add_row("Status", f"[yellow]⚠ HTTP {s_mcp.http_status}[/yellow]")
        else:
            mcp_table.add_row("Status", "[red]✗ Down[/red]")
        mcp_table.add_row("URL", str(s_mcp.url))
    else:
        mcp_table.add_row("Mode", "[cyan]Embedded[/cyan]")
        mcp_table.add_row("Status", "[green]✓ In-process[/green]")
        mcp_table.add_row("URL", "[dim]embedded[/dim]")

    ha_table = make_table()
    if s_cfg.ha_native_mcp_url:
        ha_table.add_row("HA MCP", "[green]✓ Native MCP[/green]")
        ha_table.add_row("URL", f"[dim]{s_cfg.ha_native_mcp_url}[/dim]")
        tools_val = f"[green]{s_cfg.ha_native_mcp_tools}[/green]" if s_cfg.ha_native_mcp_tools else "[dim]0[/dim]"
        ha_table.add_row("Tools", tools_val)
    elif s_cfg.ha_mcp_enabled:
        ha_table.add_row("HA MCP", f"[green]✓ Enabled[/green]")
        ha_table.add_row("URL", f"[dim]{s_cfg.ha_mcp_url}[/dim]")
    else:
        ha_table.add_row("HA MCP", "[dim]Disabled[/dim]")

    row_mcp_ha = Table.grid(expand=True)
    row_mcp_ha.add_column(ratio=1)
    row_mcp_ha.add_column(ratio=1)
    row_mcp_ha.add_row(
        Panel(mcp_table, title="[bold]MCP Memory Server[/bold]", border_style="grey39"),
        Panel(ha_table, title="[bold]HA Integration[/bold]", border_style="grey39"),
    )
    console.print(row_mcp_ha)
    console.print()

    # ── Memory + Dependencies (side-by-side) ─────────────────────
    memory_table = make_table()

    if not s_mem.postgres_connected and s_mem.postgres_error is None and s_mem.postgres_host is None:
        memory_table.add_row("Postgres", "[yellow]Not configured[/yellow]")
    elif s_mem.postgres_connected:
        pg_host = s_mem.postgres_host or ""
        if len(pg_host) > 15:
            pg_host = pg_host[:12] + "..."
        memory_table.add_row("Postgres", f"[green]✓[/green] {pg_host}")
    else:
        memory_table.add_row("Postgres", "[red]✗ Error[/red]")

    messages_display = f"[green]{s_mem.messages_count}[/green]" if s_mem.messages_count else "[dim]0[/dim]"
    memories_display = f"[green]{s_mem.memories_count}[/green]" if s_mem.memories_count else "[dim]0[/dim]"
    memory_table.add_row("Messages", messages_display)
    memory_table.add_row("Memories", memories_display)

    if s_mem.pgvector_available:
        memory_table.add_row("pgvector", "[green]✓[/green]")
    elif s_mem.postgres_connected or s_mem.postgres_error:
        memory_table.add_row("pgvector", "[red]✗ Missing[/red]")
    else:
        memory_table.add_row("pgvector", "[dim]○ N/A[/dim]")

    if s_mem.embeddings_available:
        memory_table.add_row("Embeddings", "[green]✓[/green] [dim]MiniLM[/dim]")
    else:
        memory_table.add_row("Embeddings", "[dim]○ N/A[/dim]")

    # Dependencies
    deps_table = make_table()
    missing_deps: list[str] = []

    if s_dep.cuda_version and s_dep.cuda_version not in ("error",):
        deps_table.add_row("CUDA", f"[green]✓[/green] {s_dep.cuda_version}")
    elif s_dep.cuda_version == "error":
        deps_table.add_row("CUDA", "[yellow]⚠ error[/yellow]")
    else:
        deps_table.add_row("CUDA", "[dim]○ N/A[/dim]")

    if s_dep.llama_cpp_available:
        llama_label = "[green]✓[/green]"
        if s_dep.llama_cpp_gpu is True:
            llama_label += " (GPU)"
        elif s_dep.llama_cpp_gpu is False:
            llama_label += " [dim](CPU)[/dim]"
        deps_table.add_row("llama-cpp", llama_label)
    else:
        deps_table.add_row("llama-cpp", "[red]✗ Missing[/red]")
        missing_deps.append("[cyan]make install-extras-llama[/cyan]")

    if s_dep.hf_hub_available:
        deps_table.add_row("hf-hub", "[green]✓[/green]")
    else:
        deps_table.add_row("hf-hub", "[red]✗ Missing[/red]")
        missing_deps.append("[cyan]make install-extras-hf[/cyan]")

    if s_dep.torch_available:
        deps_table.add_row("torch", "[green]✓[/green]")
    else:
        deps_table.add_row("torch", "[dim]○ Optional[/dim]")

    if s_dep.openai_key_set:
        deps_table.add_row("OpenAI", "[green]✓ Key set[/green]")
    else:
        deps_table.add_row("OpenAI", "[dim]○ No key[/dim]")

    if s_dep.newsapi_key_set:
        deps_table.add_row("NewsAPI", "[green]✓ Key set[/green]")
        deps_table.add_row("News Tool", "[green]✓ Enabled[/green]")
    else:
        deps_table.add_row("NewsAPI", "[dim]○ No key[/dim]")
        deps_table.add_row("News Tool", "[yellow]⚠ Disabled[/yellow] [dim](set LLM_BAWT_NEWSAPI_API_KEY)[/dim]")

    if s_dep.search_provider:
        deps_table.add_row("Search", f"[green]✓[/green] {s_dep.search_provider}")
    else:
        deps_table.add_row("Search", "[dim]○ None[/dim]")

    row2 = Table.grid(expand=True)
    row2.add_column(ratio=1)
    row2.add_column(ratio=1)
    row2.add_row(
        Panel(memory_table, title="[bold]Memory[/bold]", border_style="grey39"),
        Panel(deps_table, title="[bold]Dependencies[/bold]", border_style="grey39"),
    )
    console.print(row2)

    console.print()
    console.print(
        "[dim]Useful:[/dim] "
        "[cyan]--job-status[/cyan]  "
        "[cyan]--list-models[/cyan]  "
        "[cyan]--list-bots[/cyan]  "
        "[cyan]--list-users[/cyan]  "
        "[cyan]--user-profile[/cyan]  "
        "[cyan]--list-config[/cyan]"
    )

    if missing_deps:
        console.print()
        console.print(f"  [dim]Install missing:[/dim] {', '.join(missing_deps)}")


def show_bots(config: Config, service_mode: bool = False):
    """Display available bots.

    In service mode, queries the running service. Otherwise uses local YAML only.
    """
    if service_mode:
        client = get_service_client(config)
        if client and client.is_available():
            service_bots = client.list_bots()
            if service_bots is not None:
                _show_bots_from_service(config, service_bots)
                return
        console.print("[yellow]Service not reachable — showing local bots.[/yellow]")

    _show_bots_from_local(config)


def _show_bots_from_service(config: Config, bots_data: list[dict]):
    """Render bot list from service API response."""
    default_slug = config.DEFAULT_BOT or "nova"

    console.print(Panel.fit("[bold cyan]Available Bots[/bold cyan] [dim](service)[/dim]", border_style="cyan"))
    console.print()

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Slug", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Color")
    table.add_column("Description")
    table.add_column("Default Model")
    table.add_column("Memory", justify="center")

    for bot in sorted(bots_data, key=lambda b: b.get("slug", "")):
        slug = bot.get("slug", "")
        is_default = " ⭐" if slug == default_slug else ""
        requires_memory = bot.get("requires_memory", False)
        memory_icon = "[green]✓[/green]" if requires_memory else "[dim]✗[/dim]"
        default_model = bot.get("default_model") or f"[dim]{config.DEFAULT_MODEL_ALIAS or 'global'}[/dim]"
        color = bot.get("color")
        table.add_row(
            f"{slug}{is_default}",
            bot.get("name", slug),
            format_bot_type(bot.get("bot_type"), bot.get("agent_backend")),
            f"[{color}]{color}[/]" if color else "[dim]default[/dim]",
            bot.get("description", ""),
            default_model,
            memory_icon,
        )

    console.print(table)
    console.print()
    console.print(f"[dim]⭐ = default bot | ✓ = requires database | Use -b/--bot <slug> to select[/dim]")
    console.print()


def _show_bots_from_local(config: Config):
    """Render bot list from local BotManager (YAML only, no DB)."""
    bot_manager = BotManager(config, local_only=True)
    bots = bot_manager.list_bots()
    default_bot = bot_manager.get_default_bot()

    console.print(Panel.fit("[bold cyan]Available Bots[/bold cyan] [dim](local)[/dim]", border_style="cyan"))
    console.print()

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Slug", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Color")
    table.add_column("Description")
    table.add_column("Default Model")
    table.add_column("Memory", justify="center")

    for bot in bots:
        is_default = " ⭐" if bot.slug == default_bot.slug else ""
        memory_icon = "[green]✓[/green]" if bot.requires_memory else "[dim]✗[/dim]"
        selection = bot_manager.select_model(None, bot_slug=bot.slug)
        default_model = selection.alias or f"[dim]{config.DEFAULT_MODEL_ALIAS or 'global'}[/dim]"
        table.add_row(
            f"{bot.slug}{is_default}",
            bot.name,
            format_bot_type(bot.bot_type, bot.agent_backend),
            f"[{bot.color}]{bot.color}[/]" if bot.color else "[dim]default[/dim]",
            bot.description,
            default_model,
            memory_icon
        )

    console.print(table)
    console.print()
    console.print(f"[dim]⭐ = default bot | ✓ = requires database | Use -b/--bot <slug> to select[/dim]")
    console.print()


def _render_job_status_from_service(config: Config) -> bool:
    """Render job status via the service API. Returns True if successful."""
    client = get_service_client(config)
    if not client or not client.is_available():
        return False

    from datetime import datetime, timezone

    jobs_resp = client.get_jobs()
    runs_resp = client.get_job_runs(limit=10)
    if jobs_resp is None and runs_resp is None:
        return False

    jobs = (jobs_resp or {}).get("jobs", [])
    runs = (runs_resp or {}).get("runs", [])

    console.print(Panel.fit("[bold cyan]Scheduled Jobs[/bold cyan]", border_style="cyan"))
    console.print()

    if not jobs:
        console.print("[dim]No scheduled jobs found. Start the service to initialize default jobs.[/dim]")
    else:
        job_table = Table(show_header=True, box=None, padding=(0, 2))
        job_table.add_column("Job Type", style="cyan")
        job_table.add_column("Bot", style="bold")
        job_table.add_column("Enabled", justify="center")
        job_table.add_column("Interval")
        job_table.add_column("Last Run")
        job_table.add_column("Next Run")

        now = datetime.now(timezone.utc)
        for job in jobs:
            enabled_icon = "[green]✓[/green]" if job.get("enabled") else "[red]✗[/red]"
            interval_str = f"{job.get('interval_minutes', 0)}m"

            last_run_str = "[dim]never[/dim]"
            if job.get("last_run_at"):
                try:
                    last_run = datetime.fromisoformat(job["last_run_at"].replace("Z", "+00:00")).replace(tzinfo=None)
                    ago = now - last_run
                    if ago.total_seconds() < 3600:
                        last_run_str = f"{int(ago.total_seconds() // 60)}m ago"
                    elif ago.total_seconds() < 86400:
                        last_run_str = f"{int(ago.total_seconds() // 3600)}h ago"
                    else:
                        last_run_str = f"{int(ago.total_seconds() // 86400)}d ago"
                except Exception:
                    last_run_str = "[dim]?[/dim]"

            next_run_str = "[dim]pending[/dim]"
            if job.get("next_run_at"):
                try:
                    next_run = datetime.fromisoformat(job["next_run_at"].replace("Z", "+00:00")).replace(tzinfo=None)
                    until = next_run - now
                    if until.total_seconds() < 0:
                        next_run_str = "[yellow]overdue[/yellow]"
                    elif until.total_seconds() < 3600:
                        next_run_str = f"in {int(until.total_seconds() // 60)}m"
                    else:
                        next_run_str = f"in {int(until.total_seconds() // 3600)}h"
                except Exception:
                    next_run_str = "[dim]?[/dim]"

            job_table.add_row(
                job.get("job_type", "?"),
                job.get("bot_id", "?"),
                enabled_icon,
                interval_str,
                last_run_str,
                next_run_str,
            )
        console.print(job_table)

    console.print()
    console.print(Panel.fit("[bold cyan]Recent Job Runs[/bold cyan]", border_style="cyan"))
    console.print()

    if not runs:
        console.print("[dim]No job runs recorded yet.[/dim]")
    else:
        run_table = Table(show_header=True, box=None, padding=(0, 2))
        run_table.add_column("Started", style="dim")
        run_table.add_column("Bot", style="bold")
        run_table.add_column("Status", justify="center")
        run_table.add_column("Duration")
        run_table.add_column("Error")

        for run in runs:
            status_val = run.get("status", "")
            if status_val == "success":
                status_str = "[green]✓ success[/green]"
            elif status_val == "failed":
                status_str = "[red]✗ failed[/red]"
            elif status_val == "running":
                status_str = "[yellow]⟳ running[/yellow]"
            elif status_val == "skipped":
                status_str = "[dim]○ skipped[/dim]"
            else:
                status_str = "[dim]○ pending[/dim]"

            duration_ms = run.get("duration_ms")
            duration_str = f"{duration_ms}ms" if duration_ms else "[dim]-[/dim]"
            error_msg = run.get("error_message") or ""
            error_str = (error_msg[:40] + "...") if len(error_msg) > 40 else (error_msg or "[dim]-[/dim]")
            started_str = "[dim]-[/dim]"
            if run.get("started_at"):
                try:
                    started_str = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass

            run_table.add_row(started_str, run.get("bot_id", "?"), status_str, duration_str, error_str)
        console.print(run_table)

    console.print()
    console.print(f"[dim]Scheduler enabled: {config.SCHEDULER_ENABLED} | Check interval: {config.SCHEDULER_CHECK_INTERVAL_SECONDS}s[/dim]")
    console.print()
    return True


def show_job_status(config: Config):
    """Display scheduled background jobs and recent run history via service."""
    client = get_service_client(config)
    if not client or not client.is_available():
        console.print("[yellow]Service not available. Start with: llm-service[/yellow]")
        return

    if not _render_job_status_from_service(config):
        console.print("[red]Failed to load job status from service.[/red]")


def trigger_job(config: Config, job_type: str):
    """Trigger a scheduled job to run immediately via service."""
    client = get_service_client(config)
    if not client or not client.is_available():
        console.print("[yellow]Service not available. Start with: llm-service[/yellow]")
        return

    try:
        from llm_bawt.service.scheduler import JobType
        valid_types = {jt.value for jt in JobType}
    except ImportError:
        valid_types = {"profile_maintenance", "memory_consolidation", "memory_decay", "history_summarization"}
    if job_type not in valid_types:
        console.print(f"[red]Unknown job type: {job_type}[/red]")
        console.print(f"[dim]Valid types: {', '.join(sorted(valid_types))}[/dim]")
        return

    result = client.trigger_job(job_type)
    if result and result.get("success"):
        console.print(f"[green]✓ Triggered job: {job_type}[/green]")
        console.print(f"[dim]The scheduler will pick it up within {config.SCHEDULER_CHECK_INTERVAL_SECONDS} seconds.[/dim]")
        console.print(f"[dim]Run 'llm --job-status' to check the result.[/dim]")
    else:
        error = client.last_error if client else "unknown error"
        console.print(f"[red]Failed to trigger job: {error}[/red]")


