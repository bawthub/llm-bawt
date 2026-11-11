#!/usr/bin/env python3
"""
Memory management CLI tool for llm-bawt.
Thin client that proxies all operations through the llm-service.

Two types of data are managed:
  1. MEMORIES - Extracted facts with importance scores and embeddings
  2. MESSAGES - Raw conversation history (message logs)

Usage:
    # === MEMORIES (extracted facts) ===
    llm-memory "what do you know about me"       # Search memories
    llm-memory -m embedding "user"               # Search via embeddings only
    llm-memory --list-memories                   # List memories by importance
    llm-memory --delete-memory <ID>              # Delete a memory by UUID prefix
    llm-memory --consolidate                     # Merge redundant memories
    llm-memory --regenerate-embeddings           # Regenerate all embeddings

    # === ENTITY PROFILES (user or bot) ===
    llm-memory --list-profiles                   # List all profiles (users and bots)
    llm-memory --list-attrs alex                 # List attributes (auto-detects user/bot)
    llm-memory --list-attrs helper               # List attributes for helper
    llm-memory --delete-attr 42                  # Delete a profile attribute

    # === MESSAGE HISTORY (conversation logs) ===
    llm-memory --msg                             # Show recent messages
    llm-memory --msg-search "topic"              # Search message history
    llm-memory --msg-forget 5                    # Soft-delete last 5 messages
    llm-memory --msg-forget-since 30             # Soft-delete from last 30 min
    llm-memory --msg-forget-id <ID>              # Soft-delete specific message
    llm-memory --msg-restore                     # Restore soft-deleted messages
    llm-memory --msg-summarize                   # Summarize old sessions
    llm-memory --msg-rebuild-summaries 10       # Rebuild last 10 session summaries
    llm-memory --msg-summarize-preview           # Preview summarization
    llm-memory --msg-summaries                   # List existing summaries
    llm-memory --msg-delete-summary <ID>         # Delete a summary

    # === GENERAL ===
    llm-memory --stats                           # Show combined statistics
"""

import argparse
import sys
from datetime import datetime

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from llm_bawt.utils.config import config

console = Console()

# Timeout settings
DEFAULT_TIMEOUT = 30.0
LONG_TIMEOUT = 120.0  # For operations like consolidation


def _confirm(prompt: str) -> bool:
    """Prompt for y/N confirmation. Returns False if stdin is not a TTY (non-interactive)."""
    try:
        answer = console.input(prompt).strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Non-interactive stdin — treating as 'N' (cancelled)[/dim]")
        return False


def get_service_url() -> str:
    """Get the service base URL from config."""
    host = config.SERVICE_HOST or "localhost"
    port = config.SERVICE_PORT  # Default is 8642 from config
    return f"http://{host}:{port}"


def check_service_available() -> bool:
    """Check if the llm-service is running."""
    try:
        response = httpx.get(f"{get_service_url()}/health", timeout=2.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


def api_get(endpoint: str, params: dict | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """Make a GET request to the service API."""
    try:
        response = httpx.get(
            f"{get_service_url()}{endpoint}",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        return None
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        return None


def api_post(endpoint: str, json_data: dict | None = None, params: dict | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """Make a POST request to the service API."""
    try:
        response = httpx.post(
            f"{get_service_url()}{endpoint}",
            json=json_data or {},
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        return None
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        return None


def api_delete(endpoint: str, params: dict | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    """Make a DELETE request to the service API."""
    try:
        response = httpx.delete(
            f"{get_service_url()}{endpoint}",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        return None
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        return None


def display_messages_preview(messages: list[dict], max_content_length: int = 80):
    """Display a preview of messages for confirmation prompts."""
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Time", style="dim", width=19)
    table.add_column("Role", width=10)
    table.add_column("Content", overflow="fold")

    for msg in messages:
        ts = msg.get("timestamp")
        time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "?"

        role = msg.get("role", "?")
        role_style = "cyan" if role == "user" else "green" if role == "assistant" else "yellow"

        content = msg.get("content", "")
        if not isinstance(content, str):
            import json
            content = json.dumps(content, indent=2, ensure_ascii=False)

        table.add_row(time_str, f"[{role_style}]{role}[/{role_style}]", content)

    console.print(table)


def display_results(results: list, method: str, show_ids: bool = True):
    """Display memory results in a table."""
    table = Table()
    if show_ids:
        table.add_column("ID", style="dim")
    table.add_column("Rel", justify="right", style="cyan", width=5)
    table.add_column("Imp", justify="right", style="yellow", width=4)
    table.add_column("Tags", style="magenta")
    table.add_column("Content", style="white", overflow="fold")

    for r in results:
        content = r.get("content", "")
        if not isinstance(content, str):
            import json
            content = json.dumps(content, indent=2, ensure_ascii=False)
        relevance = r.get("relevance") or r.get("similarity", 0) or 0
        importance = r.get("importance", 0.5)
        tags = r.get("tags", ["misc"])
        if isinstance(tags, str):
            import json
            tags = json.loads(tags) if tags else ["misc"]

        memory_id = r.get("id", "?")

        row = []
        if show_ids:
            row.append(memory_id)
        row.extend([
            f"{relevance:.3f}",
            f"{importance:.2f}",
            ", ".join(str(tag) for tag in tags) if tags else "misc",
            content,
        ])
        table.add_row(*row)

    console.print(table)
    console.print(f"[dim]Found {len(results)} results via {method}[/dim]")
    if show_ids:
        console.print("[dim]Use --delete-memory <ID> to remove a memory[/dim]\n")


# =============================================================================
# Memory Operations (extracted facts)
# =============================================================================

def show_stats(bot_id: str):
    """Show memory and message statistics."""
    data = api_get("/v1/memory/stats", {"bot_id": bot_id})
    if not data:
        return

    messages = data.get("messages", {})
    memories = data.get("memories", {})

    mem_total = memories.get("total_count", memories.get("total", 0))
    msg_total = messages.get("total_count", messages.get("total", 0))
    msg_forgotten = messages.get("forgotten_count", messages.get("ignored", 0))

    # Get summary count
    summary_count = 0
    try:
        summ_data = api_get("/v1/history/summaries", {"bot_id": bot_id})
        if summ_data:
            summary_count = summ_data.get("total_count", 0)
    except Exception:
        pass

    console.print(Panel(f"""
[bold]Memories:[/bold] {mem_total} extracted facts
[bold]Messages:[/bold] {msg_total} total ({msg_forgotten} forgotten, {summary_count} summaries)
""", title=f"Statistics - {data.get('bot_id', bot_id)}"))

    # Tag distribution (if available)
    tag_counts = memories.get("tag_counts", {})
    if tag_counts:
        table = Table(title="Memory Tag Distribution")
        table.add_column("Tag", style="cyan")
        table.add_column("Count", justify="right")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            table.add_row(tag, str(count))
        console.print(table)

    # Importance distribution (if available)
    importance_dist = memories.get("importance_distribution", {})
    if importance_dist:
        table = Table(title="Memory Importance Distribution")
        table.add_column("Range", style="cyan")
        table.add_column("Count", justify="right")
        for range_name, count in sorted(importance_dist.items(), reverse=True):
            table.add_row(range_name, str(count))
        console.print(table)


def list_memories(bot_id: str, limit: int):
    """List memories ordered by importance."""
    data = api_get("/v1/memory", {"bot_id": bot_id, "limit": limit})
    if not data:
        return

    results = data.get("results", [])
    if not results:
        console.print("[yellow]No memories found.[/yellow]")
        return

    table = Table(title=f"Top {limit} Memories by Importance")
    table.add_column("ID", style="dim")
    table.add_column("Imp", justify="right", style="cyan", width=4)
    table.add_column("Tags", style="magenta")
    table.add_column("Content", style="white", overflow="fold")
    table.add_column("Acc", justify="right", width=3)

    for mem in results:
        content = mem.get("content", "")
        if not isinstance(content, str):
            import json
            content = json.dumps(content, indent=2, ensure_ascii=False)
        tags = mem.get("tags", ["misc"])
        memory_id = mem.get("id", "?")
        table.add_row(
            memory_id,
            f"{mem.get('importance', 0.5):.2f}",
            ", ".join(str(tag) for tag in tags),
            content,
            str(mem.get("access_count", 0)),
        )

    console.print(table)
    console.print("[dim]Use --delete-memory <ID> to remove a memory[/dim]")


def search_memories(bot_id: str, query: str, method: str, limit: int, min_importance: float):
    """Search memories using specified method."""
    console.print(f"[bold]Query:[/bold] \"{query}\"\n")

    if method == "all":
        # Show results from multiple methods
        for m in ["embedding", "high-importance"]:
            console.print(f"[bold cyan]{m.title()} Search[/bold cyan]")
            data = api_post("/v1/memory/search", {
                "query": query,
                "method": m,
                "limit": limit,
                "min_importance": min_importance if m == "high-importance" else 0.0,
                "bot_id": bot_id,
            })
            if data and data.get("results"):
                display_results(data["results"], m)
            else:
                console.print("[dim]No results[/dim]\n")
    else:
        data = api_post("/v1/memory/search", {
            "query": query,
            "method": method,
            "limit": limit,
            "min_importance": min_importance,
            "bot_id": bot_id,
        })
        if data and data.get("results"):
            display_results(data["results"], method)
        else:
            console.print("[dim]No results[/dim]")


def handle_delete_memory(bot_id: str, memory_id: str, skip_confirm: bool):
    """Delete a specific memory by ID."""
    # First, try to find the memory to show what will be deleted
    data = api_post("/v1/memory/search", {
        "query": "",
        "method": "embedding",
        "limit": 100,
        "bot_id": bot_id,
    })

    matching = None
    if data and data.get("results"):
        for mem in data["results"]:
            if mem.get("id", "").startswith(memory_id):
                matching = mem
                break

    if not matching:
        # Try listing all memories
        data = api_get("/v1/memory", {"bot_id": bot_id, "limit": 100})
        if data and data.get("results"):
            for mem in data["results"]:
                if mem.get("id", "").startswith(memory_id):
                    matching = mem
                    break

    if not matching:
        console.print(f"[red]Memory '{memory_id}' not found[/red]")
        return

    full_id = matching.get("id", memory_id)
    content = matching.get("content", "")
    if not isinstance(content, str):
        import json
        content = json.dumps(content, indent=2, ensure_ascii=False)
    tags = matching.get("tags", [])

    console.print("[bold]Memory to delete:[/bold]")
    console.print(f"  ID: [dim]{full_id}[/dim]")
    console.print(f"  Tags: [magenta]{', '.join(tags) if tags else 'none'}[/magenta]")
    console.print(f"  Content: {content}")
    console.print()

    if not skip_confirm:
        if not _confirm("[bold yellow]Delete this memory? (y/N):[/bold yellow] "):
            console.print("[dim]Cancelled[/dim]")
            return

    result = api_delete(f"/v1/memory/{full_id}", {"bot_id": bot_id})
    if result and result.get("success"):
        console.print("[green]Memory deleted[/green]")
    else:
        console.print("[red]Failed to delete memory[/red]")


def handle_regenerate_embeddings(bot_id: str):
    """Regenerate embeddings for all memories."""
    console.print(f"[bold]Regenerating embeddings for bot: {bot_id}[/bold]\n")

    with console.status("[bold green]Generating embeddings..."):
        result = api_post("/v1/memory/regenerate-embeddings", params={"bot_id": bot_id}, timeout=LONG_TIMEOUT)
    console.print()  # Newline after spinner clears

    if not result:
        return

    if result.get("success"):
        console.print(f"[green]Updated: {result.get('updated', 0)}[/green]")
        if result.get("failed", 0) > 0:
            console.print(f"[yellow]Failed: {result.get('failed', 0)}[/yellow]")
        if result.get("embedding_dim"):
            console.print(f"  Embedding dimension: {result['embedding_dim']}")
    else:
        console.print(f"[red]Error: {result.get('message', 'Unknown error')}[/red]")
        console.print("Install with: [cyan]pipx runpip llm-bawt install sentence-transformers[/cyan]")


def handle_consolidate(bot_id: str, dry_run: bool = False):
    """Find and merge redundant memories."""
    mode_str = "[yellow](DRY RUN)[/yellow] " if dry_run else ""
    console.print(f"[bold]{mode_str}Memory Consolidation for bot: {bot_id}[/bold]\n")

    with console.status("[bold green]Finding and consolidating similar memories..."):
        result = api_post(
            "/v1/memory/consolidate",
            {"dry_run": dry_run},
            {"bot_id": bot_id},
            timeout=LONG_TIMEOUT,
        )
    console.print()  # Newline after spinner clears

    if not result:
        return

    console.print()
    if dry_run:
        console.print("[yellow]Would merge:[/yellow]")
    else:
        console.print("[green]Results:[/green]")

    console.print(f"  Clusters found: {result.get('clusters_found', 0)}")
    console.print(f"  Clusters merged: {result.get('clusters_merged', 0)}")
    console.print(f"  Memories consolidated: {result.get('memories_consolidated', 0)}")
    if not dry_run:
        console.print(f"  New memories created: {result.get('new_memories_created', 0)}")

    errors = result.get("errors", [])
    if errors:
        console.print(f"\n[yellow]Errors ({len(errors)}):[/yellow]")
        for err in errors:
            console.print(f"  - {err}")


# =============================================================================
# User Profile Operations
# =============================================================================

def handle_list_attrs(entity_id: str):
    """List profile attributes with IDs for any entity (user or bot).

    Entity type is auto-detected - just provide the entity ID.

    Args:
        entity_id: The entity's ID (user_id or bot_id)
    """
    data = api_get(f"/v1/profiles/{entity_id}")

    if not data:
        console.print(f"[red]Failed to get profile for '{entity_id}'[/red]")
        return

    # Show profile summary if present
    display_name = data.get("display_name")
    summary = data.get("summary")
    entity_type = data.get("entity_type", "unknown")
    entity_label = f"{entity_type.capitalize()}: {entity_id}"

    if display_name or summary:
        console.print(Panel.fit(f"[bold cyan]Profile: {entity_label}[/bold cyan]", border_style="cyan"))
        if display_name:
            console.print(f"[bold]Name:[/bold] {display_name}")
        if summary:
            console.print("\n[bold]Summary:[/bold]")
            console.print(summary)
        console.print()

    attributes = data.get("attributes", [])
    if not attributes:
        if not summary:
            console.print(f"[yellow]No attributes found for {entity_type} '{entity_id}'[/yellow]")
        else:
            console.print("[dim]Individual attributes consolidated into summary above.[/dim]")
        return

    table = Table(title=f"Profile Attributes for {entity_label}")
    table.add_column("ID", style="cyan", justify="right", width=6)
    table.add_column("Category", style="magenta", width=15)
    table.add_column("Key", style="yellow", width=25)
    table.add_column("Value", style="white", overflow="fold")
    table.add_column("Conf", justify="right", width=4)

    for attr in attributes:
        value = attr.get("value", "")
        if not isinstance(value, str):
            import json
            value = json.dumps(value, indent=2, ensure_ascii=False)
        table.add_row(
            str(attr.get("id", "?")),
            attr.get("category", "?"),
            attr.get("key", "?"),
            value,
            f"{attr.get('confidence', 1.0):.1f}",
        )

    console.print(table)
    console.print("\n[dim]Use --delete-attr <ID> to remove an attribute[/dim]")


def handle_list_profiles():
    """List all profiles (both users and bots) with summary information."""
    # Fetch both user and bot profiles
    user_profiles_data = api_get("/v1/profiles", {"entity_type": "user"})
    bot_profiles_data = api_get("/v1/profiles", {"entity_type": "bot"})

    # Collect all profiles
    all_profiles = []

    # Process user profiles
    if user_profiles_data and user_profiles_data.get("profiles"):
        for profile in user_profiles_data["profiles"]:
            attrs = profile.get("attributes", [])
            categories = list(set(a.get("category", "?") for a in attrs)) if attrs else []
            all_profiles.append({
                "type": "user",
                "id": profile.get("entity_id", "?"),
                "name": profile.get("display_name", "-"),
                "attr_count": len(attrs),
                "categories": categories,
            })

    # Process bot profiles
    if bot_profiles_data and bot_profiles_data.get("profiles"):
        for profile in bot_profiles_data["profiles"]:
            attrs = profile.get("attributes", [])
            categories = list(set(a.get("category", "?") for a in attrs)) if attrs else []
            all_profiles.append({
                "type": "bot",
                "id": profile.get("entity_id", "?"),
                "name": profile.get("display_name", "-"),
                "attr_count": len(attrs),
                "categories": categories,
            })

    # Check if we got any data
    if not all_profiles:
        if not user_profiles_data and not bot_profiles_data:
            console.print("[red]Failed to fetch profile data from both endpoints[/red]")
        else:
            console.print("[yellow]No profiles found[/yellow]")
        return

    # Sort by type then ID
    all_profiles.sort(key=lambda p: (p["type"], p["id"]))

    # Display table
    table = Table(title="All Entity Profiles")
    table.add_column("Type", style="cyan", width=6)
    table.add_column("ID", style="yellow")
    table.add_column("Name", style="white")
    table.add_column("Attrs", justify="right", width=6)
    table.add_column("Categories", style="magenta", overflow="fold")

    for profile in all_profiles:
        # Format categories
        categories = profile.get("categories", [])
        if categories:
            cat_str = ", ".join(str(category) for category in categories)
        else:
            cat_str = "-"

        # Format name
        name = profile.get("name", "-")

        table.add_row(
            profile["type"],
            profile["id"],
            name or "-",
            str(profile.get("attr_count", 0)),
            cat_str,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(all_profiles)} profiles ({sum(1 for p in all_profiles if p['type'] == 'user')} users, {sum(1 for p in all_profiles if p['type'] == 'bot')} bots)[/dim]")
    console.print("[dim]Use --list-attrs <entity_id> to view detailed attributes[/dim]")


def handle_delete_attribute(attribute_id: int, skip_confirm: bool):
    """Delete a user profile attribute by ID."""
    result = api_delete(f"/v1/profiles/attribute/{attribute_id}")

    if not result:
        console.print(f"[red]Failed to delete attribute {attribute_id}[/red]")
        return

    if result.get("success"):
        deleted = result.get("deleted", {})
        console.print(f"[green]Deleted:[/green] {deleted.get('category', '?')}.{deleted.get('key', '?')} = {deleted.get('value', '?')}")
        console.print(f"[dim]  From: {deleted.get('entity_type', '?')}/{deleted.get('entity_id', '?')}[/dim]")
    else:
        console.print(f"[red]Failed to delete attribute: {result.get('detail', 'Unknown error')}[/red]")


# =============================================================================
# Message History Operations
# =============================================================================

def handle_show_messages(bot_id: str, limit: int):
    """Show recent conversation messages."""
    data = api_get("/v1/history", {"bot_id": bot_id, "limit": limit})

    if not data:
        console.print(f"[red]Failed to get messages for bot '{bot_id}'[/red]")
        return

    messages = data.get("messages", [])
    if not messages:
        console.print(f"[yellow]No messages found for bot '{bot_id}'[/yellow]")
        return

    console.print(Panel.fit(f"[bold]Message History[/bold] ({len(messages)} messages)", border_style="cyan"))
    for index, msg in enumerate(messages, start=1):
        ts = msg.get("timestamp", 0)
        time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
        role = msg.get("role", "?")
        role_style = "cyan" if role == "user" else "green" if role == "assistant" else "magenta" if role == "summary" else "yellow"
        msg_id = msg.get("id", "?")
        content = msg.get("content", "")

        if not isinstance(content, str):
            import json
            content = json.dumps(content, indent=2, ensure_ascii=False)

        console.print(
            Panel(
                content,
                title=f"{index}. [{role_style}]{role}[/{role_style}] • {time_str}",
                subtitle=f"ID: {msg_id}",
                border_style=role_style,
                expand=True,
            )
        )

    console.print("\n[dim]Use --msg-forget-id <ID> to soft-delete a message[/dim]")


def handle_search_messages(bot_id: str, query: str, limit: int):
    """Search message history."""
    data = api_post("/v1/history/search", params={"bot_id": bot_id, "query": query, "limit": limit})

    if not data:
        console.print(f"[red]Failed to search messages for bot '{bot_id}'[/red]")
        return

    messages = data.get("messages", [])
    if not messages:
        console.print(f"[yellow]No messages found matching '{query}'[/yellow]")
        return

    console.print(Panel.fit(f"[bold]Message Search[/bold] '{query}' ({len(messages)} matches)", border_style="cyan"))
    for index, msg in enumerate(messages, start=1):
        ts = msg.get("timestamp", 0)
        time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
        role = msg.get("role", "?")
        role_style = "cyan" if role == "user" else "green" if role == "assistant" else "magenta" if role == "summary" else "yellow"
        msg_id = msg.get("id", "?")
        content = msg.get("content", "")

        if not isinstance(content, str):
            import json
            content = json.dumps(content, indent=2, ensure_ascii=False)

        console.print(
            Panel(
                content,
                title=f"{index}. [{role_style}]{role}[/{role_style}] • {time_str}",
                subtitle=f"ID: {msg_id}",
                border_style=role_style,
                expand=True,
            )
        )

    console.print("\n[dim]Use --msg-forget-id <ID> to soft-delete a message[/dim]")


def handle_msg_forget(bot_id: str, count: int, skip_confirm: bool):
    """Soft-delete the last N messages."""
    # Preview messages first
    data = api_get("/v1/memory/preview/recent", {"bot_id": bot_id, "count": count})
    if not data or not data.get("messages"):
        console.print("[yellow]No messages to forget[/yellow]")
        return

    messages = data["messages"]
    console.print(f"[bold]Messages to forget ({len(messages)}):[/bold]\n")
    display_messages_preview(messages)

    if skip_confirm or _confirm("\n[bold yellow]Forget these messages? (y/N):[/bold yellow] "):
        result = api_post("/v1/memory/forget", {"count": count}, {"bot_id": bot_id})
        if result:
            console.print(f"[green]{result.get('message', 'Done')}[/green]")
            console.print("[dim]Use --msg-restore to undo[/dim]")
    else:
        console.print("[dim]Cancelled[/dim]")


def handle_msg_forget_since(bot_id: str, minutes: int, skip_confirm: bool):
    """Soft-delete messages from the last N minutes."""
    # Preview messages first
    data = api_get("/v1/memory/preview/minutes", {"bot_id": bot_id, "minutes": minutes})
    if not data or not data.get("messages"):
        console.print("[yellow]No messages in that time range[/yellow]")
        return

    messages = data["messages"]
    console.print(f"[bold]Messages from last {minutes} minutes to forget ({len(messages)}):[/bold]\n")
    display_messages_preview(messages)

    if skip_confirm or _confirm("\n[bold yellow]Forget these messages? (y/N):[/bold yellow] "):
        result = api_post("/v1/memory/forget", {"minutes": minutes}, {"bot_id": bot_id})
        if result:
            console.print(f"[green]{result.get('message', 'Done')}[/green]")
            console.print("[dim]Use --msg-restore to undo[/dim]")
    else:
        console.print("[dim]Cancelled[/dim]")


def handle_msg_get(bot_id: str, message_id: str):
    """Get and display a specific message by ID."""
    result = api_get("/v1/memory/message", {"bot_id": bot_id, "message_id": message_id})
    
    if not result:
        console.print(f"[red]Failed to retrieve message {message_id}[/red]")
        return
    
    # Handle error response
    if isinstance(result, dict) and result.get("error"):
        console.print(f"[red]{result.get('error')}[/red]")
        return
    
    # The message might be in result["message"] or result itself
    if isinstance(result, dict) and "message" in result:
        msg = result["message"]
    else:
        msg = result
    
    if not msg or not isinstance(msg, dict):
        console.print(f"[yellow]Message {message_id} not found[/yellow]")
        return
    
    # Display message details
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    timestamp = msg.get("timestamp", 0)
    
    from datetime import datetime
    time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S") if timestamp else "unknown"
    
    # Role color
    role_color = {"user": "cyan", "assistant": "green", "system": "yellow", "summary": "magenta"}.get(role, "white")
    
    console.print(f"[bold]Message ID:[/bold] {msg.get('id', '?')}")
    console.print(f"[bold]Role:[/bold] [{role_color}]{role}[/{role_color}]")
    console.print(f"[bold]Timestamp:[/bold] {time_str}")
    if msg.get("session_id"):
        console.print(f"[bold]Session:[/bold] {msg.get('session_id')}")
    if msg.get("summary_metadata"):
        meta = msg.get("summary_metadata")
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta)
            except Exception:
                pass
        if isinstance(meta, dict):
            console.print(f"[bold]Summary of:[/bold] {meta.get('message_count', '?')} messages")
    
    console.print("\n[bold]Content:[/bold]")
    console.print(Panel(content or "(empty)", border_style="dim"))


def handle_msg_forget_id(bot_id: str, message_id: str, skip_confirm: bool):
    """Soft-delete a specific message by ID."""
    if not skip_confirm:
        if not _confirm(f"[bold yellow]Forget message {message_id}? (y/N):[/bold yellow] "):
            console.print("[dim]Cancelled[/dim]")
            return

    result = api_post("/v1/memory/forget", {"message_id": message_id}, {"bot_id": bot_id})
    if result and result.get("success"):
        console.print(f"[green]{result.get('message', 'Message forgotten')}[/green]")
        console.print("[dim]Use --msg-restore to undo[/dim]")
    elif result and result.get("detail"):
        console.print(f"[red]{result.get('detail')}[/red]")
    else:
        console.print("[red]Failed to forget message[/red]")


def handle_msg_restore(bot_id: str, skip_confirm: bool):
    """Restore all forgotten messages."""
    # Preview forgotten messages first
    data = api_get("/v1/memory/preview/ignored", {"bot_id": bot_id})
    if not data or not data.get("messages"):
        console.print("[yellow]No forgotten messages to restore[/yellow]")
        return

    messages = data["messages"]
    console.print(f"[bold]Forgotten messages to restore ({len(messages)}):[/bold]\n")
    display_messages_preview(messages)

    if skip_confirm or _confirm("\n[bold yellow]Restore these messages? (y/N):[/bold yellow] "):
        result = api_post("/v1/memory/restore", params={"bot_id": bot_id})
        if result:
            console.print(f"[green]{result.get('message', 'Done')}[/green]")
    else:
        console.print("[dim]Cancelled[/dim]")


def handle_msg_summarize_preview(bot_id: str):
    """Preview what sessions would be summarized."""
    data = api_get("/v1/history/summarize/preview", {"bot_id": bot_id})

    if not data:
        console.print(f"[red]Failed to get summarization preview for bot '{bot_id}'[/red]")
        return

    sessions = data.get("sessions", [])
    if not sessions:
        console.print("[yellow]No sessions eligible for summarization.[/yellow]")
        console.print("[dim]Sessions must be older than 30 minutes and have at least 4 messages.[/dim]")
        return

    console.print(Panel("[bold yellow]DRY RUN[/bold yellow] - Message Summarization Preview"))
    console.print(f"\nFound [bold]{len(sessions)}[/bold] sessions eligible for summarization:\n")

    total_messages = 0
    for i, session in enumerate(sessions, 1):
        start = session.get("start_time", "?")
        end = session.get("end_time", "?")
        msg_count = session.get("message_count", 0)
        first_msg = session.get("first_message", "")
        last_msg = session.get("last_message", "")

        total_messages += msg_count

        console.print(f"[bold]Session {i}[/bold] ({start} - {end})")
        console.print(f"  Messages: {msg_count}")
        console.print(f"  First: \"{first_msg}\"")
        console.print(f"  Last: \"{last_msg}\"")
        console.print()

    console.print(f"[bold]Would summarize {total_messages} messages into {len(sessions)} summaries.[/bold]")
    console.print("[dim]Run with --msg-summarize to execute.[/dim]")


def check_model_loaded() -> tuple[bool, str | None, list[str]]:
    """Check if a model is loaded in the service.
    
    Returns:
        Tuple of (is_loaded, current_model, available_models)
    """
    status = api_get("/status")
    if not status:
        return False, None, []
    
    current_model = status.get("current_model")
    available_models = status.get("available_models", [])
    return current_model is not None, current_model, available_models


def handle_msg_summarize(bot_id: str, skip_confirm: bool):
    """Summarize eligible message sessions."""
    # First check if a model is loaded
    model_loaded, current_model, available_models = check_model_loaded()
    
    if not model_loaded:
        console.print("[yellow]No LLM model is currently loaded.[/yellow]")
        console.print("[dim]Summarization works best with an LLM. Without one, heuristic summaries will be used.[/dim]\n")
        
        if available_models:
            console.print("[bold]Available models:[/bold]")
            for model in available_models:
                console.print(f"  - {model}")
            console.print()
            console.print("[dim]To load a model, run: llm \"hello\" (any query will load the default model)[/dim]")
            console.print()
        
        if not skip_confirm:
            if not _confirm("[bold yellow]Continue with heuristic summarization? (y/N):[/bold yellow] "):
                console.print("[dim]Cancelled. Load a model first for better summaries.[/dim]")
                return
    else:
        console.print(f"[dim]Using model: {current_model}[/dim]\n")
    
    # Get preview
    preview_data = api_get("/v1/history/summarize/preview", {"bot_id": bot_id})

    if not preview_data:
        console.print(f"[red]Failed to get summarization preview for bot '{bot_id}'[/red]")
        return

    sessions = preview_data.get("sessions", [])
    if not sessions:
        console.print("[yellow]No sessions eligible for summarization.[/yellow]")
        return

    total_messages = sum(s.get("message_count", 0) for s in sessions)

    console.print(f"[bold]Found {len(sessions)} sessions ({total_messages} messages) to summarize.[/bold]\n")

    # Show ALL sessions
    for i, session in enumerate(sessions, 1):
        start = session.get("start_time", "?")
        end = session.get("end_time", "?")
        msg_count = session.get("message_count", 0)
        console.print(f"  Session {i}: {start} - {end} ({msg_count} messages)")

    console.print()

    if not skip_confirm:
        if not _confirm("[bold yellow]Summarize these sessions? (y/N):[/bold yellow] "):
            console.print("[dim]Cancelled[/dim]")
            return

    # Execute summarization (without heuristic fallback first)
    console.print()
    use_heuristic = False
    
    with console.status("[bold green]Summarizing sessions with LLM..."):
        result = api_post("/v1/history/summarize", params={"bot_id": bot_id, "use_heuristic": use_heuristic}, timeout=LONG_TIMEOUT)

    if not result:
        console.print("[red]Summarization failed. Check that the llm-service is running.[/red]")
        console.print("[dim]Run: ./server.sh status[/dim]")
        return

    sessions_done = result.get("sessions_summarized", 0)
    messages_done = result.get("messages_summarized", 0)
    errors = result.get("errors", [])

    # Check if there were LLM failures
    if errors:
        console.print(f"\n[yellow]LLM summarization failed for {len(errors)} session(s):[/yellow]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        console.print()
        
        if not skip_confirm:
            if _confirm("[bold yellow]Use heuristic summarization for failed sessions? (y/N):[/bold yellow] "):
                console.print()
                with console.status("[bold green]Summarizing with heuristic fallback..."):
                    result2 = api_post("/v1/history/summarize", params={"bot_id": bot_id, "use_heuristic": True}, timeout=LONG_TIMEOUT)
                
                if result2:
                    sessions_done += result2.get("sessions_summarized", 0)
                    messages_done += result2.get("messages_summarized", 0)

    if sessions_done > 0:
        console.print(f"[green]Summarized {sessions_done} sessions ({messages_done} messages)[/green]")
    else:
        console.print("[yellow]No sessions were summarized.[/yellow]")


def handle_msg_rebuild_summaries(
    bot_id: str,
    sessions: int,
    skip_confirm: bool,
    start_ts: float | None = None,
    end_ts: float | None = None,
    purge_existing: bool = False,
):
    """Rebuild summaries for the most recent eligible sessions."""
    target_label = "all eligible sessions" if sessions == 0 else f"last {sessions} eligible session(s)"
    console.print(f"[bold]Rebuild target:[/bold] {target_label}")
    if start_ts is not None or end_ts is not None:
        console.print(f"[bold]Date filter:[/bold] start_ts={start_ts} end_ts={end_ts}")
    if purge_existing:
        console.print("[bold yellow]Mode:[/bold yellow] purge existing historical summaries in range before rebuild")
    console.print()

    if not skip_confirm:
        if not _confirm("[bold yellow]Recalculate and replace existing summaries for these sessions? (y/N):[/bold yellow] "):
            console.print("[dim]Cancelled[/dim]")
            return

    with console.status("[bold green]Rebuilding summaries..."):
        rebuild_timeout = max(LONG_TIMEOUT, float(sessions) * 180.0) if sessions > 0 else max(LONG_TIMEOUT, 3600.0)
        params = {
            "bot_id": bot_id,
            "sessions": sessions,
            "use_heuristic": False,
            "purge_existing": purge_existing,
        }
        if start_ts is not None:
            params["start_ts"] = start_ts
        if end_ts is not None:
            params["end_ts"] = end_ts
        result = api_post(
            "/v1/history/summarize/rebuild",
            params=params,
            timeout=rebuild_timeout,
        )

    if not result:
        console.print("[red]Rebuild failed.[/red]")
        return

    sessions_done = result.get("sessions_summarized", 0)
    messages_done = result.get("messages_summarized", 0)
    targeted = result.get("sessions_targeted", sessions_done)
    replaced = result.get("summaries_replaced", 0)
    purged = result.get("summaries_purged", 0)
    errors = result.get("errors", [])

    console.print(
        f"[green]Rebuilt {sessions_done}/{targeted} sessions ({messages_done} messages), "
        f"replaced {replaced} summary row(s), purged {purged} pre-existing summary row(s).[/green]"
    )

    if errors:
        console.print(f"\n[yellow]LLM rebuild failed for {len(errors)} session(s):[/yellow]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")

        if not skip_confirm:
            if _confirm("[bold yellow]Retry with heuristic fallback? (y/N):[/bold yellow] "):
                with console.status("[bold green]Rebuilding with heuristic fallback..."):
                    params2 = {
                        "bot_id": bot_id,
                        "sessions": sessions,
                        "use_heuristic": True,
                        "purge_existing": purge_existing,
                    }
                    if start_ts is not None:
                        params2["start_ts"] = start_ts
                    if end_ts is not None:
                        params2["end_ts"] = end_ts
                    result2 = api_post(
                        "/v1/history/summarize/rebuild",
                        params=params2,
                        timeout=LONG_TIMEOUT,
                    )
                if result2:
                    console.print(
                        f"[green]Fallback rebuilt {result2.get('sessions_summarized', 0)} sessions, "
                        f"replaced {result2.get('summaries_replaced', 0)} summary row(s), "
                        f"purged {result2.get('summaries_purged', 0)} pre-existing summary row(s).[/green]"
                    )


def handle_msg_summaries(bot_id: str):
    """List existing message summaries."""
    data = api_get("/v1/history/summaries", {"bot_id": bot_id})

    if not data:
        console.print(f"[red]Failed to get summaries for bot '{bot_id}'[/red]")
        return

    summaries = data.get("summaries", [])
    if not summaries:
        console.print("[yellow]No summaries found.[/yellow]")
        console.print("[dim]Use --msg-summarize to create summaries of old sessions.[/dim]")
        return

    console.print(Panel.fit(f"[bold]Message Summaries[/bold] ({len(summaries)} total)", border_style="cyan"))

    for index, summ in enumerate(summaries, start=1):
        summ_id = summ.get("id", "?")
        start = summ.get("session_start_time", "?")
        end = summ.get("session_end_time", "?")
        msg_count = summ.get("message_count", 0)
        method = summ.get("method", "?")
        content = summ.get("content", "") or "(empty)"

        method_style = "green" if method == "llm" else "yellow"

        console.print(
            Panel(
                str(content),
                title=(
                    f"{index}. ID: {summ_id} • "
                    f"[{method_style}]{method}[/{method_style}] • "
                    f"{msg_count} messages"
                ),
                subtitle=f"Session: {start} - {end}",
                border_style=method_style,
                expand=True,
            )
        )

    console.print("\n[dim]Use --msg-summary-get <ID> to view full summary[/dim]")
    console.print("[dim]Use --msg-delete-summary <ID> to remove a summary[/dim]")


def handle_msg_summary_get(bot_id: str, summary_id_prefix: str):
    """Show full summary content for a summary ID/prefix."""
    data = api_get("/v1/history/summaries", {"bot_id": bot_id})
    if not data:
        console.print(f"[red]Failed to get summaries for bot '{bot_id}'[/red]")
        return

    summaries = data.get("summaries", [])
    if not summaries:
        console.print("[yellow]No summaries found.[/yellow]")
        return

    needle = (summary_id_prefix or "").strip().lower()
    matches = [
        s for s in summaries
        if str(s.get("id", "")).lower().startswith(needle)
    ]

    if not matches:
        console.print(f"[red]No summary found matching prefix '{summary_id_prefix}'[/red]")
        return
    if len(matches) > 1:
        console.print(f"[yellow]Prefix matched {len(matches)} summaries. Be more specific.[/yellow]")
        for s in matches:
            console.print(f"  - {str(s.get('id', ''))}")
        return

    summ = matches[0]
    summ_id = str(summ.get("id", "?"))
    start = summ.get("session_start_time", "?")
    end = summ.get("session_end_time", "?")
    msg_count = summ.get("message_count", 0)
    method = summ.get("method", "?")
    content = summ.get("content", "")

    header = (
        f"ID: {summ_id}\n"
        f"Session: {start} - {end}\n"
        f"Messages: {msg_count}\n"
        f"Method: {method}"
    )
    console.print(Panel(header, title="Summary Metadata", border_style="cyan"))
    console.print(Panel(content or "(empty)", title="Summary Content", border_style="green"))


def handle_msg_delete_summary(bot_id: str, summary_id: str, skip_confirm: bool):
    """Delete a summary and restore the original messages."""
    if not skip_confirm:
        if not _confirm(f"[bold yellow]Delete summary {summary_id} and restore original messages? (y/N):[/bold yellow] "):
            console.print("[dim]Cancelled[/dim]")
            return

    result = api_delete(f"/v1/history/summary/{summary_id}", {"bot_id": bot_id})

    if not result:
        console.print("[red]Failed to delete summary[/red]")
        return

    if result.get("success"):
        msgs_restored = result.get("messages_restored", 0)
        console.print(f"[green]Deleted summary, restored {msgs_restored} messages[/green]")
    else:
        console.print(f"[red]{result.get('detail', 'Failed to delete summary')}[/red]")


def resolve_bot_selection(bot_arg: str | None) -> str:
    """Resolve bot id from CLI arg or interactive prompt."""
    if bot_arg:
        return bot_arg

    bots_data = api_get("/v1/bots")
    bots = bots_data.get("data", []) if bots_data else []
    slugs = [b.get("slug", "").strip() for b in bots if b.get("slug")]
    slugs = [s for s in slugs if s]

    if not sys.stdin.isatty():
        console.print("[red]No --bot provided and no interactive terminal available.[/red]")
        console.print("[dim]Pass --bot <slug> explicitly.[/dim]")
        sys.exit(1)

    if slugs:
        console.print("[bold]Select a bot:[/bold]")
        for idx, slug in enumerate(slugs, 1):
            name = next((b.get("name", "") for b in bots if b.get("slug") == slug), "")
            label = f"{slug} ({name})" if name else slug
            console.print(f"  {idx}. {label}")
        choice = console.input("[bold yellow]Bot number or slug:[/bold yellow] ").strip()
        if choice.isdigit():
            i = int(choice)
            if 1 <= i <= len(slugs):
                return slugs[i - 1]
            console.print("[red]Invalid bot number.[/red]")
            sys.exit(1)
        if choice in slugs:
            return choice
        console.print(f"[red]Unknown bot '{choice}'.[/red]")
        sys.exit(1)

    manual = console.input("[bold yellow]Enter bot slug:[/bold yellow] ").strip()
    if not manual:
        console.print("[red]Bot is required.[/red]")
        sys.exit(1)
    return manual


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Memory and message history management for llm-bawt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llm-memory "user preferences"       Search memories
  llm-memory --list-memories          List all memories
  llm-memory --msg                    Show recent messages
  llm-memory --msg-summarize          Summarize old sessions
  llm-memory --stats                  Show statistics
""",
    )
    parser.add_argument("query", nargs="?", default="", help="Search query for memories")
    parser.add_argument("--bot", "-b", default=None, help="Bot ID (if omitted, prompts interactively)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Max results (default: 10)")

    # General
    gen_group = parser.add_argument_group("General")
    gen_group.add_argument("--stats", action="store_true", help="Show memory and message statistics")
    # Memory operations (extracted facts)
    mem_group = parser.add_argument_group("Memories (extracted facts)")
    mem_group.add_argument("--method", "-m", choices=["text", "embedding", "high-importance", "all"],
                           default="all", help="Memory search method")
    mem_group.add_argument("--min-importance", type=float, default=0.0, help="Min importance filter")
    mem_group.add_argument("--list-memories", "-l", action="store_true", help="List memories by importance")
    mem_group.add_argument("--delete-memory", metavar="ID", help="Delete a memory by ID prefix")
    mem_group.add_argument("--consolidate", action="store_true", help="Merge redundant memories")
    mem_group.add_argument("--consolidate-dry-run", action="store_true", help="Preview memory consolidation")
    mem_group.add_argument("--regenerate-embeddings", action="store_true", help="Regenerate memory embeddings")

    # Entity profiles
    profile_group = parser.add_argument_group("Entity Profiles")
    profile_group.add_argument("--list-profiles", action="store_true",
                               help="List all profiles (users and bots) with attribute counts")
    profile_group.add_argument("--list-attrs", metavar="ENTITY", nargs="?", const="user",
                               help="List profile attributes (auto-detects if user or bot)")
    profile_group.add_argument("--delete-attr", metavar="ID", type=int,
                               help="Delete a profile attribute by ID")

    # Message history operations
    msg_group = parser.add_argument_group("Message History (conversation logs)")
    msg_group.add_argument("--msg", action="store_true", help="Show recent messages")
    msg_group.add_argument("--msg-search", metavar="QUERY", help="Search message history")
    msg_group.add_argument("--msg-forget", type=int, metavar="N", help="Forget last N messages (reversible)")
    msg_group.add_argument("--msg-forget-since", type=int, metavar="MIN", help="Forget messages from last N minutes")
    msg_group.add_argument("--msg-get", metavar="ID", help="Get a message by ID (supports prefix match)")
    msg_group.add_argument("--msg-forget-id", metavar="ID", help="Forget a specific message by ID")
    msg_group.add_argument("--msg-restore", action="store_true", help="Restore forgotten messages")
    msg_group.add_argument("--msg-summarize", action="store_true", help="Summarize old message sessions")
    msg_group.add_argument(
        "--msg-rebuild-summaries",
        metavar="N",
        type=int,
        help="Rebuild summaries for last N eligible sessions from original history (0 = all eligible)",
    )
    msg_group.add_argument("--msg-summarize-preview", action="store_true", help="Preview message summarization")
    msg_group.add_argument("--msg-summaries", action="store_true", help="List message summaries")
    msg_group.add_argument("--msg-summary-get", metavar="ID", help="Show full summary by ID or prefix")
    msg_group.add_argument("--msg-delete-summary", metavar="ID", help="Delete a summary by ID")
    msg_group.add_argument("--msg-rebuild-start", metavar="UNIX_TS", type=float, help="Optional start timestamp filter for --msg-rebuild-summaries")
    msg_group.add_argument("--msg-rebuild-end", metavar="UNIX_TS", type=float, help="Optional end timestamp filter for --msg-rebuild-summaries")
    msg_group.add_argument("--msg-rebuild-purge", action="store_true", help="Delete existing historical summaries in range before rebuild")

    args = parser.parse_args()

    # Check service availability
    if not check_service_available():
        console.print("[red]Error: llm-service is not running.[/red]")
        console.print("[dim]Start it with: llm-service[/dim]")
        sys.exit(1)

    # === Entity profiles (no bot selection needed) ===
    if args.list_profiles:
        handle_list_profiles()
        return

    if args.list_attrs:
        handle_list_attrs(args.list_attrs)
        return

    if args.delete_attr:
        handle_delete_attribute(args.delete_attr, args.yes)
        return

    selected_bot = resolve_bot_selection(args.bot)
    console.print(f"\n[bold cyan]Memory Tool[/bold cyan] - Bot: [yellow]{selected_bot}[/yellow]\n")

    # === General ===
    if args.stats:
        show_stats(selected_bot)
        return

    # === Memory operations ===
    if args.list_memories:
        list_memories(selected_bot, args.limit)
        return

    if args.delete_memory:
        handle_delete_memory(selected_bot, args.delete_memory, args.yes)
        return

    if args.consolidate or args.consolidate_dry_run:
        handle_consolidate(selected_bot, dry_run=args.consolidate_dry_run)
        return

    if args.regenerate_embeddings:
        handle_regenerate_embeddings(selected_bot)
        return

    # === Message history operations ===
    if args.msg:
        handle_show_messages(selected_bot, args.limit)
        return

    if args.msg_search:
        handle_search_messages(selected_bot, args.msg_search, args.limit)
        return

    if args.msg_forget:
        handle_msg_forget(selected_bot, args.msg_forget, args.yes)
        return

    if args.msg_forget_since:
        handle_msg_forget_since(selected_bot, args.msg_forget_since, args.yes)
        return

    if args.msg_get:
        handle_msg_get(selected_bot, args.msg_get)
        return

    if args.msg_forget_id:
        handle_msg_forget_id(selected_bot, args.msg_forget_id, args.yes)
        return

    if args.msg_restore:
        handle_msg_restore(selected_bot, args.yes)
        return

    if args.msg_summarize_preview:
        handle_msg_summarize_preview(selected_bot)
        return

    if args.msg_summarize:
        handle_msg_summarize(selected_bot, args.yes)
        return

    if args.msg_rebuild_summaries is not None:
        if args.msg_rebuild_summaries < 0:
            console.print("[red]--msg-rebuild-summaries requires N >= 0[/red]")
            sys.exit(1)
        handle_msg_rebuild_summaries(
            selected_bot,
            args.msg_rebuild_summaries,
            args.yes,
            start_ts=args.msg_rebuild_start,
            end_ts=args.msg_rebuild_end,
            purge_existing=bool(args.msg_rebuild_purge),
        )
        return

    if args.msg_summaries:
        handle_msg_summaries(selected_bot)
        return

    if args.msg_summary_get:
        handle_msg_summary_get(selected_bot, args.msg_summary_get)
        return

    if args.msg_delete_summary:
        handle_msg_delete_summary(selected_bot, args.msg_delete_summary, args.yes)
        return

    # === Memory search (default action) ===
    if not args.query and args.method != "high-importance":
        parser.print_help()
        console.print("\n[yellow]Tip:[/yellow] Use a query to search memories, or --msg to view messages")
        sys.exit(1)

    search_memories(selected_bot, args.query, args.method, args.limit, args.min_importance)


if __name__ == "__main__":
    main()
