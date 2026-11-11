"""CLI commands for managing Nextcloud Talk bot integrations."""

import asyncio
import click
import httpx
from rich.console import Console
from rich.table import Table

from .manager import get_nextcloud_manager
from .provisioner import get_provisioner_client
from ...utils.config import Config

console = Console()


@click.group(name='nextcloud')
def nextcloud_cli():
    """Manage Nextcloud Talk bot integrations."""
    pass


@nextcloud_cli.command(name='list')
def list_bots():
    """List all configured Nextcloud Talk bots."""
    manager = get_nextcloud_manager()
    bots = manager.list_bots()

    if not bots:
        console.print("[yellow]No Nextcloud bots configured.[/yellow]")
        console.print("Provision one with: [cyan]llm nextcloud provision --bot nova[/cyan]")
        return

    table = Table(title="Nextcloud Talk Bots")
    table.add_column("Bot", style="cyan")
    table.add_column("NC Bot ID", style="magenta")
    table.add_column("Conversation", style="blue")
    table.add_column("Enabled", style="yellow")

    for bot in bots:
        table.add_row(
            bot.llm_bawt_bot,
            str(bot.nextcloud_bot_id),
            bot.conversation_token[:12] + "..." if len(bot.conversation_token) > 12 else bot.conversation_token,
            "✓" if bot.enabled else "✗",
        )

    console.print(table)


@nextcloud_cli.command(name='reload')
def reload_bots():
    """Reload Nextcloud bot configuration (local and remote service)."""
    # Reload local manager
    manager = get_nextcloud_manager()
    manager.reload()
    bots = manager.list_bots()
    console.print(f"[green]✓ Local config reloaded ({len(bots)} bots)[/green]")

    # Notify running service
    try:
        config = Config()
        service_url = f"http://{config.SERVICE_HOST}:{config.SERVICE_PORT}"
        reload_resp = httpx.post(f"{service_url}/admin/nextcloud-talk/reload", timeout=5.0)
        if reload_resp.status_code == 200:
            data = reload_resp.json()
            console.print(f"[green]✓ Service reloaded ({data.get('bots_count', 0)} bots)[/green]")
        else:
            console.print(f"[yellow]⚠ Service returned status {reload_resp.status_code}[/yellow]")
    except httpx.ConnectError:
        console.print("[dim]ℹ Service not running[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not notify service: {e}[/yellow]")


@nextcloud_cli.command(name='provision')
@click.option('--bot', 'llm_bawt_bot', required=True, help='llm-bawt bot ID (e.g., nova, monika)')
@click.option('--room-name', help='Room name (default: bot name)')
@click.option('--bot-name', help='Bot display name (default: bot ID title-cased)')
@click.option('--owner', default='user', help='Room owner user ID')
def provision(llm_bawt_bot, room_name, bot_name, owner):
    """Provision a new Nextcloud Talk room and bot via provisioner service."""
    manager = get_nextcloud_manager()

    # Check if bot already has Nextcloud config
    if manager.get_bot(llm_bawt_bot):
        console.print(f"[red]Error:[/red] Bot '{llm_bawt_bot}' already has Nextcloud config")
        console.print(f"Remove it first with: [cyan]llm nextcloud remove --bot {llm_bawt_bot}[/cyan]")
        return

    # Defaults
    if not room_name:
        room_name = llm_bawt_bot.title()
    if not bot_name:
        bot_name = llm_bawt_bot.title()

    console.print("[cyan]Provisioning Nextcloud Talk room...[/cyan]")
    console.print(f"  Bot: {llm_bawt_bot}")
    console.print(f"  Room: {room_name}")
    console.print(f"  Owner: {owner}")

    try:
        provisioner = get_provisioner_client()

        # Call provisioner
        result = asyncio.run(
            provisioner.provision_talk_room_and_bot(
                room_name=room_name,
                bot_name=bot_name,
                owner_user_id=owner,
            )
        )

        # Save to config
        bot = manager.add_bot(
            llm_bawt_bot=llm_bawt_bot,
            nextcloud_bot_id=result.bot_id,
            secret=result.bot_secret,
            conversation_token=result.room_token,
        )

        console.print("\n[green]✓ Provisioned successfully![/green]")
        console.print(f"  Bot ID: {result.bot_id}")
        console.print(f"  Room token: {result.room_token}")
        console.print(f"  Room URL: [link]{result.room_url}[/link]")

        # Notify running service to reload bot config
        try:
            config = Config()
            service_url = f"http://{config.SERVICE_HOST}:{config.SERVICE_PORT}"
            reload_resp = httpx.post(f"{service_url}/admin/nextcloud-talk/reload", timeout=5.0)
            if reload_resp.status_code == 200:
                console.print("[dim]✓ Service reloaded bot config[/dim]")
            else:
                console.print(f"[yellow]⚠ Could not reload service (status {reload_resp.status_code})[/yellow]")
        except httpx.ConnectError:
            console.print("[dim]ℹ Service not running - config will be loaded on next start[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not notify service: {e}[/yellow]")

        console.print("\n[green]Test it by sending a message in the room![/green]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
    except Exception as e:
        console.print(f"[red]Provisioning failed:[/red] {e}")


@nextcloud_cli.command(name='remove')
@click.option('--bot', 'llm_bawt_bot', required=True, help='llm-bawt bot ID to remove Nextcloud config from')
def remove_bot(llm_bawt_bot):
    """Remove Nextcloud Talk configuration from a bot."""
    manager = get_nextcloud_manager()

    if not manager.get_bot(llm_bawt_bot):
        console.print(f"[red]Error:[/red] Bot '{llm_bawt_bot}' has no Nextcloud config")
        return

    if click.confirm(f"Remove Nextcloud config from bot '{llm_bawt_bot}'?"):
        manager.remove_bot(llm_bawt_bot)
        console.print(f"[green]✓[/green] Removed Nextcloud config from bot: {llm_bawt_bot}")


@nextcloud_cli.command(name='rename')
@click.option('--bot', 'llm_bawt_bot', required=True, help='llm-bawt bot ID')
@click.option('--name', 'new_name', required=True, help='New room name')
def rename_room(llm_bawt_bot, new_name):
    """Rename a Nextcloud Talk room."""
    manager = get_nextcloud_manager()

    bot_config = manager.get_bot(llm_bawt_bot)
    if not bot_config:
        console.print(f"[red]Error:[/red] Bot '{llm_bawt_bot}' has no Nextcloud config")
        return

    try:
        provisioner = get_provisioner_client()

        asyncio.run(
            provisioner.rename_room(
                room_token=bot_config.conversation_token,
                new_name=new_name,
            )
        )

        console.print(f"[green]✓[/green] Renamed room to '{new_name}'")

    except Exception as e:
        console.print(f"[red]Rename failed:[/red] {e}")
