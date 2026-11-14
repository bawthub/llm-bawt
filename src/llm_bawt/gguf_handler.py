import re
import pathlib
import importlib.util
from typing import List
from rich.prompt import Prompt, Confirm
from rich.console import Console

from .utils.config import Config

console = Console()

def normalize_for_match(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())


def get_or_download_gguf_model(repo_id: str, filename: str, config: Config) -> str | None:
    """Get path to GGUF model, downloading if needed.
    
    Returns the local file path, or None if download failed.
    """
    cache_dir = pathlib.Path(config.MODEL_CACHE_DIR).expanduser()
    model_repo_cache_dir = cache_dir / repo_id
    local_model_path = model_repo_cache_dir / filename
    
    if local_model_path.is_file():
        return str(local_model_path)
    
    # Need to download
    if importlib.util.find_spec("huggingface_hub") is None:
        console.print("[bold red]Error:[/bold red] `huggingface-hub` required for download but not found.")
        return None
    
    from huggingface_hub import hf_hub_download
    
    console.print(f"Downloading '[yellow]{filename}[/yellow]' from {repo_id}...")
    model_repo_cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_repo_cache_dir),
        )
        console.print("[green]Download complete[/green]")
        return downloaded_path
    except Exception as e:
        console.print(f"[bold red]Error downloading '{filename}':[/bold red] {e}")
        return None


def generate_gguf_alias(repo_id: str, filename: str, existing_aliases: List[str]) -> str:
    repo_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id
    base_filename = filename.lower().replace(".gguf", "")
    base_filename = re.sub(r'[._]q\d[._]\w+', '', base_filename)
    base_filename = re.sub(r'[._]q\d+k?[sm]?', '', base_filename)
    base_filename = base_filename.replace(repo_name.lower(), '').strip(' .-_/')
    alias_base = re.sub(r'[^a-z0-9]+', '-', f"{repo_name}-{base_filename}".lower()).strip('-')
    alias_base = alias_base or repo_name.lower()

    alias = alias_base
    counter = 1
    while alias in existing_aliases:
        alias = f"{alias_base}-{counter}"
        counter += 1
    return alias

def handle_add_gguf(repo_id: str, config: Config, service_client=None) -> bool:
    if service_client is None:
        console.print(
            "[bold red]Adding a model needs the running service:[/bold red] "
            "models are stored in the catalog. Start it with [yellow]llm-service[/yellow]."
        )
        return False
    if importlib.util.find_spec("huggingface_hub") is None:
        console.print("[bold red]Error:[/bold red] `huggingface-hub` is required to add GGUF models.")
        console.print("Install with: `pip install huggingface-hub`")
        return False

    console.print(f"Attempting to add GGUF from repository: [cyan]{repo_id}[/cyan]")

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError

        api = HfApi()
        console.print("Listing files in repository...")
        # Get file info including sizes
        repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)
        gguf_files_info = []
        for sibling in repo_info.siblings:
            if sibling.rfilename.lower().endswith(".gguf"):
                gguf_files_info.append({
                    "filename": sibling.rfilename,
                    "size": sibling.size or 0
                })
        gguf_files_info = sorted(gguf_files_info, key=lambda x: x["filename"])
    except HfHubHTTPError as e:
        console.print(f"[bold red]Error accessing HF repo '{repo_id}':[/bold red] {e}")
        return False
    except Exception as e:
        console.print(f"[bold red]Error listing files in '{repo_id}':[/bold red] {e}")
        return False

    if not gguf_files_info:
        console.print(f"[bold red]Error:[/bold red] No GGUF files found in '{repo_id}'.")
        return False

    selected_filename = _select_gguf_file(repo_id, gguf_files_info)
    if not selected_filename:
        return False

    download_success = _download_gguf_file(repo_id, selected_filename, config)
    if not download_success:
        return False

    return _add_to_catalog(repo_id, selected_filename, config, service_client)


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.1f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _select_gguf_file(repo_id: str, gguf_files_info: list[dict]) -> str | None:
    """Select a GGUF file from the list with alternating colors and file sizes."""
    if len(gguf_files_info) == 1:
        info = gguf_files_info[0]
        selected_filename = info["filename"]
        size_str = _format_size(info["size"])
        console.print(f"Found single GGUF file: [green]{selected_filename}[/green] ({size_str})")
        if not Confirm.ask(f"Add this file?\n  Repo: {repo_id}\n  File: {selected_filename}", default=True):
             console.print("Add operation cancelled.")
             return None
        return selected_filename
    else:
        console.print("Multiple GGUF files found. Select one:\n")
        choices_map = {str(i+1): info["filename"] for i, info in enumerate(gguf_files_info)}
        for i, info in enumerate(gguf_files_info):
            size_str = _format_size(info["size"])
            num = f"{i+1:3}"
            fname = info['filename']
            # Alternating colors for readability
            if i % 2 == 0:
                console.print(f"  [bold cyan]{num}[/bold cyan]  [white]{fname:<60}[/white]  [green]{size_str:>10}[/green]")
            else:
                console.print(f"  [bold cyan]{num}[/bold cyan]  [dim]{fname:<60}[/dim]  [green]{size_str:>10}[/green]")
        console.print()
        while True:
            try:
                choice_str = Prompt.ask("Enter the number of the file", default="1")
                if not choice_str.isdigit():
                    console.print("[yellow]Invalid input. Please enter a number.[/yellow]")
                    continue
                choice_int = int(choice_str)
                if 1 <= choice_int <= len(gguf_files_info):
                    return choices_map[choice_str]
                else:
                    console.print(f"[yellow]Invalid selection. Please enter a number between 1 and {len(gguf_files_info)}.[/yellow]")
            except (ValueError, KeyError):
                console.print("[yellow]Invalid selection.[/yellow]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[red]Selection cancelled.[/red]")
                return None

def _download_gguf_file(repo_id: str, filename: str, config: Config, max_retries: int = 3) -> bool:
    if importlib.util.find_spec("huggingface_hub") is None:
        console.print("[bold red]Error:[/bold red] `huggingface-hub` required for download but not found.")
        return False
    from huggingface_hub import hf_hub_download
    import time

    console.print(f"Downloading '[yellow]{filename}[/yellow]'...")
    cache_dir = pathlib.Path(config.MODEL_CACHE_DIR).expanduser()
    model_repo_cache_dir = cache_dir / repo_id
    local_model_path = model_repo_cache_dir / filename

    if local_model_path.is_file():
         console.print("[dim]File already exists in cache.[/dim]")
         return True
    
    model_repo_cache_dir.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                console.print(f"[yellow]Retry {attempt}/{max_retries}...[/yellow]")
            downloaded_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(model_repo_cache_dir),
            )
            console.print(f"[green]Download complete:[/green] {downloaded_path_str}")
            return True
        except KeyboardInterrupt:
            console.print("\n[red]Download cancelled.[/red]")
            return False
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries:
                console.print(f"[yellow]Download error (attempt {attempt}/{max_retries}):[/yellow] {error_msg}")
                console.print("[dim]Waiting 5 seconds before retry...[/dim]")
                time.sleep(5)
            else:
                console.print(f"[bold red]Download failed after {max_retries} attempts:[/bold red] {error_msg}")
                console.print("Cannot add model to config without successful download.")
                return False
    
    return False

def _add_to_catalog(repo_id: str, selected_filename: str, config: Config, service_client) -> bool:
    """Register a downloaded GGUF file in the model catalog (TASK-2282)."""
    from .model_manager import ModelManager

    entry = {
        "type": "gguf",
        "repo_id": repo_id,
        "filename": selected_filename,
        "description": f"{repo_id}/{selected_filename} (GGUF)",
    }
    return ModelManager(config, service_client=service_client).add_catalog_model(
        entry,
        suggest_alias=lambda existing: generate_gguf_alias(repo_id, selected_filename, existing),
        is_duplicate=lambda d: d.get("type") == "gguf"
        and d.get("repo_id") == repo_id and d.get("filename") == selected_filename,
    )

