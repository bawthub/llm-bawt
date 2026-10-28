"""Argument parsing for llm-bawt CLI."""

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.config import Config


def parse_arguments(config_obj: "Config") -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        config_obj: Configuration object for defaults
        
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Query LLM models from the command line using model aliases defined in models.yaml"
    )
    
    # Model selection
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help=f"Model alias defined in {config_obj.MODELS_CONFIG_PATH}. "
             f"Supports partial matching. (Default: bot's default or {config_obj.DEFAULT_MODEL_ALIAS or 'None'})"
    )
    
    # Model management
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model aliases defined in the configuration file and exit."
    )
    parser.add_argument(
        "--add-gguf",
        type=str,
        metavar="REPO_ID",
        help="(Deprecated: use --add-model gguf) Add a GGUF model from a Hugging Face repo ID."
    )
    parser.add_argument(
        "--add-model",
        type=str,
        choices=['ollama', 'openai', 'codex', 'gguf', 'vllm'],
        metavar="TYPE",
        help="Add models: 'ollama' (refresh from server), 'openai' (query API), 'codex' (install Codex agent model aliases), 'gguf' (add from HuggingFace repo), 'vllm' (add vLLM model from HuggingFace)"
    )
    parser.add_argument(
        "--add-bot",
        type=str,
        choices=["chat"],
        metavar="TYPE",
        help="Add bots: 'chat' (create a chat bot profile)"
    )
    parser.add_argument(
        "--delete-model",
        type=str,
        metavar="ALIAS",
        help="Delete the specified model alias from the configuration file after confirmation."
    )
    parser.add_argument(
        "--set-context-window",
        nargs=2,
        metavar=("ALIAS", "TOKENS"),
        help="Set per-model context window in models.yaml (creates missing aliases)."
    )
    
    # Configuration management
    parser.add_argument(
        "--config-set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Set a configuration value (e.g., DEFAULT_MODEL_ALIAS) in the .env file."
    )
    parser.add_argument(
        "--list-config",
        action="store_true",
        help="List the current effective configuration settings."
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Walk through client .env settings interactively (pre-filled with current values)."
    )
    
    # Main query input
    parser.add_argument(
        "question",
        nargs="*",
        help="Your question for the LLM model"
    )
    
    # Output control
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with rich pipeline information"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging with detailed API/tool I/O"
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Use plain text output (no Rich formatting)"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming output"
    )
    
    # History management
    parser.add_argument(
        "-dh", "--delete-history",
        action="store_true",
        help="Clear chat history"
    )
    parser.add_argument(
        "-ph", "--print-history",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        help="Print chat history (optional: number of recent pairs)"
    )
    
    # Command integration
    parser.add_argument(
        "-c", "--command",
        help="Execute command and add output to question"
    )
    
    # Mode selection
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local filesystem for history instead of database"
    )
    parser.add_argument(
        "--service",
        action="store_true",
        help="Route queries through the background service (if running)"
    )
    
    # Bot selection
    parser.add_argument(
        "-b", "--bot",
        type=str,
        default=None,
        help="Bot slug to use. Use --list-bots to see all."
    )
    parser.add_argument(
        "--list-bots",
        action="store_true",
        help="List available bots and exit"
    )
    parser.add_argument(
        "--inspect-config",
        type=str,
        metavar="BOT",
        default=None,
        help="Print the effective resolved system prompt + settings (with source "
             "provenance) for BOT and exit (TASK-487). Uses --user for the user id."
    )

    # Status and info
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show memory system status and configuration"
    )
    parser.add_argument(
        "--job-status",
        action="store_true",
        help="Show scheduled background jobs and recent run history"
    )

    # User profile management
    parser.add_argument(
        "--user",
        type=str,
        default=None,  # Will use config.DEFAULT_USER if not specified
        help="User profile to use (creates if not exists). Defaults to config.DEFAULT_USER"
    )
    parser.add_argument(
        "--list-users",
        action="store_true",
        help="List all user profiles"
    )
    parser.add_argument(
        "--user-profile",
        action="store_true",
        help="Show current user profile"
    )
    parser.add_argument(
        "--user-profile-set",
        metavar="FIELD=VALUE",
        help="Set a user profile field (e.g., name=\"Nick\")"
    )
    parser.add_argument(
        "--user-profile-setup",
        action="store_true",
        help="Run user profile setup wizard"
    )
    
    return parser.parse_args()
