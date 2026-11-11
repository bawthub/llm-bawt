"""Migration utilities for llm-bawt memory schema changes.

Run from command line:
    python -m llm_bawt.memory.migrations backfill_meaning --bot nova
    python -m llm_bawt.memory.migrations all --bot nova

Or import and use programmatically:
    from llm_bawt.memory.migrations import run_all_migrations
    run_all_migrations(backend)

This module is a thin orchestration + compatibility facade (TASK-553). The
individual migrations live in cohesive per-domain modules and are re-exported
here so every existing ``from llm_bawt.memory.migrations import <fn>`` import
keeps working:

    migrations_memory     — memory/history schema (tags, meaning embeddings,
                            recalled_history column, sessions backfill)
    migrations_media      — media_assets table + messages.attachments column
"""

import json
import logging

# Re-export every public migration so the historical import surface
# (`from llm_bawt.memory.migrations import X`) is preserved unchanged.
from .migrations_memory import (
    backfill_empty_tags,
    backfill_meaning_embeddings,
    add_recalled_history_column,
    backfill_sessions,
)
from .migrations_media import (
    add_media_assets_table,
    drop_media_assets_table,
    add_attachments_to_messages,
)

logger = logging.getLogger(__name__)

__all__ = [
    "backfill_empty_tags",
    "backfill_meaning_embeddings",
    "add_recalled_history_column",
    "backfill_sessions",
    "add_media_assets_table",
    "drop_media_assets_table",
    "add_attachments_to_messages",
    "run_all_migrations",
    "main",
]


def run_all_migrations(backend, dry_run: bool = False) -> dict:
    """Run all pending migrations."""
    results = {}

    logger.debug("Running tag backfill...")
    results["tags"] = backfill_empty_tags(backend, dry_run=dry_run)

    logger.debug("Running meaning embedding generation...")
    results["meaning_embeddings"] = backfill_meaning_embeddings(backend, dry_run=dry_run)

    logger.debug("Running recalled_history column migration...")
    results["recalled_history"] = add_recalled_history_column(backend, dry_run=dry_run)

    logger.debug("Running sessions backfill...")
    results["sessions"] = backfill_sessions(backend, dry_run=dry_run)

    logger.debug("Creating media_assets table...")
    results["media_assets_table"] = add_media_assets_table(backend, dry_run=dry_run)

    logger.debug("Adding attachments column to messages tables...")
    results["messages_attachments"] = add_attachments_to_messages(backend, dry_run=dry_run)

    return results


# CLI entry point
def main():
    import argparse
    from ..utils.config import Config
    from .postgresql import PostgreSQLMemoryBackend

    parser = argparse.ArgumentParser(description="Run memory schema migrations")
    parser.add_argument(
        "command",
        choices=[
            "backfill_tags",
            "backfill_meaning",
            "backfill_sessions",
            "media_assets_up",
            "media_assets_down",
            "messages_attachments",
            "all",
        ],
    )
    parser.add_argument("--bot", required=True, help="Bot ID to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be done")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    config = Config()
    backend = PostgreSQLMemoryBackend(config=config, bot_id=args.bot)

    if args.command == "backfill_tags":
        result = backfill_empty_tags(backend, dry_run=args.dry_run)
    elif args.command == "backfill_meaning":
        result = backfill_meaning_embeddings(backend, dry_run=args.dry_run)
    elif args.command == "backfill_sessions":
        result = backfill_sessions(backend, dry_run=args.dry_run)
    elif args.command == "media_assets_up":
        result = add_media_assets_table(backend, dry_run=args.dry_run)
    elif args.command == "media_assets_down":
        result = drop_media_assets_table(backend, dry_run=args.dry_run)
    elif args.command == "messages_attachments":
        result = add_attachments_to_messages(backend, dry_run=args.dry_run)
    elif args.command == "all":
        result = run_all_migrations(backend, dry_run=args.dry_run)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
