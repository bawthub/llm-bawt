#!/usr/bin/env python3
"""One-bot ownership rollout; defaults to a counts-only dry run.

No configuration/profile enumeration, wildcard selection or automatic activation.
See docs/MEMORY_OWNERSHIP.md for the maintenance-window and rollback procedure.
"""
from __future__ import annotations

import argparse
import json

from sqlalchemy import text

from llm_bawt.memory import ownership_migration as migration

REMEDIES = {
    "unknown_bot": "Select one existing canonical bot identity.",
    "storage_identity_collision": "Resolve the conflicting storage identities before activation.",
    "legacy_evidence_requires_attestation": "Use explicit human-owner attestation only after operator verification.",
    "unproven_evidence_requires_attestation": "Verify uncertain provenance; attest one human owner only if justified.",
    "no_human_evidence_requires_attestation": "Supply explicit human-owner attestation only after operator verification.",
    "unproven_row_provenance_requires_attestation": "Inspect provenance through approved operator tools; attest only if justified.",
    "attestation_requires_human_profile": "Select a registered non-service human owner.",
    "other_human_owner": "Do not attest over another human; preserve ambiguous rows or resolve the evidence.",
    "other_explicit_owner": "Do not overwrite another explicit owner; preserve existing ownership.",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="counts of profiles and unmarked bots, no identities")
    activate = commands.add_parser("activate", help="one explicit bot; dry run unless --apply")
    activate.add_argument("--bot", required=True)
    activate.add_argument("--actor", required=True, help="operator attester identity retained in the receipt")
    mode = activate.add_mutually_exclusive_group()
    mode.add_argument("--attest-user", help="explicit human owner; never inferred from a default")
    mode.add_argument("--mark-only", action="store_true",
                      help="activate without stamping any row, including uncertain legacy provenance")
    activate.add_argument("--apply", action="store_true")
    for name, help_text in (
        ("expand", "nullable schema only; never stamps rows or activates bots"),
        ("index", "concurrent child indexes outside the activation transaction"),
        ("archive-down", "after code rollback: archive ownership before removing schema"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--apply", action="store_true")
    return result


def run(engine, args) -> dict:
    if args.command in {"expand", "index", "archive-down"} and not args.apply:
        return {"status": "not_applied", "operation": args.command, "dry_run": True}
    if args.command == "index":
        migration.build_owner_index(engine)
        return {"status": "indexed"}
    with engine.begin() as conn:
        # Fail visibly instead of waiting indefinitely on an unplanned writer.
        conn.execute(text("SET LOCAL lock_timeout = '5s'"))
        if args.command == "status":
            return {"status": "ok", **migration.status(conn)}
        if args.command == "expand":
            migration.ensure_schema(conn)
            return {"status": "expanded"}
        if args.command == "archive-down":
            archive = migration.downgrade(conn)
            return {"status": "archived_down" if archive else "already_down", "archive_id": archive}
        result = migration.activate(conn, args.bot, actor_id=args.actor,
                                    attest_user=args.attest_user, mark_only=args.mark_only,
                                    dry_run=not args.apply)
        if result.get("reason"):
            result["remedy"] = REMEDIES[result["reason"]]
            if result["reason"] not in {"unknown_bot", "storage_identity_collision"}:
                result["remedy"] += " Explicit --mark-only can activate without assigning any legacy owner."
        return result


def main(argv=None, *, engine=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "activate":
        # Validate before connecting, not after any SQL has been issued.
        try:
            migration._identity(args.bot, "bot")
            migration._identity(args.actor, "actor")
            if args.attest_user is not None:
                migration._identity(args.attest_user, "attest-user")
        except ValueError as exc:
            parser().error(str(exc))
    try:
        if engine is None:
            from llm_bawt.utils.config import config
            from llm_bawt.utils.db import get_shared_engine
            engine = get_shared_engine(config)
            if engine is None:
                raise RuntimeError("database unavailable")
        result = run(engine, args)
    except Exception:
        # Driver failures may contain credentials or row contents. Never print
        # them into this counts-only operator receipt.
        print(json.dumps({"status": "failed", "reason": "operation_failed",
                          "remedy": "Check the selected schema, isolation and maintenance window using approved operator tools."}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 2 if result["status"] == "refused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
