"""The backfill script's imports resolve (TASK-1807).

scripts/ is not a package and nothing else imports this script, so a removed
dependency (TASK-1329 dropped ``PROTECTED_BOT_IDS``) once broke it silently
until an operator ran it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_message_authors.py"


def test_script_imports_and_offers_both_kinds():
    spec = importlib.util.spec_from_file_location("backfill_message_authors_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # runs top-level imports only; main() needs a DB
    assert set(module.KINDS) == {"legacy", "continuation"}
    assert callable(module.read_protected_slugs)
