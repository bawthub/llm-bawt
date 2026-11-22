"""Dependency validation for generated Setup seed data (TASK-2516)."""
from dataclasses import asdict

import pytest

from llm_bawt.seeding.bot_presets import BotPreset, PresetError
from llm_bawt.seeding.setup_role_dependencies import required_roles


def bot(slug, *roles):
    return BotPreset("starter", slug, slug, slug.title(), "Assistant", "Helps.", "Be helpful.", tags=roles)


def role(slug, **changes):
    return {"slug": slug, "name": slug.title(), "description": "Reviews changes.", **changes}


def test_dependencies_are_sorted_deduplicated_and_never_include_grants():
    rows = {"audit": role("audit", prompt_key="role.audit", capabilities=["approve_ops"]),
            "triage": role("triage"), "unused": role("unused", prompt_key="role.unreviewed")}
    actual = required_roles([bot("alpha", "triage", "audit"), bot("beta", "audit")], rows, {"role.audit"})
    assert [asdict(row) for row in actual] == [
        {"slug": "audit", "name": "Audit", "description": "Reviews changes.",
         "color": None, "prompt_key": "role.audit"},
        {"slug": "triage", "name": "Triage", "description": "Reviews changes.",
         "color": None, "prompt_key": None},
    ]
    assert required_roles([bot("alpha")], rows, set()) == ()


def test_missing_role_or_portable_prompt_holds_instead_of_dropping_dependency():
    bots = [bot("alpha", "audit")]
    with pytest.raises(PresetError, match="missing from the portable role catalog"):
        required_roles(bots, {}, set())
    with pytest.raises(PresetError, match="no released or exportable role prompt"):
        required_roles(bots, {"audit": role("audit", prompt_key="role.audit")}, set())


def test_nonportable_role_text_does_not_echo_the_value():
    private = "someone@example.test"
    with pytest.raises(PresetError, match="description contains an email address") as exc:
        required_roles([bot("alpha", "audit")], {"audit": role("audit", description=private)}, set())
    assert private not in str(exc.value)


def test_local_catalog_identity_collisions_are_not_exportable():
    bots = [bot("alpha", "audit", "triage")]
    with pytest.raises(PresetError, match="duplicate role name"):
        required_roles(bots, {"audit": role("audit"), "triage": role("triage", name="AUDIT")}, set())
    rows = {slug: role(slug, prompt_key="role.shared") for slug in ("audit", "triage")}
    with pytest.raises(PresetError, match="duplicate role prompt key"):
        required_roles(bots, rows, {"role.shared"})
