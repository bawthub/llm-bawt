"""Generated stock catalog stages deterministically and fails closed (TASK-2516)."""
import json
from dataclasses import replace

import pytest

from llm_bawt.seeding.bot_presets import PresetError
from llm_bawt.seeding.setup_seed_sync import SetupSeedSnapshot, stage_catalog


def bot(slug, *roles, **extra):
    return {"key": slug, "slug": slug, "name": slug.title(), "role": "Assistant",
            "summary": "Describes useful work.", "system_prompt": "Be useful.",
            "tags": list(roles), **extra}


def role(slug, **extra):
    return {"slug": slug, "name": slug.title(), "description": "Reviews work.", **extra}


def test_sync_uses_full_source_snapshot_and_tracks_add_edit_remove(tmp_path):
    target = tmp_path / "generated.json"
    roles = {"audit": role("audit", prompt_key="role.audit", capabilities=["example_grant"]),
             "coach": role("coach")}
    before = SetupSeedSnapshot({"first", "second"},
                               [bot("second", "coach"), bot("first", "audit", endpoint_id=88)],
                               roles, {"role.audit"})
    assert stage_catalog(lambda: before, target)
    original = target.read_bytes()
    assert not stage_catalog(lambda: replace(before, bots=list(reversed(before.bots))), target)
    assert target.read_bytes() == original
    initial = json.loads(original)
    assert [entry["slug"] for entry in initial["presets"]] == ["first", "second"]
    assert [(entry["slug"], entry["prompt_key"]) for entry in initial["roles"]] == [
        ("audit", "role.audit"), ("coach", None),
    ]
    assert b"example_grant" not in original and b"endpoint_id" not in original

    edited = SetupSeedSnapshot({"first", "third"},
                               [bot("third", "triage"), bot("first", "audit", name="Edited")],
                               {**roles, "audit": role("audit", prompt_key="role.audit2"),
                                "triage": role("triage")}, {"role.audit2"})
    assert stage_catalog(lambda: edited, target)
    updated = json.loads(target.read_bytes())
    assert [(entry["slug"], entry["name"]) for entry in updated["presets"]] == [
        ("first", "Edited"), ("third", "Third"),
    ]
    assert [(entry["slug"], entry["prompt_key"]) for entry in updated["roles"]] == [
        ("audit", "role.audit2"), ("triage", None),
    ]


def test_role_assignment_order_and_duplicates_do_not_change_catalog(tmp_path):
    target = tmp_path / "generated.json"
    roles = {"audit": role("audit"), "triage": role("triage")}
    snapshot = SetupSeedSnapshot({"first"}, [bot("first", "triage", "audit", "audit")], roles, set())
    assert stage_catalog(lambda: snapshot, target)
    assert not stage_catalog(lambda: replace(snapshot, bots=[bot("first", "audit", "triage")]), target)
    assert json.loads(target.read_bytes())["presets"][0]["tags"] == ["audit", "triage"]


@pytest.mark.parametrize("failure", ["source", "partial", "blocked_prompt", "private_text"])
def test_failed_or_held_catalog_keeps_last_good_bytes(tmp_path, failure):
    target = tmp_path / "generated.json"
    existing = SetupSeedSnapshot({"first"}, [bot("first", "audit")],
                                 {"audit": role("audit", prompt_key="role.audit")}, {"role.audit"})
    assert stage_catalog(lambda: existing, target)
    prior = target.read_bytes()

    def read():
        if failure == "source":
            raise ConnectionError("source unavailable")
        if failure == "partial":
            return replace(existing, eligible_slugs={"first", "missing"})
        if failure == "blocked_prompt":
            return replace(existing, portable_prompt_keys=set())
        return replace(existing, bots=[bot("first", "audit", summary="Don't share user@example.test")])

    with pytest.raises((ConnectionError, PresetError)) as exc:
        stage_catalog(read, target)
    assert "user@example.test" not in str(exc.value)
    assert target.read_bytes() == prior
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("field", ["bot", "role"])
def test_install_terms_hold_candidate_without_echoing_term(tmp_path, field):
    target = tmp_path / "generated.json"
    first = SetupSeedSnapshot({"first"}, [bot("first", "audit")], {"audit": role("audit")}, set())
    assert stage_catalog(lambda: first, target)
    prior = target.read_bytes()
    private = "ExampleInternalProject"
    held = (replace(first, bots=[bot("first", "audit", summary=f"Reviews {private}.")]) if field == "bot"
            else replace(first, roles={"audit": role("audit", description=f"Reviews {private}.")}))
    with pytest.raises(PresetError, match="contains an installation-specific term") as exc:
        stage_catalog(lambda: held, target, install_terms=[private])
    assert private not in str(exc.value)
    assert target.read_bytes() == prior


def test_complete_empty_roster_can_remove_last_bot_and_its_roles(tmp_path):
    target = tmp_path / "generated.json"
    assert stage_catalog(lambda: SetupSeedSnapshot({"first"}, [bot("first")], {}, set()), target)
    assert stage_catalog(lambda: SetupSeedSnapshot(set(), [], {}, set()), target)
    doc = json.loads(target.read_bytes())
    assert doc["presets"] == doc["roles"] == []
