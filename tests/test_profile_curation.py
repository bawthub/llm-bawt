"""TASK-1720: candidate profile facts, curated rendering and diff-based curation."""

import json
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from llm_bawt.memory.profile_maintenance import (
    UNADDRESSED, ProfileCurationService, parse_ops, plan_changes,
)
from llm_bawt.profiles import (
    STATUS_CANDIDATE, STATUS_LIVE, STATUS_RETIRED, EntityProfile, EntityType, ProfileAttribute,
    ProfileManager,
)

U = EntityType.USER


@pytest.fixture
def pm():
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine, tables=[ProfileAttribute.__table__,
                                                 EntityProfile.__table__])
    manager = ProfileManager.__new__(ProfileManager)
    manager.engine = engine
    return manager


def _summary_profile(pm, summary="Legacy summary text.\nSecond line."):
    profile, _ = pm.get_or_create_profile(U, "alex")
    with Session(pm.engine) as session:
        row = session.get(EntityProfile, profile.id)
        row.display_name = "Alex"
        row.summary = summary
        session.add(row)
        session.commit()


def test_legacy_render_is_the_stored_summary_byte_for_byte(pm):
    """Bots without the curated setting keep exactly the old summary output."""
    _summary_profile(pm)
    pm.set_attribute(U, "alex", "fact", "occupation", "Engineer", source="explicit")
    pm.add_candidate(U, "alex", "fact", "age", "45", 0.9, "extracted", "m1")

    assert pm.get_user_profile_summary("alex") == "User's name: Alex\nLegacy summary text.\nSecond line."


def test_curated_render_shows_live_rows_only(pm):
    _summary_profile(pm)
    pm.set_attribute(U, "alex", "preference", "replies", "Short answers", source="explicit")
    pm.add_candidate(U, "alex", "fact", "age", "45-year-old", 0.9, "extracted", "m1")

    text = pm.get_user_profile_summary("alex", curated=True)
    assert "Short answers" in text
    assert "Legacy summary" not in text
    assert "45-year-old" not in text


def test_candidate_never_overwrites_live_row(pm):
    pm.set_attribute(U, "alex", "fact", "city", "Boston", source="explicit")
    pm.add_candidate(U, "alex", "fact", "city", "Denver", 0.9, "extracted", "m1")

    assert pm.get_attribute(U, "alex", "fact", "city").value == "Boston"
    rows = pm.attributes_with_status(U, "alex", (STATUS_CANDIDATE,))
    assert [r.value for r in rows] == ["Denver"]


def _rows(pm):
    pm.set_attribute(U, "alex", "fact", "city", "Boston", source="explicit")
    live = {a.id: a for a in pm.attributes_with_status(U, "alex", (STATUS_LIVE,))}
    for key, value, mid in (("city", "Moved to Denver", "m1"), ("dog", "Has a dog", "m2"),
                            ("age", "45", "m3"), ("misc", "noise", "m4")):
        pm.add_candidate(U, "alex", "fact", key, value, 0.9, "extracted", mid)
    cands = {a.id: a for a in pm.attributes_with_status(U, "alex", (STATUS_CANDIDATE,))}
    return live, cands


def test_plan_validates_ops_and_retires_unaddressed(pm):
    live, cands = _rows(pm)
    (lid,) = live
    c_city, c_dog, c_age, c_misc = cands
    ops = [
        {"op": "update", "id": lid, "candidate_ids": [c_city], "value": "Denver", "reason": "moved"},
        {"op": "add", "candidate_id": c_dog, "key": "pets", "value": "Has a dog", "reason": "stable"},
        {"op": "retire", "id": c_age, "reason": "not stated as lasting"},
        {"op": "add", "candidate_id": c_age, "reason": "reuse"},          # already used
        {"op": "retire", "id": lid, "reason": "drop it"},                 # live without citation
        {"op": "add", "candidate_id": 9999, "reason": "unknown"},
        {"op": "add", "candidate_id": c_misc},                             # no reason
        {"op": "delete", "id": lid, "reason": "x"},
    ]
    changes, applied, rejected = plan_changes(ops, live, cands)

    assert len(rejected) == 5
    by_id = {c["id"]: c for c in changes}
    assert by_id[lid] == {"id": lid, "value": "Denver", "source_message_id": "m1"}
    assert by_id[c_city]["status"] == STATUS_RETIRED
    assert by_id[c_dog]["status"] == STATUS_LIVE and by_id[c_dog]["key"] == "pets"
    assert by_id[c_age]["retire_reason"] == "not stated as lasting"
    assert by_id[c_misc]["retire_reason"] == UNADDRESSED
    assert all(op["sources"] for op in applied)


def test_add_rejects_key_that_collides_with_live(pm):
    live, cands = _rows(pm)
    c_city = next(iter(cands))
    _, applied, rejected = plan_changes(
        [{"op": "add", "candidate_id": c_city, "reason": "r"}], live, cands)
    assert rejected and rejected[0]["rejected"].startswith("a live entry")


def test_parse_ops_handles_fenced_and_bad_replies():
    assert parse_ops('```json\n{"ops": [{"op": "add"}]}\n```') == [{"op": "add"}]
    assert parse_ops('{"ops": []}') == []
    assert parse_ops("no json here") is None


def test_run_applies_one_candidate_diff_with_sources(pm):
    live, cands = _rows(pm)
    (lid,) = live
    c_city, c_dog, c_age, c_misc = cands
    reply = json.dumps({"ops": [
        {"op": "update", "id": lid, "candidate_ids": [c_city], "value": "Denver", "reason": "moved"},
        {"op": "add", "candidate_id": c_dog, "reason": "stable"},
        {"op": "retire", "id": c_age, "reason": "trivia"},
        {"op": "retire", "id": c_misc, "reason": "noise"},
    ]})
    client = MagicMock()
    client.query.return_value = reply
    with patch("llm_bawt.memory.profile_maintenance.get_profile_curation_prompt",
               return_value="prompt"):
        result = ProfileCurationService(pm, client).run("alex")

    assert result.error is None and not result.rejected
    assert pm.get_attribute(U, "alex", "fact", "city").value == "Denver"
    assert pm.get_attribute(U, "alex", "fact", "city").source_message_id == "m1"
    assert pm.get_attribute(U, "alex", "fact", "dog").value == "Has a dog"
    assert not pm.attributes_with_status(U, "alex", (STATUS_CANDIDATE,))
    assert "45" not in pm.get_user_profile_summary("alex", curated=True)


def test_run_makes_no_model_call_without_candidates(pm):
    pm.set_attribute(U, "alex", "fact", "city", "Boston", source="explicit")
    client = MagicMock()
    result = ProfileCurationService(pm, client).run("alex")
    client.query.assert_not_called()
    assert result.candidates == 0


def test_apply_curation_rejects_foreign_rows(pm):
    pm.set_attribute(U, "other", "fact", "city", "Paris", source="explicit")
    (row,) = pm.attributes_with_status(U, "other", (STATUS_LIVE,))
    with pytest.raises(ValueError):
        pm.apply_curation(U, "alex", [{"id": row.id, "status": STATUS_RETIRED}])
