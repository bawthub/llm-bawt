"""Private-PG proof of the dormant guidance run (TASK-893 slice 4)."""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlmodel import Session

from llm_bawt.bot_tags import catalog
from llm_bawt.memory import guidance as g
from llm_bawt.memory import guidance_run as gr
from llm_bawt.runtime_settings import BotProfile
from tests.test_guidance_pg import Rig, engine  # noqa: F401  (module fixture)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig(engine):  # noqa: F811
    with engine.begin() as conn:
        catalog.ensure_schema(conn)
        catalog.seed(conn)
    r = Rig(engine)
    # The run resolves protection from the DB: tag the protected bot private.
    with Session(engine) as s:
        for slug in (r.bot, r.prot):
            s.add(BotProfile(slug=slug, name=slug, system_prompt=""))
        s.commit()
    with engine.begin() as conn:
        catalog.assign(conn, r.prot, ["private"], "test")
    return r


def _cursor(rig):
    with rig.engine.connect() as c:
        return c.execute(text("SELECT cursor_message_id FROM history_read_cursors "
                              "WHERE bot_id=:b AND user_id=:u"),
                         {"b": rig.bot, "u": rig.user}).scalar()


def _receipts(rig):
    with rig.engine.connect() as c:
        return c.execute(text("SELECT status FROM guidance_run_receipts "
                              "WHERE bot_id=:b AND user_id=:u ORDER BY id"),
                         {"b": rig.bot, "u": rig.user}).scalars().all()


def _adds(statement="short answers"):
    def drafter(base, evidence):
        ev = [{"bot_id": e["bot_id"], "message_id": e["message_id"]} for e in evidence[:1]]
        return {"preferences": [*base["preferences"],
                                {"id": f"p{len(base['preferences'])}", "statement": statement,
                                 "status": "active", "evidence": ev}],
                "reason": "stated"}
    return drafter


def _run(rig, **kw):
    kw.setdefault("drafter", _adds())
    return gr.run_pair(rig.engine, bot_id=rig.bot, user_id=rig.user, dry_run=False, **kw)


def test_tagged_bot_is_protected_without_extras(rig):
    with rig.engine.connect() as c:
        assert rig.prot in gr.resolve_protected(c)
        assert rig.prot in gr.resolve_protected(c, ["someone-else"])
    with pytest.raises(ValueError):
        gr.run_pair(rig.engine, bot_id=rig.prot, user_id=rig.user, dry_run=False)


def test_partition_collision_with_tagged_bot_is_refused(rig):
    twin = rig.prot[:3] + "-" + rig.prot[3:]  # same storage partition
    with Session(rig.engine) as s:
        s.add(BotProfile(slug=twin, name=twin, system_prompt=""))
        s.commit()
    with pytest.raises(ValueError, match="protected bot's storage partition"):
        gr.run_pair(rig.engine, bot_id=twin, user_id=rig.user)


def test_unknown_bot_is_refused(rig):
    with pytest.raises(ValueError, match="no profile"):
        gr.run_pair(rig.engine, bot_id=f"nobot{uuid.uuid4().hex[:6]}", user_id=rig.user)


def test_capacity_caps_the_page_so_no_candidate_is_skipped(rig):
    ids = [rig.msg(f"I prefer thing {i}", ts=float(i + 1))["message_id"] for i in range(5)]
    out = _run(rig, max_evidence=2, page_limit=10)
    assert out["stop"] == "capacity" and out["evidence"] == 2
    assert _cursor(rig) == ids[1]  # only offered rows committed
    assert out["proposal"]["status"] == "pending"
    out2 = _run(rig, max_evidence=10, page_limit=2, drafter=lambda b, e: {
        "preferences": b["preferences"], "reason": "none"})
    assert out2["evidence"] == 3 and _cursor(rig) == ids[4]
    assert out2["proposal"]["status"] == "no_change"
    assert _receipts(rig) == ["committed"] * 3


def test_dry_run_writes_nothing(rig):
    rig.msg()
    out = gr.run_pair(rig.engine, bot_id=rig.bot, user_id=rig.user)
    assert out["status"] == "preview_only" and out["evidence"] == 1
    assert _cursor(rig) is None and _receipts(rig) == []


def test_base_moved_during_drafting_is_reported_without_touching_receipts(rig):
    seed = rig.msg("I like lists", ts=0.5)
    rig.msg(ts=1.0)

    def racing(base, evidence):
        rig.approve(rig.propose(rig.doc(rig.pref("other", [seed]))))
        return _adds()(base, evidence)

    out = _run(rig, drafter=racing)
    assert out["proposal"]["status"] == "base_moved"
    assert _receipts(rig) == ["committed"]


def test_citations_outside_offered_evidence_are_dropped(rig):
    old = rig.msg("older words", ts=0.5)
    _run(rig, drafter=lambda b, e: {"preferences": b["preferences"], "reason": "x"})
    rig.msg(ts=2.0)

    def cites_old(base, evidence):
        return {"preferences": [{"id": "p", "statement": "s", "status": "active",
                                 "evidence": [old]}], "reason": "x"}

    out = _run(rig, drafter=cites_old)
    assert out["proposal"]["status"] == "no_change"
    assert out["proposal"]["dropped"][0]["id"] == "p"


def test_untrusted_draft_is_reported_not_raised(rig):
    rig.msg()
    out = _run(rig, drafter=lambda b, e: {"preferences": "nope"})
    assert out["proposal"]["status"] == "invalid_draft"
    assert _receipts(rig) == ["committed"]


def test_expected_base_is_checked_in_propose(rig):
    ev = rig.msg()
    with pytest.raises(g.BaseMoved):
        g.propose(rig.engine, user_id=rig.user, new_doc=rig.doc(rig.pref("p1", [ev])),
                  reason="r", proposed_by="al", excluded_bot_ids=[rig.prot],
                  expected_base_row_id=123456789)


def _ref(mid):
    return {"bot_id": "b", "message_id": mid}


def _base(*prefs):
    return {"mandate": "m", "preferences": list(prefs)}


def _p(statement, refs, **kw):
    return {"id": "p", "statement": statement, "status": "active",
            "evidence": [_ref(r) for r in refs], **kw}


OFFERED = {("b", "new")}


def test_rewrite_keeping_only_old_refs_is_dropped():
    base = _base(_p("always ask", ["old"]))
    doc, dropped = gr._constrain(base, {"preferences": [_p("never ask", ["old"])]}, OFFERED)
    assert doc["preferences"] == base["preferences"] and dropped[0]["id"] == "p"


def test_erasing_citations_is_dropped():
    base = _base(_p("always ask", ["old"]))
    for draft in (_p("always ask", [], status="superseded"), _p("never ask", [])):
        doc, dropped = gr._constrain(base, {"preferences": [draft]}, OFFERED)
        assert doc["preferences"] == base["preferences"] and dropped


def test_supersession_and_offered_additions_survive():
    base = _base(_p("always ask", ["old"]))
    ok = [_p("always ask", ["old"], status="superseded", superseded_by="q"),
          _p("always ask", ["old", "new"]), _p("never ask", ["new"])]
    for draft in ok:
        doc, dropped = gr._constrain(base, {"preferences": [draft]}, OFFERED)
        assert doc["preferences"] == [draft] and not dropped
