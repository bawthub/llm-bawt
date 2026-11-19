"""Private-PG proof of the guidance prompt feed loader (TASK-893 slice 6)."""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlmodel import Session

from llm_bawt.bot_tags import catalog
from llm_bawt.memory import guidance_feed as gf
from llm_bawt.runtime_settings import BotProfile
from tests.test_guidance_pg import Rig, engine  # noqa: F401  (module fixture)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig(engine):  # noqa: F811
    with engine.begin() as conn:
        catalog.ensure_schema(conn)
        catalog.seed(conn)
    r = Rig(engine)
    with Session(engine) as s:
        for slug in (r.bot, r.prot):
            s.add(BotProfile(slug=slug, name=slug, system_prompt=""))
        s.commit()
    with engine.begin() as conn:
        catalog.assign(conn, r.prot, ["private"], "test")
    return r


def _load(rig, bot=None):
    return gf.load(rig.engine, bot_id=bot or rig.bot, user_id=rig.user)


def test_no_guidance_renders_nothing(rig):
    assert _load(rig).body is None


def test_approved_version_feeds_and_pending_never_does(rig):
    ev1, ev2 = rig.msg(), rig.msg("actually, detailed answers", ts=2.0)
    rig.approve(rig.propose(rig.doc(rig.pref("p1", [ev1]))))
    first = _load(rig)
    assert "short answers" in first.body
    old = rig.pref("p1", [ev1], status="superseded", superseded_by="p2")
    new = rig.pref("p2", [ev2], statement="detailed answers", supersedes="p1")
    pid = rig.propose(rig.doc(old, new))
    assert _load(rig).body == first.body  # pending is not fed
    rig.approve(pid)
    after = _load(rig)
    assert "detailed answers" in after.body and "short answers" not in after.body
    assert after.provenance["version_row_id"] != first.provenance["version_row_id"]
    with rig.engine.connect() as c:
        sha = c.execute(text("SELECT body FROM prompt_template_versions WHERE id=:i"),
                        {"i": after.provenance["version_row_id"]}).scalar_one()
    assert gf._sha(sha) == after.provenance["source_sha256"]


def test_protected_and_unknown_bots_are_never_fed(rig):
    rig.approve(rig.propose(rig.doc(rig.pref("p1", [rig.msg()]))))
    assert _load(rig).body
    assert _load(rig, rig.prot).provenance == {"reason": "protected_bot"}
    assert _load(rig, rig.prot.upper()).body is None
    assert _load(rig, "no-such-bot").provenance == {"reason": "unknown_bot"}
