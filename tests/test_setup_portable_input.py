"""Product sync reads only approved portable content on a structural fixture."""
import json
from hashlib import sha256

import pytest

from llm_bawt.prompt_synced_defaults import write_manifest
from llm_bawt.seeding import product_sync as sync
from llm_bawt.seeding.bot_presets import PresetError
from llm_bawt.seeding.setup_portable_input import load_portable_input
from llm_bawt.seeding.setup_seed_sync import approved_setup_snapshot
from llm_bawt.seeding.setup_structure import SetupSourceStructure


@pytest.fixture
def source():
    return SetupSourceStructure({"fixture": "1" * 64}, {"review": "2" * 64}, {"fixture": {"review"}})


@pytest.fixture
def document():
    return {"schema": 1, "profiles": {"fixture": {
        "revision": "1" * 64, "roles": ["review"], "preset": {
            "name": "Fixture", "role": "Reviewer", "summary": "Checks changes.",
            "system_prompt": "Review independently.", "icon": "🔎",
        },
    }}, "roles": {"review": {
        "revision": "2" * 64, "name": "Review", "description": "Review fixture changes.",
        "color": None, "prompt_key": "role.review", "prompt_sha256": "3" * 64,
    }}}


def write(path, payload):
    if not isinstance(payload, bytes):
        payload = json.dumps(payload).encode()
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


@pytest.mark.parametrize("field", ["grants", "endpoint_id", "prompt_override_id", "nextcloud_config", "history"])
def test_rejects_nonportable_fields_even_with_a_matching_file_digest(tmp_path, document, field):
    document["profiles"]["fixture"]["preset"][field] = "not reviewed"
    path = tmp_path / "input.json"
    with pytest.raises(PresetError, match="invalid version or field schema"):
        load_portable_input(path, write(path, document))


def test_rejects_duplicate_entries_and_does_not_query_with_unapproved_bytes(tmp_path, document, monkeypatch):
    path = tmp_path / "input.json"
    digest = write(path, document)
    monkeypatch.setattr(sync, "read_setup_structure", lambda *_a, **_kw: pytest.fail("unexpected source read"))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(PresetError, match="approved bytes"):
        sync._read_configured_source({}, path=path, sha=digest, excluded=(), denied_roles=())
    duplicate = b'{"schema":1,"roles":{},"profiles":{},"profiles":{}}'
    with pytest.raises(PresetError, match="invalid version or field schema"):
        load_portable_input(path, write(path, duplicate))
    document["roles"]["review"]["capabilities"] = ["operate"]
    with pytest.raises(PresetError, match="invalid version or field schema"):
        load_portable_input(path, write(path, document))


@pytest.mark.parametrize("variant, reason", [
    ("missing profile", "preset 1 has no approved"),
    ("stale profile", "preset 1 needs review"),
    ("changed membership", "preset 1 needs review"),
    ("missing role", "role 1 has no approved"),
    ("stale role", "role 1 needs review"),
    ("wrong prompt hash", "no matching approved product prompt"),
    ("partial roster", "incomplete profile memberships"),
    ("partial role revisions", "incomplete role revisions"),
])
def test_holds_instead_of_using_unapproved_or_incomplete_content(tmp_path, document, source, variant, reason):
    expected = {"role.review": "3" * 64}
    if variant == "missing profile":
        document["profiles"].clear()
    elif variant == "stale profile":
        source.profiles["fixture"] = "4" * 64
    elif variant == "changed membership":
        source.memberships["fixture"] = set()
    elif variant == "missing role":
        document["roles"].clear()
    elif variant == "stale role":
        source.roles["review"] = "4" * 64
    elif variant == "wrong prompt hash":
        expected["role.review"] = "4" * 64
    elif variant == "partial roster":
        source.memberships.clear()
    elif variant == "partial role revisions":
        source.roles.clear()
    path = tmp_path / "input.json"
    approved = load_portable_input(path, write(path, document))
    with pytest.raises(PresetError, match=reason):
        approved_setup_snapshot(source, approved, expected)


def test_sync_pins_exact_prompt_and_never_reuses_held_bytes(tmp_path, document, source, monkeypatch):
    shipped, stage = tmp_path / "shipped", tmp_path / "staged"
    shipped.mkdir()
    stage.mkdir(mode=0o700)
    write_manifest({"role.review": {"title": "Review", "category": "bot_tag", "format": "plain_text",
                                    "required_vars": []}}, shipped)
    body = shipped / "role.review.txt"
    body.write_bytes(b"Check the evidence.\n")
    digest = sync.portable_role_prompt_revisions(shipped)["role.review"]
    document["roles"]["review"]["prompt_sha256"] = digest
    path = tmp_path / "input.json"
    approved_sha = write(path, document)
    observed_scopes = []

    def read_structure(_config, *, excluded, allowed_roles, denied_roles):
        observed_scopes.append((list(excluded), list(allowed_roles), list(denied_roles)))
        return source

    monkeypatch.setattr(sync, "read_setup_structure", read_structure)

    def run():
        return sync.sync_product(api="", keys=(), shipped=shipped, stage=stage, install_terms=(),
                                 read_snapshot=lambda revisions: sync._read_configured_source(
                                     revisions, path=path, sha=approved_sha, excluded=["fixture-protected"],
                                     denied_roles=["held_role"]))["catalog"]

    assert run()["status"] == "staged"
    assert observed_scopes == [(["fixture-protected"], ["review"], ["held_role"])]
    target = stage / sync.CATALOG_FILE
    prior = target.read_bytes()
    assert b"Check the evidence" not in prior and digest.encode() not in prior
    assert run()["status"] == "unchanged"
    body.write_bytes(b"Check the evidence.\r\n")
    held = run()
    assert held["status"] == "held" and held["previous_catalog_reusable"] is False and "file" not in held
    assert target.read_bytes() == prior
    # A complete, approved removal prunes the role even with old approvals present.
    source.profiles.clear()
    source.memberships.clear()
    source.roles.clear()
    assert run()["status"] == "staged"
    final = json.loads(target.read_bytes())
    assert final["presets"] == final["roles"] == []
