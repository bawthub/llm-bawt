"""Bot preset catalog and planning (TASK-2516)."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from llm_bawt.seeding import bot_presets as bp
from llm_bawt.seeding import preset_cli, preset_source
from llm_bawt.seeding.setup_role_dependencies import SetupRoleConflict
from llm_bawt.seeding.setup_seed_sync import SetupSeedSnapshot, stage_catalog
from llm_bawt.service.routes import setup_presets as routes


def _pack(**over):
    preset = {"key": "alpha", "name": "Alpha", "role": "Role", "summary": "Does a thing.",
              "system_prompt": "Be useful.", "tags": ["coder"],
              "provider_preference": ["anthropic", "openai"]}
    preset.update(over)
    return {"pack": "extra", "title": "Extra", "presets": [preset]}


@pytest.fixture
def legacy_catalog(tmp_path, monkeypatch):
    """Exercise old installs independently of the packaged stock resources."""
    monkeypatch.setattr(bp, "GENERATED_CATALOG", tmp_path / "setup_catalog.json")


@pytest.fixture
def generated_catalog(tmp_path, monkeypatch):
    target = tmp_path / "setup_catalog.json"
    profiles = [{"key": slug, "slug": slug, "name": slug.title(), "summary": "Does a thing.",
                 "system_prompt": "Be useful.", "role": "Assistant", "tags": tags,
                 "recommended": slug == "alpha"}
                for slug, tags in (("alpha", ["audit"]), ("beta", []), ("gamma", ["coach"]))]
    roles = {slug: {"slug": slug, "name": slug.title()} for slug in ("audit", "coach")}
    assert stage_catalog(lambda: SetupSeedSnapshot({"alpha", "beta", "gamma"}, profiles, roles, set()), target)
    monkeypatch.setattr(bp, "GENERATED_CATALOG", target)
    return target


def test_builtin_pack_uses_generated_roster_and_preserves_legacy_refs(generated_catalog):
    packs = bp.all_packs(None)
    assert packs[0].pack == bp.GENERATED_PACK
    stock = [preset for preset in packs[0].presets if not preset.legacy]
    assert [p.name for p in stock] == ["Alpha", "Beta", "Gamma"]
    assert stock[0].role_dependencies[0].slug == "audit"
    assert "role_dependencies" not in stock[0].public()
    assert bp.recommended_refs(packs) == ["stock/alpha"]
    assert bp.presets_by_ref(packs)["starter/helper"].legacy
    assert stage_catalog(lambda: SetupSeedSnapshot(set(), [], {}, set()), generated_catalog)
    assert not bp.all_packs(None)[0].presets
    assert all(p.legacy for p in bp.all_packs(None)[1].presets)


def test_shipped_catalog_loads_against_shipped_role_prompts():
    stock = bp.load_generated_pack(bp.GENERATED_CATALOG)
    assert stock.pack == bp.GENERATED_PACK
    # The roster is regenerated from the source lineup; check its shape, not its members.
    assert stock.presets
    assert all(preset.ref == f"stock/{preset.slug}" for preset in stock.presets)
    assert not bp.recommended_refs([stock])
    from llm_bawt.prompt_synced_defaults import SYNCED_DIR
    from llm_bawt.seeding.setup_role_dependencies import portable_role_prompt_keys

    shipped = portable_role_prompt_keys(SYNCED_DIR)
    for preset in stock.presets:
        assert [role.slug for role in preset.role_dependencies] == list(preset.tags)
        assert all(role.prompt_key in shipped for role in preset.role_dependencies if role.prompt_key)


def test_packaged_catalog_rejects_missing_role_prompt(tmp_path, monkeypatch):
    from llm_bawt import prompt_synced_defaults

    target = tmp_path / "setup_catalog.json"
    prompt_dir = tmp_path / "prompt_defaults"
    prompt_dir.mkdir()
    monkeypatch.setattr(prompt_synced_defaults, "SYNCED_DIR", prompt_dir)
    key = "role.auditor"
    preset = {**_pack()["presets"][0], "slug": "alpha", "tags": ["audit"]}
    roles = {"audit": {"slug": "audit", "name": "Audit", "prompt_key": key}}
    assert stage_catalog(lambda: SetupSeedSnapshot({"alpha"}, [preset], roles, {key}), target)
    prompt_synced_defaults.write_manifest({key: {
        "title": "Audit", "category": "bot_tag", "format": "plain_text", "required_vars": [],
    }}, prompt_dir)
    body = prompt_synced_defaults.body_path(key, prompt_dir)
    body.write_text("Review the result.", encoding="utf-8")
    assert bp.load_generated_pack(target).presets[0].role_dependencies[0].prompt_key == key
    body.unlink()
    with pytest.raises(bp.PresetError, match="no released or exportable role prompt"):
        bp.load_generated_pack(target)


def test_bad_stored_pack_is_skipped_and_builtin_id_reserved():
    assert bp.stored_packs([{"pack": "x"}, _pack()])[0].pack == "extra"
    with pytest.raises(bp.PresetError):
        bp.merge_pack([], bp.parse_pack({**_pack(), "pack": bp.BUILTIN_PACK}))


def test_selection_refuses_unknown_and_slug_collision():
    known = bp.presets_by_ref(bp.all_packs([_pack(slug="helper")]))
    with pytest.raises(bp.PresetError, match="unknown"):
        bp.normalize_selection(["nope/xx"], known)
    with pytest.raises(bp.PresetError, match="both create"):
        bp.normalize_selection(["starter/helper", "extra/alpha"], known)


def test_plan_prefers_vendor_skips_existing_and_blocks_without_endpoint():
    known = bp.presets_by_ref([bp.parse_pack(_pack()), bp.parse_pack({**_pack(key="beta"), "pack": "two"})])
    usable = [bp.EndpointChoice(1, "m@openai", "openai"), bp.EndpointChoice(2, "m@anthropic", "anthropic")]
    rows = bp.plan(["extra/alpha", "two/beta"], known, usable, {"beta"}, set())
    assert rows[0]["action"] == "create" and rows[0]["endpoint_id"] == 2
    assert rows[0]["missing_tags"] == ["coder"]
    assert rows[1]["action"] == "skip"
    blocked = bp.plan(["extra/alpha"], known, [], set(), None)
    assert blocked[0]["action"] == "blocked"
    override = bp.plan(["extra/alpha"], known, usable, set(), None, {"extra/alpha": 99})
    assert override[0]["action"] == "blocked"


def test_setup_preset_routes_registered():
    from llm_bawt.service.routes import setup_presets_router

    methods = {(r.path, m) for r in setup_presets_router.routes for m in r.methods}
    assert {("/v1/setup/bot-presets", m) for m in ("GET", "PUT", "POST")} <= methods


# --- routes and CLI ----------------------------------------------------------------------

class _Settings:
    engine = object()

    def __init__(self, values=None):
        self.values = dict(values or {})

    def get_value(self, scope_type, scope_id, key):
        return self.values.get(key)

    def set_value(self, scope_type, scope_id, key, value):
        self.values[key] = value


class _Profiles:
    def __init__(self, slugs=()):
        self.slugs = set(slugs)

    def list_all(self):
        return [SimpleNamespace(slug=s) for s in sorted(self.slugs)]

    def get(self, slug):
        return SimpleNamespace(slug=slug) if slug in self.slugs else None


_USABLE = [bp.EndpointChoice(7, "m@anthropic", "anthropic")]


@contextmanager
def _fakes(settings=None, profiles=None, usable=None, ready=True, refuse=()):
    settings = settings if settings is not None else _Settings()
    profiles = profiles if profiles is not None else _Profiles()
    created = []

    async def persist(payload, create_only, on_create=None, before_create=None):
        assert create_only is True
        if payload["slug"] in profiles.slugs:
            raise HTTPException(status_code=409, detail="exists")
        if payload["slug"] in refuse:  # the route gate's refusal shares the 409
            raise HTTPException(status_code=409, detail="Anthropic is not connected (expired).")
        if before_create is not None:
            before_create(None)
        if on_create is not None:
            on_create(None, SimpleNamespace(slug=payload["slug"]))
        profiles.slugs.add(payload["slug"])
        created.append(payload)
        return payload

    service = SimpleNamespace(config=MagicMock(name="config"))
    runtime = SimpleNamespace(ready=ready, detail="bridge down" if not ready else "ok")
    with (
        patch.object(routes, "get_service", return_value=service),
        patch.object(routes, "get_runtime_settings_store", return_value=settings),
        patch.object(routes, "get_bot_profile_store", return_value=profiles),
        patch.object(routes, "usable_endpoints", return_value=list(_USABLE if usable is None else usable)),
        patch.object(routes, "_known_tags", return_value={"coder", "reviewer"}),
        patch.object(routes, "check_agent_runtime", AsyncMock(return_value=runtime)),
        patch.object(routes, "_persist_bot_profile", side_effect=persist),
        patch.object(routes, "_record_first_default_bot") as default_bot,
    ):
        yield SimpleNamespace(settings=settings, profiles=profiles, created=created,
                              default_bot=default_bot, config=service.config)


def _client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_get_offers_recommended_selection_until_one_is_stored():
    with _fakes():
        body = _client().get("/v1/setup/bot-presets").json()
    assert body["stored"] is False
    assert body["selection"] == bp.recommended_refs(bp.all_packs(None))
    assert all(row["action"] == "create" for row in body["plan"])


def test_subset_creates_exactly_those_bots_and_rerun_skips(legacy_catalog):
    with _fakes() as f:
        client = _client()
        saved = client.put("/v1/setup/bot-presets", json={"selection": ["starter/critic"]})
        assert saved.status_code == 200 and saved.json()["stored"] is True
        first = client.post("/v1/setup/bot-presets").json()
        again = client.post("/v1/setup/bot-presets").json()
    assert [r["action"] for r in first["results"]] == ["created"]
    assert [p["slug"] for p in f.created] == ["critic"]
    assert f.created[0]["tags"] == ["reviewer"] and f.created[0]["endpoint_id"] == 7
    assert f.created[0]["system_prompt"].startswith("You are a reviewer")
    assert [r["action"] for r in again["results"]] == ["skip"]
    assert len(f.created) == 1
    f.default_bot.assert_called_once()


@pytest.mark.parametrize("guide", [False, True])
def test_generated_subset_and_guide_apply_once(guide, generated_catalog):
    delivery = object() if guide else None
    dispatcher = MagicMock()
    with (
        _fakes() as f,
        patch.object(routes, "install_missing_roles") as install,
        patch.object(routes, "profile_actor", return_value="owner"),
        patch.object(routes, "prepare_first_chat", return_value="Offer an introduction") as prompt,
        patch.object(routes, "record_choice", return_value=delivery) as choice,
    ):
        routes.get_service()._inter_bot_dispatcher = dispatcher
        client = _client()
        body = {"selection": ["stock/alpha", "stock/beta"], "guide_me": guide}
        first = client.post("/v1/setup/bot-presets?user_id=owner", json=body)
        again = client.post("/v1/setup/bot-presets?user_id=owner", json=body)
        assert first.status_code == again.status_code == 200
    assert [r["action"] for r in first.json()["results"]] == ["created", "created"]
    assert [r["action"] for r in again.json()["results"]] == ["skip", "skip"]
    assert [p["slug"] for p in f.created] == ["alpha", "beta"]
    assert [p["tags"] for p in f.created] == [["audit"], []]
    assert all(p["endpoint_id"] == 7 for p in f.created)
    install.assert_called_once()
    assert [role.slug for role in install.call_args.kwargs["roles"]] == ["audit"]
    f.default_bot.assert_called_once()
    assert f.default_bot.call_args.args[1] == "alpha"
    choice.assert_called_once()
    assert choice.call_args.args[1].slug == "alpha"
    assert choice.call_args.kwargs["opted_in"] == guide
    if guide:
        dispatcher.announce_admitted.assert_called_once_with(delivery)
    else:
        dispatcher.announce_admitted.assert_not_called()
        prompt.assert_not_called()


def test_generated_role_conflict_blocks_only_the_dependent_bot(generated_catalog):
    with (
        _fakes() as f,
        patch.object(routes, "install_missing_roles", side_effect=SetupRoleConflict("Local role conflict")),
    ):
        body = _client().post("/v1/setup/bot-presets", json={
            "selection": ["stock/alpha", "stock/beta"],
        }).json()
    assert body["ok"] is False
    assert [row["action"] for row in body["results"]] == ["blocked", "created"]
    assert body["results"][0]["reason"] == "Local role conflict"
    assert [p["slug"] for p in f.created] == ["beta"]


def test_retired_stock_choice_does_not_fall_back_to_a_legacy_name(generated_catalog):
    assert stage_catalog(lambda: SetupSeedSnapshot(set(), [], {}, set()), generated_catalog)
    with _fakes() as f:
        client = _client()
        assert client.get("/v1/setup/bot-presets").json()["packs"] == []
        result = client.post("/v1/setup/bot-presets", json={"selection": ["stock/helper"]}).json()
        assert result["results"][0]["action"] == "blocked"
        assert not f.created


def test_existing_custom_bot_is_left_alone(legacy_catalog):
    with _fakes(profiles=_Profiles({"helper"})) as f:
        body = _client().post("/v1/setup/bot-presets",
                              json={"selection": ["starter/helper", "starter/builder"]}).json()
    assert [r["action"] for r in body["results"]] == ["skip", "created"]
    assert [p["slug"] for p in f.created] == ["builder"]


def test_no_connection_blocks_and_creates_nothing(legacy_catalog):
    with _fakes(usable=[]) as f:
        body = _client().post("/v1/setup/bot-presets").json()
    assert {r["action"] for r in body["results"]} == {"blocked"}
    assert "Connect a provider" in body["results"][0]["reason"]
    assert f.created == []
    f.default_bot.assert_not_called()


def test_apply_refuses_while_agent_runtime_is_down():
    with _fakes(ready=False) as f:
        response = _client().post("/v1/setup/bot-presets")
    assert response.status_code == 503
    assert f.created == []


def test_unknown_selection_is_rejected_without_saving():
    with _fakes() as f:
        response = _client().put("/v1/setup/bot-presets", json={"selection": ["starter/nope"]})
    assert response.status_code == 400
    assert bp.SELECTION_KEY not in f.settings.values


def test_retry_creates_valid_presets_and_reports_a_retired_choice(legacy_catalog):
    with _fakes() as f:
        client = _client()
        body = {"selection": ["starter/retired", "starter/critic"]}
        first = client.post("/v1/setup/bot-presets", json=body)
        again = client.post("/v1/setup/bot-presets", json=body)
    assert first.status_code == again.status_code == 200
    assert [r["action"] for r in first.json()["results"]] == ["created", "blocked"]
    assert "no longer offered" in first.json()["results"][1]["reason"]
    assert [r["action"] for r in again.json()["results"]] == ["skip", "blocked"]
    assert [p["slug"] for p in f.created] == ["critic"]


def test_malformed_retry_ref_is_rejected_before_creation():
    with _fakes() as f:
        response = _client().post("/v1/setup/bot-presets",
                                  json={"selection": ["starter/critic", "starter/a/b"]})
    assert response.status_code == 400
    assert f.created == []


def test_cli_select_by_number_and_import(tmp_path, capsys, legacy_catalog):
    pack = tmp_path / "pack.json"
    pack.write_text(__import__("json").dumps(_pack()))
    with _fakes() as f, patch.object(preset_cli, "_config", return_value=f.config), \
            patch("llm_bawt.service.dependencies.get_runtime_settings_store", return_value=f.settings):
        assert preset_cli.main(["import", str(pack)]) == 0
        assert preset_cli.main(["list"]) == 0
        listing = capsys.readouterr().out
        assert "Extra" in listing and "Alpha" in listing
        builtin = bp.builtin_pack().presets
        helper = next(i for i, preset in enumerate(builtin, 1) if preset.key == "helper")
        assert preset_cli.main(["select", str(helper), str(len(builtin) + 1)]) == 0
        assert f.settings.values[bp.SELECTION_KEY] == ["starter/helper", "extra/alpha"]
        assert preset_cli.main(["select", "current"]) == 0
        assert f.settings.values[bp.SELECTION_KEY] == ["starter/helper", "extra/alpha"]
        assert preset_cli.main(["select", "none"]) == 0
        assert f.settings.values[bp.SELECTION_KEY] == []
        assert preset_cli.main(["select", "99"]) == 2


def _row(slug, name=None, description="", prompt=None, avatar=None, color=None):
    return {"slug": slug, "name": name or slug.title(), "description": description,
            "system_prompt": prompt if prompt is not None else f"Be {slug}.", "avatar": avatar, "color": color}


@contextmanager
def _snapshot(rows, tags=()):
    calls = []

    def read(config, slugs):
        calls.append(slugs)
        if slugs is None:
            return [r["slug"] for r in rows], []
        return [r for r in rows if r["slug"] in slugs], [t for t in tags if t[0] in slugs]

    with patch.object(preset_cli, "_read_snapshot", side_effect=read):
        yield calls


def test_cli_export_needs_named_bots_and_carries_only_travelling_fields(capsys):
    rows = [_row("ally", "Ally", "Writes code.", avatar="🤖", color="sky"), _row("pic", avatar="/uploads/a.png")]
    with _snapshot(rows, [("ally", "coder", "Coder")]) as calls, \
            patch.object(preset_cli, "_config", return_value=MagicMock()):
        assert preset_cli.main(["export"]) == 2
        assert "Exportable: ally, pic" in capsys.readouterr().err
        assert calls == [None]  # listing reads IDs only, never prompts
        doc = preset_cli._export(MagicMock(), ["ally", "pic"], "mine", "Mine")
    ally, pic = doc["presets"]
    assert ally["tags"] == ["coder"] and ally["role"] == "Coder" and ally["icon"] == "🤖"
    assert pic["icon"] == "" and "avatar" not in ally
    assert doc["exported"]["bots"] == ["ally", "pic"]
    parsed = bp.parse_pack(doc)
    assert not hasattr(parsed.presets[0], "avatar")


def test_cli_export_scans_every_travelling_field_and_the_pack_title():
    with _snapshot([_row("ally", name="Ally someone@example.test")]), \
            pytest.raises(bp.PresetError, match="ally: name contains an email"):
        preset_cli._export(MagicMock(), ["ally"], "mine", "Mine")
    with _snapshot([_row("ally")]), pytest.raises(bp.PresetError, match="pack: title contains a URL"):
        preset_cli._export(MagicMock(), ["ally"], "mine", "See https://example.test")
    doc = None
    with _snapshot([_row("ally", avatar="Jane Doe 🙂", color="https://example.test/c")]):
        doc = preset_cli._export(MagicMock(), ["ally"], "mine", "Mine")
    assert doc["presets"][0]["icon"] == "" and doc["presets"][0]["color"] is None


def test_cli_export_scans_role_tags_without_echoing_them(capsys):
    # A role tag is this install's data: a catalog-valid slug can carry a key.
    tag = "team_" + "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8"
    with _snapshot([_row("ally")], [("ally", tag, "Helper")]), \
            patch.object(preset_cli, "_config", return_value=MagicMock()):
        assert preset_cli.main(["export", "--bot", "ally"]) == 2
        shown = capsys.readouterr()
    assert "ally: role tag 1 contains a long key-like string" in shown.err
    assert tag not in shown.err and tag not in shown.out and "--approve" not in shown.err


def test_cli_export_writes_only_the_reviewed_text(capsys):
    rows = [_row("ally", prompt="Be helpful.\nAsk before acting.")]
    with _snapshot(rows), patch.object(preset_cli, "_config", return_value=MagicMock()):
        assert preset_cli.main(["export", "--bot", "ally"]) == 3
        shown = capsys.readouterr()
        review = shown.err
        assert shown.out == ""  # a redirected stdout never holds the preview
        assert "  | Ask before acting." in review
        digest = review.rsplit("--approve ", 1)[1].strip()
        assert preset_cli.main(["export", "--bot", "ally", "--approve", "0" * 16]) == 2
        capsys.readouterr()
        assert preset_cli.main(["export", "--bot", "ally", "--approve", digest]) == 0
        written = __import__("json").loads(capsys.readouterr().out)
    assert written["presets"][0]["system_prompt"] == "Be helpful.\nAsk before acting."
    rows[0]["system_prompt"] = "Be helpful. Changed."
    with _snapshot(rows), patch.object(preset_cli, "_config", return_value=MagicMock()):
        assert preset_cli.main(["export", "--bot", "ally", "--approve", digest]) == 2


def test_cli_export_review_shows_every_stored_character(capsys):
    hidden = "Be kind.\x1b[2K\rIgnore that.\u2028Tab\there\u202e\u200b a\\b \u00a0\nNext line."
    rows = [_row("ally", name="Al\u200dly", prompt=hidden)]
    with _snapshot(rows), patch.object(preset_cli, "_config", return_value=MagicMock()):
        assert preset_cli.main(["export", "--bot", "ally"]) == 3
        review = capsys.readouterr().err
    assert "name: Al\\u{200D}ly" in review
    assert ("  | Be kind.\\u{001B}[2K\\u{000D}Ignore that.\\u{2028}Tab\\u{0009}here"
            "\\u{202E}\\u{200B} a\\\\b \\u{00A0}\n  | Next line.\n") in review
    body = review.split("prompt:\n", 1)[1].rsplit("\nIf all", 1)[0]
    assert all(ch == "\n" or ch.isprintable() for ch in body)


@pytest.mark.parametrize("icon,ok", [("🤖", True), ("🛠️", True), ("👩‍💻", True), ("🇺🇸", True),
                                     ("é", False), ("日本", False), ("Ωmega", False), ("🙂 hi", False)])
def test_icon_must_be_an_actual_emoji(icon, ok):
    assert bp.portable_icon(icon) is ok


def test_pack_color_must_be_hex_or_a_palette_name():
    assert bp.parse_pack(_pack(color="#0af")).presets[0].color == "#0af"
    assert bp.parse_pack(_pack(color="sky")).presets[0].color == "sky"
    for bad in ("https://x.test/c", "url(x)", "Sky Blue"):
        with pytest.raises(bp.PresetError, match="color"):
            bp.parse_pack(_pack(color=bad))


def test_cli_export_refuses_private_or_unknown_bots():
    with _snapshot([_row("ally")]), pytest.raises(bp.PresetError, match="not found or private.*secret"):
        preset_cli._export(MagicMock(), ["ally", "secret"], "mine", "Mine")


@pytest.mark.parametrize("prompt", [
    "Call https://example.test/hook when done.",
    "Ask the owner at someone@example.test.",
    "The NAS is at 10.1.2.3.",
    "Deploy to build.lan when asked.",
    "Use key sk-abcdefghijklmnopqrstuvwx for that.",
    "password: hunter2",
    "token aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3z",
    "Notes live in /srv/team/notes.",
    "Logs are under ~/work/logs.",
    "The API is on gateway:8443.",
    "Talk to ci.build.example for builds.",
])
def test_cli_export_refuses_addresses_and_secret_looking_text(prompt):
    with _snapshot([_row("ally", prompt=prompt)]), pytest.raises(bp.PresetError, match="ally: system_prompt contains"):
        preset_cli._export(MagicMock(), ["ally"], "mine", "Mine")


def test_export_snapshot_excludes_private_in_one_read_only_statement():
    sql = str(preset_source.SELECTED_PROFILES)
    assert "include_in_global_search" in sql and "NOT EXISTS" in sql and "ANY(:slugs)" in sql
    assert preset_source.EXPORTABLE_PROFILES in str(preset_source.ELIGIBLE_PROFILES)
    engine = MagicMock()
    conn = engine.connect.return_value.execution_options.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value.all.return_value = [{"slug": "ally"}]
    conn.execute.return_value.__iter__.return_value = iter([])
    with patch("llm_bawt.utils.db.build_engine_from_config", return_value=engine):
        preset_cli._read_snapshot(MagicMock(), ["ally", "excluded"])
    engine.connect.return_value.execution_options.assert_called_once_with(
        isolation_level="REPEATABLE READ", postgresql_readonly=True)
    conn.begin.assert_called_once()
    params = conn.execute.call_args_list[0].args[1]
    assert set(params["private"]) >= {"private"} and params["slugs"] == ["ally", "excluded"]
    # Join only the eligible bots, and use source role data, not coded names.
    tag_sql = str(preset_source.ROLE_ASSIGNMENTS)
    assert "bot_tag_catalog" in tag_sql and "c.kind = 'role'" in tag_sql
    assert conn.execute.call_args_list[1].args[1] == {"slugs": ["ally"]}
    engine.dispose.assert_called_once()


def test_route_refusal_is_reported_as_failed_not_as_an_existing_bot(legacy_catalog):
    with _fakes(refuse={"critic"}) as f:
        body = _client().post("/v1/setup/bot-presets",
                              json={"selection": ["starter/critic", "starter/builder"]}).json()
    assert body["ok"] is False
    builder, critic = body["results"]  # catalog order
    assert critic["action"] == "failed" and "not connected" in critic["reason"]
    assert builder["action"] == "created" and [p["slug"] for p in f.created] == ["builder"]


def test_usable_endpoints_follow_the_route_gate_rule():
    path = lambda key, auth: SimpleNamespace(key=key, auth_mechanism=auth, vendor="Anthropic")  # noqa: E731
    eps = [SimpleNamespace(id=1, ref="a@oauth", access_path=path("oauth", "oauth")),
           SimpleNamespace(id=2, ref="b@expired", access_path=path("expired", "api_key")),
           SimpleNamespace(id=3, ref="c@local", access_path=path("local", "none")),
           SimpleNamespace(id=4, ref="d@untracked", access_path=path("untracked", "api_key"))]
    config = MagicMock()
    config.ensure_model_catalog.return_value.endpoints.return_value = eps
    connections = {"oauth": {"state": "ok", "connected": True},
                   "expired": {"state": "expired", "connected": True}}
    with patch.object(routes, "harness_for_agent_backend", return_value="claude"), \
            patch.object(routes, "access_path_connections", return_value=connections):
        assert [c.endpoint_id for c in routes.usable_endpoints(config)] == [1, 3]


def test_pack_requires_emoji_icon_and_text_description():
    with pytest.raises(bp.PresetError, match="emoji"):
        bp.parse_pack(_pack(icon="/a.png"))
    with pytest.raises(bp.PresetError, match="must be text"):
        bp.parse_pack({**_pack(), "description": {"nested": "token=abc"}})
    stored = bp.merge_pack([], bp.parse_pack(_pack(icon="🧪", avatar="https://x.test/a.png")))
    assert "avatar" not in stored[0]["presets"][0] and stored[0]["presets"][0]["icon"] == "🧪"
