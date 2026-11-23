"""One source-only run stages prompts and the eligible Setup roster independently."""
import json
import stat

import pytest

from llm_bawt.prompt_synced_defaults import write_manifest
from llm_bawt.seeding import product_sync as sync
from llm_bawt.seeding.setup_role_dependencies import portable_role_prompt_keys
from llm_bawt.seeding.setup_seed_sync import SetupSeedSnapshot


META = {"title": "Review work", "category": "bot_tag", "format": "plain_text", "required_vars": []}


@pytest.fixture
def paths(tmp_path):
    shipped = tmp_path / "shipped"
    shipped.mkdir()
    stage = tmp_path / "review"
    stage.mkdir(mode=0o700)
    return shipped, stage


def roster(keys, prompt="role.review"):
    return SetupSeedSnapshot({"example"}, [{
        "key": "example", "slug": "example", "name": "Example", "summary": "Reviews work.",
        "system_prompt": "Be helpful.", "role": "Review", "tags": ["review"],
    }], {"review": {"slug": "review", "name": "Review", "prompt_key": prompt}}, keys)


def run(paths, read_snapshot=roster, *, keys=(), terms=()):
    shipped, stage = paths
    return sync.sync_product(api="https://example.test", keys=keys, shipped=shipped,
                             stage=stage, install_terms=terms, read_snapshot=read_snapshot)


def test_explicit_new_prompt_and_catalog_are_staged_together(paths, monkeypatch):
    shipped, stage = paths
    write_manifest({"role.old": META}, shipped)
    (shipped / "role.old.txt").write_text("Old default.")
    calls = []

    def fetch(api, keys):
        calls.append((api, keys))
        return {"promotion_candidates": [
            {**META, "key": "role.review", "kind": "new", "scope_type": "global", "scope_id": "*",
             "sync_class": "product", "body": "Review for errors."},
        ]}

    monkeypatch.setattr(sync, "_fetch_selected", fetch)
    result = run(paths, keys=("role.review",))
    assert calls == [("https://example.test", ("role.review",))]
    assert result["catalog"]["status"] == "staged"
    prompt_dir = stage / result["prompts"]["directory"]
    assert result["prompts"]["status"] == "staged"
    assert (prompt_dir / "role.review.txt").read_text() == "Review for errors."
    assert not (shipped / "role.review.txt").exists()
    assert not (prompt_dir / "role.old.txt").exists()
    assert "role.old" in json.loads((prompt_dir / "manifest.json").read_bytes())["prompts"]
    catalog = stage / sync.CATALOG_FILE
    assert b"Review for errors." not in catalog.read_bytes()
    assert json.loads(catalog.read_bytes())["roles"][0]["prompt_key"] == "role.review"
    for candidate in [catalog, *prompt_dir.iterdir()]:
        assert stat.S_IMODE(candidate.stat().st_mode) == 0o600


@pytest.mark.parametrize("prompt_status", ["failed", "held"])
def test_prompt_failure_does_not_hide_other_catalog_status(paths, monkeypatch, prompt_status):
    shipped, stage = paths
    write_manifest({"role.review": META}, shipped)
    (shipped / "role.review.txt").write_text("Review independently.")

    def fetch(_api, keys):
        if prompt_status == "failed":
            raise RuntimeError("should not be logged")
        return {"promotion_candidates": [
            {**META, "key": keys[0], "kind": "new", "scope_type": "global", "scope_id": "*",
             "sync_class": "product", "body": "mentions ExampleInternalTopic"},
        ]}

    monkeypatch.setattr(sync, "_fetch_selected", fetch)
    independently_ready = run(paths, keys=("unrelated.prompt",), terms=("ExampleInternalTopic",))
    assert independently_ready["prompts"]["status"] == prompt_status
    assert independently_ready["catalog"]["status"] == "staged"
    target = stage / sync.CATALOG_FILE
    prior = target.read_bytes()
    dependent = run(paths, keys=("role.review",), terms=("ExampleInternalTopic",))
    assert dependent["prompts"]["status"] == prompt_status
    assert dependent["catalog"]["status"] == "held"
    assert "ExampleInternalTopic" not in str(dependent) and "should not be logged" not in str(dependent)
    assert target.read_bytes() == prior
    assert list(stage.iterdir()) == [target]


def test_only_matching_portable_artifacts_count(paths, monkeypatch):
    shipped, stage = paths
    write_manifest({"role.review": META, "role.missing": META,
                    "role.private": {**META, "category": "install"}}, shipped)
    for key in ("role.orphan", "role.review", "role.private"):
        (shipped / f"{key}.txt").write_text("A default.")
    monkeypatch.setattr(sync, "_fetch_selected", lambda *_: pytest.fail("no implicit prompt reads"))
    assert portable_role_prompt_keys(shipped) == {"role.review"}
    assert run(paths)["prompts"]["status"] == "not_selected"
    assert run(paths)["catalog"]["status"] == "unchanged"
    assert run(paths, lambda keys: roster(keys, "role.orphan"))["catalog"]["status"] == "held"
    assert (stage / sync.CATALOG_FILE).is_file()


def test_main_requires_reviewed_terms_and_reports_exit(paths, monkeypatch, capsys):
    _, stage = paths
    monkeypatch.setattr(sync, "SYNCED_DIR", paths[0])
    monkeypatch.delenv("PROMPT_SYNC_INSTALL_TERMS", raising=False)
    monkeypatch.setattr(sync, "_read_configured_source", lambda keys, **_: roster(keys))
    argv = ["--stage", str(stage), "--setup-catalog"]
    with pytest.raises(SystemExit) as exc:
        sync.main(argv)
    assert exc.value.code == 2
    assert "PROMPT_SYNC_INSTALL_TERMS" in capsys.readouterr().err
    monkeypatch.setenv("PROMPT_SYNC_INSTALL_TERMS", "ExampleInternalTopic")
    with pytest.raises(SystemExit) as exc:
        sync.main(argv)
    assert exc.value.code == 2
    assert "--excluded-bots" in capsys.readouterr().err
    argv += ["--portable-input", "reviewed.json", "--portable-sha256", "a" * 64, "--excluded-bots", ""]
    with pytest.raises(SystemExit) as exc:
        sync.main(argv)
    assert exc.value.code == 2
    assert "--excluded-roles" in capsys.readouterr().err
    argv += ["--excluded-roles", ""]
    assert sync.main(argv) == 1  # missing role dependency
    catalog = json.loads(capsys.readouterr().out)["catalog"]
    assert catalog["status"] == "held" and catalog["previous_catalog_reusable"] is False
    monkeypatch.setattr(sync, "_read_configured_source", lambda keys, **_: roster(keys, None))
    stage.chmod(0o755)
    with pytest.raises(SystemExit) as unsafe:
        sync.main(argv)
    assert unsafe.value.code == 2
    assert not list(stage.iterdir())
