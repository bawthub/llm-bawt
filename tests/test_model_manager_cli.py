"""Offline regressions for model-manager CLI catalog and deletion behavior."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt import model_manager as cli


def manager_with(models=None, service=True):
    manager = object.__new__(cli.ModelManager)
    manager.models_data = {"models": models or {}}
    manager._changed_aliases = set()
    manager.service_client = Mock() if service else None
    return manager


@pytest.mark.parametrize("provider", ["codex", "openai", "grok", "ollama"])
def test_add_model_carries_discovered_context_to_service(monkeypatch, provider):
    manager = manager_with()
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: kwargs["default"])

    assert manager._prepare_new_model_entries(provider, [{"id": "new-model", "context_length": 123456}]) == 1
    assert manager.save_config(added=1)
    manager.service_client.upsert_catalog_model.assert_called_once()
    alias, entry = manager.service_client.upsert_catalog_model.call_args.args
    assert alias == "new-model"
    assert entry["context_window"] == 123456


def test_add_model_prompts_for_missing_context_and_rejects_invalid_values(monkeypatch):
    manager = manager_with()
    answers = iter(["new-model", "nope", "0", "-3", "64000"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(answers))

    assert manager._prepare_new_model_entries("openai", [{"id": "new-model"}]) == 1
    assert manager.save_config(added=1)
    assert manager.service_client.upsert_catalog_model.call_args.args[1]["context_window"] == 64000


def test_add_model_can_skip_unknown_context_without_catalog_write(monkeypatch):
    manager = manager_with()
    answers = iter(["new-model", ""])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(answers))

    assert manager._prepare_new_model_entries("openai", [{"id": "new-model"}]) == 0
    assert manager.models_data == {"models": {}}
    assert manager._changed_aliases == set()
    manager.service_client.upsert_catalog_model.assert_not_called()


def test_add_local_model_without_context_does_not_prompt_for_window(monkeypatch):
    manager = manager_with(service=False)
    prompt = Mock(return_value="local-model")
    monkeypatch.setattr(cli.Prompt, "ask", prompt)

    assert manager._prepare_new_model_entries("openai", [{"id": "local-model"}]) == 1
    prompt.assert_called_once()
    assert "context_window" not in manager.models_data["models"]["local-model"]


def test_codex_compat_entry_is_not_reoffered_or_matched_as_openai(monkeypatch):
    entry = {"type": "openai", "access_path": "openai-oauth", "model_id": "codex-model", "description": "Catalog model"}
    manager = manager_with({"existing": entry})
    monkeypatch.setattr(manager, "_fetch_api_models", lambda provider: (True, [{"id": "codex-model", "summary": "Catalog model"}]))
    prompt = Mock(side_effect=AssertionError("Existing model must not be reoffered"))
    monkeypatch.setattr(manager, "_prompt_for_new_models", prompt)

    assert manager._update_provider_models("codex")
    assert cli.ModelManager._entry_matches_provider(entry, "codex")
    assert not cli.ModelManager._entry_matches_provider(entry, "openai")
    assert cli.ModelManager._entry_matches_provider({"type": "agent_backend", "backend": "codex"}, "codex")
    assert not cli.ModelManager._entry_matches_provider({"type": "openai", "access_path": "openai-api"}, "codex")
    prompt.assert_not_called()


def test_model_id_is_highlighted_in_description():
    manager = manager_with()
    assert manager._format_model_details("openai", {
        "model_id": "gpt-test.1", "description": "Openai gpt-test.1 (Created: today)",
    }) == "[bright_blue]gpt-test.1[/bright_blue] (Created: today)"


def gguf_manager(monkeypatch, tmp_path, *, shared=False, filename="first.gguf"):
    config = SimpleNamespace(MODEL_CACHE_DIR=tmp_path)
    models = {
        "first": {"type": "gguf", "repo_id": "owner/repo", "filename": filename},
        "second": {"type": "gguf", "repo_id": "owner/repo", "filename": filename if shared else "second.gguf"},
    }
    manager = manager_with(models)
    manager.config = config
    monkeypatch.setattr(cli, "ModelManager", lambda *args, **kwargs: manager)
    repo = tmp_path / "owner" / "repo"
    repo.mkdir(parents=True)
    (repo / "first.gguf").write_bytes(b"first")
    (repo / "second.gguf").write_bytes(b"second")
    (repo / "metadata.json").write_text("{}")
    return manager, config, repo


def test_gguf_delete_removes_only_selected_file(monkeypatch, tmp_path):
    manager, config, repo = gguf_manager(monkeypatch, tmp_path)
    confirm = Mock(return_value=True)
    monkeypatch.setattr(cli.Confirm, "ask", confirm)

    assert cli.delete_model("first", config)
    assert not (repo / "first.gguf").exists()
    assert (repo / "second.gguf").read_bytes() == b"second"
    assert (repo / "metadata.json").exists()
    assert "second" in manager.models_data["models"]
    assert confirm.call_count == 2
    manager.service_client.delete_catalog_model.assert_called_once_with("first")


def test_gguf_delete_preserves_file_shared_by_another_alias(monkeypatch, tmp_path):
    manager, config, repo = gguf_manager(monkeypatch, tmp_path, shared=True)
    confirm = Mock(return_value=True)
    monkeypatch.setattr(cli.Confirm, "ask", confirm)

    assert cli.delete_model("first", config)
    assert (repo / "first.gguf").exists()
    assert "second" in manager.models_data["models"]
    confirm.assert_called_once()


@pytest.mark.parametrize("answers", [[False], [True, False]])
def test_gguf_delete_respects_alias_and_file_confirmation(monkeypatch, tmp_path, answers):
    manager, config, repo = gguf_manager(monkeypatch, tmp_path)
    confirm = Mock(side_effect=answers)
    monkeypatch.setattr(cli.Confirm, "ask", confirm)

    assert cli.delete_model("first", config)
    assert (repo / "first.gguf").exists()
    assert (repo / "second.gguf").exists()
    assert confirm.call_count == len(answers)
    if not answers[0]:
        assert "first" in manager.models_data["models"]
        manager.service_client.delete_catalog_model.assert_not_called()


def test_gguf_delete_does_not_remove_file_when_catalog_delete_fails(monkeypatch, tmp_path):
    manager, config, repo = gguf_manager(monkeypatch, tmp_path)
    manager.service_client.delete_catalog_model.return_value = None
    confirm = Mock(return_value=True)
    monkeypatch.setattr(cli.Confirm, "ask", confirm)

    assert not cli.delete_model("first", config)
    assert (repo / "first.gguf").exists()
    confirm.assert_called_once()


def test_gguf_delete_refuses_path_outside_cache(monkeypatch, tmp_path):
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"keep")
    manager, config, repo = gguf_manager(monkeypatch, tmp_path / "cache", filename=str(outside))
    confirm = Mock(return_value=True)
    monkeypatch.setattr(cli.Confirm, "ask", confirm)

    assert cli.delete_model("first", config)
    assert outside.read_bytes() == b"keep"
    assert (repo / "second.gguf").exists()
    confirm.assert_called_once()


def _oauth_only_manager(**model_fields):
    """Service-mode manager over a catalog whose gpt-5.5 lives only on openai-oauth."""
    from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity

    endpoint = ModelEndpoint(
        id=7,
        model=ModelIdentity(
            3, "gpt-5.5", "openai", "GPT-5.5",
            description=model_fields.get("description", "gpt-5.5 (Created: 2026-01-01)"),
            default_context_window=272000,
        ),
        access_path=AccessPath(
            5, "openai-oauth", "openai", "openai-responses",
            "https://chatgpt.com/backend-api/codex", "oauth",
        ),
        upstream_model_id="gpt-5.5",
        serving_config={"reasoning_effort": "high"},
        context_window_override=None,
        tool_support_override="native",
    )
    catalog = ModelCatalog([endpoint])
    manager = manager_with(catalog.compatibility_mapping())
    manager._catalog = catalog
    manager.service_client.patch_catalog_endpoint.return_value = {"id": 7}
    manager.service_client.put_catalog_model.return_value = {"key": "gpt-5.5"}
    return manager


def test_service_set_context_window_patches_existing_endpoint_only():
    manager = _oauth_only_manager()

    assert manager.set_context_window("gpt-5.5", 400000)

    client = manager.service_client
    client.upsert_catalog_model.assert_not_called()
    client.put_catalog_model.assert_not_called()
    client.patch_catalog_endpoint.assert_called_once_with(
        "gpt-5.5", "openai-oauth", {"endpoint_id": 7, "context_window_override": 400000},
    )


def test_service_description_refresh_keeps_display_name_and_endpoint():
    manager = _oauth_only_manager()
    manager.models_data["models"]["gpt-5.5"]["description"] = "gpt-5.5 (Created: 2026-02-02)"
    manager._changed_aliases.add("gpt-5.5")

    assert manager.save_config(updated=1)

    client = manager.service_client
    client.upsert_catalog_model.assert_not_called()
    client.patch_catalog_endpoint.assert_not_called()
    client.put_catalog_model.assert_called_once_with("gpt-5.5", {
        "vendor": "openai",
        "display_name": "GPT-5.5",
        "description": "gpt-5.5 (Created: 2026-02-02)",
        "default_context_window": 272000,
        "default_tool_support": None,
    })


def test_service_set_context_window_on_new_alias_still_upserts():
    manager = _oauth_only_manager()

    assert manager.set_context_window("gpt-9", 128000)

    manager.service_client.upsert_catalog_model.assert_called_once()
    assert manager.service_client.upsert_catalog_model.call_args.args[0] == "gpt-9"


# TASK-2282: models resolve only from the catalog, so models.yaml is never read.

def test_add_catalog_model_writes_gguf_through_service(monkeypatch):
    answers = iter(["my-gguf", "8192"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: next(answers))
    manager = manager_with({"other": {"type": "openai", "model_id": "gpt-x"}})

    entry = {"type": "gguf", "repo_id": "org/repo", "filename": "m.gguf", "description": "d"}
    assert manager.add_catalog_model(entry, lambda existing: "repo", lambda d: False)

    alias, sent = manager.service_client.upsert_catalog_model.call_args.args
    assert alias == "my-gguf"
    assert sent == {**entry, "context_window": 8192}


def test_add_catalog_model_without_service_refuses_and_writes_nothing(tmp_path):
    manager = manager_with(service=False)
    manager.config_path = tmp_path / "models.yaml"

    assert not manager.add_catalog_model({"type": "vllm"}, lambda e: "x", lambda d: False)
    assert not manager.config_path.exists()


def test_add_gguf_without_service_refuses_before_download(monkeypatch):
    from llm_bawt import gguf_handler

    monkeypatch.setattr(gguf_handler, "_download_gguf_file", Mock(side_effect=AssertionError))
    assert not gguf_handler.handle_add_gguf("org/repo", SimpleNamespace(), service_client=None)


def test_add_vllm_without_service_refuses():
    from llm_bawt.cli import vllm_handler

    assert not vllm_handler.handle_add_vllm(SimpleNamespace(), service_client=None)


def test_save_without_service_does_not_overwrite_models_yaml(tmp_path):
    manager = manager_with({"gpt-5.5": {"type": "openai", "endpoint_id": 7}}, service=False)
    manager.local_only = False
    manager.config_path = tmp_path / "models.yaml"
    manager.config_path.write_text("models:\n  mine: {type: gguf}\n")

    assert not manager.save_config(deleted=1)
    assert manager.config_path.read_text() == "models:\n  mine: {type: gguf}\n"


def test_local_list_reads_catalog_not_models_yaml(monkeypatch):
    built = []

    class FakeManager:
        def __init__(self, config, local_only=False, service_client=None):
            built.append(local_only)

        def list_available_models(self):
            pass

    monkeypatch.setattr(cli, "ModelManager", FakeManager)
    cli.list_models(SimpleNamespace(), service_mode=False)
    assert built == [False]


def test_add_catalog_model_rejects_bare_key_of_multi_endpoint_model(monkeypatch):
    answers = iter(["gpt-5.5", "gpt-5.5", "fresh", "4096"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: next(answers))
    manager = manager_with({
        "gpt-5.5@openai-oauth": {"type": "openai"},
        "gpt-5.5@openai-api": {"type": "openai"},
    })
    manager._catalog = SimpleNamespace(model_keys=lambda: ("gpt-5.5",))

    assert manager.add_catalog_model({"type": "vllm"}, lambda e: "x", lambda d: False)
    assert manager.service_client.upsert_catalog_model.call_args.args[0] == "fresh"
