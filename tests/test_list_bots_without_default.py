"""Local --list-bots works without a configured default bot (TASK-2413)."""

from types import SimpleNamespace

import llm_bawt.cli.display_cmd as display


class _Manager:
    def __init__(self, config, local_only=False):
        self._bots = [SimpleNamespace(slug="alpha", name="Alpha", requires_memory=False,
                                      bot_type="chat", agent_backend=None, color=None,
                                      cli_color=None, description="")]

    def get_default_bot(self):  # pragma: no cover - must not be called
        raise AssertionError("listing must not require a default bot")

    def list_bots(self):
        return self._bots

    def select_model(self, alias, bot_slug=None):
        return SimpleNamespace(alias=None)


def _rendered(monkeypatch, default_slug):
    monkeypatch.setattr(display, "BotManager", _Manager)
    monkeypatch.setattr(display, "optional_default_bot_id", lambda config: default_slug)
    with display.console.capture() as captured:
        display._show_bots_from_local(SimpleNamespace(DEFAULT_MODEL_ALIAS=None))
    return captured.get()


def test_lists_bots_without_a_default(monkeypatch):
    out = _rendered(monkeypatch, None)
    assert "alpha" in out and "alpha ⭐" not in out


def test_marks_the_configured_default(monkeypatch):
    out = _rendered(monkeypatch, "alpha")
    assert "alpha ⭐" in out
