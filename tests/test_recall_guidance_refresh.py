"""TASK-1495: registry edits to recall guidance reach the next assembled prompt
without rebuilding the bot."""

from types import SimpleNamespace

from llm_bawt.core.prompt_builder import PromptBuilder
from llm_bawt.core.prompt_manifest import PER_TURN, PROMPT_MANIFEST
from llm_bawt.core.prompt_sections import PromptSectionsMixin


def test_global_instructions_is_per_turn():
    spec = next(s for s in PROMPT_MANIFEST if s.name == "global_instructions")
    assert spec.stage == PER_TURN


class _Resolver:
    def __init__(self):
        self.bodies = {}
        self.calls = []

    def resolve_body(self, key, scope_type="global", scope_id=None):
        self.calls.append((key, scope_type, scope_id))
        body = self.bodies.get((scope_type, scope_id)) or self.bodies.get(("global", None))
        if body is None:
            return None
        return SimpleNamespace(body=body, source="db_override")


class _Bot(PromptSectionsMixin):
    def __init__(self, resolver):
        self.bot = SimpleNamespace(requires_memory=True)
        self.bot_id = "fixture"
        self.config_resolver = resolver


def _render(bot):
    b = PromptBuilder()
    bot._sec_global_instructions(b, "")
    return b.build()


def test_v1_to_v2_update_reaches_next_render_and_bot_scope_wins():
    resolver = _Resolver()
    bot = _Bot(resolver)
    resolver.bodies[("global", None)] = "RECALL v1"
    assert "RECALL v1" in _render(bot)

    resolver.bodies[("global", None)] = "RECALL v2"
    out = _render(bot)
    assert "RECALL v2" in out and "RECALL v1" not in out
    assert resolver.calls[-1] == ("chat.global_recall_guidance", "bot", "fixture")

    resolver.bodies[("bot", "fixture")] = "RECALL bot"
    assert "RECALL bot" in _render(bot)
