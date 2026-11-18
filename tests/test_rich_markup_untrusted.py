"""TASK-2434: untrusted text must never be parsed as Rich markup.

A message with a closing-tag-shaped token in its first 300 characters aborted
the agent turn: the verbose "-> LLM" preview printed it through a Rich console
(quiet off a TTY, but Rich parses before it discards) and MarkupError escaped
into the turn worker. Logging must render such text literally and must never
raise into the turn it describes.
"""

from __future__ import annotations

import io
import logging
from types import SimpleNamespace as NS

import pytest

from llm_bawt.service import logging as service_logging
from llm_bawt.shared.rich_safe import SafeConsole, SafeRichHandler

# Built from parts so this file never carries the literal near a message start.
CLOSING = "[" + "/api/auth" + "]"
STYLED = "[bold red]" + "loud" + "[/bold red]"


def _message(token: str, at: int = 190) -> str:
    text = "x" * at + token + " trailing words"
    assert text.index(token) == at
    return text


def _messages(user_text: str) -> list:
    return [
        NS(role="system", content="system prompt"),
        NS(role="user", content="earlier"),
        NS(role="assistant", content="ok"),
        NS(role="user", content=user_text),
    ]


@pytest.fixture
def verbose_logger(monkeypatch):
    """A verbose ServiceLogger on the console the service really builds."""

    def make(*, tty: bool):
        monkeypatch.setattr(service_logging, "_console", None)
        monkeypatch.setattr(service_logging._logfmt, "is_tty", lambda: tty)
        console = service_logging.get_console()
        console.file = io.StringIO()
        console.width = 400
        if tty:
            # Same console class, but plain output so assertions read text.
            monkeypatch.setattr(console, "_force_terminal", False)
            monkeypatch.setattr(console, "_color_system", None)
        monkeypatch.setattr(service_logging, "_verbose", True)
        return service_logging.ServiceLogger("test.task2434"), console

    return make


@pytest.fixture
def production_handler(monkeypatch):
    """The Rich handler setup_service_logging installs on a terminal."""
    monkeypatch.setattr(service_logging, "_console", None)
    monkeypatch.setattr(service_logging._logfmt, "is_tty", lambda: True)
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    saved_levels = {
        name: lg.level for name, lg in logging.root.manager.loggerDict.items()
        if isinstance(lg, logging.Logger)
    }
    monkeypatch.setenv("LLM_BAWT_LOG_PREFIX", "")
    for var in ("HF_HUB_DISABLE_PROGRESS_BARS", "TRANSFORMERS_NO_ADVISORY_WARNINGS",
                "TRANSFORMERS_VERBOSITY", "TOKENIZERS_PARALLELISM", "TQDM_DISABLE"):
        monkeypatch.setenv(var, "1")
    try:
        service_logging.setup_service_logging(verbose=False, debug=False)
        handler = root.handlers[0]
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)
    handler.console.file = io.StringIO()
    handler.console.width = 400
    monkeypatch.setattr(handler.console, "_force_terminal", False)
    monkeypatch.setattr(handler.console, "_color_system", None)
    return handler


@pytest.mark.parametrize("token", [CLOSING, STYLED])
def test_quiet_console_never_parses_untrusted_preview(verbose_logger, token):
    # Production shape: app container, stdout not a TTY -> quiet console.
    log, console = verbose_logger(tty=False)
    assert console.quiet
    log.llm_context(_messages(_message(token)))
    log.llm_response(_message(token))
    assert console.file.getvalue() == ""


@pytest.mark.parametrize("token", [CLOSING, STYLED])
def test_terminal_preview_prints_untrusted_text_literally(verbose_logger, token):
    log, console = verbose_logger(tty=True)
    text = _message(token)
    log.llm_context(_messages(text))
    log.api_request(
        service_logging.RequestContext(
            request_id="r1", method="POST", path="/v1/chat/completions",
            bot_id="caid" + CLOSING, model=STYLED,
        ),
        {"messages": [{"role": "user", "content": token + " first"}]},
    )
    log.tool_result("tool" + CLOSING, "one\n" + token + "\nthree")
    log._log_result_item({"role": "user", "content": token}, 0)
    log._log_result_item({"content": token, "tags": ["t"]}, 1)
    out = console.file.getvalue()
    # Every untrusted string survives as plain characters, token included.
    assert text in out
    assert (token + " first") in out
    assert out.count(token) >= 5


def test_structured_log_methods_never_raise(verbose_logger, monkeypatch):
    log, console = verbose_logger(tty=True)

    def boom(*_a, **_k):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(console, "print", boom)
    # Every structured method swallows its own failure.
    log.llm_context(_messages("hi"))
    log.llm_response("hi")
    log.tool_result("t", "r")
    log.api_request(service_logging.RequestContext(request_id="r", method="GET", path="/"),
                    {"messages": [{"role": "user", "content": "hi"}]})


def test_safe_console_falls_back_to_literal_on_bad_markup():
    console = SafeConsole(file=io.StringIO(), force_terminal=False, width=200)
    console.print("prefix " + CLOSING)
    assert ("prefix " + CLOSING) in console.file.getvalue()


def _emit(handler, msg: str, extra: dict | None = None) -> str:
    logger = logging.getLogger("test.task2434.handler")
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        logger.warning("got %s", msg, extra=extra)
    finally:
        logger.handlers[:] = []
    return handler.console.file.getvalue()


@pytest.mark.parametrize("token", [CLOSING, STYLED])
def test_terminal_log_handler_renders_untrusted_records_literally(production_handler, token):
    assert ("got " + token) in _emit(production_handler, token)


def test_markup_record_with_bad_markup_still_logs():
    handler = SafeRichHandler(
        console=SafeConsole(file=io.StringIO(), force_terminal=False, width=400),
        show_time=False, show_path=False, show_level=False,
    )
    # A record that opted into markup but carries a bad tag prints literally.
    assert CLOSING in _emit(handler, CLOSING, {"markup": True})


def test_turn_worker_preview_call_survives_token(verbose_logger, monkeypatch):
    """The exact production call site: TurnStreamWorker logs llm_context."""
    import llm_bawt.service.turn_stream_worker as worker_mod

    _, console = verbose_logger(tty=False)
    worker_log = worker_mod.log
    assert isinstance(worker_log, service_logging.ServiceLogger)
    monkeypatch.setattr(worker_log, "_console", console)
    worker_log.llm_context(_messages(_message(CLOSING)))
    worker_log.llm_context(_messages(_message(STYLED)))


CONCEAL = "[conceal]" + "hidden" + "[/conceal]"


def _plain_console(monkeypatch, module):
    console = SafeConsole(file=io.StringIO(), force_terminal=False, width=400)
    monkeypatch.setattr(module, "console", console)
    return console


@pytest.mark.parametrize("token", [STYLED, CONCEAL])
def test_profile_previews_print_profile_text_literally(monkeypatch, token):
    """Valid tags parse without error, so they must be escaped, not just survived."""
    from llm_bawt.core import prompt_sections
    from llm_bawt.core.prompt_builder import PromptBuilder

    console = _plain_console(monkeypatch, prompt_sections)
    profiles = NS(
        get_user_profile_summary=lambda user_id, curated=False: "likes " + token,
        get_bot_profile_summary=lambda bot_id: "tends " + token,
    )
    owner = NS(
        _db_available=True, profile_manager=profiles, verbose=True,
        user_id="user-1" + token, bot_id="bot-1" + token,
        _resolve_setting=lambda _key, default=None: default,
    )
    builder = PromptBuilder()
    prompt_sections.PromptSectionsMixin._sec_user_context(owner, builder, "")
    prompt_sections.PromptSectionsMixin._sec_bot_traits(owner, builder, "")
    out = console.file.getvalue()
    for literal in ("likes " + token, "tends " + token, "user-1" + token, "bot-1" + token):
        assert literal in out
    # The prompt itself still carries the profile text unchanged.
    assert ("likes " + token) in builder.build()


def test_persona_title_prints_literally(monkeypatch):
    from llm_bawt.core import base as core_base
    import llm_bawt.prompt_registry as registry

    console = _plain_console(monkeypatch, core_base)
    row = NS(category=registry.PERSONA_CATEGORY, title="Persona " + CONCEAL,
             key="k", body="persona body")
    monkeypatch.setattr(registry, "PromptTemplateStore",
                        lambda _config: NS(get_by_id=lambda _id: row))
    owner = NS(bot=NS(prompt_override_id=7, system_prompt="base"), config=None,
               verbose=True, bot_id="bot-1")
    assert core_base.BaseLLMBawt._resolve_base_prompt(owner) == "persona body"
    assert ("Persona " + CONCEAL) in console.file.getvalue()
