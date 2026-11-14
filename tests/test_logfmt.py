"""Standard log line (TASK-2225)."""

from __future__ import annotations

import io
import logging
import re

import pytest

from agent_bridge import logfmt


@pytest.fixture
def capture():
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    buf = io.StringIO()
    logfmt.configure_logging("app", stream=buf)
    yield buf
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])


def lines(buf: io.StringIO) -> list[str]:
    return [line for line in buf.getvalue().splitlines() if line]


TS = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z"


def test_line_layout_utc_ms_level_component(capture):
    logging.getLogger("claude_code_bridge.proxy.routes").info("proxy_stream_complete status=%s", 200)
    logging.getLogger("llm_bawt.service.x").warning("slow_upstream ms=9000")
    out = lines(capture)
    assert re.fullmatch(TS + r" INFO proxy proxy_stream_complete status=200", out[0])
    assert re.fullmatch(TS + r" WARN app slow_upstream ms=9000", out[1])


def test_correlation_fields_fixed_order_and_never_capped(capture):
    long_req = "req_delivery_" + "a" * 300
    with logfmt.log_context(task="TASK-1", bot="al", req=long_req):
        logging.getLogger("llm_bawt.x").info("turn_start blob=%s", "b" * 500)
    (line,) = lines(capture)
    assert line.endswith(f"req={long_req} bot=al task=TASK-1")
    assert "blob=" + "b" * 200 + "… blob_len=500" in line


def test_existing_inline_correlation_not_duplicated(capture):
    with logfmt.log_context(req="r1"):
        logging.getLogger("llm_bawt.x").info("turn_end req=r1 ms=5")
    (line,) = lines(capture)
    assert line.count("req=r1") == 1


def test_no_ansi_markup_or_newlines(capture):
    logging.getLogger("llm_bawt.x").info("[request_id]\\[abc][/request_id] \x1b[32mok\x1b[0m\nnext")
    (line,) = lines(capture)
    assert "\x1b" not in line and "[/" not in line
    assert line.endswith("[abc] ok\\nnext")


def test_traceback_only_for_error(capture):
    log = logging.getLogger("llm_bawt.x")
    try:
        raise ValueError("boom")
    except ValueError:
        log.warning("retrying", exc_info=True)
        log.error("failed", exc_info=True)
    out = capture.getvalue().splitlines()
    assert out[0].endswith('retrying exc="ValueError: boom"')
    assert out[1].endswith("ERROR app failed")
    assert out[2].startswith("Traceback")


def test_redaction(capture):
    logging.getLogger("llm_bawt.x").info("call Authorization: Bearer abcdefghijklmnop token=supersecret123")
    (line,) = lines(capture)
    assert "abcdefghijklmnop" not in line and "supersecret123" not in line


def test_access_filter_drops_polling_and_keeps_errors(capture):
    access = logging.getLogger("uvicorn.access")
    msg = '%s - "%s %s HTTP/%s" %d'
    access.info(msg, "1.2.3.4:5", "GET", "/v1/providers/claude/token", "1.1", 200)
    access.info(msg, "1.2.3.4:5", "GET", "/v1/health/checks?x=1", "1.1", 200)
    access.info(msg, "1.2.3.4:5", "POST", "/v1/chat/completions?k=secret", "1.1", 200)
    access.info(msg, "1.2.3.4:5", "GET", "/v1/campus", "1.1", 503)
    out = lines(capture)
    assert len(out) == 2
    assert out[0].endswith("INFO http http_access method=POST path=/v1/chat/completions status=200")
    assert out[1].endswith("WARN http http_access method=GET path=/v1/campus status=503")


def test_debug_stays_out_of_stdout_but_reaches_debug_file(tmp_path):
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    buf = io.StringIO()
    path = tmp_path / "d.log"
    try:
        logfmt.configure_logging("app", stream=buf, debug_file=str(path))
        logging.getLogger("llm_bawt.x").debug("token_fetch state=ok")
        for h in root.handlers:
            h.flush()
        assert buf.getvalue() == ""
        assert "DEBUG app token_fetch state=ok" in path.read_text()
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])


def test_json_mode(monkeypatch, capture):
    import json

    root = logging.getLogger()
    buf = io.StringIO()
    monkeypatch.setenv("LOG_FORMAT", "json")
    logfmt.configure_logging("app", stream=buf)
    with logfmt.log_context(req="r9"):
        logging.getLogger("llm_bawt.x").info("turn_end")
    rec = json.loads(buf.getvalue())
    assert rec["level"] == "INFO" and rec["msg"] == "turn_end" and rec["req"] == "r9"
    root.handlers.clear()


def test_json_summary_values_are_not_cut(capture):
    summary = '{"a":' + '"' + "x" * 300 + '","observer_errors":0,"last_productive_ms":12}'
    logging.getLogger("claude_code_bridge.proxy").info("responses_progress req=r1 summary=%s", summary)
    (line,) = lines(capture)
    assert line.endswith("summary=" + summary)


def test_whole_message_cap_keeps_inline_correlation(capture):
    logging.getLogger("llm_bawt.x").error("event %s req=keep-this-request session=al:nick", "y" * 4100)
    (line,) = lines(capture)
    assert "req=keep-this-request" in line and "session=al:nick" in line
    assert "len=" in line


def test_capped_quoted_value_stays_well_formed(capture):
    logging.getLogger("llm_bawt.x").info("event detail=%s", logfmt.quote("x " * 160))
    (line,) = lines(capture)
    value = re.search(r'detail=("(?:[^"\\]|\\.)*") detail_len=320', line)
    assert value and value.group(1).endswith('…"')


def test_redaction_covers_quoted_values_and_exceptions(capture):
    log = logging.getLogger("llm_bawt.x")
    log.info('call api_key="SYNTHETIC_KEY_12345"')
    try:
        raise ValueError("Bearer SYNTHETIC_TOKEN_12345")
    except ValueError:
        log.warning("upstream", exc_info=True)
        log.error("upstream", exc_info=True)
    text = capture.getvalue()
    assert "SYNTHETIC_KEY_12345" not in text and "SYNTHETIC_TOKEN_12345" not in text
    assert "Traceback" in text


def test_uvicorn_config_keeps_error_access_records_without_debug(capture, monkeypatch):
    """uvicorn.Config with the service's non-TTY level must not mute 5xx."""
    import uvicorn

    uvicorn.Config(app=lambda *a: None, log_config=None, log_level="info")
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "1.2.3.4:5", "GET", "/v1/campus", "1.1", 503)
    assert lines(capture)[-1].endswith("WARN http http_access method=GET path=/v1/campus status=503")


@pytest.mark.parametrize("json_mode", [False, True])
def test_redaction_of_quoted_secret_in_exception_summary(monkeypatch, json_mode):
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    buf = io.StringIO()
    if json_mode:
        monkeypatch.setenv("LOG_FORMAT", "json")
    try:
        logfmt.configure_logging("app", stream=buf)
        try:
            raise ValueError('api_key="SYNTHETIC_KEY_12345"')
        except ValueError:
            logging.getLogger("llm_bawt.x").warning("upstream", exc_info=True)
            logging.getLogger("llm_bawt.x").error("upstream", exc_info=True)
        assert "SYNTHETIC_KEY_12345" not in buf.getvalue()
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
