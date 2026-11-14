"""Handler prefixes must not mutate records or leak Rich escaping to files."""
import io
import logging
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler

from llm_bawt.shared.logging import LogConfig


def test_debug_handlers_prefix_once_without_mutating_record(monkeypatch, tmp_path):
    from llm_bawt.shared import logging as module
    root = logging.getLogger()
    old_handlers, old_level = list(root.handlers), root.level
    names = (*LogConfig.NOISY_LOGGERS, *LogConfig.VERY_NOISY_LOGGERS, "llm_bawt")
    old_levels = {name: logging.getLogger(name).level for name in names}
    output = io.StringIO()
    monkeypatch.setattr(module, "Console", lambda **kwargs: Console(file=output, color_system=None, width=120))
    monkeypatch.setattr("llm_bawt.shared.logfmt.is_tty", lambda: True)
    monkeypatch.setenv("LLM_BAWT_LOG_PREFIX", "fixture")
    monkeypatch.setenv("LLM_BAWT_LOG_DIR", str(tmp_path))
    try:
        LogConfig.configure(debug=True)
        record = logging.LogRecord("fixture.logger", logging.INFO, __file__, 1, "Hello %s", ("world",), None)
        root.handle(record)
        assert record.msg == "Hello %s" and record.args == ("world",)
        assert output.getvalue().count("[fixture]") == 1
        assert "Hello world" in output.getvalue()
        file_handler = next(h for h in root.handlers if isinstance(h, RotatingFileHandler))
        file_handler.flush()
        line = (tmp_path / "fixture.debug.log").read_text()
        assert line.count("[fixture]") == 1 and "\\[fixture]" not in line
        console_handler = next(h for h in root.handlers if isinstance(h, RichHandler))
        # Formatting order must not change either handler's result.
        assert file_handler.format(record).count("[fixture]") == 1
        assert console_handler.format(record).count("[fixture]") == 1
        assert record.msg == "Hello %s" and record.args == ("world",)
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)
