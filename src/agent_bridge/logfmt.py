"""One standard log line for every llm-bawt process (TASK-2225).

Layout (the only place it is defined; ``LOG_FORMAT=json`` switches to JSON):

    <UTC ISO ts with ms>Z <LEVEL> <component> <message> [req=… bot=… task=… session=… tool_use=…]

* One line per event. Newlines inside a message are escaped; an ERROR's
  traceback is the only multi-line output and follows its record once.
* No ANSI colour, Rich markup, padding or wrapping.
* Correlation fields come from :func:`log_context` (a contextvar) and from
  ``extra={"req": …}``; they are never truncated. Other ``key=value``
  values longer than ``VALUE_CAP`` are cut with a ``<key>_len=`` marker.
* A redaction filter masks bearer tokens and API keys as a backstop.

Readers are bots grepping ``docker logs``: grep ``req=<id>`` to follow a
turn, `` ERROR `` or `` WARN `` for problems, or an event name.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import sys
import time
from typing import Any, Iterator, TextIO

CORRELATION_KEYS: tuple[str, ...] = ("req", "bot", "task", "session", "tool_use")
VALUE_CAP = 200
MESSAGE_CAP = 4000

_LEVEL_NAMES = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "ERROR",
}

# Logger-name prefix -> short component. Longest prefix wins.
COMPONENTS: dict[str, str] = {
    "llm_bawt.mcp_server": "mcp",
    "llm_bawt.ops": "ops",
    "llm_bawt.service.inter_bot": "dispatch",
    "llm_bawt": "app",
    "claude_code_bridge.proxy": "proxy",
    "claude_code_bridge": "bridge",
    "codex_bridge": "bridge",
    "agent_bridge": "bridge",
    "claude_agent_sdk": "sdk",
    "local_model_bridge": "local",
    "uvicorn": "http",
    "httpx": "http",
    "fastmcp": "mcp",
    "mcp": "mcp",
}

_ctx: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar("llm_bawt_log_ctx", default={})


@contextlib.contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach correlation fields (req, bot, task, session, tool_use) to every
    record logged inside the block, including from awaited coroutines."""
    merged = dict(_ctx.get())
    merged.update({k: str(v) for k, v in fields.items() if v not in (None, "")})
    token = _ctx.set(merged)
    try:
        yield
    finally:
        _ctx.reset(token)


def bind_log_context(**fields: Any) -> None:
    """Set correlation fields for the rest of the current task/context."""
    merged = dict(_ctx.get())
    merged.update({k: str(v) for k, v in fields.items() if v not in (None, "")})
    _ctx.set(merged)


def utc_timestamp(created: float) -> str:
    t = time.gmtime(created)
    return f"{time.strftime('%Y-%m-%dT%H:%M:%S', t)}.{int((created % 1) * 1000):03d}Z"


def component_for(logger_name: str, default: str) -> str:
    best = ""
    for prefix in COMPONENTS:
        if (logger_name == prefix or logger_name.startswith(prefix + ".")) and len(prefix) > len(best):
            best = prefix
    return COMPONENTS[best] if best else default


_MARKUP_TAG = re.compile(r"\[/?[a-zA-Z_.][\w. ]*\]")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_KV = re.compile(r"(?<![\w.])([A-Za-z_][\w.]*)=(\"(?:[^\"\\]|\\.)*\"|\S+)")


def _plain(msg: str, markup: bool) -> str:
    msg = _ANSI.sub("", msg)
    if markup and "[/" in msg:
        parts = msg.split("\\[")  # an escaped bracket is literal text, never a tag
        msg = "[".join(_MARKUP_TAG.sub("", p) if i == 0 else p[:1] + _MARKUP_TAG.sub("", p[1:]) for i, p in enumerate(parts))
    return msg


def quote(value: Any) -> str:
    s = str(value)
    if s == "" or any(c in s for c in ' "=\n\t'):
        return json.dumps(s, ensure_ascii=False)
    return s


def _is_correlation(key: str) -> bool:
    return key in CORRELATION_KEYS or key in ("request_id", "session_id") or key.endswith("_id")


def cap_values(msg: str) -> str:
    """Cut long ``key=value`` values. Correlation ids are never cut, and
    JSON object/array values (bounded diagnostic summaries) are kept whole.
    A cut quoted value is decoded, shortened and re-quoted."""

    def _cut(m: re.Match[str]) -> str:
        key, value = m.group(1), m.group(2)
        if _is_correlation(key) or len(value) <= VALUE_CAP or value[:1] in "{[":
            return m.group(0)
        if value.startswith('"'):
            try:
                raw = json.loads(value)
            except ValueError:
                raw = value[1:]
            return f"{key}={json.dumps(raw[:VALUE_CAP] + '…', ensure_ascii=False)} {key}_len={len(raw)}"
        return f"{key}={value[:VALUE_CAP]}… {key}_len={len(value)}"

    return _KV.sub(_cut, msg)


def cap_message(msg: str) -> str:
    """Whole-message cap that keeps every correlation ``key=value`` token."""
    if len(msg) <= MESSAGE_CAP:
        return msg
    head = msg[:MESSAGE_CAP]
    kept = [m.group(0) for m in _KV.finditer(msg)
            if _is_correlation(m.group(1)) and m.group(0) not in head]
    return " ".join([f"{head}… len={len(msg)}", *kept])


_REDACTIONS = (
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1***"),
    (re.compile(r"\b(sk-(?:ant-)?[A-Za-z0-9_-]{6})[A-Za-z0-9_-]{10,}"), r"\1***"),
    (re.compile(r"(?i)\b((?:access_|refresh_|api_)?(?:token|key|secret|password))=(\"[^\"]{6,}\"|'[^']{6,}'|[^\s&\"']{6,})"), r"\1=***"),
    (re.compile(r"(?i)([\"']?(?:access_token|refresh_token|api_key|apikey|secret|password|authorization)[\"']?\s*:\s*)([\"'])[^\"']{6,}\2"), r"\1\2***\2"),
)


def redact(text: str) -> str:
    for pat, rep in _REDACTIONS:
        text = pat.sub(rep, text)
    return text


class RedactFilter(logging.Filter):
    """Backstop on the message; the formatter also redacts the final line
    (including exception text), so this only keeps other handlers clean."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        red = redact(msg)
        if red != msg:
            record.msg, record.args = red, ()
        return True


class LogfmtFormatter(logging.Formatter):
    """The standard line. ``json_mode`` renders the same fields as JSON."""

    def __init__(self, component: str, *, json_mode: bool = False) -> None:
        super().__init__()
        self.component = component
        self.json_mode = json_mode

    def fields(self, record: logging.LogRecord) -> dict[str, Any]:
        msg = _plain(record.getMessage(), getattr(record, "markup", True) is not False)
        corr = dict(_ctx.get())
        for key in CORRELATION_KEYS:
            val = getattr(record, key, None)
            if val not in (None, ""):
                corr[key] = str(val)
        return {
            "ts": utc_timestamp(record.created),
            "level": _LEVEL_NAMES.get(record.levelno, record.levelname),
            "component": getattr(record, "component", None) or component_for(record.name, self.component),
            "msg": msg,
            "corr": corr,
        }

    def format(self, record: logging.LogRecord) -> str:
        f = self.fields(record)
        exc = None
        if record.exc_info and record.levelno >= logging.ERROR:
            exc = self.formatException(record.exc_info)
        elif record.exc_info:
            e = record.exc_info[1]
            f["corr"]["exc"] = f"{type(e).__name__}: {e}" if e else "unknown"
        # Redact raw strings before quoting/JSON escaping hides their shape.
        f["corr"] = {k: redact(v) for k, v in f["corr"].items()}
        if exc:
            exc = redact(exc)
        if self.json_mode:
            out: dict[str, Any] = {"ts": f["ts"], "level": f["level"], "component": f["component"], "msg": f["msg"]}
            out.update(f["corr"])
            if exc:
                out["traceback"] = exc
            return redact(json.dumps(out, ensure_ascii=False, default=str))
        msg = cap_message(cap_values(f["msg"]).replace("\r", "").replace("\n", "\\n"))
        present = [k for k in CORRELATION_KEYS if k in f["corr"] and f"{k}=" not in msg]
        extras = [k for k in f["corr"] if k not in CORRELATION_KEYS]
        tail = " ".join(f"{k}={quote(f['corr'][k])}" for k in present + extras)
        line = f"{f['ts']} {f['level']} {f['component']} {msg}"
        if tail:
            line = f"{line} {tail}"
        if exc:
            line = f"{line}\n{exc}"
        line = redact(line)
        return line


# Polling endpoints whose successful access lines are pure noise.
QUIET_ACCESS_PATHS = re.compile(
    r"^/(?:v1/)?(?:health(?:/|$)|providers/(?:claude/token|health)|campus|models|turn-logs|"
    r"tool-approval-policies/bundle|tool-approval-requests|tool-calls|admin/gates|ops/jobs/live|"
    r"inter-bot-deliveries/[^/]+$|history|mcp$)"
)


class AccessLogFilter(logging.Filter):
    """Rewrite uvicorn access records as ``http_access`` and drop quiet ones."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not (isinstance(args, tuple) and len(args) == 5):
            return True
        _client, method, full_path, _ver, status = args
        try:
            code = int(status)
        except (TypeError, ValueError):
            code = 0
        path = str(full_path).split("?", 1)[0]
        if code < 400 and QUIET_ACCESS_PATHS.match(path):
            return False
        record.msg = "http_access method=%s path=%s status=%s"
        record.args = (method, path, code)
        record.levelno, record.levelname = (
            (logging.WARNING, "WARNING") if code >= 500 else (record.levelno, record.levelname)
        )
        return True


NOISY_LIBRARIES: tuple[str, ...] = (
    "httpx", "httpcore", "urllib3", "asyncio", "openai", "anthropic", "huggingface_hub",
    "sentence_transformers", "transformers", "filelock", "multipart", "watchfiles",
    "sse_starlette", "mcp.server.lowlevel.server", "mcp.server.streamable_http_manager",
)


def env_level(default: int = logging.INFO) -> int:
    name = os.getenv("LLM_BAWT_LOG_LEVEL", "").strip().upper()
    return getattr(logging, "WARNING" if name == "WARN" else name, default) if name else default


def configure_logging(
    component: str,
    *,
    level: int | None = None,
    debug_file: str | None = None,
    stream: TextIO | None = None,
) -> logging.Handler:
    """Install the standard stdout handler on the root logger.

    ``level`` is the stdout level (default ``LLM_BAWT_LOG_LEVEL`` or INFO).
    ``debug_file`` additionally keeps full DEBUG in a rotating file, which is
    what the ``--debug`` flags mean now; DEBUG never goes to stdout unless
    ``LLM_BAWT_LOG_LEVEL=DEBUG``.
    """
    from logging.handlers import RotatingFileHandler

    stdout_level = env_level() if level is None else level
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(LogfmtFormatter(component, json_mode=os.getenv("LOG_FORMAT", "").lower() == "json"))
    handler.setLevel(stdout_level)
    handler.addFilter(RedactFilter())
    root.addHandler(handler)
    root_level = stdout_level
    if debug_file:
        try:
            os.makedirs(os.path.dirname(debug_file) or ".", exist_ok=True)
            fh = RotatingFileHandler(debug_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
            fh.setFormatter(LogfmtFormatter(component))
            fh.setLevel(logging.DEBUG)
            fh.addFilter(RedactFilter())
            root.addHandler(fh)
            root_level = logging.DEBUG
        except OSError as exc:
            root.warning("debug_log_unavailable path=%s error=%s", debug_file, exc)
    root.setLevel(root_level)
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(logging.INFO)
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, AccessLogFilter) for f in access.filters):
        access.addFilter(AccessLogFilter())
    logging.captureWarnings(True)
    return handler


def is_tty(stream: TextIO | None = None) -> bool:
    s = stream or sys.stderr
    try:
        return bool(s.isatty())
    except Exception:
        return False
