"""Server-mode JSON-RPC transport for the memory client facade (TASK-1222)."""

from __future__ import annotations

import logging
from typing import Any

from llm_bawt.bot_access.service import SERVICE_HEADER

# Logger identity is part of the extraction contract: MCP call logs keep the
# ``llm_bawt.mcp_server.client`` name (and its level/handler config) after the
# move, for both the stdlib logger and the ServiceLogger summary line.
_CLIENT_LOGGER_NAME = "llm_bawt.mcp_server.client"
logger = logging.getLogger(_CLIENT_LOGGER_NAME)


class MemoryClientTransportMixin:
    """HTTP ``tools/call`` client plus its debug response logging."""

    # -------------------------------------------------------------------------
    # Server Mode Helpers
    # -------------------------------------------------------------------------

    def _call_server(self, method: str, params: dict) -> Any:
        """Call an MCP server method via HTTP.

        This is a simple HTTP client for the MCP server. For production use,
        consider using the official MCP client library.
        """
        import json
        import time
        import urllib.request
        import urllib.error
        from uuid import uuid4

        url = f"{self.server_url}/mcp"
        call_id = uuid4().hex[:8]
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": method,
                "arguments": params,
            },
            "id": call_id,
        }

        # If we're running inside llm-service, propagate its per-request id into logs.
        request_id = None
        try:
            from llm_bawt.service.logging import get_request_id  # type: ignore

            request_id = get_request_id()
        except Exception:
            request_id = None

        # Use ServiceLogger for human-friendly output if available, otherwise standard logging
        try:
            from llm_bawt.service.logging import get_service_logger
            slog = get_service_logger(_CLIENT_LOGGER_NAME)
            # Log a human-friendly summary (call + result combined after the fact)
            use_service_logger = True
        except Exception:
            slog = None
            use_service_logger = False

        # For DEBUG mode, log technical details
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("MCP request -> tools/%s params=%s", method, params)

        data = json.dumps(payload).encode('utf-8')
        headers = {
            "Content-Type": "application/json",
            # MCP streamable-http transport requires Accept header to include text/event-stream
            "Accept": "application/json, text/event-stream",
            SERVICE_HEADER: self._service_principal.mint(method, params),
        }

        start = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                duration_ms = (time.perf_counter() - start) * 1000

                if "error" in result:
                    logger.error(f"MCP server error: {result['error']}")
                    if use_service_logger and slog:
                        slog.mcp_operation(method, bot_id=self.bot_id, duration_ms=duration_ms, success=False)
                    raise RuntimeError(result['error'].get('message', 'Unknown error'))

                # FastMCP returns JSON-RPC result of the form:
                # {"result": {"content": [...], "structuredContent": {"result": <actual_data>}}}
                rpc_result = result.get("result")
                # An explicit JSON null is a decoded domain value (notably an
                # absent sessions_get row), not a signal to return the MCP wrapper.
                undecoded = object()
                parsed_result = undecoded
                if isinstance(rpc_result, dict):
                    if rpc_result.get("isError"):
                        # MCP tool failures are valid JSON-RPC responses, not
                        # successful domain values. Never pass their text to
                        # MessageResult.from_dict (or other typed consumers).
                        details = "; ".join(
                            item["text"] for item in (rpc_result.get("content") or [])
                            if isinstance(item, dict) and isinstance(item.get("text"), str)
                        )
                        raise RuntimeError(f"MCP tool {method} failed: {details or 'unknown tool error'}")
                    # Prefer structuredContent which has the actual typed data
                    structured = rpc_result.get("structuredContent")
                    if isinstance(structured, dict) and "result" in structured:
                        parsed_result = structured["result"]
                    # Fallback to parsing content array
                    elif "content" in rpc_result:
                        content_items = rpc_result.get("content") or []
                        if content_items and isinstance(content_items, list):
                            first = content_items[0]
                            if isinstance(first, dict):
                                if first.get("type") == "json" and "json" in first:
                                    parsed_result = first.get("json")
                                elif first.get("type") == "text" and "text" in first:
                                    text = first.get("text")
                                    if isinstance(text, str):
                                        stripped = text.lstrip()
                                        if stripped.startswith("{") or stripped.startswith("["):
                                            try:
                                                parsed_result = json.loads(text)
                                            except Exception:
                                                parsed_result = text
                                        else:
                                            parsed_result = text

                if parsed_result is undecoded:
                    parsed_result = rpc_result

                # Human-friendly log via ServiceLogger (shows "Fetching history" instead of raw tool names)
                if use_service_logger and slog:
                    # Count items if result is a list
                    count = len(parsed_result) if isinstance(parsed_result, list) else None
                    # Pass params and result for verbose mode detail display
                    slog.mcp_operation(
                        method,
                        bot_id=self.bot_id,
                        duration_ms=duration_ms,
                        count=count,
                        params=params,
                        result=parsed_result,
                    )
                elif logger.isEnabledFor(logging.DEBUG):
                    # Fallback to debug logging
                    logger.debug("MCP result <- tools/%s %.1fms", method, duration_ms)

                # Verbose logging - show response data (DEBUG level)
                if logger.isEnabledFor(logging.DEBUG):
                    self._log_response_debug(method, parsed_result)

                return parsed_result
        except urllib.error.URLError as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise ConnectionError(f"Failed to connect to MCP server at {self.server_url}: {e}")

    def _log_response_debug(self, method: str, result: Any) -> None:
        """Log full response data at DEBUG level without truncation."""
        import json

        if result is None:
            logger.debug("MCP response: None")
            return

        try:
            logger.debug("MCP response: %s", json.dumps(result, default=str, indent=2))
        except Exception:
            logger.debug("MCP response: %s", result)
