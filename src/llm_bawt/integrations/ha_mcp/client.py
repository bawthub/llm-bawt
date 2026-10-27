"""Home Assistant MCP client.

Two modes:
1. Native MCP — connects directly to HA's /api/mcp via Streamable HTTP.
   Discovers tools dynamically. This is the preferred mode.
2. Legacy MCP — connects to the custom FastMCP server via SSE (backward compat).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from ...utils.config import Config
from ..ha_credentials import HACredentials, HomeAssistantUnavailable, resolve_ha_credentials

logger = logging.getLogger(__name__)


@dataclass
class HAToolDefinition:
    """A tool discovered from HA's native MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


class HomeAssistantNativeClient:
    """Client that connects to HA's native MCP server at /api/mcp.

    On init, connects via Streamable HTTP, calls list_tools() to discover
    available tools, caches them, and provides call_tool() for execution.
    """

    def __init__(self, config: Config):
        # URL and token are resolved for every discover/call (TASK-1476), so a
        # token saved or rotated under Integrations is used without a restart.
        self._config = config
        self._timeout = float(max(1, int(getattr(config, "HA_MCP_TIMEOUT", 10) or 10)))

        # Parse exclude list from config
        exclude_str = (getattr(config, "HA_MCP_TOOL_EXCLUDE", "") or "").strip()
        self._exclude_names: set[str] = set()
        if exclude_str:
            self._exclude_names = {n.strip() for n in exclude_str.split(",") if n.strip()}

        # Cached tool definitions (populated by discover_tools())
        self._tools: list[HAToolDefinition] = []
        self._tool_names: set[str] = set()
        self._initialized = False

    def _credentials(self) -> HACredentials | None:
        """Current credentials; ``None`` when none are configured or usable."""
        try:
            return resolve_ha_credentials(self._config)
        except HomeAssistantUnavailable as error:
            logger.warning("HA native MCP unavailable: %s", error.code)
            return None

    @property
    def available(self) -> bool:
        """True if URL and token are configured."""
        return self._credentials() is not None

    @property
    def initialized(self) -> bool:
        """True if tools have been discovered."""
        return self._initialized

    @property
    def tools(self) -> list[HAToolDefinition]:
        """Return cached tool definitions."""
        return self._tools

    @property
    def tool_names(self) -> set[str]:
        """Return set of available tool names."""
        return self._tool_names

    def is_ha_tool(self, name: str) -> bool:
        """Check if a tool name belongs to HA native tools."""
        return name in self._tool_names

    def discover_tools(self) -> list[HAToolDefinition]:
        """Connect to HA MCP, list tools, cache and return them.

        Filters out tools in the exclude list. Safe to call multiple times
        (will re-discover each time).
        """
        credentials = self._credentials()
        if credentials is None:
            logger.warning("HA native MCP not configured (missing URL or token)")
            return []

        try:
            raw_tools = self._run_async(self._discover_tools_async(credentials))
            self._tools = []
            self._tool_names = set()

            raw_tool_names = {t.name for t in raw_tools}
            for t in raw_tools:
                if t.name in self._exclude_names:
                    logger.debug(f"Excluding HA tool: {t.name}")
                    continue
                tool_def = HAToolDefinition(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {"type": "object", "properties": {}},
                )
                self._tools.append(tool_def)
                self._tool_names.add(t.name)

            self._initialized = True
            excluded_count = len(self._exclude_names & raw_tool_names)
            logger.info(
                f"Discovered {len(self._tools)} HA native MCP tools "
                f"(excluded {excluded_count})"
            )
            return self._tools

        except Exception as e:
            logger.error(f"Failed to discover HA native MCP tools: {e}")
            self._initialized = False
            return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call an HA tool by name and return the text result."""
        credentials = self._credentials()
        if credentials is None:
            return "Error: HA native MCP not configured"
        if not self._initialized:
            return "Error: HA native MCP tools not discovered yet"
        if name not in self._tool_names:
            return f"Error: Unknown HA tool '{name}'"

        try:
            result = self._run_async(self._call_tool_async(credentials, name, arguments))
            return self._extract_text(result)
        except Exception as e:
            logger.error(f"HA tool call failed: {name}({arguments}): {e}")
            return f"Error calling HA tool {name}: {e}"

    # --- Async internals ---

    async def _discover_tools_async(self, credentials: HACredentials):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"Authorization": f"Bearer {credentials.token}"}
        async with streamablehttp_client(credentials.mcp_url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools

    async def _call_tool_async(self, credentials: HACredentials, name: str, arguments: dict[str, Any]):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"Authorization": f"Bearer {credentials.token}"}
        async with streamablehttp_client(credentials.mcp_url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)

    # --- Sync/async bridge ---

    @staticmethod
    def _run_async(coro):
        """Run an async coroutine from sync context, handling existing event loops."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Any = None
        error: BaseException | None = None

        def _runner():
            nonlocal result, error
            try:
                result = asyncio.run(coro)
            except BaseException as exc:
                error = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(timeout=30)
        if error is not None:
            raise error
        return result

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Extract text content from MCP tool result."""
        if hasattr(result, "content"):
            parts: list[str] = []
            for item in getattr(result, "content", []) or []:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            joined = "\n".join(p for p in parts if p).strip()
            if joined:
                return joined

        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined

        if isinstance(result, str):
            return result.strip()

        return json.dumps(result, default=str, ensure_ascii=False)


# --- Legacy client kept for backward compatibility ---


class HomeAssistantMCPClient:
    """Typed client for Home Assistant MCP tool calls."""

    def __init__(self, config: Config):
        self._base_url = (getattr(config, "HA_MCP_URL", "") or "").rstrip("/")
        self._timeout = float(max(1, int(getattr(config, "HA_MCP_TIMEOUT", 10) or 10)))
        self._auth_token = (getattr(config, "HA_MCP_AUTH_TOKEN", "") or "").strip()  # Reserved for future authenticated transports.

    @property
    def available(self) -> bool:
        return bool(self._base_url)

    def status(self) -> str:
        return self._call_text_tool("status", {})

    def status_raw(self) -> str:
        """Return raw MCP payload for status tool (debugging)."""
        result = self._call_tool(name="status", arguments={})
        return json.dumps(self._to_serializable(result), indent=2, default=str, ensure_ascii=False)

    def query(self, pattern: str | None = None, domain: str | None = None) -> str:
        args: dict[str, Any] = {}
        if pattern:
            args["pattern"] = pattern
        if domain:
            args["domain"] = domain
        return self._call_text_tool("query", args)

    def get(self, entity: str) -> str:
        return self._call_text_tool("get", {"entity": entity})

    def set(self, entity: str, state: str, brightness: int | None = None) -> str:
        args: dict[str, Any] = {"entity": entity, "state": state}
        if brightness is not None:
            args["brightness"] = brightness
        return self._call_text_tool("set", args)

    def scene(self, name: str) -> str:
        return self._call_text_tool("scene", {"name": name})

    def _call_text_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = self._call_tool(name=name, arguments=arguments)
        text = self._extract_text(result)
        if text:
            return text
        return json.dumps(result, default=str)

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._base_url:
            raise RuntimeError("HA MCP URL is not configured")
        sse_url = f"{self._base_url}/sse"
        try:
            return self._run_async(self._call_tool_async(sse_url, name, arguments))
        except Exception as e:
            raise ConnectionError(f"HA MCP request failed: {e}") from e

    async def _call_tool_async(self, sse_url: str, name: str, arguments: dict[str, Any]) -> Any:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Any = None
        error: BaseException | None = None

        def _runner():
            nonlocal result, error
            try:
                result = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover - defensive
                error = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if error is not None:
            raise error
        return result

    @staticmethod
    def _extract_text(result: Any) -> str:
        if hasattr(result, "content"):
            parts: list[str] = []
            for item in getattr(result, "content", []) or []:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(p for p in parts if p).strip()

        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif item.get("type") == "json":
                            parts.append(json.dumps(item.get("json"), default=str))
                return "\n".join(p for p in parts if p).strip()

            structured = result.get("structuredContent")
            if isinstance(structured, dict) and "result" in structured:
                embedded = structured["result"]
                if isinstance(embedded, str):
                    return embedded.strip()

        if isinstance(result, str):
            return result.strip()
        return ""

    @classmethod
    def _to_serializable(cls, value: Any) -> Any:
        """Convert MCP SDK objects to JSON-serializable structures."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {str(k): cls._to_serializable(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [cls._to_serializable(item) for item in value]

        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
                return cls._to_serializable(dumped)
            except Exception:
                pass

        if hasattr(value, "dict"):
            try:
                dumped = value.dict()
                return cls._to_serializable(dumped)
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                return cls._to_serializable(vars(value))
            except Exception:
                pass

        return str(value)
