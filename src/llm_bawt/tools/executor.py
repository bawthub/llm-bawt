"""Tool executor for LLM tool calls.

Executes tool calls by routing them to the appropriate backend
(llm-bawt MCP server, profiles, web search, model management, etc.) and returning formatted results.

Consolidated tools (10 total):
- memory: action-based (search/store/delete)
- history: action-based (search/recent/forget) with date filtering
- profile: action-based (get/set/delete)
- self: action-based (get/set/delete) for bot personality development
- search: type-based (web/news/reddit)
- news: action-based (search/headlines) via NewsAPI
- web_fetch: fetch and read web page content via Crawl4AI
- home: action-based (status/query/get/set/scene) for Home Assistant
- model: action-based (list/current/switch)
- time: current time
"""

import logging
import re
from difflib import SequenceMatcher
from itertools import zip_longest
from datetime import datetime
from typing import Callable, TYPE_CHECKING

from agent_bridge.content_provenance import (
    external_source, provenance_envelope, remote_mcp_source,
)

from .parser import ToolCall, format_tool_result, tool_result_is_error
from .taint import (
    confirmation_required_message,
    label_untrusted,
    needs_confirmation_reason,
)
from .definitions import TOOL_SEARCH_NAME, normalize_legacy_tool_call

if TYPE_CHECKING:
    from ..integrations.ha_mcp.client import HomeAssistantMCPClient, HomeAssistantNativeClient
    from ..integrations.newsapi.client import NewsAPIClient
    from ..integrations.web_fetch.client import WebFetchClient
    from ..search.base import SearchClient
    from ..core.model_lifecycle import ModelLifecycleManager
    from ..utils.config import Config

logger = logging.getLogger(__name__)

# Try to get service logger for verbose tool result logging
try:
    from ..service.logging import get_service_logger
    slog = get_service_logger(__name__)
except ImportError:
    slog = None


class ToolExecutor:
    """Executes local tool calls and discovered (BawtHub and remote) MCP tools.

    Routes tool calls to the appropriate backend and formats results
    for injection back into the conversation.
    """

    # Default maximum number of tool calls per conversation turn (prevent infinite loops)
    DEFAULT_MAX_TOOL_CALLS = 20

    def __init__(
        self,
        search_client: "SearchClient | None" = None,
        home_client: "HomeAssistantMCPClient | None" = None,
        ha_native_client: "HomeAssistantNativeClient | None" = None,
        news_client: "NewsAPIClient | None" = None,
        web_fetch_client: "WebFetchClient | None" = None,
        model_lifecycle: "ModelLifecycleManager | None" = None,
        config: "Config | None" = None,
        user_id: str = "",  # Required - must be passed explicitly
        bot_id: str = "",
        trigger_message_id: str | None = None,
        turn_id: str | None = None,
        content_store=None,
        bawthub_mcp=None,
        turn_capability: str | None = None,
        remote_mcp=None,
        inline_loaded_schemas: bool = False,
    ):
        """Initialize the executor.

        Args:
            search_client: Search client for web search operations.
            home_client: Home Assistant MCP client.
            ha_native_client: Home Assistant native MCP client.
            news_client: NewsAPI client for news search/headlines.
            web_fetch_client: Web fetch client for reading web page content.
            model_lifecycle: Model lifecycle manager for model switching.
            user_id: Current user ID for profile operations (required).
            bot_id: Current bot ID for bot personality operations.
            inline_loaded_schemas: Text-format (ReAct) loops have no schema
                channel, so tool_search also returns the loaded tools' usage.
        """
        if not user_id:
            raise ValueError("user_id is required for ToolExecutor")
        self.search_client = search_client
        self.home_client = home_client
        self.ha_native_client = ha_native_client
        self.news_client = news_client
        self.web_fetch_client = web_fetch_client
        self.model_lifecycle = model_lifecycle
        self.config = config
        self.user_id = user_id
        self.bot_id = bot_id.lower().strip()  # Normalize to match ProfileManager
        self._call_count = 0
        # Per-turn external-content taint (see tools/taint.py).
        self.trigger_message_id = trigger_message_id
        self.turn_id = turn_id
        self._content_store = content_store
        # BawtHub MCP (TASK-2599): allowlisted, called under this turn's own
        # identity; schemas a tool_search loaded join later rounds.
        self.bawthub_mcp = bawthub_mcp
        self.turn_capability = turn_capability
        # Granted remote MCP servers (TASK-2610): a per-turn RemoteMcpTurn whose
        # <slug>__<tool> schemas load through the same tool_search.
        self.remote_mcp = remote_mcp
        self.mcp_loaded: dict = {}
        self.inline_loaded_schemas = inline_loaded_schemas
        self.external_content_seen = False
        # Get max tool calls from config (0 = unlimited)
        self._max_tool_calls = getattr(config, 'MAX_TOOL_CALLS_PER_TURN', self.DEFAULT_MAX_TOOL_CALLS) if config else self.DEFAULT_MAX_TOOL_CALLS

        # Tool dispatch table - maps tool names to handler methods
        # New consolidated tools
        self._handlers: dict[str, Callable[[ToolCall], str]] = {
            # Consolidated tools (new)
            "search": self._execute_search,
            "news": self._execute_news,
            "web_fetch": self._execute_web_fetch,
            "home": self._execute_home,
            "model": self._execute_model,
            "time": self._execute_time,

            # Legacy tools (backward compatibility) - route through consolidated
            "web_search": self._execute_search,
            "news_search": self._execute_search,
            "list_models": self._execute_model,
            "get_current_model": self._execute_model,
            "switch_model": self._execute_model,
            "get_current_time": self._execute_time,
        }

    def reset_call_count(self):
        """Reset local state; durable exposure for the same turn stays sticky."""
        self._call_count = 0
        self.external_content_seen = False

    def _provenance_store(self):
        if self._content_store is None:
            password = getattr(self.config, "POSTGRES_PASSWORD", "")
            if isinstance(password, str) and password.strip():
                from ..service.dependencies import get_content_provenance_store

                self._content_store = get_content_provenance_store(self.config)
        return self._content_store

    def _content_turn(self):
        from ..content_provenance import ContentTurn

        # Non-service callers use their persisted triggering message as the
        # stable turn key. Missing identity is rejected by the store.
        return ContentTurn(
            self.bot_id, self.user_id, self.turn_id or self.trigger_message_id or "",
            self.trigger_message_id or "",
        )

    def _admit_content(self, call: ToolCall, source: str | None) -> str | None:
        """Register external exposure before dispatch.

        Local tools make no prompt-persistent writes: memory and trait writes
        are BawtHub MCP calls, which its server admits against this same turn.
        """
        try:
            store = self._provenance_store()
            if store is not None:
                turn = self._content_turn()
                if source:
                    store.expose(turn)
                else:
                    self.external_content_seen |= store.seen(turn)
            if source:
                self.external_content_seen = True
        except Exception:
            logger.warning("Content provenance source/audit unavailable; native call held")
            return "Held: content provenance is unavailable. Nothing was executed."
        return None

    def can_execute_more(self) -> bool:
        """Check if more tool calls are allowed this turn."""
        # 0 means unlimited
        if self._max_tool_calls == 0:
            return True
        return self._call_count < self._max_tool_calls

    def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return formatted result.

        Args:
            tool_call: The parsed tool call to execute.

        Returns:
            Formatted result string for injection into conversation.
        """
        if not self.can_execute_more():
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Too many tool calls this turn (max {self._max_tool_calls})"
            )
        self._call_count += 1

        # An exact advertised BawtHub MCP name dispatches as itself, before
        # legacy aliases (e.g. memory_search -> memory) can rewrite it.
        mcp = self.bawthub_mcp
        remote = self.remote_mcp
        remote_exact = remote is not None and remote.allows(tool_call.name)
        mcp_exact = remote_exact or (
            (mcp is not None or remote is not None) and tool_call.name == TOOL_SEARCH_NAME
        ) or (mcp is not None and mcp.allows(tool_call.name))

        # Normalize legacy tool names to consolidated tools
        original_name = tool_call.name
        new_name, merged_args = (
            (tool_call.name, tool_call.arguments) if mcp_exact
            else normalize_legacy_tool_call(tool_call.name, tool_call.arguments)
        )

        # Create normalized tool call
        normalized_call = ToolCall(
            name=new_name,
            arguments=merged_args,
            raw_text=tool_call.raw_text,
            tool_use_id=tool_call.tool_use_id,
        )

        if original_name != new_name:
            logger.debug(f"Normalized legacy tool '{original_name}' -> '{new_name}'")

        logger.debug("Executing native tool: %s", normalized_call.name)

        result: str
        source = (remote_mcp_source(normalized_call.name) if remote_exact
                  else external_source(normalized_call.name, native=True))
        try:
            content_hold = self._admit_content(normalized_call, source)
            if content_hold:
                return format_tool_result(normalized_call.name, None, error=content_hold)
            is_ha_native = bool(
                self.ha_native_client and self.ha_native_client.is_ha_tool(normalized_call.name)
            )
            held = self._held_after_external_content(
                normalized_call.name, normalized_call.arguments, ha_native=is_ha_native,
            )
            if held:
                return held

            if is_ha_native:
                return self._execute_ha_native_tool(normalized_call)
            if remote_exact:
                # Falls through to the provenance label below.
                result = self._execute_remote_mcp(normalized_call)
            elif mcp_exact:
                if normalized_call.name == TOOL_SEARCH_NAME:
                    return self._execute_tool_search(normalized_call)
                return self._execute_bawthub_mcp(normalized_call)
            elif handler := self._handlers.get(normalized_call.name):
                result = handler(normalized_call)
            else:
                result = format_tool_result(
                    normalized_call.name,
                    None,
                    error=f"Unknown tool: {normalized_call.name}"
                )
        except Exception as e:
            logger.exception(f"Tool execution failed: {normalized_call.name}")
            result = format_tool_result(normalized_call.name, None, error=str(e))

        if source:
            # Keep the native status wrapper intact and label errors too: a
            # failed read can return an attacker-controlled response body.
            from .parser import strip_tool_result_tags

            data = provenance_envelope(strip_tool_result_tags(result), source)
            labelled = label_untrusted(data)
            result = format_tool_result(
                normalized_call.name, None if tool_result_is_error(result) else labelled,
                error=labelled if tool_result_is_error(result) else None,
            )

        # Log the result in verbose mode
        if slog:
            slog.tool_result(normalized_call.name, result)

        return result

    def mcp_tool_definitions(self) -> list:
        """Tools a tool_search loaded this turn, for the next round's schema."""
        from .definitions import ha_tools_to_tool_definitions

        return ha_tools_to_tool_definitions(list(self.mcp_loaded.values()))

    def _execute_tool_search(self, tool_call: ToolCall) -> str:
        """Deferred discovery: load matching tool schemas for the next step.

        ``select:a,b`` loads exact names; any other query searches names and
        descriptions. The result is the tool_reference list the chat renders.
        """
        import json

        from ..integrations.bawthub_mcp.client import BawtHubMcpToolError

        query = str(tool_call.arguments.get("query") or "").strip()
        try:
            limit = max(1, min(10, int(tool_call.arguments.get("max_results") or 5)))
        except (TypeError, ValueError):
            limit = 5
        selecting = query.lower().startswith("select:")
        wanted = [n.strip() for n in query[7:].split(",") if n.strip()] if selecting else []
        found: list = []
        errors: list[str] = []
        if self.bawthub_mcp is not None:
            try:
                if selecting:
                    catalog = {t.name: t for t in self.bawthub_mcp.search_tools(
                        self.turn_capability, "", limit=len(self.bawthub_mcp.allowlist) or 1)}
                    found = [catalog[n] for n in wanted if n in catalog]
                else:
                    found = self.bawthub_mcp.search_tools(self.turn_capability, query, limit=limit)
            except BawtHubMcpToolError as error:
                errors.append(str(error))
        remote = self.remote_mcp
        if remote is not None:
            # Remote names, descriptions and schemas are server-supplied and
            # reach the model, so the turn is marked exposed before they do.
            extra = remote.select(wanted) if selecting else remote.search_tools(query, limit=limit)
            if extra and not self._expose_remote_discovery():
                extra = []
                errors.append("Remote MCP tools withheld: content provenance is unavailable")
            if selecting:
                found = found + extra
            else:
                # Interleave the two ranked lists so neither starves the other.
                merged = [t for pair in zip_longest(found, extra) for t in pair if t is not None]
                found = merged[:limit]
            if remote.unavailable:
                errors.append("Unavailable MCP servers: " + ", ".join(remote.unavailable))
        if not found:
            if errors:
                return format_tool_result(tool_call.name, None, error="; ".join(errors))
            return format_tool_result(tool_call.name, "No matching tools found.")
        for tool in found:
            self.mcp_loaded[tool.name] = tool
        text = json.dumps([{"type": "tool_reference", "tool_name": tool.name} for tool in found])
        if self.inline_loaded_schemas:
            from .definitions import ha_tools_to_tool_definitions

            usage = "\n".join(t.to_prompt_string() for t in ha_tools_to_tool_definitions(found))
            text += f"\n\nLoaded tools, now callable:\n{usage}"
        return format_tool_result(tool_call.name, text)

    def _execute_bawthub_mcp(self, tool_call: ToolCall) -> str:
        from ..integrations.bawthub_mcp.client import BawtHubMcpToolError

        arguments = {k: v for k, v in (tool_call.arguments or {}).items() if v is not None}
        # Agents are told their slug in the prompt; a regular bot isn't, so
        # default an omitted bot_id to the caller. The server still authorizes.
        loaded = self.mcp_loaded.get(tool_call.name)
        if (self.bot_id and "bot_id" not in arguments and loaded is not None
                and "bot_id" in (loaded.input_schema.get("properties") or {})):
            arguments["bot_id"] = self.bot_id
        if tool_call.name == "profile" and not arguments.get("entity_id"):
            # "My profile" is this bot's; the user's is this turn's user.
            bot_entity = arguments.get("entity_type") == "bot"
            arguments["entity_id"] = self.bot_id if bot_entity else self.user_id
        try:
            text = self.bawthub_mcp.call_tool(
                self.turn_capability, tool_call.name, arguments, tool_use_id=tool_call.tool_use_id)
        except BawtHubMcpToolError as error:
            return format_tool_result(tool_call.name, None, error=str(error))
        return format_tool_result(tool_call.name, text)

    def _expose_remote_discovery(self) -> bool:
        """Record untrusted exposure before remote tool metadata reaches the model."""
        try:
            store = self._provenance_store()
            if store is not None:
                store.expose(self._content_turn())
        except Exception:
            logger.warning("Content provenance unavailable; remote MCP discovery withheld")
            return False
        self.external_content_seen = True
        return True

    def _execute_remote_mcp(self, tool_call: ToolCall) -> str:
        """Call a granted remote MCP server's tool; the caller labels the result untrusted."""
        from ..integrations.mcp_servers.tools import RemoteMcpToolError

        # The remote schema may distinguish null from missing: send what was submitted.
        arguments = dict(tool_call.arguments or {})
        try:
            text = self.remote_mcp.call_tool(tool_call.name, arguments)
        except RemoteMcpToolError as error:
            return format_tool_result(tool_call.name, None, error=str(error))
        return format_tool_result(tool_call.name, text)

    def _held_after_external_content(
        self, tool_name: str, arguments: dict, *, ha_native: bool = False,
    ) -> str | None:
        """Error result if this is a physical-security action in a tainted turn."""
        if not self.external_content_seen:
            return None
        reason = needs_confirmation_reason(tool_name, arguments, ha_native=ha_native)
        if not reason:
            return None
        logger.warning("Held %s after external content this turn: %s", tool_name, reason)
        return format_tool_result(tool_name, None, error=confirmation_required_message(reason))

    def _execute_search(self, tool_call: ToolCall) -> str:
        """Execute search tool - web, news, or reddit search."""
        self._ensure_search_client()
        if not self.search_client:
            error_msg = "Web search not available"
            if self.config:
                try:
                    from ..search import get_search_unavailable_reason
                    error_msg = get_search_unavailable_reason(self.config)
                except Exception:
                    pass
            return format_tool_result(
                tool_call.name,
                None,
                error=error_msg
            )

        search_type = tool_call.arguments.get("type", "web").lower()
        query = tool_call.arguments.get("query", "")
        max_results = tool_call.arguments.get("max_results", tool_call.arguments.get("n_results", 5))

        if not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: query"
            )

        try:
            if search_type == "news":
                time_range = tool_call.arguments.get("time_range", "w")
                results = self.search_client.search_news(
                    query,
                    max_results=max_results,
                    time_range=time_range,
                )
                logger.info(f"News search '{query}' returned {len(results)} results")
            elif search_type == "reddit":
                time_range = tool_call.arguments.get("time_range", "w")
                results = self.search_client.search_reddit(
                    query,
                    max_results=max_results,
                    time_range=time_range,
                )
                logger.info(f"Reddit search '{query}' returned {len(results)} results")
            elif search_type == "web":
                results = self.search_client.search(query, max_results=max_results)
                logger.info(f"Web search '{query}' returned {len(results)} results")
            else:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Invalid search type: '{search_type}'. Use 'web', 'news', or 'reddit'."
                )

            formatted = self.search_client.format_results_for_llm(results)
            note = self._search_fallback_note()
            if note:
                formatted = f"{note}\n\n{formatted}"
            return format_tool_result(tool_call.name, formatted)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _search_fallback_note(self) -> str | None:
        """Say so when a stand-in served the configured provider's search.

        With SEARCH_PROVIDER set to a provider that can't be used (e.g. Brave
        disconnected), ``get_search_client`` falls back to DuckDuckGo; the
        results must not read as that provider's (TASK-1562).
        """
        preferred = str(getattr(self.config, "SEARCH_PROVIDER", None) or "").lower()
        served = getattr(getattr(self.search_client, "PROVIDER", None), "value", None)
        if not preferred or not served or served == preferred:
            return None
        try:
            from ..search import get_search_unavailable_reason
            reason = get_search_unavailable_reason(self.config, preferred)
        except Exception:  # noqa: BLE001 — the note is advisory
            reason = f"{preferred} is unavailable."
        return f"Note: {reason} These results are from {served} instead."

    def _ensure_search_client(self) -> None:
        """Lazy-init the search client, and keep it on what the factory would serve now.

        Every call re-checks the selection and stored credentials, so a
        disconnect, reconnect, rotation or provider change applies to an
        existing executor; on a change the client is rebuilt.
        """
        if not self.config:
            return
        if self.search_client:
            from ..search.factory import client_is_current

            try:
                if client_is_current(self.config, self.search_client):
                    return
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to re-check the search credentials: {type(e).__name__}")
            self.search_client = None
        try:
            from ..search import get_search_client
            self.search_client = get_search_client(self.config)
        except Exception as e:
            logger.warning(f"Failed to initialize search client: {e}")

    def _execute_news(self, tool_call: ToolCall) -> str:
        """Execute NewsAPI tool - article search or headlines."""
        self._ensure_news_client()
        if not self.news_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="NewsAPI not available. Connect a NewsAPI key in Settings → Providers.",
            )

        action = tool_call.arguments.get("action", "search").lower()
        query = tool_call.arguments.get("query", "")
        max_results = tool_call.arguments.get("max_results", tool_call.arguments.get("n_results", 5))

        try:
            if action == "headlines":
                country = tool_call.arguments.get("country", "us")
                category = tool_call.arguments.get("category")
                result = self.news_client.headlines(
                    query=query or None,
                    max_results=max_results,
                    country=country,
                    category=category,
                )
                logger.info(f"News headlines (country={country}, category={category}) returned")
            else:
                if not query:
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error="Missing required parameter: query (required for action='search')",
                    )
                sort_by = tool_call.arguments.get("sort_by", "publishedAt")
                result = self.news_client.search(
                    query=query,
                    max_results=max_results,
                    sort_by=sort_by,
                )
                logger.info(f"News search '{query}' returned")

            return format_tool_result(tool_call.name, result)

        except Exception as e:
            logger.error(f"News tool failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _ensure_news_client(self) -> None:
        """Match the NewsAPI client to the key resolved for this call.

        Resolved on every call, so a key connected, rotated or disconnected
        in Settings takes effect without a restart. Without a config the
        client handed in (if any) is used as is.
        """
        if not self.config:
            return
        try:
            from ..search.credentials import resolve_key
            api_key = resolve_key(self.config, "newsapi").value
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to resolve the NewsAPI key: {e}")
            self.news_client = None
            return
        if not api_key:
            self.news_client = None
            return
        if self.news_client is None or getattr(self.news_client, "_api_key", None) != api_key:
            from ..integrations.newsapi.client import NewsAPIClient
            self.news_client = NewsAPIClient(api_key=api_key)
            logger.debug("NewsAPI client initialized")

    def _execute_web_fetch(self, tool_call: ToolCall) -> str:
        """Execute web fetch tool - retrieve web page content."""
        if not self.web_fetch_client:
            self._ensure_web_fetch_client()
        if not self.web_fetch_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Web fetch service not available. Is the crawl4ai container running?",
            )

        url = tool_call.arguments.get("url", "")
        if not url:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: url",
            )

        try:
            result = self.web_fetch_client.fetch(url)
            logger.info(f"Web fetch '{url}' returned {len(result)} chars")
            return format_tool_result(tool_call.name, result)
        except Exception as e:
            logger.error(f"Web fetch tool failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _ensure_web_fetch_client(self) -> None:
        """Lazy-init web fetch client if service is available."""
        if self.web_fetch_client:
            return
        try:
            from ..integrations.web_fetch.client import WebFetchClient
            client = WebFetchClient()
            if client.is_available():
                self.web_fetch_client = client
                logger.debug("Web fetch client initialized")
            else:
                logger.warning("Crawl4AI service not reachable")
        except Exception as e:
            logger.warning(f"Failed to initialize web fetch client: {e}")

    def _execute_home(self, tool_call: ToolCall) -> str:
        """Execute Home Assistant tool."""
        if not self.home_client:
            self._ensure_home_client()
        if not self.home_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Home Assistant integration not available",
            )

        action = str(tool_call.arguments.get("action", "")).strip().lower()
        if not action:
            return format_tool_result(
                tool_call.name,
                None,
                error="Missing required parameter: action",
            )

        try:
            if action == "status":
                text = self.home_client.status()
                battery_match = re.search(r"Battery:\s*([-+]?\d+(?:\.\d+)?)%", text)
                if battery_match:
                    try:
                        battery_pct = float(battery_match.group(1))
                        if battery_pct < 0 or battery_pct > 100:
                            text += (
                                "\n\n[warning] Reported battery percent is outside 0-100. "
                                "This likely indicates the MCP status formatter is reading a non-percentage source sensor."
                            )
                    except ValueError:
                        pass
                return format_tool_result(tool_call.name, text)

            if action == "status_raw":
                text = self.home_client.status_raw()
                return format_tool_result(tool_call.name, text)

            if action == "query":
                pattern = tool_call.arguments.get("pattern")
                domain = tool_call.arguments.get("domain")
                text = self.home_client.query(pattern=pattern, domain=domain)
                return format_tool_result(tool_call.name, text)

            if action == "get":
                entity = str(tool_call.arguments.get("entity", "")).strip()
                if not entity:
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error="action='get' requires 'entity'",
                    )
                text = self.home_client.get(entity=entity)
                attempted_resolution = False
                if self._home_needs_resolution(text):
                    resolved = self._home_resolve_entity(entity)
                    if resolved:
                        attempted_resolution = True
                        text = self.home_client.get(entity=resolved)

                if self._home_needs_resolution(text):
                    suggestions = self._home_suggest_entities(entity)
                    hint = ""
                    if suggestions:
                        hint = f" Try one of: {', '.join(suggestions)}"
                    retry_note = " after auto-resolution" if attempted_resolution else ""
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error=(
                            f"Entity '{entity}' not found{retry_note}. "
                            f"Call home with action='query' first to find the exact entity ID.{hint}"
                        ),
                    )
                return format_tool_result(tool_call.name, text)

            if action == "set":
                entity = str(tool_call.arguments.get("entity", "")).strip()
                state = str(tool_call.arguments.get("state", "")).strip().lower()
                if not entity or not state:
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error="action='set' requires 'entity' and 'state'",
                    )
                valid_states = {"on", "off", "toggle", "open", "close", "stop", "lock", "unlock"}
                if state not in valid_states:
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error=f"state must be one of: {', '.join(sorted(valid_states))}",
                    )

                brightness_raw = tool_call.arguments.get("brightness")
                brightness: int | None = None
                if brightness_raw is not None:
                    brightness = max(0, min(100, int(brightness_raw)))

                text = self.home_client.set(entity=entity, state=state, brightness=brightness)
                attempted_resolution = False
                if "multiple matches" in (text or "").lower():
                    # Never pick a device for a state change the user didn't choose.
                    suggestions = self._home_suggest_entities(entity, max_items=5)
                    hint = f" Candidates: {', '.join(suggestions)}" if suggestions else ""
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error=(
                            f"Entity '{entity}' matches multiple devices; nothing was changed. "
                            f"Ask which one, then retry with the exact entity ID.{hint}"
                        ),
                    )

                if self._home_needs_resolution(text):
                    resolved = self._home_resolve_entity(entity)
                    if resolved:
                        # Re-gate on the real target: a harmless alias can resolve to a lock/cover.
                        held = self._held_after_external_content(
                            "home", {"action": "set", "entity": resolved, "state": state},
                        )
                        if held:
                            return held
                        attempted_resolution = True
                        text = self.home_client.set(entity=resolved, state=state, brightness=brightness)

                if self._home_needs_resolution(text):
                    suggestions = self._home_suggest_entities(entity)
                    hint = ""
                    if suggestions:
                        hint = f" Try one of: {', '.join(suggestions)}"
                    retry_note = " after auto-resolution" if attempted_resolution else ""
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error=(
                            f"Entity '{entity}' not found{retry_note}. "
                            f"Call home with action='query' first to find the exact entity ID.{hint}"
                        ),
                    )
                return format_tool_result(tool_call.name, text)

            if action == "scene":
                scene_name = str(
                    tool_call.arguments.get("scene_name")
                    or tool_call.arguments.get("name")
                    or ""
                ).strip()
                if not scene_name:
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error="action='scene' requires 'scene_name'",
                    )
                text = self.home_client.scene(name=scene_name)
                return format_tool_result(tool_call.name, text)

            return format_tool_result(
                tool_call.name,
                None,
                error="Invalid action. Use: status, status_raw, query, get, set, or scene.",
            )

        except Exception as e:
            logger.error(f"Home tool failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _ensure_home_client(self) -> None:
        """Lazy-init Home Assistant MCP client if enabled in config."""
        if self.home_client or not self.config:
            return
        if not getattr(self.config, "HA_MCP_ENABLED", False):
            return
        try:
            from ..integrations.ha_mcp.client import HomeAssistantMCPClient

            client = HomeAssistantMCPClient(self.config)
            if client.available:
                self.home_client = client
        except Exception as e:
            logger.warning(f"Failed to initialize Home Assistant MCP client: {e}")

    def _execute_ha_native_tool(self, tool_call: ToolCall) -> str:
        """Execute an HA native MCP tool call (passthrough to HA)."""
        if not self.ha_native_client:
            return format_tool_result(tool_call.name, None, error="HA native MCP client not available")

        tool_name = tool_call.name
        arguments = tool_call.arguments or {}

        # Strip None values — HA doesn't want them
        clean_args = {k: v for k, v in arguments.items() if v is not None}

        logger.info(f"HA native tool call: {tool_name}({clean_args})")
        try:
            result = self.ha_native_client.call_tool(tool_name, clean_args)
            result_len = len(result) if isinstance(result, str) else 0
            logger.debug(f"HA native tool result length: {result_len} chars")
            return format_tool_result(tool_name, result or "Done")
        except Exception as e:  # HAToolError, or anything unexpected
            logger.error(f"HA native tool call failed: {tool_name}: {e}")
            return format_tool_result(tool_name, None, error=str(e))

    @staticmethod
    def _home_needs_resolution(text: str) -> bool:
        lower = (text or "").lower()
        return "not found" in lower or "multiple matches" in lower

    def _home_resolve_entity(self, entity: str) -> str | None:
        """Resolve guessed entity names to real HA entity IDs via query()."""
        if not self.home_client:
            return None

        domain = None
        raw = entity.strip()
        if "." in raw:
            domain = raw.split(".", 1)[0].strip().lower()

        # Build candidate search phrases from guessed IDs/names.
        tail = raw.split(".", 1)[1] if "." in raw else raw
        normalized = tail.replace("_", " ").replace(".", " ").strip()
        compact = normalized.replace(" ", "")
        no_numbers = re.sub(r"\b\d+\b", " ", normalized).strip()
        no_words = re.sub(r"\b(light|lights|lamp|lamps|switch|switches)\b", " ", no_numbers, flags=re.IGNORECASE)
        room_hint = " ".join([t for t in re.split(r"\s+", no_words.strip()) if len(t) >= 3])
        candidates = [
            raw,
            tail,
            normalized,
            compact,
            normalized.replace("sun room", "sunroom"),
            no_numbers,
            room_hint,
        ]
        # de-dup while preserving order
        deduped: list[str] = []
        for c in candidates:
            c = c.strip()
            if c and c not in deduped:
                deduped.append(c)

        found_ids: list[str] = []
        for candidate in deduped[:4]:
            try:
                result = self.home_client.query(pattern=candidate, domain=domain)
            except Exception:
                continue
            ids = self._home_extract_entity_ids(result)
            for eid in ids:
                if eid not in found_ids:
                    found_ids.append(eid)
            if found_ids:
                break

        if not found_ids:
            return None

        wanted_tokens = self._home_normalized_tokens(tail)

        # Rank by similarity to requested name, prefer exact domain and token overlap.
        def score_and_overlap(eid: str) -> tuple[float, bool]:
            eid_tail = eid.split(".", 1)[-1].lower()
            base = SequenceMatcher(None, tail.lower(), eid_tail).ratio()
            if domain and eid.startswith(f"{domain}."):
                base += 0.2
            if "all_" in eid and "light" in eid:
                base += 0.05
            overlap = self._home_has_token_overlap(wanted_tokens, self._home_normalized_tokens(eid_tail))
            if overlap:
                base += 0.3
            return base, overlap

        ranked = sorted(found_ids, key=lambda eid: score_and_overlap(eid)[0], reverse=True)
        best = ranked[0]
        best_score, has_overlap = score_and_overlap(best)
        if not has_overlap or best_score < 0.55:
            return None
        if len(ranked) > 1:
            runner_score, runner_overlap = score_and_overlap(ranked[1])
            if runner_overlap and best_score - runner_score < 0.05:
                return None  # ambiguous: no unique best match
        return best

    def _home_suggest_entities(self, entity: str, max_items: int = 3) -> list[str]:
        """Return likely entity IDs for user-facing recovery hints."""
        if not self.home_client:
            return []

        domain = None
        raw = entity.strip()
        if "." in raw:
            domain = raw.split(".", 1)[0].strip().lower()

        tail = raw.split(".", 1)[1] if "." in raw else raw
        simplified = re.sub(r"\b\d+\b", " ", tail.replace("_", " ")).strip()
        tokens = [t for t in re.split(r"\s+", simplified) if len(t) >= 3]
        if not tokens:
            tokens = [tail.replace("_", " ").strip()]

        phrase = " ".join(tokens[:2]).strip()
        if not phrase:
            phrase = tail.replace("_", " ").strip()

        try:
            result = self.home_client.query(pattern=phrase, domain=domain)
        except Exception:
            return []

        ids = self._home_extract_entity_ids(result)
        if not ids:
            return []

        def score(eid: str) -> float:
            return SequenceMatcher(None, tail.lower(), eid.split(".", 1)[-1].lower()).ratio()

        ranked = sorted(ids, key=score, reverse=True)
        return ranked[:max_items]

    @staticmethod
    def _home_normalized_tokens(value: str) -> list[str]:
        """Extract meaningful normalized tokens from an entity-ish string."""
        cleaned = value.lower().replace(".", " ").replace("_", " ")
        cleaned = re.sub(r"\b(light|lights|lamp|lamps|switch|switches|entity|device)\b", " ", cleaned)
        cleaned = re.sub(r"\b\d+\b", " ", cleaned)
        parts = [p for p in re.split(r"\s+", cleaned) if len(p) >= 3]
        normalized: list[str] = []
        for token in parts:
            compact = token.strip()
            if compact and compact not in normalized:
                normalized.append(compact)
        joined = "".join(parts)
        if len(joined) >= 5 and joined not in normalized:
            normalized.append(joined)
        return normalized

    @staticmethod
    def _home_has_token_overlap(wanted: list[str], candidate: list[str]) -> bool:
        """True when any meaningful token overlaps between requested and candidate IDs."""
        if not wanted or not candidate:
            return False
        for w in wanted:
            for c in candidate:
                if w == c or w in c or c in w:
                    return True
        return False

    @staticmethod
    def _home_extract_entity_ids(query_text: str) -> list[str]:
        """Extract entity IDs from HA query output lines."""
        if not query_text:
            return []

        ids: list[str] = []
        # Matches both "(light.foo_bar)" and bare "light.foo_bar"
        pattern = re.compile(r"\b([a-z_]+\.[a-z0-9_]+)\b")
        for match in pattern.findall(query_text):
            if match not in ids:
                ids.append(match)
        return ids

    @staticmethod
    def _compact_home_text(text: str, max_lines: int = 8, max_chars_per_line: int = 180) -> str:
        """Normalize Home Assistant MCP output without truncating content."""
        if not text:
            return ""

        lines: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            lines.append(line)

        return "\n".join(lines)

    def _execute_model(self, tool_call: ToolCall) -> str:
        """Execute model tool - list, current, or switch models."""
        if not self.model_lifecycle:
            return format_tool_result(
                tool_call.name,
                None,
                error="Model management not available"
            )

        action = tool_call.arguments.get("action", "").lower()
        
        # Default to 'current' if action is empty (common when LLM doesn't provide required arg)
        if not action:
            logger.debug("Model tool called with empty action, defaulting to 'current'")
            action = "current"

        if action == "list":
            return self._model_list(tool_call)
        elif action == "current":
            return self._model_current(tool_call)
        elif action == "switch":
            return self._model_switch(tool_call)
        else:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid action: '{action}'. Use 'list', 'current', or 'switch'."
            )

    def _model_list(self, tool_call: ToolCall) -> str:
        """List available models."""
        try:
            models = self.model_lifecycle.get_available_models()
            current = self.model_lifecycle.current_model

            if not models:
                return format_tool_result(
                    tool_call.name,
                    "No models configured."
                )

            lines = ["Available models:"]
            for model in sorted(models):
                if model == current:
                    lines.append(f"  * {model} (current)")
                else:
                    lines.append(f"  * {model}")

            return format_tool_result(tool_call.name, "\n".join(lines))

        except Exception as e:
            logger.error(f"List models failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _model_current(self, tool_call: ToolCall) -> str:
        """Get current model info."""
        try:
            current = self.model_lifecycle.current_model

            if not current:
                return format_tool_result(
                    tool_call.name,
                    "No model currently loaded."
                )

            model_info = self.model_lifecycle.get_model_info(current)
            if model_info:
                model_type = model_info.get("type", "unknown")
                model_id = model_info.get("model_id", model_info.get("repo_id", ""))
                description = model_info.get("description", "")

                result = f"Current model: {current}\n"
                result += f"  Type: {model_type}\n"
                if model_id:
                    result += f"  Model ID: {model_id}\n"
                if description:
                    result += f"  Description: {description}"
            else:
                result = f"Current model: {current}"

            return format_tool_result(tool_call.name, result)

        except Exception as e:
            logger.error(f"Get current model failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _model_switch(self, tool_call: ToolCall) -> str:
        """Switch to a different model."""
        model_name = tool_call.arguments.get("model_name", "")

        if not model_name:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='switch' requires 'model_name' parameter"
            )

        try:
            success, message = self.model_lifecycle.switch_model(
                model_name, owner=(self.bot_id, self.user_id)
            )

            if success:
                logger.info(f"Model switch requested: {model_name}")
                return format_tool_result(tool_call.name, message)
            else:
                return format_tool_result(tool_call.name, None, error=message)

        except Exception as e:
            logger.error(f"Switch model failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_time(self, tool_call: ToolCall) -> str:
        """Execute time tool - returns current date and time."""
        now = datetime.now()
        # Format: "Thursday, January 23, 2026 at 6:45 PM"
        datetime_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
        # Also include timezone if available
        try:
            import time
            tz_name = time.tzname[time.daylight] if time.daylight else time.tzname[0]
            datetime_str += f" ({tz_name})"
        except Exception:
            pass

        return format_tool_result(tool_call.name, datetime_str)
