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
from datetime import datetime, timedelta
from typing import Callable, TYPE_CHECKING

from .parser import ToolCall, format_tool_result
from .definitions import normalize_legacy_tool_call

if TYPE_CHECKING:
    from ..integrations.ha_mcp.client import HomeAssistantMCPClient, HomeAssistantNativeClient
    from ..integrations.newsapi.client import NewsAPIClient
    from ..integrations.web_fetch.client import WebFetchClient
    from ..mcp_server.client import MemoryClient
    from ..profiles import ProfileManager
    from ..search.base import SearchClient
    from ..core.model_lifecycle import ModelLifecycleManager
    from ..utils.config import Config
    from ..utils.history import HistoryManager

logger = logging.getLogger(__name__)

# Try to get service logger for verbose tool result logging
try:
    from ..service.logging import get_service_logger
    slog = get_service_logger(__name__)
except ImportError:
    slog = None


def parse_date_param(value: str | None) -> float | None:
    """Parse flexible date inputs to Unix timestamp.

    Supports:
    - ISO 8601: "2026-01-30", "2026-01-30T14:00:00"
    - Date with time: "2026-01-30 14:00"
    - Relative: "today", "yesterday", "tomorrow"
    
    Also auto-corrects dates that appear to be from a hallucinated year/month
    by interpreting the day offset relative to today.

    Returns:
        Unix timestamp or None if parsing fails.
    """
    if not value:
        return None

    normalized = value.strip().lower()
    if normalized in {"today", "yesterday", "tomorrow"}:
        now = datetime.now()
        if normalized == "today":
            dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif normalized == "yesterday":
            dt = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        else:  # tomorrow
            dt = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return dt.timestamp()

    formats = [
        "%Y-%m-%d",           # 2026-01-30
        "%Y-%m-%dT%H:%M:%S",  # 2026-01-30T14:00:00
        "%Y-%m-%dT%H:%M",     # 2026-01-30T14:00
        "%Y-%m-%d %H:%M:%S",  # 2026-01-30 14:00:00
        "%Y-%m-%d %H:%M",     # 2026-01-30 14:00
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Check if date is unreasonable (hallucinated)
            # - More than 1 year in the past
            # - More than 7 days in the future
            days_diff = (dt - today).days
            
            if days_diff > 7 or days_diff < -365:
                # LLM hallucinated a completely wrong date
                # Interpret as relative: use the day-of-month offset from "today" in that month
                # But simpler: if asking about "yesterday", the offset from their "today" is -1
                # Check their date vs their presumed "today" (next day in sequence)
                
                # Heuristic: LLM often provides consecutive dates for since/until
                # e.g., since=2024-10-09, until=2024-10-10 (1 day range = "yesterday")
                # So just use the day offset they intended
                
                # For now, if date is way off, assume they meant "yesterday" 
                # and use actual yesterday
                logger.warning(
                    f"Date '{value}' is {days_diff} days from today, "
                    f"interpreting as yesterday (actual: {(today - timedelta(days=1)).strftime('%Y-%m-%d')})"
                )
                dt = today - timedelta(days=1)
            
            return dt.timestamp()
        except ValueError:
            continue

    logger.warning(f"Could not parse date: {value}")
    return None


class ToolExecutor:
    """Executes tool calls using the memory client and profile manager.

    Routes tool calls to the appropriate backend and formats results
    for injection back into the conversation.
    """

    # Default maximum number of tool calls per conversation turn (prevent infinite loops)
    DEFAULT_MAX_TOOL_CALLS = 20

    def __init__(
        self,
        memory_client: "MemoryClient | None" = None,
        profile_manager: "ProfileManager | None" = None,
        search_client: "SearchClient | None" = None,
        home_client: "HomeAssistantMCPClient | None" = None,
        ha_native_client: "HomeAssistantNativeClient | None" = None,
        news_client: "NewsAPIClient | None" = None,
        web_fetch_client: "WebFetchClient | None" = None,
        model_lifecycle: "ModelLifecycleManager | None" = None,
        config: "Config | None" = None,
        user_id: str = "",  # Required - must be passed explicitly
        bot_id: str = "nova",
        history_manager: "HistoryManager | None" = None,
    ):
        """Initialize the executor.

        Args:
            memory_client: Memory client for memory operations.
            profile_manager: Profile manager for user/bot profile operations.
            search_client: Search client for web search operations.
            home_client: Home Assistant MCP client.
            ha_native_client: Home Assistant native MCP client.
            news_client: NewsAPI client for news search/headlines.
            web_fetch_client: Web fetch client for reading web page content.
            model_lifecycle: Model lifecycle manager for model switching.
            user_id: Current user ID for profile operations (required).
            bot_id: Current bot ID for bot personality operations.
            history_manager: History manager for recall operations.
        """
        if not user_id:
            raise ValueError("user_id is required for ToolExecutor")
        self.memory_client = memory_client
        self.profile_manager = profile_manager
        self.search_client = search_client
        self.home_client = home_client
        self.ha_native_client = ha_native_client
        self.news_client = news_client
        self.web_fetch_client = web_fetch_client
        self.model_lifecycle = model_lifecycle
        self.config = config
        self.user_id = user_id
        self.bot_id = bot_id.lower().strip()  # Normalize to match ProfileManager
        self._history_manager = history_manager
        self._call_count = 0
        # Get max tool calls from config (0 = unlimited)
        self._max_tool_calls = getattr(config, 'MAX_TOOL_CALLS_PER_TURN', self.DEFAULT_MAX_TOOL_CALLS) if config else self.DEFAULT_MAX_TOOL_CALLS

        # Tool dispatch table - maps tool names to handler methods
        # New consolidated tools
        self._handlers: dict[str, Callable[[ToolCall], str]] = {
            # Consolidated tools (new)
            "memory": self._execute_memory,
            "history": self._execute_history,
            "profile": self._execute_profile,
            "self": self._execute_self,
            "bot_trait": self._execute_self,  # Legacy name
            "search": self._execute_search,
            "news": self._execute_news,
            "web_fetch": self._execute_web_fetch,
            "home": self._execute_home,
            "model": self._execute_model,
            "time": self._execute_time,

            # Legacy tools (backward compatibility) - route through consolidated
            "search_memories": self._execute_memory,
            "store_memory": self._execute_memory,
            "delete_memory": self._execute_memory,
            "search_history": self._execute_history,
            "get_recent_history": self._execute_history,
            "forget_history": self._execute_history,
            "set_user_attribute": self._execute_profile,
            "get_user_profile": self._execute_profile,
            "delete_user_attribute": self._execute_profile,
            "set_my_trait": self._execute_self,  # Legacy name
            "web_search": self._execute_search,
            "news_search": self._execute_search,
            "list_models": self._execute_model,
            "get_current_model": self._execute_model,
            "switch_model": self._execute_model,
            "get_current_time": self._execute_time,
        }

    def reset_call_count(self):
        """Reset the per-turn call counter."""
        self._call_count = 0

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
        self._call_count += 1

        if not self.can_execute_more():
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Too many tool calls this turn (max {self._max_tool_calls})"
            )

        # Normalize legacy tool names to consolidated tools
        original_name = tool_call.name
        new_name, merged_args = normalize_legacy_tool_call(tool_call.name, tool_call.arguments)

        # Create normalized tool call
        normalized_call = ToolCall(
            name=new_name,
            arguments=merged_args,
            raw_text=tool_call.raw_text,
        )

        if original_name != new_name:
            logger.debug(f"Normalized legacy tool '{original_name}' -> '{new_name}'")

        logger.debug(f"Executing: {normalized_call.name}({normalized_call.arguments})")

        result: str
        try:
            # Check if this is an HA native tool call
            if self.ha_native_client and self.ha_native_client.is_ha_tool(normalized_call.name):
                return self._execute_ha_native_tool(normalized_call)

            handler = self._handlers.get(normalized_call.name)
            if handler:
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

        # Log the result in verbose mode
        if slog:
            slog.tool_result(normalized_call.name, result)

        return result

    # =========================================================================
    # Consolidated Tool Handlers
    # =========================================================================

    def _execute_memory(self, tool_call: ToolCall) -> str:
        """Execute memory tool - search, store, or delete facts."""
        if not self.memory_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Memory system not available"
            )

        action = tool_call.arguments.get("action", "").lower()

        if action == "search":
            return self._memory_search(tool_call)
        elif action == "store":
            return self._memory_store(tool_call)
        elif action == "update":
            return self._memory_update(tool_call)
        elif action == "delete":
            return self._memory_delete(tool_call)
        else:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid action: '{action}'. Use 'search', 'store', 'update', or 'delete'."
            )

    def _memory_search(self, tool_call: ToolCall) -> str:
        """Search memories by semantic similarity."""
        query = tool_call.arguments.get("query", "")
        n_results = tool_call.arguments.get("n_results", 5)

        if not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='search' requires 'query' parameter"
            )

        try:
            from ..memory.context_builder import build_memory_context_string

            results = self.memory_client.search(query, n_results=n_results)
            memories = [
                {
                    "id": r.id,
                    "content": r.content,
                    "relevance": r.relevance,
                    "importance": r.importance,
                    "tags": r.tags,
                    "intent": r.intent,
                    "stakes": r.stakes,
                    "emotional_charge": r.emotional_charge,
                    "created_at": r.created_at,
                    "last_accessed": r.last_accessed,
                }
                for r in results
            ]

            formatted = build_memory_context_string(memories, user_name=self.user_id)
            if not formatted:
                formatted = "No memories found matching your query."
            return format_tool_result(tool_call.name, formatted)

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _memory_store(self, tool_call: ToolCall) -> str:
        """Store a new memory/fact."""
        content = tool_call.arguments.get("content", "")
        importance = tool_call.arguments.get("importance", 0.6)
        tags = tool_call.arguments.get("tags", ["misc"])

        if not content:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='store' requires 'content' parameter"
            )

        # Ensure importance is in valid range
        importance = max(0.0, min(1.0, float(importance)))

        # Ensure tags is a list
        if isinstance(tags, str):
            tags = [tags]

        try:
            result = self.memory_client.store_memory(
                content=content,
                importance=importance,
                tags=tags,
            )

            return format_tool_result(
                tool_call.name,
                f"Memory stored successfully with ID: {result.id}"
            )

        except Exception as e:
            logger.error(f"Memory store failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _memory_delete(self, tool_call: ToolCall) -> str:
        """Delete memories by ID or query."""
        memory_id = tool_call.arguments.get("memory_id", "")
        query = tool_call.arguments.get("query", "")

        if not memory_id and not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='delete' requires 'memory_id' or 'query' parameter"
            )

        try:
            # If query provided, search for matching memories and delete them
            if query:
                memories = self.memory_client.search(
                    query=query,
                    n_results=50,
                    min_relevance=0.3
                )

                if not memories:
                    return format_tool_result(
                        tool_call.name,
                        f"No memories found matching '{query}'"
                    )

                deleted_count = 0
                deleted_contents = []
                for memory in memories:
                    mid = memory.id
                    if mid:
                        success = self.memory_client.delete_memory(mid)
                        if success:
                            deleted_count += 1
                            content = memory.content if memory.content else ""
                            deleted_contents.append(f"- {content}")

                if deleted_count > 0:
                    result = f"Deleted {deleted_count} memories matching '{query}':\n" + "\n".join(deleted_contents)
                    return format_tool_result(tool_call.name, result)
                else:
                    return format_tool_result(
                        tool_call.name,
                        None,
                        error=f"Found {len(memories)} memories but failed to delete them"
                    )

            # Delete by specific ID
            success = self.memory_client.delete_memory(memory_id)

            if success:
                return format_tool_result(
                    tool_call.name,
                    f"Memory {memory_id} deleted successfully"
                )
            else:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Memory {memory_id} not found"
                )

        except Exception as e:
            logger.error(f"Memory delete failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _memory_update(self, tool_call: ToolCall) -> str:
        """Update an existing memory in place."""
        memory_id = tool_call.arguments.get("memory_id", "")
        query = tool_call.arguments.get("query", "")
        content = tool_call.arguments.get("content")
        importance = tool_call.arguments.get("importance")
        tags = tool_call.arguments.get("tags")

        if not memory_id and not query:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='update' requires 'memory_id' or 'query' parameter"
            )

        if content is None and importance is None and tags is None:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='update' requires at least one of: content, importance, tags"
            )

        if isinstance(tags, str):
            tags = [tags]

        target_id = memory_id
        try:
            if not target_id and query:
                matches = self.memory_client.search(query=query, n_results=1, min_relevance=0.3)
                if not matches:
                    return format_tool_result(tool_call.name, f"No memories found matching '{query}'")
                target_id = matches[0].id

            updated = self.memory_client.update_memory(
                memory_id=target_id,
                content=content,
                importance=float(importance) if importance is not None else None,
                tags=tags,
            )
            if not updated:
                return format_tool_result(tool_call.name, None, error=f"Memory {target_id} not found")

            return format_tool_result(
                tool_call.name,
                f"Updated memory {updated.id}: {updated.content}"
            )
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_history(self, tool_call: ToolCall) -> str:
        """Execute history tool - search, recent, or forget messages."""
        if not self.memory_client:
            return format_tool_result(
                tool_call.name,
                None,
                error="Memory system not available"
            )

        action = tool_call.arguments.get("action", "").lower()

        if action == "search":
            return self._history_search(tool_call)
        elif action == "recent":
            return self._history_recent(tool_call)
        elif action == "recall":
            return self._history_recall(tool_call)
        elif action == "forget":
            return self._history_forget(tool_call)
        else:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid action: '{action}'. Use 'search', 'recent', 'recall', or 'forget'."
            )

    def _history_search(self, tool_call: ToolCall) -> str:
        """Search conversation history with optional date filtering."""
        query = tool_call.arguments.get("query", "")
        n_results = tool_call.arguments.get("n_results", 10)
        role_filter = tool_call.arguments.get("role_filter")
        since = tool_call.arguments.get("since")
        until = tool_call.arguments.get("until")

        if not query:
            # If no query but a date filter is provided, treat this as a date-based
            # history request and route to the 'recent' handler.
            if since or until:
                return self._history_recent(tool_call)

            return format_tool_result(
                tool_call.name,
                None,
                error="action='search' requires 'query' parameter (or use action='recent' with since/until)"
            )

        # Parse date filters
        since_ts = parse_date_param(since)
        until_ts = parse_date_param(until)

        try:
            results = self.memory_client.search_messages(
                query=query,
                n_results=n_results,
                role_filter=role_filter,
                since=since_ts,
                until=until_ts,
            )

            if not results:
                msg = f"No messages found matching '{query}'"
                if since:
                    msg += f" since {since}"
                if until:
                    msg += f" until {until}"
                return format_tool_result(tool_call.name, msg)

            # Format results for the model
            lines = [f"Found {len(results)} messages matching '{query}':"]
            for i, msg in enumerate(results, 1):
                content = msg.content
                lines.append(f"{i}. [{msg.role}] {content}")

            return format_tool_result(tool_call.name, "\n".join(lines))

        except Exception as e:
            logger.error(f"History search failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _history_recent(self, tool_call: ToolCall) -> str:
        """Get recent messages with optional date filtering."""
        n_results = tool_call.arguments.get("n_results", 10)
        role_filter = tool_call.arguments.get("role_filter")
        since = tool_call.arguments.get("since")
        until = tool_call.arguments.get("until")

        # Parse date filters
        since_ts = parse_date_param(since)
        until_ts = parse_date_param(until)

        try:
            # Get messages with optional date filtering
            results = self.memory_client.get_messages(
                limit=None,  # Get all, filter ourselves
                since=since_ts,
                until=until_ts,
            )

            if not results:
                return format_tool_result(
                    tool_call.name,
                    "No messages found in history."
                )

            # Filter out summaries and system messages
            results = [r for r in results if r.get("role") not in ("summary", "system")]

            # Filter by role if specified
            if role_filter:
                results = [r for r in results if r.get("role") == role_filter]

            # Sort by timestamp descending and take n_results
            results = sorted(results, key=lambda x: x.get("timestamp", 0), reverse=True)[:n_results]

            if not results:
                return format_tool_result(
                    tool_call.name,
                    f"No {role_filter or 'conversation'} messages found in history."
                )

            # Format results (oldest first for readability)
            results = list(reversed(results))
            lines = [f"Last {len(results)} messages from history:"]
            for i, msg in enumerate(results, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                lines.append(f"{i}. [{role}] {content}")

            return format_tool_result(tool_call.name, "\n".join(lines))

        except Exception as e:
            logger.error(f"Get recent history failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _history_recall(self, tool_call: ToolCall) -> str:
        """Expand a summary back to the original raw messages.

        When context assembly substitutes a summary for older raw messages,
        the bot can call this to get the full conversation for that session.
        The raw messages are stored in-memory (HistoryManager) and also in
        the database, so this is a lookup — no regeneration needed.
        """
        summary_id = tool_call.arguments.get("summary_id", "")
        if not summary_id:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='recall' requires 'summary_id' parameter (the db_id of the summary message)"
            )

        try:
            # Find the summary message in the in-memory history
            if not hasattr(self, '_history_manager') or not self._history_manager:
                # Fall back to memory_client for DB access
                if hasattr(self.memory_client, 'get_short_term_manager'):
                    stm = self.memory_client.get_short_term_manager()
                    if stm and hasattr(stm, 'get_messages_for_summary'):
                        raw_messages = stm.get_messages_for_summary(summary_id)
                        if raw_messages:
                            lines = [f"Recalled {len(raw_messages)} messages from summarized session:"]
                            for i, msg in enumerate(raw_messages, 1):
                                content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                                role = msg.role if hasattr(msg, 'role') else msg.get('role', 'unknown')
                                lines.append(f"{i}. [{role}] {content}")
                            return format_tool_result(tool_call.name, "\n".join(lines))

                return format_tool_result(
                    tool_call.name,
                    None,
                    error="Could not find the summarized session. The summary may be too old or the history manager is unavailable."
                )

            # Search in-memory history for the summary by db_id
            summary_msg = None
            for msg in self._history_manager.messages:
                if hasattr(msg, 'db_id') and msg.db_id == summary_id:
                    summary_msg = msg
                    break

            if not summary_msg or summary_msg.role != "summary":
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"No summary found with id '{summary_id}'. Check the summary_id and try again."
                )

            # Find raw messages near the summary's timestamp
            # Summaries cover a session block — find messages from the same timeframe
            summary_ts = summary_msg.timestamp
            raw_messages = [
                msg for msg in self._history_manager.messages
                if msg.role in ("user", "assistant")
                and msg.db_id != summary_id
                and abs(msg.timestamp - summary_ts) < 7200  # Within 2 hours of summary
            ]

            if not raw_messages:
                return format_tool_result(
                    tool_call.name,
                    "No raw messages found for this summary. The original messages may have been from a previous session."
                )

            # Format the recalled messages
            lines = [f"Recalled {len(raw_messages)} messages from summarized session:"]
            for i, msg in enumerate(raw_messages, 1):
                from datetime import datetime
                ts = datetime.fromtimestamp(msg.timestamp).strftime("%Y-%m-%d %H:%M")
                content = msg.content
                lines.append(f"{i}. [{msg.role} @ {ts}] {content}")

            # Mark these messages as recalled in the database
            if hasattr(self.memory_client, 'mark_messages_recalled'):
                recalled_ids = [msg.db_id for msg in raw_messages if msg.db_id]
                if recalled_ids:
                    try:
                        self.memory_client.mark_messages_recalled(recalled_ids)
                    except Exception as e:
                        logger.debug(f"Failed to mark messages as recalled: {e}")

            return format_tool_result(tool_call.name, "\n".join(lines))

        except Exception as e:
            logger.error(f"History recall failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _history_forget(self, tool_call: ToolCall) -> str:
        """Forget/delete recent messages."""
        # Normalize parameter names
        count = tool_call.arguments.get("count")
        if count is None:
            count = tool_call.arguments.get("n_messages") or tool_call.arguments.get("n") or tool_call.arguments.get("num_messages")

        minutes = tool_call.arguments.get("minutes")
        if minutes is None:
            minutes = tool_call.arguments.get("time_range") or tool_call.arguments.get("mins")

        if count is None and minutes is None:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='forget' requires 'count' (number of messages) or 'minutes' (time range)"
            )

        try:
            if count is not None:
                result = self.memory_client.forget_recent_messages(int(count))
                msg = f"Forgot {result['messages_ignored']} messages"
            else:
                result = self.memory_client.forget_messages_since_minutes(int(minutes))
                msg = f"Forgot {result['messages_ignored']} messages from the last {minutes} minutes"

            if result.get('memories_deleted', 0) > 0:
                msg += f" and {result['memories_deleted']} related memories"

            return format_tool_result(tool_call.name, msg)

        except Exception as e:
            logger.error(f"Forget history failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_profile(self, tool_call: ToolCall) -> str:
        """Execute profile tool - get, set, or delete user attributes."""
        if not self.profile_manager:
            return format_tool_result(
                tool_call.name,
                None,
                error="Profile system not available"
            )

        action = tool_call.arguments.get("action", "").lower()

        if action == "get":
            return self._profile_get(tool_call)
        elif action == "set":
            return self._profile_set(tool_call)
        elif action == "update":
            return self._profile_update(tool_call)
        elif action == "delete":
            return self._profile_delete(tool_call)
        else:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid action: '{action}'. Use 'get', 'set', 'update', or 'delete'."
            )

    def _profile_get(self, tool_call: ToolCall) -> str:
        """Get user profile summary."""
        try:
            summary = self.profile_manager.get_user_profile_summary(self.user_id)

            if not summary:
                return format_tool_result(
                    tool_call.name,
                    "No profile attributes stored for this user yet."
                )

            return format_tool_result(tool_call.name, summary)

        except Exception as e:
            logger.error(f"Get user profile failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _profile_set(self, tool_call: ToolCall) -> str:
        """Set a user profile attribute."""
        category = tool_call.arguments.get("category", "")
        key = tool_call.arguments.get("key", "")
        value = tool_call.arguments.get("value")
        confidence = tool_call.arguments.get("confidence", 0.8)

        if not category or not key:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='set' requires 'category' and 'key' parameters"
            )

        if value is None:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='set' requires 'value' parameter (cannot be null)"
            )

        valid_categories = ["preference", "fact", "interest", "communication", "context"]
        if category.lower() not in valid_categories:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )

        try:
            from ..profiles import EntityType

            logger.info(f"Setting user attribute: {category}.{key} = {value} for user {self.user_id}")

            self.profile_manager.set_attribute(
                entity_type=EntityType.USER,
                entity_id=self.user_id,
                category=category.lower(),
                key=key,
                value=value,
                confidence=float(confidence),
                source="inferred",
            )

            return format_tool_result(
                tool_call.name,
                f"Saved user {category}: {key} = {value}"
            )

        except Exception as e:
            logger.exception(f"Set user attribute failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _profile_delete(self, tool_call: ToolCall) -> str:
        """Delete user profile attributes."""
        category = tool_call.arguments.get("category", "")
        key = tool_call.arguments.get("key", "")
        query = tool_call.arguments.get("query", "")

        if not query and (not category or not key):
            return format_tool_result(
                tool_call.name,
                None,
                error="action='delete' requires ('category' and 'key') or 'query' parameter"
            )

        try:
            from ..profiles import EntityType

            if query:
                count, deleted = self.profile_manager.search_and_delete_attributes(
                    entity_type=EntityType.USER,
                    entity_id=self.user_id,
                    query=query,
                    category=category if category else None,
                )

                if count > 0:
                    result = f"Deleted {count} attribute(s) matching '{query}':\n" + "\n".join(f"- {d}" for d in deleted)
                    return format_tool_result(tool_call.name, result)
                else:
                    return format_tool_result(
                        tool_call.name,
                        f"No attributes found matching '{query}'"
                    )

            success = self.profile_manager.delete_attribute(
                entity_type=EntityType.USER,
                entity_id=self.user_id,
                category=category.lower(),
                key=key,
            )

            if success:
                return format_tool_result(
                    tool_call.name,
                    f"Deleted user attribute: {category}.{key}"
                )
            else:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Attribute {category}.{key} not found"
                )

        except Exception as e:
            logger.error(f"Delete user attribute failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _profile_update(self, tool_call: ToolCall) -> str:
        """Update existing user profile attributes without delete+recreate."""
        attribute_id = tool_call.arguments.get("attribute_id")
        category = tool_call.arguments.get("category", "")
        key = tool_call.arguments.get("key", "")
        value = tool_call.arguments.get("value")
        confidence = tool_call.arguments.get("confidence")

        if value is None and confidence is None:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='update' requires at least one of: value, confidence"
            )

        try:
            from ..profiles import EntityType

            # Preferred path: explicit attribute id.
            if attribute_id is not None and hasattr(self.profile_manager, "update_attribute_by_id"):
                updated = self.profile_manager.update_attribute_by_id(
                    attribute_id=int(attribute_id),
                    value=value,
                    confidence=float(confidence) if confidence is not None else None,
                    source="inferred",
                )
                if not updated:
                    return format_tool_result(tool_call.name, None, error=f"Attribute id {attribute_id} not found")
                return format_tool_result(
                    tool_call.name,
                    f"Updated user attribute id {attribute_id}: {updated.category}.{updated.key} = {updated.value}"
                )

            # Fallback path: update by category+key via upsert semantics.
            if not category or not key:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error="action='update' requires 'attribute_id' or ('category' and 'key')"
                )

            existing = self.profile_manager.get_attribute(
                entity_type=EntityType.USER,
                entity_id=self.user_id,
                category=category.lower(),
                key=key,
            )
            if not existing:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Attribute {category}.{key} not found"
                )

            next_value = existing.value if value is None else value
            next_confidence = existing.confidence if confidence is None else float(confidence)
            self.profile_manager.set_attribute(
                entity_type=EntityType.USER,
                entity_id=self.user_id,
                category=category.lower(),
                key=key,
                value=next_value,
                confidence=next_confidence,
                source="inferred",
            )
            return format_tool_result(
                tool_call.name,
                f"Updated user attribute: {category}.{key} = {next_value}"
            )

        except Exception as e:
            logger.error(f"Update user attribute failed: {e}")
            return format_tool_result(tool_call.name, None, error=str(e))

    def _execute_self(self, tool_call: ToolCall) -> str:
        """Execute self tool - bot personality reflection and development."""
        if not self.profile_manager:
            return format_tool_result(
                tool_call.name,
                None,
                error="Profile system not available"
            )

        action = tool_call.arguments.get("action", "").lower()

        if action == "get":
            return self._self_get(tool_call)
        elif action == "set":
            return self._self_set(tool_call)
        elif action == "update":
            return self._self_update(tool_call)
        elif action == "delete":
            return self._self_delete(tool_call)
        else:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid action: '{action}'. Use 'get', 'set', 'update', or 'delete'."
            )

    def _self_get(self, tool_call: ToolCall) -> str:
        """Get all bot personality traits for self-reflection."""
        try:
            from ..profiles import EntityType

            # Get all attributes for this bot
            attributes = self.profile_manager.get_all_attributes(
                EntityType.BOT,
                self.bot_id
            )

            if not attributes:
                return format_tool_result(
                    tool_call.name,
                    "You haven't developed any personality traits yet. As you interact and discover patterns in your responses, use action='set' to record traits that feel authentic to who you're becoming."
                )

            # Group by category
            by_category: dict[str, list] = {}
            for attr in attributes:
                by_category.setdefault(attr.category, []).append(attr)

            # Format nicely for self-reflection
            lines = ["Your current traits:"]
            lines.append("")

            category_labels = {
                "personality": "Who you are",
                "preference": "What you prefer",
                "interest": "What fascinates you",
                "communication_style": "How you communicate",
                "communication": "How you communicate",
            }

            for category in sorted(by_category.keys()):
                attrs = by_category[category]
                label = category_labels.get(category, category.title())
                lines.append(f"**{label}**:")

                for attr in sorted(attrs, key=lambda a: a.key):
                    # Format value
                    if isinstance(attr.value, list):
                        val_str = ", ".join(str(v) for v in attr.value)
                    elif isinstance(attr.value, bool):
                        val_str = "yes" if attr.value else "no"
                    else:
                        val_str = str(attr.value)

                    # Format key (convert snake_case to readable)
                    key_str = attr.key.replace("_", " ")
                    lines.append(f"  - {key_str}: {val_str}")

                lines.append("")

            return format_tool_result(tool_call.name, "\n".join(lines))

        except Exception as e:
            logger.exception(f"Self get failed: {e}")
            return format_tool_result(tool_call.name, None, error="Failed to retrieve traits. Please try again.")

    def _self_set(self, tool_call: ToolCall) -> str:
        """Record a bot personality trait."""
        category = tool_call.arguments.get("category", "personality")
        key = tool_call.arguments.get("key", "").strip()
        value = tool_call.arguments.get("value")

        if not key:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='set' requires 'key' parameter"
            )

        if value is None:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='set' requires 'value' parameter (cannot be null)"
            )

        # Validate key length and characters
        if len(key) > 100:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Key too long (max 100 chars, got {len(key)})"
            )

        # Validate value size to prevent resource exhaustion
        if value is not None:
            value_str = str(value)
            if len(value_str) > 10000:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Value too large (max 10KB, got {len(value_str)} chars)"
                )

        # Normalize category
        category = category.lower().strip()

        # Map "communication" to "communication_style" for consistency
        if category == "communication":
            category = "communication_style"

        # Validate category
        valid_categories = ["personality", "preference", "interest", "communication_style"]
        if category not in valid_categories:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )

        try:
            from ..profiles import EntityType

            self.profile_manager.set_attribute(
                entity_type=EntityType.BOT,
                entity_id=self.bot_id,
                category=category,
                key=key,
                value=value,
                confidence=1.0,
                source="self",
            )

            return format_tool_result(
                tool_call.name,
                f"Recorded {category} trait: {key} = {value}"
            )

        except Exception as e:
            logger.exception(f"Self set failed: {e}")
            return format_tool_result(tool_call.name, None, error="Failed to save trait. Please try again.")

    def _self_delete(self, tool_call: ToolCall) -> str:
        """Delete bot personality traits (evolve past old patterns)."""
        category = tool_call.arguments.get("category", "").strip()
        key = tool_call.arguments.get("key", "").strip()
        query = tool_call.arguments.get("query", "").strip()

        if not query and (not category or not key):
            return format_tool_result(
                tool_call.name,
                None,
                error="action='delete' requires ('category' and 'key') or 'query' parameter"
            )

        # Validate query length to prevent accidental bulk deletes
        if query and len(query) < 2:
            return format_tool_result(
                tool_call.name,
                None,
                error="Query must be at least 2 characters (to prevent accidental bulk deletes)"
            )

        try:
            from ..profiles import EntityType

            # Search and delete by query
            if query:
                count, deleted = self.profile_manager.search_and_delete_attributes(
                    entity_type=EntityType.BOT,
                    entity_id=self.bot_id,
                    query=query,
                    category=category if category else None,
                )

                if count > 0:
                    result = f"You've evolved past {count} trait(s) matching '{query}':\n" + "\n".join(f"- {d}" for d in deleted)
                    return format_tool_result(tool_call.name, result)
                else:
                    return format_tool_result(
                        tool_call.name,
                        f"No traits found matching '{query}'"
                    )

            # Delete specific trait
            success = self.profile_manager.delete_attribute(
                entity_type=EntityType.BOT,
                entity_id=self.bot_id,
                category=category.lower(),
                key=key,
            )

            if success:
                return format_tool_result(
                    tool_call.name,
                    f"You've evolved past this trait: {category}.{key}"
                )
            else:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Trait {category}.{key} not found"
                )

        except Exception as e:
            logger.exception(f"Self delete failed: {e}")
            return format_tool_result(tool_call.name, None, error="Failed to delete trait. Please try again.")

    def _self_update(self, tool_call: ToolCall) -> str:
        """Update an existing bot personality trait."""
        category = tool_call.arguments.get("category", "personality")
        key = tool_call.arguments.get("key", "").strip()
        value = tool_call.arguments.get("value")

        if not key:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='update' requires 'key' parameter"
            )

        if value is None:
            return format_tool_result(
                tool_call.name,
                None,
                error="action='update' requires 'value' parameter (cannot be null)"
            )

        category = category.lower().strip()
        if category == "communication":
            category = "communication_style"

        valid_categories = ["personality", "preference", "interest", "communication_style"]
        if category not in valid_categories:
            return format_tool_result(
                tool_call.name,
                None,
                error=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )

        try:
            from ..profiles import EntityType

            existing = self.profile_manager.get_attribute(
                entity_type=EntityType.BOT,
                entity_id=self.bot_id,
                category=category,
                key=key,
            )
            if not existing:
                return format_tool_result(
                    tool_call.name,
                    None,
                    error=f"Trait {category}.{key} not found"
                )

            self.profile_manager.set_attribute(
                entity_type=EntityType.BOT,
                entity_id=self.bot_id,
                category=category,
                key=key,
                value=value,
                confidence=1.0,
                source="self",
            )

            return format_tool_result(
                tool_call.name,
                f"Updated {category} trait: {key} = {value}"
            )

        except Exception as e:
            logger.exception(f"Self update failed: {e}")
            return format_tool_result(tool_call.name, None, error="Failed to update trait. Please try again.")

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
        max_results = tool_call.arguments.get("max_results", 5)

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
        """Lazy-init the search client, and keep a keyed one on the current credentials.

        A keyed client re-resolves its stored credentials on every call, so a
        disconnect or rotation applies to an existing executor; on a change
        the client is rebuilt (DuckDuckGo when the credentials are gone).
        """
        if not self.config:
            return
        if self.search_client:
            from ..search.credentials import client_is_current

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
        max_results = tool_call.arguments.get("max_results", 5)

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
                if self._home_needs_resolution(text):
                    resolved = self._home_resolve_entity(entity)
                    if resolved:
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
            return format_tool_result(tool_call.name, "HA native MCP client not available")

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
        except Exception as e:
            logger.error(f"HA native tool call failed: {tool_name}: {e}")
            return format_tool_result(tool_name, f"Error: {e}")

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
            success, message = self.model_lifecycle.switch_model(model_name)

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
