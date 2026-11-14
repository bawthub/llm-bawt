"""Tool calling loop for LLM conversations.

Handles the iterative process of:
1. Querying the LLM
2. Detecting tool calls in the response
3. Executing tools via MCP
4. Feeding results back to the LLM
5. Repeating until final response
"""

import json
import logging
from typing import TYPE_CHECKING

from .executor import ToolExecutor
from .formats import ToolFormat, get_format_handler, ToolCallRequest
from .parser import ToolCall, format_tool_result, tool_result_is_error

if TYPE_CHECKING:
    from ..models.message import Message
    from ..clients import LLMClient
    from ..integrations.ha_mcp.client import HomeAssistantMCPClient, HomeAssistantNativeClient
    from ..integrations.newsapi.client import NewsAPIClient
    from ..integrations.web_fetch.client import WebFetchClient
    from ..mcp_server.client import MemoryClient
    from ..profiles import ProfileManager
    from ..search.base import SearchClient
    from ..core.model_lifecycle import ModelLifecycleManager
    from ..utils.config import Config
    from ..utils.history import HistoryManager
    from ..adapters import ModelAdapter

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_ITERATIONS = 20


class ToolLoop:
    """Manages tool calling iterations for a single conversation turn."""
    
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
        bot_id: str = "",
        max_iterations: int | None = None,
        tool_format: ToolFormat | str = ToolFormat.REACT,
        tools: list | None = None,
        adapter: "ModelAdapter | None" = None,
        history_manager: "HistoryManager | None" = None,
        generation_kwargs: dict | None = None,
        ha_mode: bool = False,
    ):
        """
        Args:
            memory_client: Client for memory operations.
            profile_manager: Manager for profile operations.
            search_client: Client for web search operations.
            home_client: Home Assistant MCP client.
            ha_native_client: Home Assistant native MCP client.
            news_client: NewsAPI client.
            web_fetch_client: Web fetch client for reading web pages.
            model_lifecycle: Manager for model lifecycle operations.
            config: Application config (used for lazy search setup).
            user_id: Current user ID (required).
            bot_id: Current bot ID.
            max_iterations: Maximum tool call iterations per turn (None = use config default).
            adapter: Model adapter for model-specific stop sequences and output cleaning.
            history_manager: History manager for recall operations.
            ha_mode: If True, force tool_choice="required" on first iteration.
        """
        if not user_id:
            raise ValueError("user_id is required for ToolLoop")
        self.executor = ToolExecutor(
            memory_client=memory_client,
            profile_manager=profile_manager,
            search_client=search_client,
            home_client=home_client,
            ha_native_client=ha_native_client,
            news_client=news_client,
            web_fetch_client=web_fetch_client,
            model_lifecycle=model_lifecycle,
            config=config,
            user_id=user_id,
            bot_id=bot_id,
            history_manager=history_manager,
        )
        # Use config value if not explicitly provided (0 = unlimited, cap at reasonable max)
        if max_iterations is None:
            config_max = getattr(config, 'MAX_TOOL_CALLS_PER_TURN', DEFAULT_MAX_ITERATIONS) if config else DEFAULT_MAX_ITERATIONS
            self.max_iterations = config_max if config_max > 0 else 100
        else:
            self.max_iterations = max_iterations
        # HA-mode: cap iterations — device commands need at most a few tool calls
        if ha_mode and self.max_iterations > 3:
            self.max_iterations = 3
        self.tool_context: list[dict] = []  # Track tool interactions for history
        self.tool_call_details: list[dict] = []  # Per-call detail for debug logging
        self.tool_format = tool_format
        self.tools = tools
        self.format_handler = get_format_handler(tool_format)
        if self.format_handler is None:
            raise ValueError(f"ToolLoop requires a valid tool format, got: {tool_format}")
        self._using_native_tools = False
        self.adapter = adapter
        self.generation_kwargs = generation_kwargs or {}
        self.ha_mode = ha_mode
    
    def run(
        self,
        messages: list["Message"],
        client: "LLMClient",
        stream_final: bool = True,
    ) -> str:
        """Run the tool calling loop.
        
        Args:
            messages: Initial conversation messages.
            client: LLM client instance.
            stream_final: Whether to stream the final response.
            
        Returns:
            Final response text (after all tool calls resolved).
        """
        
        self.executor.reset_call_count()
        self.tool_context = []  # Reset tool context
        self.tool_call_details = []  # Reset per-call details
        current_messages = messages.copy()
        handler = self.format_handler
        self._using_native_tools = self._should_use_native_tools() and client.supports_native_tools()
        
        if self._using_native_tools:
            logger.info("🔧 Using native tool calling")

        has_executed_tools = False  # Track if we've already run tools
        
        for iteration in range(1, self.max_iterations + 1):
            # Only log iteration if we're past the first one (indicates tool use)
            if iteration > 1:
                logger.info(f"🔄 Tool loop iteration {iteration}/{self.max_iterations}")
            
            # Never stream in tool loop - we need to check for tool calls before rendering
            # The caller (LLMBawt.query) handles rendering the final response
            # After tools have been executed, don't use stop sequences - let model complete naturally
            response = self._query_llm(
                client=client,
                messages=current_messages,
                handler=handler,
                use_native=self._using_native_tools,
                skip_stop_sequences=has_executed_tools,  # Don't stop early if already have tool results
            )
            
            if not response:
                logger.debug("Empty response from LLM")
                return ""
            
            # Handle native tool calls (response is a dict with tool_calls)
            if isinstance(response, dict) and "tool_calls" in response:
                native_tool_calls = response["tool_calls"]
                content = response.get("content", "")
                
                # Convert native tool calls to our ToolCallRequest format
                tool_calls = []
                for tc in native_tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    # Parse JSON string arguments into dict
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    elif not isinstance(args, dict):
                        args = {}
                    tool_calls.append(ToolCallRequest(
                        name=func.get("name", ""),
                        arguments=args,
                        raw_text="",
                        tool_call_id=tc.get("id"),
                    ))
                
                if tool_calls:
                    # Log before execution so we see what's being attempted
                    for tc in tool_calls:
                        logger.debug(f"🔧 {tc.name}({tc.arguments})")
                    
                    # Execute tools
                    tool_messages, tool_results = self._execute_tools(tool_calls, handler)
                    
                    # Track tool interactions for history/context
                    tool_summary = "\n\n".join(tool_results)
                    self.tool_context.append({
                        "tools_called": [tc.name for tc in tool_calls],
                        "results": tool_summary,
                    })
                    # Track per-call detail for debug logging
                    for tc_req, result_text in zip(tool_calls, tool_results):
                        self.tool_call_details.append({
                            "tool": tc_req.name,
                            "parameters": tc_req.arguments,
                            "result": result_text,
                            "is_error": tool_result_is_error(result_text),
                            "iteration": iteration,
                        })
                    
                    # Build continuation messages
                    assistant_message = self._build_assistant_message(response, tool_calls)
                    current_messages.append(assistant_message)
                    current_messages.extend(tool_messages)
                    continue  # Next iteration
                
                # No tool calls in dict, return content
                return content if content else ""
            
            # Parse tool calls from text response (ReAct/XML format)
            tool_calls, remaining_text = handler.parse_response(response)
            effective_handler = handler

            # Fallback parsing if a native-capable model emits ReAct text.
            if not tool_calls and isinstance(response, str):
                lower = response.lower()
                if "action:" in lower and "action input:" in lower:
                    from .formats.react import ReActFormatHandler

                    react_handler = ReActFormatHandler()
                    tool_calls, remaining_text = react_handler.parse_response(response)
                    if tool_calls:
                        logger.warning("Detected ReAct tool call while using %s handler", type(handler).__name__)
                        effective_handler = react_handler

            if not tool_calls:
                logger.debug("No tool calls found - returning final response")
                final_text = remaining_text or (response if isinstance(response, str) else "")
                if isinstance(response, str):
                    lower = response.lower()
                    if "action:" in lower and "action input:" in lower:
                        from .formats.react import ReActFormatHandler
                        return ReActFormatHandler().sanitize_response(final_text)
                return handler.sanitize_response(final_text)
            
            # Log before execution so we see what's being attempted
            for tc in tool_calls:
                logger.info(f"🔧 {tc.name}({tc.arguments})")
            
            # Execute tools
            tool_messages, tool_results = self._execute_tools(tool_calls, effective_handler)
            has_executed_tools = True  # Mark that we've executed tools
            
            # Track tool interactions for history/context
            tool_summary = "\n\n".join(tool_results)
            self.tool_context.append({
                "tools_called": [tc.name for tc in tool_calls],
                "results": tool_summary,
            })
            # Track per-call detail for debug logging
            for tc_req, result_text in zip(tool_calls, tool_results):
                self.tool_call_details.append({
                    "tool": tc_req.name,
                    "parameters": tc_req.arguments,
                    "result": result_text,
                    "is_error": tool_result_is_error(result_text),
                    "iteration": iteration,
                })
            
            # Build continuation messages
            # For ReAct format: add assistant's action, then observation as user message
            # The model should then continue with "Thought: I now have..." or "Final Answer:"
            assistant_message = self._build_assistant_message(response, tool_calls)
            current_messages.append(assistant_message)
            current_messages.extend(tool_messages)
        
        # Max iterations reached
        logger.warning(f"Tool loop: max iterations ({self.max_iterations}) reached")
        if 'response' in dir() and isinstance(response, str):
            return handler.sanitize_response(response)
        return ""
    
    def get_tool_context_summary(self) -> str:
        """Return a summary of tool interactions for saving to history."""
        if not self.tool_context:
            return ""
        
        summaries = []
        for ctx in self.tool_context:
            tools = ", ".join(ctx["tools_called"])
            summaries.append(f"[Tools used: {tools}]\n{ctx['results']}")
        return "\n\n".join(summaries)

    def get_tool_call_details(self) -> list[dict]:
        """Return per-call tool details for debug logging.

        Each entry contains:
            tool: Tool name
            parameters: Full argument dict
            result: Result text for this specific call
            iteration: Which tool loop iteration this occurred in
        """
        return list(self.tool_call_details)
    
    def _execute_tools(self, tool_calls: list, handler) -> tuple[list["Message"], list[str]]:
        """Execute a list of tool calls and return formatted results."""
        from ..models.message import Message
        results = []
        tool_messages: list[Message] = []
        first_tool = True  # Track first tool to add guidance
        
        for tc in tool_calls:
            if not self.executor.can_execute_more():
                formatted = handler.format_result(
                    tc.name, 
                    None,
                    tool_call_id=getattr(tc, "tool_call_id", None),
                    error="Too many tool calls this turn",
                )
                results.append(format_tool_result(tc.name, None, error="Too many tool calls this turn"))
                tool_messages.append(self._result_to_message(formatted, tc, add_guidance=False))
                break
            
            tool_call = ToolCall(name=tc.name, arguments=tc.arguments, raw_text=tc.raw_text or "")
            result = self.executor.execute(tool_call)
            results.append(result)
            formatted = handler.format_result(
                tc.name,
                result,
                tool_call_id=getattr(tc, "tool_call_id", None),
            )
            # Add guidance only for first tool result to avoid repetition
            tool_messages.append(self._result_to_message(formatted, tc, add_guidance=first_tool))
            first_tool = False
            logger.debug(f"Tool {tc.name} executed: {str(result)[:100]}...")
        
        return tool_messages, results

    def _result_to_message(self, formatted_result, tool_call, add_guidance: bool = False) -> "Message":
        from ..models.message import Message
        if isinstance(formatted_result, dict):
            content = formatted_result.get("content", "")
            # Add guidance to help model provide final answer
            if add_guidance and content:
                content += "\n\nBased on this information, provide your answer to the user's question."
            return Message(
                role=formatted_result.get("role", "tool"),
                content=content,
                tool_call_id=formatted_result.get("tool_call_id"),
            )
        content = str(formatted_result)
        if add_guidance:
            content += "\n\nBased on this information, provide your answer to the user's question."
        return Message(role="user", content=content)

    def _build_assistant_message(self, response, tool_calls: list) -> "Message":
        from ..models.message import Message
        if self._using_native_tools:
            tool_payloads = [
                {
                    "id": getattr(tc, "tool_call_id", None) or "",
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in tool_calls
            ]
            content = ""
            if isinstance(response, dict):
                content = response.get("content") or ""
            elif isinstance(response, str):
                content = response
            return Message(role="assistant", content=content or "", tool_calls=tool_payloads)
        
        # For ReAct/text format: truncate response at the end of the tool call
        # to prevent hallucinated continuations from confusing the model
        content = response if isinstance(response, str) else ""
        if tool_calls and content:
            # Find where the last tool call ends and truncate there
            # Look for the closing brace of the last Action Input
            last_tc = tool_calls[-1]
            if last_tc.raw_text:
                # Use the raw_text to find where to truncate
                idx = content.find(last_tc.raw_text)
                if idx >= 0:
                    content = content[:idx + len(last_tc.raw_text)]
            else:
                # Fallback: try to find Action Input JSON and truncate after it
                import re
                # Find the last Action Input: {...} and truncate after it
                matches = list(re.finditer(r'Action\s*Input\s*:\s*(\{[^}]*\}|\S+)', content, re.IGNORECASE))
                if matches:
                    last_match = matches[-1]
                    content = content[:last_match.end()]
        
        return Message(role="assistant", content=content)

    def _should_use_native_tools(self) -> bool:
        """Check if we should use native tool calling.
        
        Native tool calling is only used for NATIVE_OPENAI format (OpenAI API).
        For GGUF/local models using REACT format, we use text-based ReAct parsing
        because llama-cpp-python's native function calling has issues with
        multi-turn tool conversations (empty responses after tool execution).
        """
        # Normalize to string for comparison
        format_str = self.tool_format.value if isinstance(self.tool_format, ToolFormat) else str(self.tool_format).lower()
        
        # Only use native for NATIVE_OPENAI (OpenAI API)
        # Do NOT use for REACT - llama-cpp-python's chatml-function-calling
        # doesn't handle multi-turn tool conversations well
        if format_str == ToolFormat.NATIVE_OPENAI.value or format_str == "native":
            return True
        return False

    def _query_llm(self, client: "LLMClient", messages: list["Message"], handler, use_native: bool, skip_stop_sequences: bool = False):
        """Query the LLM, potentially with native tool calling.
        
        Args:
            client: LLM client instance.
            messages: Conversation messages.
            handler: Format handler for tools.
            use_native: Whether to use native tool calling.
            skip_stop_sequences: If True, don't use stop sequences (for follow-up after tools).
        
        Returns:
            Either a string response or a dict with tool_calls for native mode.
        """
        if use_native and hasattr(handler, "get_tools_schema"):
            tools_schema = handler.get_tools_schema(self.tools or [])
            content, tool_calls = client.query_with_tools(
                messages,
                tools_schema=tools_schema,
                tool_choice="auto",
                stop=None,
                **self.generation_kwargs,
            )
            
            # If we got tool_calls from native mode, return them in a dict
            if tool_calls:
                return {"content": content, "tool_calls": tool_calls}
            
            # No tool calls, return content as string
            return content if content else ""

        # For text-based parsing, use stop sequences only on first iteration
        # After tools have run, let the model complete its response naturally
        if skip_stop_sequences:
            stop_sequences = None
        else:
            # Combine format handler stop sequences with model adapter stop sequences
            stop_sequences = list(handler.get_stop_sequences())
            if self.adapter:
                adapter_stops = self.adapter.get_stop_sequences()
                if adapter_stops:
                    stop_sequences.extend(adapter_stops)
        return client.query(messages, plaintext_output=True, stream=False, stop=stop_sequences, **self.generation_kwargs)


def query_with_tools(
    messages: list["Message"],
    client: "LLMClient",
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
    bot_id: str = "",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    stream: bool = True,
    tool_format: ToolFormat | str = ToolFormat.REACT,
    tools: list | None = None,
    adapter: "ModelAdapter | None" = None,
    history_manager: "HistoryManager | None" = None,
    generation_kwargs: dict | None = None,
    ha_mode: bool = False,
) -> tuple[str, str, list[dict]]:
    """Convenience function for tool-enabled queries.

    Args:
        messages: Conversation messages.
        client: LLM client instance.
        memory_client: Memory client for tool execution.
        profile_manager: Profile manager for user/bot profile tools.
        search_client: Search client for web search tools.
        home_client: Home Assistant MCP client for smart-home tools.
        ha_native_client: Home Assistant native MCP client.
        news_client: NewsAPI client for news tools.
        web_fetch_client: Web fetch client for reading web pages.
        model_lifecycle: Model lifecycle manager for model switching tools.
        config: Application config (used for lazy search setup).
        user_id: Current user ID (required).
        bot_id: Current bot ID.
        max_iterations: Max tool iterations.
        stream: Whether to stream the final response.
        tool_format: Tool format for this model.
        tools: Tool definitions to include in schema/formatting.
        adapter: Model adapter for model-specific stop sequences.
        history_manager: History manager for recall operations.
        ha_mode: If True, force tool_choice="required" on first iteration.

    Returns:
        Tuple of (final_response, tool_context_summary, tool_call_details).
        tool_context_summary contains the tool results that should be saved to history.
        tool_call_details is a list of per-call dicts for debug logging.
    """
    if not user_id:
        raise ValueError("user_id is required for query_with_tools")
    loop = ToolLoop(
        memory_client=memory_client,
        profile_manager=profile_manager,
        search_client=search_client,
        home_client=home_client,
        ha_native_client=ha_native_client,
        news_client=news_client,
        web_fetch_client=web_fetch_client,
        model_lifecycle=model_lifecycle,
        config=config,
        user_id=user_id,
        bot_id=bot_id,
        max_iterations=max_iterations,
        tool_format=tool_format,
        tools=tools,
        adapter=adapter,
        history_manager=history_manager,
        generation_kwargs=generation_kwargs,
        ha_mode=ha_mode,
    )
    response = loop.run(messages, client, stream_final=stream)
    tool_context = loop.get_tool_context_summary()
    tool_details = loop.get_tool_call_details()
    return response, tool_context, tool_details
