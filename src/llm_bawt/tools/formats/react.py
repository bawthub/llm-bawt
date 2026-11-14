from __future__ import annotations

import json
import logging
import re
from typing import Any

from .base import ToolCallRequest, ToolFormatHandler
from ...shared.output_sanitizer import strip_tool_protocol_leakage

logger = logging.getLogger(__name__)


_THOUGHT_MARKERS = (
    "thought:",
    "action:",
    "action input:",
    "observation:",
)

# Alternative format markers that some models use
_ALT_TOOL_MARKERS = (
    "# tool:",
    "tool:",
    "function:",
    "# function:",
)

# Map of common tool name variations to actual tool names
# Models often invent their own tool names; this normalizes them
TOOL_NAME_ALIASES = {
    # History retrieval variations
    "retrieve_conversation_history": "get_recent_history",
    "get_conversation_history": "get_recent_history",
    "get_history": "get_recent_history",
    "conversation_history": "get_recent_history",
    "fetch_history": "get_recent_history",
    "show_history": "get_recent_history",
    "list_history": "get_recent_history",
    "get_messages": "get_recent_history",
    "retrieve_messages": "get_recent_history",
    # Memory search variations
    "search_memory": "search_memories",
    "memory_search": "search_memories",
    "query_memories": "search_memories",
    "find_memories": "search_memories",
    # Memory storage variations
    "save_memory": "store_memory",
    "add_memory": "store_memory",
    "remember": "store_memory",
    # Profile variations
    "get_profile": "get_user_profile",
    "user_profile": "get_user_profile",
    "set_attribute": "set_user_attribute",
    "set_preference": "set_user_attribute",
    # Web search variations
    "search": "web_search",
    "internet_search": "web_search",
    "google": "web_search",
    # Model tool variations
    "current_model": "get_current_model",
    "get_model": "get_current_model",
    # Keep the list intent; the executor's legacy map supplies action=list.
    "list_models": "list_models",
    "show_models": "list_models",
    "available_models": "list_models",
}


def _normalize_tool_name(name: str) -> str:
    """Normalize tool name to match actual tool definitions."""
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    # Check aliases
    if normalized in TOOL_NAME_ALIASES:
        return TOOL_NAME_ALIASES[normalized]
    return name.strip()


def _try_fix_json(json_str: str) -> str | None:
    original = json_str

    open_count = json_str.count("{")
    close_count = json_str.count("}")
    if close_count > open_count:
        excess = close_count - open_count
        while excess > 0 and json_str.rstrip().endswith("}"):
            json_str = json_str.rstrip()[:-1]
            excess -= 1

    if open_count > close_count:
        json_str = json_str + ("}" * (open_count - close_count))

    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    if json_str != original:
        return json_str
    return None


def _load_json_payload(payload: str) -> dict[str, Any] | None:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    try:
        import json5  # type: ignore

        return json5.loads(payload)
    except Exception:
        pass

    fixed = _try_fix_json(payload)
    if fixed:
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    return None


def _extract_json_object(text: str, start_idx: int) -> tuple[str | None, int | None]:
    brace_idx = text.find("{", start_idx)
    if brace_idx == -1:
        return None, None

    depth = 0
    in_string = False
    escape = False
    for i in range(brace_idx, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_idx : i + 1], i + 1

    # Incomplete JSON - try to fix by adding missing braces
    # This handles cases where model stopped mid-JSON (e.g., stop sequence or max tokens)
    if depth > 0:
        incomplete_json = text[brace_idx:]
        # Try adding closing braces
        fixed = incomplete_json + ("}" * depth)
        # Verify it's valid JSON
        try:
            json.loads(fixed)
            logger.debug(f"Fixed incomplete JSON by adding {depth} closing brace(s)")
            return fixed, len(text)
        except json.JSONDecodeError:
            pass

    return None, None


def _tool_list_to_prompt(tools: list) -> str:
    lines: list[str] = []
    for tool in tools or []:
        if hasattr(tool, "to_prompt_string"):
            lines.append(tool.to_prompt_string())
        elif isinstance(tool, dict):
            name = tool.get("name", "")
            desc = tool.get("description", "")
            params = tool.get("parameters") or []
            param_lines = []
            for param in params:
                if not isinstance(param, dict):
                    continue
                req = "(required)" if param.get("required", True) else "(optional)"
                param_lines.append(
                    f"    - {param.get('name')} ({param.get('type')}): {param.get('description')} {req}"
                )
            if param_lines:
                lines.append(f"- **{name}**: {desc}\n" + "\n".join(param_lines))
            else:
                lines.append(f"- **{name}**: {desc}")
        else:
            lines.append(str(tool))

    return "\n".join(lines)


def _tool_names(tools: list) -> str:
    names: list[str] = []
    for tool in tools or []:
        if hasattr(tool, "name"):
            names.append(tool.name)
        elif isinstance(tool, dict):
            name = tool.get("name")
            if name:
                names.append(name)
    return ", ".join(names)


class ReActFormatHandler(ToolFormatHandler):
    """Text-based Thought/Action/Action Input tool calling."""

    def get_system_prompt(self, tools: list) -> str:
        tool_descriptions = _tool_list_to_prompt(tools)
        tool_names = _tool_names(tools)
        return (
            "## Tools\n\n"
            "You have access to the following tools:\n\n"
            f"{tool_descriptions}\n\n"
            "## How to Use Tools\n\n"
            "When you need to use a tool, respond with this EXACT format:\n\n"
            "Thought: [explain what you need to do and why]\n"
            "Action: [tool_name from the list above]\n"
            "Action Input: {\"param\": \"value\"}\n\n"
            "Then STOP. The system will execute the tool and provide an Observation.\n\n"
            "## When You Have the Answer\n\n"
            "Thought: I now have the information needed\n"
            "Final Answer: [your complete response to the user]\n\n"
            "## Important Rules\n\n"
            "1. Use EXACTLY ONE tool per response - no more\n"
            "2. STOP after Action Input - do not continue\n"
            "3. Wait for Observation before your next Thought\n"
            f"4. Tool names must be exactly: {tool_names}\n"
            "5. Action Input must be valid JSON\n"
            "6. When presenting tool results to the user, quote the ACTUAL data from the Observation - do NOT paraphrase or summarize from memory\n"
        )

    def get_stop_sequences(self) -> list[str]:
        # After Action Input: {...}, the model should STOP and wait for tool execution.
        # These sequences catch continuations after the closing brace of Action Input JSON.
        return [
            # After Action Input JSON closes, stop before any continuation
            "}\nObservation",
            "}\n\nObservation",
            "}\nFinal Answer",
            "}\n\nFinal Answer",
            "}\nThought",
            "}\n\nThought",
            # Fallback patterns (no brace prefix)
            "\n\nObservation:",
            "\n\nObservation",
            # Note: Model-specific stop sequences (e.g., [HUMAN], [INST]) are handled
            # by the ModelAdapter, not the format handler.
        ]

    def parse_response(self, response: str) -> tuple[list[ToolCallRequest], str]:
        if not response:
            return [], ""

        # Try ReAct format first: Action: / Action Input:
        result = self._parse_react_format(response)
        if result[0]:
            return result

        # Try alternative format: # Tool: / # Arguments: (Dolphin, etc.)
        result = self._parse_alt_tool_format(response)
        if result[0]:
            return result

        # Try code block with JSON tool call
        result = self._parse_code_block_format(response)
        if result[0]:
            return result

        # Check for Final Answer
        final_match = re.search(r"(?is)Final\s*Answer\s*:\s*(.*)$", response)
        if final_match:
            return [], final_match.group(1).strip()

        return [], response.strip()

    def _parse_react_format(self, response: str) -> tuple[list[ToolCallRequest], str]:
        """Parse standard ReAct format: Action: / Action Input:"""
        # Match Action: tool_name - tool name can be on same line or next line
        action_match = re.search(r"(?im)^\s*Action\s*:\s*(.+?)\s*$", response)
        if not action_match:
            # Try matching when tool name is on next line: "Action:\n  history"
            action_match = re.search(r"(?im)^\s*Action\s*:\s*\n\s*(\S+)", response)
        if action_match:
            tool_name = _normalize_tool_name(action_match.group(1))
            search_start = action_match.end()
            action_input_match = re.search(
                r"(?im)^\s*Action\s*Input\s*:\s*", response[search_start:]
            )
            if action_input_match:
                input_start = search_start + action_input_match.end()
                payload, end_idx = _extract_json_object(response, input_start)
                if payload and end_idx:
                    parsed = _load_json_payload(payload)
                    if isinstance(parsed, dict):
                        # Normalize argument keys too
                        parsed = self._normalize_arguments(tool_name, parsed)
                        raw_text = response[action_match.start() : end_idx]
                        remaining = (response[: action_match.start()] + response[end_idx:]).strip()
                        return (
                            [
                                ToolCallRequest(
                                    name=tool_name,
                                    arguments=parsed,
                                    raw_text=raw_text,
                                )
                            ],
                            remaining,
                        )
            else:
                # No Action Input found - treat as tool with no arguments
                # This handles tools like get_user_profile that don't require params
                raw_text = response[action_match.start() : action_match.end()]
                remaining = (response[: action_match.start()] + response[action_match.end():]).strip()
                return (
                    [
                        ToolCallRequest(
                            name=tool_name,
                            arguments={},
                            raw_text=raw_text,
                        )
                    ],
                    remaining,
                )
        return [], response

    def _parse_alt_tool_format(self, response: str) -> tuple[list[ToolCallRequest], str]:
        """Parse alternative format: # Tool: / # Arguments: (Dolphin, Mistral, etc.)"""
        # Match: # Tool: tool_name (with optional markdown code block)
        tool_match = re.search(r"(?im)^[`\s]*#?\s*Tool\s*:\s*(.+?)\s*$", response)
        if not tool_match:
            return [], response

        tool_name = _normalize_tool_name(tool_match.group(1))
        search_start = tool_match.end()

        # Look for # Arguments: or Arguments: with JSON
        args_match = re.search(
            r"(?im)^[`\s]*#?\s*Arguments?\s*:\s*", response[search_start:]
        )
        if args_match:
            input_start = search_start + args_match.end()
            payload, end_idx = _extract_json_object(response, input_start)
            if payload and end_idx:
                parsed = _load_json_payload(payload)
                if isinstance(parsed, dict):
                    parsed = self._normalize_arguments(tool_name, parsed)
                    raw_text = response[tool_match.start() : end_idx]
                    remaining = (response[: tool_match.start()] + response[end_idx:]).strip()
                    # Strip code block markers from remaining
                    remaining = re.sub(r"^```\s*\n?", "", remaining)
                    remaining = re.sub(r"\n?```\s*$", "", remaining)
                    logger.info(f"Parsed alt tool format: {tool_name} with args {parsed}")
                    return (
                        [
                            ToolCallRequest(
                                name=tool_name,
                                arguments=parsed,
                                raw_text=raw_text,
                            )
                        ],
                        remaining.strip(),
                    )
        return [], response

    def _parse_code_block_format(self, response: str) -> tuple[list[ToolCallRequest], str]:
        """Parse JSON tool call in a code block."""
        # Match: ```json or ``` followed by {"name": "...", "arguments": {...}}
        code_block_match = re.search(
            r"```(?:json)?\s*\n?\s*(\{.*?\})\s*\n?```",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if code_block_match:
            payload = code_block_match.group(1)
            parsed = _load_json_payload(payload)
            if isinstance(parsed, dict) and "name" in parsed:
                tool_name = _normalize_tool_name(str(parsed.get("name", "")))
                arguments = parsed.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                arguments = self._normalize_arguments(tool_name, arguments)
                raw_text = code_block_match.group(0)
                remaining = (
                    response[: code_block_match.start()]
                    + response[code_block_match.end() :]
                ).strip()
                logger.info(f"Parsed code block tool format: {tool_name}")
                return (
                    [
                        ToolCallRequest(
                            name=tool_name,
                            arguments=arguments,
                            raw_text=raw_text,
                        )
                    ],
                    remaining,
                )
        return [], response

    def _normalize_arguments(self, tool_name: str, args: dict) -> dict:
        """Normalize argument keys for a tool call."""
        # Tool-specific argument mappings
        # Different tools use different parameter names for similar concepts
        TOOL_ARG_ALIASES = {
            "get_recent_history": {
                "count": "n_messages",
                "limit": "n_messages",
                "num_messages": "n_messages",
                "number_of_messages": "n_messages",
            },
            "forget_history": {
                "n_messages": "count",
                "num_messages": "count",
                "number_of_messages": "count",
                "n": "count",
                "limit": "count",
            },
        }
        
        # Generic aliases (applied if no tool-specific mapping)
        GENERIC_ARG_ALIASES = {
            "num_results": "n_results",
            "number_of_results": "n_results",
            "max_results": "n_results",
            "search_query": "query",
            "q": "query",
            "text": "content",
            "message": "content",
        }
        
        # Get tool-specific aliases, fall back to generic
        tool_aliases = TOOL_ARG_ALIASES.get(tool_name, {})
        
        normalized = {}
        for key, value in args.items():
            normalized_key = key.lower().replace("-", "_").replace(" ", "_")
            # Check tool-specific first, then generic
            if normalized_key in tool_aliases:
                normalized_key = tool_aliases[normalized_key]
            elif normalized_key in GENERIC_ARG_ALIASES:
                normalized_key = GENERIC_ARG_ALIASES[normalized_key]
            normalized[normalized_key] = value
        return normalized

    def format_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
        error: str | None = None,
    ) -> str:
        if error:
            return f"Observation: ERROR: {error}"

        result_str = self._normalize_result(result)
        return f"Observation: {result_str}"

    def sanitize_response(self, response: str) -> str:
        if not response:
            return ""

        # Extract Final Answer if present
        final_match = re.search(r"(?is)Final\s*Answer\s*:\s*(.*)$", response)
        if final_match:
            response = final_match.group(1).strip()

        # Remove ReAct markers (Thought:, Action:, Action Input:, Observation:)
        cleaned_lines = []
        skip_until_blank = False
        for line in response.splitlines():
            stripped = line.strip().lower()
            
            # Skip ReAct single-line markers (don't set skip_until_blank - these are one line)
            if stripped.startswith(_THOUGHT_MARKERS):
                continue
            
            # Skip alternative tool format markers (these have multi-line JSON after them)
            if any(stripped.startswith(marker) for marker in _ALT_TOOL_MARKERS):
                skip_until_blank = True
                continue
            if stripped.startswith("# arguments:") or stripped.startswith("arguments:"):
                skip_until_blank = True
                continue
            
            # Resume after blank line (end of tool block)
            if skip_until_blank:
                if not stripped:
                    skip_until_blank = False
                continue
            
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines)
        
        # Remove XML-style tool calls
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<function_call>.*?</function_call>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove code blocks containing tool calls
        cleaned = re.sub(
            r"```(?:json)?\s*\n?\s*#\s*Tool:.*?```",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(
            r"```(?:json)?\s*\n?\s*\{[^}]*\"name\"[^}]*\}.*?```",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        
        # Note: Model-specific markers (e.g., [HUMAN], [INST], BBCode) are handled
        # by the ModelAdapter.clean_output(), not the format handler.
        
        return strip_tool_protocol_leakage(cleaned)

    def _normalize_result(self, result: Any) -> str:
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        result_str = str(result)
        # Strip legacy tool_result wrappers if present
        if "<tool_result" in result_str:
            result_str = re.sub(r"<tool_result[^>]*>", "", result_str, flags=re.IGNORECASE)
            result_str = re.sub(r"</tool_result>", "", result_str, flags=re.IGNORECASE)
            result_str = re.sub(r"\\[IMPORTANT:.*?\\]", "", result_str, flags=re.DOTALL)
        return result_str.strip()
