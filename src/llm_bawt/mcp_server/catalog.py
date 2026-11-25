"""Compact discovery metadata without changing dispatch or validation.

FastMCP still owns callable signatures and validation. The public list_tools seam
only replaces model-facing descriptions and removes generated schema titles.
Titles are annotations, not field names; never recursively strip arbitrary keys.

TASK-1555: call_tool also re-emits JSON result text without whitespace.
FastMCP renders a tool's dict/list result with ``indent=2``, and every
newline and indent is a model token: the same payload costs ~45% more.
"""

import json
from copy import deepcopy
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool

from mcp.server.mcpserver.exceptions import ToolError

from llm_bawt.memory_switch import current_turn_memory_enabled, is_memory_tool, memory_system_enabled
from .approval_interceptor import ApprovalAwareFastMCP, PolicyResult
from .catalog_contracts import TOOL_SUMMARIES, tool_description
from .bot_access import ToolBotAccess, invocation_context
from .content_provenance import ToolContentProvenance
from agent_bridge.content_provenance import external_source, provenance_envelope
from agent_bridge.mcp_call_context import MCP_CALL_CONTEXT_KEY

MEMORY_DISABLED_ERROR = (
    "The memory system is disabled (memory_system_enabled=false). "
    "Use context_recall or messages_find for history."
)

# JSON Schema locations that hold schemas (not arbitrary user data/defaults).
_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_SINGLES = (
    "additionalProperties", "unevaluatedProperties", "propertyNames", "items",
    "contains", "not", "if", "then", "else", "additionalItems", "unevaluatedItems",
)


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a schema, dropping only presentation titles at schema nodes."""
    result = deepcopy(schema)

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node.pop("title", None)
        for key in _SCHEMA_MAPS:
            children = node.get(key)
            if isinstance(children, dict):
                for child in children.values():
                    visit(child)
        for key in _SCHEMA_LISTS:
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    visit(child)
        for key in _SCHEMA_SINGLES:
            child = node.get(key)
            if isinstance(child, list):  # draft-07 tuple items
                for entry in child:
                    visit(entry)
            else:
                visit(child)

    visit(result)
    return result


def compact_json_text(block: Any) -> Any:
    """``block`` with JSON text re-serialized without whitespace; else as-is.

    Only text that parses as a JSON object or array changes, and it decodes to
    the same value: nothing is dropped or reordered. Non-ASCII stays literal,
    as FastMCP wrote it.
    """
    if not isinstance(block, TextContent) or block.text[:1] not in ("{", "["):
        return block
    try:
        value = json.loads(block.text)
    except ValueError:
        return block
    compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return block if compact == block.text else block.model_copy(update={"text": compact})


def compact_call_result(result: Any) -> Any:
    """A tool's ``CallToolResult`` with every JSON text content block compacted.

    Structured content and any other shape (an input-required result) pass
    through unchanged.
    """
    if not isinstance(result, CallToolResult) or isinstance(result, PolicyResult):
        return result
    content = [compact_json_text(block) for block in result.content]
    if all(new is old for new, old in zip(content, result.content)):
        return result
    return result.model_copy(update={"content": content})


def label_external_result(result: Any, source: str | None) -> Any:
    """Keep output schemas and image blocks intact; label every result view."""
    if source is None or not isinstance(result, CallToolResult) or isinstance(result, PolicyResult):
        return result
    provenance = {
        "untrusted_content": True, "source": source,
        "applies_to": ["content", "structuredContent"],
    }
    content = [
        block.model_copy(update={"text": provenance_envelope(block.text, source)})
        if isinstance(block, TextContent) else block
        for block in result.content
    ]
    # Standard MCP result metadata scopes provenance over BOTH representations.
    # Do not change structuredContent: clients validate it against outputSchema.
    return result.model_copy(update={
        "content": content,
        "meta": {**(result.meta or {}), "content_provenance": provenance},
    })


class CatalogFastMCP(ApprovalAwareFastMCP):
    """Approval-aware server with a compact, skill-linked discovery catalog."""

    def __init__(self, *args, bot_policy_resolver=None, content_store_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot_access = ToolBotAccess(bot_policy_resolver)
        self.content_provenance = ToolContentProvenance(content_store_provider)

    def add_tool(self, fn, name=None, *args, **kwargs):
        wrapped = self.content_provenance.register(name or fn.__name__, fn)
        wrapped = self.bot_access.register(name or fn.__name__, wrapped)
        return super().add_tool(wrapped, name, *args, **kwargs)

    async def call_tool(self, name, arguments, context=None):
        public = dict(arguments or {})
        public.pop(MCP_CALL_CONTEXT_KEY, None)
        if is_memory_tool(name) and not current_turn_memory_enabled():
            raise ToolError(MEMORY_DISABLED_ERROR)
        with invocation_context(name, public):
            # Avoid doomed ASK cards. The registered callable checks again at
            # execution, including approved execution's bypass of call_tool.
            await self.bot_access.preflight(name, public)
            await self.content_provenance.check(name, public, preflight=True)
            result = compact_call_result(await super().call_tool(name, arguments, context))
            return label_external_result(result, external_source(name))

    async def call_approved_tool(self, name, arguments, **kwargs):
        # Approved replays run outside the original turn: resolve the switch
        # for the server-owned caller recorded on the approval (TASK-2653).
        caller = kwargs.get("caller_context")
        if is_memory_tool(name) and not memory_system_enabled(
                bot_id=getattr(caller, "bot_id", None) or None):
            raise ToolError(MEMORY_DISABLED_ERROR)
        with invocation_context(name, arguments):
            result = compact_call_result(await super().call_approved_tool(name, arguments, **kwargs))
            return label_external_result(result, external_source(name))

    async def list_tools(self) -> list[Tool]:
        tools = await super().list_tools()
        if not current_turn_memory_enabled():
            tools = [tool for tool in tools if not is_memory_tool(tool.name)]
        return [
            tool.model_copy(update={
                # Unmigrated/new tools remain callable and discoverable. Tests
                # require explicit coverage before they can land in this repo.
                "description": tool_description(tool.name)
                if tool.name in TOOL_SUMMARIES else tool.description,
                "input_schema": compact_schema(tool.input_schema),
            })
            for tool in tools
        ]
