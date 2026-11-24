"""Tool definitions for LLM tool calling.

Defines the available tools that bots can use, with their descriptions
and parameters in a format suitable for prompt injection.

Consolidated tools (9 total):
- memory: Search/store/delete facts (action-based)
- history: Search/retrieve/forget messages (action-based, with date filtering)
- profile: Get/set/delete user attributes (action-based)
- self: Bot personality reflection and development (action-based)
- search: Web/news/reddit search (type-based)
- web_fetch: Fetch and read web page content
- home: Home Assistant control/status (action-based)
- model: List/current/switch models (action-based)
- time: Get current time
"""

from dataclasses import dataclass, field
from typing import Any

from .formats import ToolFormat, get_format_handler


@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    """Definition of a tool that the LLM can call."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_prompt_string(self) -> str:
        """Format tool for inclusion in system prompt."""
        params_str = ""
        if self.parameters:
            param_lines = []
            for p in self.parameters:
                req = "(required)" if p.required else "(optional)"
                param_lines.append(f"    - {p.name} ({p.type}): {p.description} {req}")
            params_str = "\n" + "\n".join(param_lines)

        return f"- **{self.name}**: {self.description}{params_str}"


# =============================================================================
# Local tool definitions (memory, history and profiles come from BawtHub MCP)
# =============================================================================

# Search tool - combines web_search, news_search
SEARCH_TOOL = Tool(
    name="search",
    description="Search the internet for information. Use type='web' for general search, 'news' for recent news, or 'reddit' for Reddit threads/posts.",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query"
        ),
        ToolParameter(
            name="type",
            type="string",
            description="'web' (default), 'news', or 'reddit'",
            required=False,
            default="web"
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Max results (default 5)",
            required=False,
            default=5
        ),
        ToolParameter(
            name="time_range",
            type="string",
            description="For news/reddit freshness filters when supported: 'd' (day), 'w' (week), 'm' (month), 'y' (year)",
            required=False,
            default="w"
        ),
    ]
)

# News tool (NewsAPI)
NEWS_TOOL = Tool(
    name="news",
    description="Get news articles and headlines via NewsAPI. Use action='search' to search articles or action='headlines' for top headlines by country/category.",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="'search' (search articles by keyword) or 'headlines' (top headlines)",
        ),
        ToolParameter(
            name="query",
            type="string",
            description="Search keywords (required for search, optional for headlines)",
            required=False,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Max results (default 5)",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="country",
            type="string",
            description="For headlines: 2-letter country code (e.g. 'us', 'gb', 'de'). Default 'us'",
            required=False,
            default="us",
        ),
        ToolParameter(
            name="category",
            type="string",
            description="For headlines: business, entertainment, general, health, science, sports, technology",
            required=False,
        ),
        ToolParameter(
            name="sort_by",
            type="string",
            description="For search: 'publishedAt' (default), 'relevancy', or 'popularity'",
            required=False,
            default="publishedAt",
        ),
    ]
)

# Home Assistant tool
HOME_TOOL = Tool(
    name="home",
    description=(
        "Control Home Assistant devices and scenes. "
        "Use action='status', 'status_raw', 'query', 'get', 'set', or 'scene'. "
        "For status/raw requests, return exact tool output; do not synthesize missing JSON fields."
    ),
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="'status', 'status_raw', 'query', 'get', 'set', or 'scene'",
        ),
        ToolParameter(
            name="pattern",
            type="string",
            description="Entity search term for action='query', like 'bedroom' or 'garage'",
            required=False,
        ),
        ToolParameter(
            name="domain",
            type="string",
            description="Optional entity domain for action='query': light, switch, sensor, automation",
            required=False,
        ),
        ToolParameter(
            name="entity",
            type="string",
            description="Entity name/ID for action='get' and action='set'",
            required=False,
        ),
        ToolParameter(
            name="state",
            type="string",
            description="Required for action='set'. Lights/switches: on, off, toggle. Covers/blinds: open, close, stop. Locks: lock, unlock.",
            required=False,
        ),
        ToolParameter(
            name="brightness",
            type="integer",
            description="Optional brightness 0-100 for action='set' (lights only)",
            required=False,
        ),
        ToolParameter(
            name="scene_name",
            type="string",
            description="Scene name for action='scene', like 'chill' or 'theater'",
            required=False,
        ),
    ],
)

# Model tool - combines list_models, get_current_model, switch_model
MODEL_TOOL = Tool(
    name="model",
    description="Manage AI models. Check current model (action='current'), list all available models (action='list'), or switch to a different model (action='switch').",
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="REQUIRED. Must be one of: 'current' (show current model), 'list' (show all available models), 'switch' (change model)",
            required=True
        ),
        ToolParameter(
            name="model_name",
            type="string",
            description="Model name to switch to (only required when action='switch')",
            required=False
        ),
    ]
)

# Web fetch tool - retrieve web page content
WEB_FETCH_TOOL = Tool(
    name="web_fetch",
    description="Fetch a web page and read its content as text. Use this to read articles, documentation, blog posts, or any URL the user references.",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The full URL to fetch (must start with http:// or https://)",
        ),
    ],
)

# Time tool - simple utility
TIME_TOOL = Tool(
    name="time",
    description="Get current date and time.",
    parameters=[]
)


# =============================================================================
# Tool Categories (for selective inclusion)
# =============================================================================

# Core tools always included
CORE_TOOLS = [TIME_TOOL]

# Optional tool categories
SEARCH_TOOLS = [SEARCH_TOOL]
NEWS_TOOLS = [NEWS_TOOL]
WEB_FETCH_TOOLS = [WEB_FETCH_TOOL]
HOME_TOOLS = [HOME_TOOL]
MODEL_TOOLS = [MODEL_TOOL]

# All tools combined
ALL_TOOLS = CORE_TOOLS + SEARCH_TOOLS + NEWS_TOOLS + WEB_FETCH_TOOLS + HOME_TOOLS + MODEL_TOOLS


# =============================================================================
# Legacy Tool Name Mapping (for backward compatibility)
# =============================================================================

# Maps old tool names to (new_tool_name, default_arguments)
LEGACY_TOOL_MAP = {
    # Search tools
    "web_search": ("search", {"type": "web"}),
    "news_search": ("search", {"type": "news"}),
    "reddit_search": ("search", {"type": "reddit"}),
    # Model tools
    "list_models": ("model", {"action": "list"}),
    "get_current_model": ("model", {"action": "current"}),
    "switch_model": ("model", {"action": "switch"}),
    # Time
    "get_current_time": ("time", {}),
}


# BawtHub MCP deferred discovery (TASK-2599): only this tool is always offered;
# the schemas it loads join the following rounds.
TOOL_SEARCH_NAME = "tool_search"
TOOL_SEARCH_TOOL = Tool(
    name=TOOL_SEARCH_NAME,
    description=(
        "Find and load more tools (memory, message history, profiles, other "
        "BawtHub tools and tools from connected MCP servers). Search by keywords, or load exact names with "
        "'select:name_a,name_b'. Matching tools become callable on your next step."
    ),
    parameters=[
        ToolParameter(name="query", type="string",
                      description="Keywords, or 'select:' followed by exact tool names"),
        ToolParameter(name="max_results", type="integer",
                      description="Maximum tools to load (1-10, default 5)", required=False),
    ],
)


# =============================================================================
# Tool Selection Functions
# =============================================================================

def ha_tools_to_tool_definitions(ha_tools: list) -> list[Tool]:
    """Convert HAToolDefinition objects to Tool dataclass instances.

    Takes the tool definitions discovered from HA's native MCP server and
    converts them into the same Tool format used by all other llm-bawt tools.
    This allows them to be included in tool lists and schema generation.

    Args:
        ha_tools: List of HAToolDefinition from HomeAssistantNativeClient.

    Returns:
        List of Tool dataclass instances.
    """
    tools = []
    for ha_tool in ha_tools:
        params = []
        schema = ha_tool.input_schema or {}
        properties = schema.get("properties", {})
        required_names = set(schema.get("required", []))

        for prop_name, prop_def in properties.items():
            # Map JSON schema types to our ToolParameter types
            prop_type = prop_def.get("type", "string")
            if prop_type == "array":
                items = prop_def.get("items", {})
                item_type = items.get("type", "string")
                prop_type = f"array[{item_type}]"
            elif prop_type == "number":
                prop_type = "number"

            description = prop_def.get("description", "")
            # Add enum values to description if present
            if "enum" in prop_def:
                enum_str = ", ".join(str(v) for v in prop_def["enum"])
                description = f"{description} (one of: {enum_str})".strip()
            elif "items" in prop_def and "enum" in prop_def.get("items", {}):
                enum_str = ", ".join(str(v) for v in prop_def["items"]["enum"])
                description = f"{description} (values: {enum_str})".strip()

            # Add min/max to description if present
            if "minimum" in prop_def or "maximum" in prop_def:
                range_parts = []
                if "minimum" in prop_def:
                    range_parts.append(f"min={prop_def['minimum']}")
                if "maximum" in prop_def:
                    range_parts.append(f"max={prop_def['maximum']}")
                description = f"{description} ({', '.join(range_parts)})".strip()

            params.append(ToolParameter(
                name=prop_name,
                type=prop_type,
                description=description or f"Parameter '{prop_name}'",
                required=prop_name in required_names,
            ))

        tools.append(Tool(
            name=ha_tool.name,
            description=ha_tool.description or f"Home Assistant tool: {ha_tool.name}",
            parameters=params,
        ))

    return tools


def get_tools_list(
    tools: list[Tool] | None = None,
    include_search_tools: bool = False,
    include_news_tools: bool = False,
    include_web_fetch_tools: bool = False,
    include_home_tools: bool = False,
    include_model_tools: bool = False,
    ha_native_tools: list[Tool] | None = None,
) -> list[Tool]:
    """Return the tool list based on selection flags.

    Args:
        ha_native_tools: Pre-converted HA native MCP tool definitions.
            When provided, these replace HOME_TOOLS entirely.
    """
    if tools is not None:
        return tools

    resolved = CORE_TOOLS.copy()
    if include_search_tools:
        resolved.extend(SEARCH_TOOLS)
    if include_news_tools:
        resolved.extend(NEWS_TOOLS)
    if include_web_fetch_tools:
        resolved.extend(WEB_FETCH_TOOLS)
    if ha_native_tools:
        # Native HA tools replace the legacy home tool
        resolved.extend(ha_native_tools)
    elif include_home_tools:
        # Fallback to legacy home tool
        resolved.extend(HOME_TOOLS)
    if include_model_tools:
        resolved.extend(MODEL_TOOLS)
    return resolved


def get_tools_prompt(
    tools: list[Tool] | None = None,
    include_search_tools: bool = False,
    include_news_tools: bool = False,
    include_web_fetch_tools: bool = False,
    include_home_tools: bool = False,
    include_model_tools: bool = False,
    tool_format: ToolFormat | str = ToolFormat.REACT,
    ha_native_tools: list[Tool] | None = None,
) -> str:
    """Generate the tools instruction prompt.

    Args:
        tools: List of tools to include. If None, auto-selects based on flags.
        include_search_tools: Whether to include web search tools (default False).
        include_news_tools: Whether to include NewsAPI tools (default False).
        include_web_fetch_tools: Whether to include web fetch tools (default False).
        include_home_tools: Whether to include Home Assistant tools (default False).
        include_model_tools: Whether to include model management tools (default False).
        tool_format: Tool format to use for prompt instructions.
        ha_native_tools: Pre-converted HA native MCP tool definitions.

    Returns:
        Formatted prompt string to inject into system message.
    """
    tools = get_tools_list(
        tools=tools,
        include_search_tools=include_search_tools,
        include_news_tools=include_news_tools,
        include_web_fetch_tools=include_web_fetch_tools,
        include_home_tools=include_home_tools,
        include_model_tools=include_model_tools,
        ha_native_tools=ha_native_tools,
    )

    handler = get_format_handler(tool_format)
    if handler is None:
        return ""  # tool_format="none" — no tool prompt needed
    return handler.get_system_prompt(tools)


def get_tool_by_name(name: str, tools: list[Tool] | None = None) -> Tool | None:
    """Get a tool definition by name."""
    if tools is None:
        tools = ALL_TOOLS
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def normalize_legacy_tool_call(name: str, arguments: dict) -> tuple[str, dict]:
    """Convert legacy tool names to new consolidated tools.

    Args:
        name: Tool name (may be legacy or new)
        arguments: Tool arguments

    Returns:
        Tuple of (new_tool_name, merged_arguments)
    """
    if name in LEGACY_TOOL_MAP:
        new_name, default_args = LEGACY_TOOL_MAP[name]
        # Merge defaults with provided arguments (provided takes priority)
        merged = {**default_args, **arguments}
        return new_name, merged
    return name, arguments
