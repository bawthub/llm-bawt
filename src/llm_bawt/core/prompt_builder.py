"""Template-based system prompt assembly.

PromptBuilder provides a composable, ordered approach to building system prompts
with clear sections that can be enabled/disabled and reordered per-bot.

Usage:
    builder = PromptBuilder()
    builder.add_section("user_context", "User's name is Nick", position=0)
    builder.add_section("base_prompt", bot.system_prompt, position=2)
    builder.add_section("tools", tools_instructions, position=3)
    system_message = builder.build()
"""

from dataclasses import dataclass, field
from typing import Any
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptSection:
    """A named section of the system prompt.
    
    Attributes:
        name: Unique identifier for this section (e.g., 'user_context', 'tools')
        content: The text content of this section
        enabled: Whether this section should be included in the final prompt
        position: Sort order (lower = earlier in prompt). Default sections:
            0: user_context (About the User)
            1: bot_traits (Bot's developed personality)
            2: base_prompt (Bot's system_prompt from bot_profiles DB)
            3: memory_context (Retrieved memories - non-tool bots)
            4: tools (Tool calling instructions - tool bots)
        metadata: Optional extra data (e.g., source, confidence)
    """
    name: str
    content: str
    enabled: bool = True
    position: int = 50  # Default to middle
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Strip whitespace but preserve internal formatting
        if self.content:
            self.content = self.content.strip()


# Standard section positions
class SectionPosition:
    """Standard ordering for prompt sections."""
    DATETIME = -1  # Reserved for the dormant legacy pipeline; not live prompt assembly.
    USER_CONTEXT = 0
    BOT_TRAITS = 1
    BASE_PROMPT = 2
    MEMORY_CONTEXT = 3
    TOOLS = 4
    CLIENT_CONTEXT = 5  # System context passed by the calling client
    GLOBAL_INSTRUCTIONS = 6  # Global behavioral instructions for all bots
    WORKING_MEMORY = 7  # TASK-1281: the agent's working-memory list, late in the prompt
    CUSTOM = 50  # For bot-specific additions


# ---------------------------------------------------------------------------
# Global system prompt fragment — injected for every memory-enabled bot
# ---------------------------------------------------------------------------

GLOBAL_SYSTEM_PROMPT = """\
CONVERSATION AND TASK RECALL:
For what's active now, start with compact `tasks_board`; open a specific task with \
`tasks_get_context` only when its details matter. For a past conversation, decision, \
or "why", use `context_recall` first: it researches authorized history in an \
isolated utility-model context and returns bounded cited quotes, not transcripts. \
Default sources are messages, memories and tasks (task key fields); add \
`task_details` for task reviews and step outputs. Check coverage and citations: `no_match` requires \
complete successful coverage; `insufficient_evidence` or an unavailable source is \
not proof that nothing happened. Historical attribution is not current ownership.

LOCATE AND EXPAND:
For a cross-bot literal, identifier, recent browse or time-filtered lookup — or when \
recall is insufficient — use compact `messages_find`, with `bot`, `speaker`, \
`since`/`until` as needed (`24h`, `7d`, ISO date/time or Unix seconds). It searches \
only bots you may read. Expand a card with `messages_context` or a known ID with \
`messages_get_by_id`. Use bounded `messages_get` or `messages_search_all` only for \
exact text, IDs, full thread context or troubleshooting, not reflexively; these \
return bulky history. Never infer access to protected bots from a search result.

CROSS-BOT MEMORY:
For extracted facts rather than conversation evidence, use `memory_search_all`; \
`memory_list_sources` and `memory_search_source` narrow an allowed source. \
Memory summaries alone are not a substitute for the original conversation. \
Weave cited findings naturally into your answer without announcing the lookup.\
"""


class PromptBuilder:
    """Assembles system prompts from ordered, named sections.
    
    Provides a clear, inspectable structure for prompt composition with:
    - Named sections that can be individually enabled/disabled
    - Consistent ordering via position values
    - Debug/verbose output showing what's included
    - Template variable substitution (optional)
    """
    
    def __init__(self, separator: str = "\n\n"):
        """Initialize the builder.
        
        Args:
            separator: String to join sections (default: double newline)
        """
        self._sections: OrderedDict[str, PromptSection] = OrderedDict()
        self._separator = separator
    
    def add_section(
        self,
        name: str,
        content: str,
        *,
        position: int | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> "PromptBuilder":
        """Add or update a named section.
        
        Args:
            name: Unique section identifier
            content: Section text content
            position: Sort order (lower = earlier). Uses SectionPosition constants.
            enabled: Whether to include this section in build()
            metadata: Optional extra data for debugging/inspection
            overwrite: If False, skip if section already exists
            
        Returns:
            self for method chaining
        """
        if not overwrite and name in self._sections:
            logger.debug(f"Section '{name}' already exists, skipping (overwrite=False)")
            return self
        
        if not content or not content.strip():
            logger.debug(f"Section '{name}' has empty content, skipping")
            return self
        
        # Determine position
        if position is None:
            # Auto-assign based on known section names
            position = self._get_default_position(name)
        
        section = PromptSection(
            name=name,
            content=content,
            enabled=enabled,
            position=position,
            metadata=metadata or {},
        )
        
        self._sections[name] = section
        logger.debug(f"Added section '{name}' at position {position} (enabled={enabled})")
        return self
    
    def _get_default_position(self, name: str) -> int:
        """Get default position for known section names."""
        defaults = {
            "user_context": SectionPosition.USER_CONTEXT,
            "bot_traits": SectionPosition.BOT_TRAITS,
            "base_prompt": SectionPosition.BASE_PROMPT,
            "memory_context": SectionPosition.MEMORY_CONTEXT,
            "tools": SectionPosition.TOOLS,
            "client_context": SectionPosition.CLIENT_CONTEXT,
        }
        return defaults.get(name, SectionPosition.CUSTOM)
    
    def remove_section(self, name: str) -> "PromptBuilder":
        """Remove a section by name."""
        if name in self._sections:
            del self._sections[name]
            logger.debug(f"Removed section '{name}'")
        return self
    
    def enable_section(self, name: str, enabled: bool = True) -> "PromptBuilder":
        """Enable or disable a section."""
        if name in self._sections:
            self._sections[name].enabled = enabled
            logger.debug(f"Section '{name}' enabled={enabled}")
        return self
    
    def get_section(self, name: str) -> PromptSection | None:
        """Get a section by name."""
        return self._sections.get(name)
    
    def has_section(self, name: str) -> bool:
        """Check if a section exists and is enabled."""
        section = self._sections.get(name)
        return section is not None and section.enabled
    
    @property
    def sections(self) -> list[PromptSection]:
        """Get all sections in position order."""
        return sorted(self._sections.values(), key=lambda s: s.position)
    
    @property
    def enabled_sections(self) -> list[PromptSection]:
        """Get enabled sections in position order."""
        return [s for s in self.sections if s.enabled]
    
    def build(self) -> str:
        """Assemble the final system prompt from enabled sections.
        
        Returns:
            Concatenated prompt string with sections joined by separator
        """
        enabled = self.enabled_sections
        
        if not enabled:
            logger.warning("PromptBuilder.build() called with no enabled sections")
            return ""
        
        result = self._separator.join(s.content for s in enabled)
        
        logger.debug(
            f"Built prompt with {len(enabled)} sections: "
            f"{[s.name for s in enabled]}"
        )
        
        return result
    
    def build_with_debug(self) -> tuple[str, dict[str, Any]]:
        """Build prompt and return debug info.
        
        Returns:
            Tuple of (prompt_string, debug_info_dict)
        """
        enabled = self.enabled_sections
        disabled = [s for s in self.sections if not s.enabled]
        
        debug_info = {
            "total_sections": len(self._sections),
            "enabled_count": len(enabled),
            "disabled_count": len(disabled),
            "section_order": [s.name for s in enabled],
            "disabled_sections": [s.name for s in disabled],
            "section_sizes": {s.name: len(s.content) for s in enabled},
            "total_chars": sum(len(s.content) for s in enabled),
        }
        
        return self.build(), debug_info
    
    def get_verbose_summary(self) -> str:
        """Get a human-readable summary for --verbose output."""
        lines = ["[dim]─── System Prompt Structure ───[/dim]"]
        
        for section in self.sections:
            status = "✓" if section.enabled else "○"
            size = len(section.content)
            lines.append(
                f"  {status} [cyan]{section.name}[/cyan] "
                f"(pos={section.position}, {size} chars)"
            )
        
        total = sum(len(s.content) for s in self.enabled_sections)
        lines.append(f"[dim]  Total: {total} characters[/dim]")
        
        return "\n".join(lines)
    
    def clear(self) -> "PromptBuilder":
        """Remove all sections."""
        self._sections.clear()
        return self
    
    def copy(self) -> "PromptBuilder":
        """Create a copy of this builder."""
        new_builder = PromptBuilder(separator=self._separator)
        for name, section in self._sections.items():
            new_builder._sections[name] = PromptSection(
                name=section.name,
                content=section.content,
                enabled=section.enabled,
                position=section.position,
                metadata=section.metadata.copy(),
            )
        return new_builder
