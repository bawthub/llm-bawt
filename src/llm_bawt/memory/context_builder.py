"""Structured memory context builder for LLM injection.

This module transforms raw memory results into structured, meaningful context
that helps the LLM understand and utilize memories effectively.
"""

from dataclasses import dataclass, field


@dataclass
class MemoryContext:
    """Structured memory context ready for LLM injection."""
    core_context: list[dict] = field(default_factory=list)      # High-stakes foundational memories
    active_concerns: list[dict] = field(default_factory=list)   # Recent seeking-help/advice intents
    preferences: list[dict] = field(default_factory=list)       # Communication/interaction preferences
    background: list[dict] = field(default_factory=list)        # General context/background info
    
    def is_empty(self) -> bool:
        return not any([self.core_context, self.active_concerns, self.preferences, self.background])
    
    def total_count(self) -> int:
        return len(self.core_context) + len(self.active_concerns) + len(self.preferences) + len(self.background)


def categorize_memory(memory: dict) -> str:
    """Categorize a memory based on its meaning fields and tags.
    
    Categories:
    - core_context: High-stakes, foundational facts (name, profession, etc.)
    - active_concerns: User is seeking help/advice/understanding
    - preferences: Communication/interaction preferences
    - background: General context
    
    Priority: Intent-based categories first, then stakes-based.
    """
    intent = (memory.get("intent") or "").lower()
    stakes = (memory.get("stakes") or "").lower()
    tags = memory.get("tags") or []
    importance = memory.get("importance", 0.5)
    content = (memory.get("content") or memory.get("document") or "").lower()
    
    # 1. Intent-based categorization (highest priority)
    # Active concerns - user is seeking something
    seeking_intents = ["seek", "asking", "request", "help", "advice", "understand", 
                       "problem", "concern", "vent", "frustrat", "struggle"]
    if any(s in intent for s in seeking_intents):
        return "active_concerns"
    
    # Preferences - communication style, interaction expectations
    preference_intents = ["prefer", "expect", "style", "communication", "interaction", 
                          "want", "like", "dislike", "value"]
    preference_tags = ["preference", "communication", "lifestyle"]
    if any(p in intent for p in preference_intents) or any(t in preference_tags for t in tags):
        return "preferences"
    
    # 2. Stakes-based categorization
    # Critical/high stakes are core context
    if "critical" in stakes or "high" in stakes:
        return "core_context"
    
    # Very high importance facts are core context
    if importance >= 0.85:
        return "core_context"
    
    # Core identity facts (name, profession) - check content
    identity_keywords = ["name is", "works as", "profession", "occupation", "job is"]
    if any(kw in content for kw in identity_keywords):
        return "core_context"
    
    # Important context but not critical
    if "important" in stakes and importance >= 0.6:
        return "core_context"
    
    # Default to background
    return "background"


def build_structured_context(memories: list[dict]) -> MemoryContext:
    """Transform raw memory results into structured context.
    
    Args:
        memories: List of memory dicts from search results
        
    Returns:
        MemoryContext with categorized memories
    """
    context = MemoryContext()
    
    for mem in memories:
        category = categorize_memory(mem)
        
        if category == "core_context":
            context.core_context.append(mem)
        elif category == "active_concerns":
            context.active_concerns.append(mem)
        elif category == "preferences":
            context.preferences.append(mem)
        else:
            context.background.append(mem)
    
    return context


def format_memory_for_context(memory: dict, include_intent: bool = True) -> str:
    """Format a single memory for inclusion in structured context.
    
    Args:
        memory: Memory dict with content, intent, stakes, etc.
        include_intent: Whether to add contextual intent hint
        
    Returns:
        Formatted string for LLM consumption
    """
    content = memory.get("content", "")
    intent = memory.get("intent")
    
    if include_intent and intent:
        # Add subtle context about why this was shared
        return f"• {content} (shared while {intent})"
    else:
        return f"• {content}"


def format_structured_context(context: MemoryContext, user_name: str | None = None) -> str:
    """Format structured memory context for LLM system prompt injection.
    
    Creates a hierarchical, meaningful presentation of memories that
    helps the LLM understand context and respond appropriately.
    
    Args:
        context: Structured MemoryContext
        user_name: Optional user name for personalization
        
    Returns:
        Formatted string ready for system prompt injection
    """
    if context.is_empty():
        return ""
    
    sections = []
    user_ref = user_name if user_name else "the user"
    
    # Header
    sections.append(f"## What You Remember About {user_ref}\n")
    
    # Core context - always show first, most important
    if context.core_context:
        sections.append("### Core Context")
        for mem in context.core_context:
            sections.append(format_memory_for_context(mem, include_intent=False))
        sections.append("")
    
    # Active concerns - things they may want follow-up on
    if context.active_concerns:
        sections.append("### Active Concerns (may want follow-up)")
        for mem in context.active_concerns:
            sections.append(format_memory_for_context(mem, include_intent=True))
        sections.append("")
    
    # Preferences - how to interact with them
    if context.preferences:
        sections.append("### Their Preferences")
        for mem in context.preferences:
            sections.append(format_memory_for_context(mem, include_intent=False))
        sections.append("")
    
    # Background - general context, less prominent
    if context.background:
        sections.append("### Background")
        for mem in context.background:
            sections.append(format_memory_for_context(mem, include_intent=False))
        sections.append("")
    
    return "\n".join(sections)


def build_memory_context_string(
    memories: list[dict],
    user_name: str | None = None,
) -> str:
    """Main entry point: Convert raw memories to formatted context string.
    
    Args:
        memories: Raw memory dicts from search
        user_name: Optional user name
        
    Returns:
        Formatted context string for LLM injection
    """
    if not memories:
        return ""
    
    context = build_structured_context(memories)
    return format_structured_context(context, user_name)
