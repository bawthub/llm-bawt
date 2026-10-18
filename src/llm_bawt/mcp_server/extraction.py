"""Extraction service wrapper for llm-bawt MCP server.

Wraps the existing MemoryExtractionService for use with MCP tools.
Handles LLM client initialization and async interface.
Also extracts profile attributes from high-importance facts.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from llm_bawt.memory.extraction.service import MemoryExtractionService, ExtractedFact
from llm_bawt.utils.config import Config

logger = logging.getLogger(__name__)

# Lazy-loaded extraction service
_service: MemoryExtractionService | None = None
_llm_client: Any = None


# Minimum importance to consider for profile attributes (fallback)
PROFILE_ATTRIBUTE_MIN_IMPORTANCE = 0.6

# Allowed profile attribute keys - ONLY core identity/personality traits
# Project details, tools, one-time requests should be memories, not profile attributes
ALLOWED_PROFILE_KEYS = {
    # Core identity facts
    "name", "age", "location", "occupation", "job", "employer",
    "family", "pets", "relationship_status", "gender", "pronouns",
    "nationality", "languages", "timezone",
    # Core health (persistent conditions only, not temporary)
    "health_condition", "disability", "chronic_condition", "allergies",
    # Core preferences (persistent personality/preferences)
    "communication_style", "content_tone", "conversation_style",
    "values", "boundaries", "preferences_summary",
    # Core interests (high-level, not specific projects)
    "hobbies", "interests", "favorite_genres", "gaming_preferences",
}

# Blocked patterns - content that should NEVER be profile attributes.
# IMPORTANT: Patterns must be targeted enough to avoid blocking identity facts.
# e.g. "has 2 dogs" and "has chronic back issues" must NOT be blocked.
BLOCKED_PROFILE_PATTERNS = [
    # Projects, tools, technical systems (but NOT "has/have" alone — too broad)
    r"(?:(?:built|developing|working on|maintains)\s+(?:a |an |the )?(?:[\w-]+\s+){0,3}(?:project|system|app|tool|framework|platform|server|bot|integration|chatbot|assistant))",
    r"(?:uses\s+(?:a |an |the )?(?:tool|app|system|framework|platform|service|stack|server))",
    r"(?:(?:search|internet)\s+(?:integration|provider|capability))",  # Technical capabilities
]


def _extract_attribute_key(content: str) -> str | None:
    """Best-effort profile attribute key extraction for compatibility.

    Older debugging scripts import this private helper directly. Keep a
    lightweight heuristic mapping so those scripts keep working even though
    primary extraction now comes from LLM-provided ``profile_attribute``.
    """
    text = (content or "").strip().lower()
    if not text:
        return None

    patterns: list[tuple[str, str]] = [
        (r"\bname\b|\bcalled\b|\bnamed\b", "name"),
        (r"\bage\b|\byears old\b", "age"),
        (r"\blive\b|\blocation\b|\bbased in\b", "location"),
        (r"\bwork\b|\bjob\b|\boccupation\b|\bemployer\b", "occupation"),
        (r"\bpet\b|\bdog\b|\bcat\b", "pets"),
        (r"\bhobby\b|\benjoy\b|\binterested in\b", "hobbies"),
        (r"\bfavorite\b", "preferences_summary"),
        (r"\blanguage\b", "languages"),
        (r"\btimezone\b", "timezone"),
        (r"\ballerg\b", "allergies"),
    ]

    for pattern, key in patterns:
        if re.search(pattern, text):
            return key if key in ALLOWED_PROFILE_KEYS else None

    return None


def extract_profile_attributes_from_fact(
    fact: ExtractedFact | dict,
    user_id: str,  # Required - must be passed explicitly
    config: Config | None = None,
) -> bool:
    """Extract profile attributes from a fact if the LLM identified it as one.
    
    The LLM extraction prompt now includes profile_attribute classification.
    This function reads that classification and stores the attribute.
    
    Args:
        fact: ExtractedFact object or dict with content, tags, importance, profile_attribute
        user_id: The user to associate the attribute with
        config: Config object for database connection
        
    Returns:
        True if an attribute was created, False otherwise
    """
    config = config or Config()
    
    # Extract fact properties
    if isinstance(fact, dict):
        content = fact.get("content", "")
        importance = fact.get("importance", 0.5)
        profile_attr = fact.get("profile_attribute")
    else:
        content = fact.content
        importance = fact.importance
        profile_attr = getattr(fact, "profile_attribute", None)
    
    # Check if profile extraction is enabled
    if not getattr(config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True):
        logger.debug("[Profile] Profile extraction disabled")
        return False
    
    # Check if LLM identified this as a profile attribute
    if not profile_attr or not isinstance(profile_attr, dict):
        logger.debug(f"[Profile] No profile_attribute from LLM for: '{content[:50]}...'")
        return False
    
    category = profile_attr.get("category")
    key = profile_attr.get("key")
    
    if not category or not key:
        logger.debug(f"[Profile] Invalid profile_attribute format: {profile_attr}")
        return False
    
    # Validate that this key is allowed as a profile attribute
    # Only core identity/personality traits should be profile attributes
    key_lower = key.lower()
    if key_lower not in ALLOWED_PROFILE_KEYS:
        logger.debug(f"[Profile] Key '{key}' not in allowed profile keys, storing as memory only")
        return False
    
    # Check for blocked patterns in content (project details, tools, etc.)
    content_lower = content.lower()
    for pattern in BLOCKED_PROFILE_PATTERNS:
        if re.search(pattern, content_lower):
            logger.debug(f"[Profile] Content matches blocked pattern '{pattern}', storing as memory only")
            return False
    
    # Require higher importance for profile attributes (core identity only)
    if importance < 0.7:
        logger.debug(f"[Profile] Importance {importance:.2f} < 0.7 threshold for profile attribute")
        return False
    
    logger.info(f"[Profile] Validated attribute: {category}.{key} = '{content[:50]}...'")
    
    # Store as profile attribute
    try:
        from llm_bawt.profiles import EntityType
        # TASK-202: reuse the cached MCP-process ProfileManager singleton
        # rather than constructing a new one per fact extraction. Each
        # ProfileManager owns a 5-connection SQLAlchemy pool.
        from llm_bawt.mcp_server.server import _get_profile_manager
        manager = _get_profile_manager()
        attr = manager.set_attribute(
            entity_type=EntityType.USER,
            entity_id=user_id,
            category=category,
            key=key,
            value=content,  # Store full content as value
            confidence=importance,  # Map importance to confidence
            source="extracted",
        )
        
        # If this is a name attribute, also set display_name on the profile
        if key.lower() == "name":
            # Try to extract just the name from content like "User's name is Nick"
            # (re is already imported at module level)
            name_match = re.search(r"(?:name\s+is\s+|named\s+|I'?m\s+|call\s+me\s+)([A-Z][a-z]+)", content, re.IGNORECASE)
            if name_match:
                name = name_match.group(1)
            else:
                # Fallback: use the content directly if it looks like a name
                name = content.strip()
                if len(name) > 30 or " is " in name.lower():
                    # Too long or still a sentence - extract first capitalized word
                    words = [w for w in name.split() if w[0].isupper()]
                    name = words[-1] if words else None  # Last capitalized word is often the name
            
            if name:
                # Don't set display_name to a bot name — extraction sometimes
                # confuses the bot's name with the user's name from conversation context
                from llm_bawt.bots import load_bots
                try:
                    loaded_bots = load_bots()
                    bot_names = {b.get("name", "").lower() for b in loaded_bots.values()}
                    bot_names |= set(loaded_bots.keys())  # also slug names
                except Exception:
                    # No hard-coded identities: an unavailable registry is not
                    # evidence that an unregistered fallback bot exists.
                    bot_names = set()
                
                if name.lower() in bot_names:
                    logger.warning(
                        f"[Profile] Skipping display_name update: '{name}' matches a bot name"
                    )
                else:
                    manager.set_display_name(EntityType.USER, user_id, name)
                    logger.info(f"[Profile] ✓ Set display_name to '{name}' for user {user_id}")

        # Determine if this was a create or update based on timestamps
        is_new = attr.created_at == attr.updated_at
        action = "Created" if is_new else "Updated"
        logger.info(f"[Profile] ✓ {action} attribute: {category}.{key} for user {user_id}")
        return is_new  # Return True only if newly created
        
    except Exception as e:
        logger.exception(f"[Profile] Failed to create profile attribute: {e}")
        return False


def _get_extraction_client(config: Config) -> Any:
    """Get or create an LLM client for extraction.
    
    Prefers a lightweight model for extraction to avoid loading
    large models just for fact extraction.
    """
    global _llm_client
    
    if _llm_client is not None:
        return _llm_client
    
    # Check if there's a configured extraction model (global runtime setting,
    # TASK-522; job-specific first, then the shared maintenance/default job
    # model). Empty falls through to the model-discovery heuristic below.
    from ..runtime_settings import resolve_job_model

    extraction_model = (
        resolve_job_model(config, "extraction_model")
        or resolve_job_model(config, "maintenance_model")
    )
    
    if not extraction_model:
        # Try to find a suitable model (prefer smaller GGUF or OpenAI)
        models = config.defined_models.get("models", {})
        
        # Priority: extraction-specific > small GGUF > any OpenAI > any available
        for alias, info in models.items():
            if "extract" in alias.lower() or "small" in alias.lower():
                extraction_model = alias
                break
        
        if not extraction_model:
            # Fall back to any OpenAI model (fast, no VRAM)
            for alias, info in models.items():
                if info.get("type") == "openai":
                    extraction_model = alias
                    break
        
        if not extraction_model:
            # Fall back to first available
            if models:
                extraction_model = next(iter(models.keys()))
    
    if not extraction_model:
        logger.warning("No model available for extraction, using heuristics only")
        return None
    
    # Initialize the client
    try:
        from llm_bawt.core import LLMBawt
        
        # Create a minimal LLMBawt instance for extraction
        # This will load the model if needed
        llm_bawt = LLMBawt(
            resolved_model_alias=extraction_model,
            bot_id="nova",  # Use nova for extraction context
            config=config,
        )
        _llm_client = llm_bawt.client
        logger.debug(f"Initialized extraction client with model: {extraction_model}")
        return _llm_client
        
    except Exception as e:
        logger.warning(f"Failed to initialize extraction client: {e}")
        return None


def get_extraction_service(config: Config | None = None) -> MemoryExtractionService:
    """Get the singleton extraction service."""
    global _service
    
    if _service is None:
        config = config or Config()
        llm_client = _get_extraction_client(config)
        _service = MemoryExtractionService(llm_client=llm_client)
    
    return _service


async def extract_facts_from_messages(
    messages: list[dict],
    config: Config | None = None,
    use_llm: bool = True,
    user_id: str = "",  # Required - must be passed explicitly
    extract_profile_attributes: bool | None = None,
) -> list[dict]:
    """Extract facts from conversation messages.
    
    Args:
        messages: List of message dicts with role/content.
        config: Config object (uses default if not provided).
        use_llm: Whether to use LLM extraction (falls back to heuristics if False).
        user_id: User ID for profile attribute extraction (required).
        extract_profile_attributes: Whether to also create profile attributes from facts.
        
    Returns:
        List of extracted fact dicts.
    """
    if not user_id:
        raise ValueError("user_id is required for extract_facts_from_messages")
    config = config or Config()
    
    # Check if extraction is enabled
    if not config.MEMORY_EXTRACTION_ENABLED:
        logger.debug("Memory extraction is disabled")
        return []
    
    service = get_extraction_service(config)
    
    try:
        facts = service.extract_from_conversation(
            messages=messages,
            use_llm=use_llm and service.llm_client is not None,
        )
        
        # Filter by minimum importance
        min_importance = config.MEMORY_EXTRACTION_MIN_IMPORTANCE
        facts = [f for f in facts if f.importance >= min_importance]
        
        # Also extract profile attributes from high-importance facts
        if extract_profile_attributes is None:
            extract_profile_attributes = getattr(config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True)

        if extract_profile_attributes:
            for fact in facts:
                try:
                    created = extract_profile_attributes_from_fact(
                        fact=fact,
                        user_id=user_id,
                        config=config,
                    )
                    if created:
                        logger.debug(f"Created profile attribute from: {fact.content[:50]}...")
                except Exception as e:
                    logger.warning(f"Profile attribute extraction failed: {e}")
        
        return [f.to_dict() for f in facts]
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return []
