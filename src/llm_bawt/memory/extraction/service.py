"""
Memory extraction service for distilling important facts from conversations.
Uses LLM to extract facts and handle memory updates/conflicts.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol

from ..speaker_attribution import extraction_input, human_user_turns_present
from .prompts import (
    MEMORY_TAGS,
    get_fact_extraction_prompt,
    get_memory_update_prompt,
    get_summary_extraction_prompt,
    estimate_importance,
)

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients that can be used for extraction."""
    
    def query(self, messages: list, plaintext_output: bool = True, **kwargs) -> str:
        """Query the LLM with messages."""
        ...


@dataclass
class ExtractedFact:
    """A fact extracted from a conversation."""
    content: str
    tags: list[str]
    importance: float
    source_message_ids: list[str] = field(default_factory=list)
    meaning: Optional["MeaningAssociation"] = None
    profile_attribute: dict | None = None  # If set, should be stored as user profile attribute
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "tags": self.tags,
            "importance": self.importance,
            "source_message_ids": self.source_message_ids,
            "meaning": self.meaning.to_dict() if self.meaning else None,
            "profile_attribute": self.profile_attribute,
        }


@dataclass
class MeaningAssociation:
    """Meaning-level attributes for a memory."""
    intent: str | None = None
    stakes: str | None = None
    emotional_charge: float | None = None  # 0-1 scale
    recurrence_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "stakes": self.stakes,
            "emotional_charge": self.emotional_charge,
            "recurrence_keywords": self.recurrence_keywords,
        }

    def to_embedding_text(self) -> str:
        parts = []
        if self.intent:
            parts.append(f"intent: {self.intent}")
        if self.stakes:
            parts.append(f"stakes: {self.stakes}")
        if self.emotional_charge is not None:
            parts.append(f"emotional_charge: {self.emotional_charge:.2f}")
        if self.recurrence_keywords:
            parts.append(f"recurrence: {' '.join(self.recurrence_keywords)}")
        return " | ".join(parts)


@dataclass
class MemoryAction:
    """An action to take on memory (ADD, UPDATE, DELETE, NONE)."""
    action: str  # ADD, UPDATE, DELETE, NONE
    fact: ExtractedFact | None = None
    target_memory_id: str | None = None
    reason: str = ""


class MemoryExtractionService:
    """
    Service for extracting important facts from conversations and managing memory updates.
    
    This service uses an LLM to:
    1. Extract discrete facts from conversation exchanges
    2. Compare new facts against existing memories
    3. Determine appropriate actions (ADD, UPDATE, DELETE, NONE)
    """
    
    def __init__(self, llm_client: LLMClientProtocol | None = None):
        """
        Initialize the extraction service.
        
        Args:
            llm_client: Optional LLM client for extraction. If not provided,
                       will use heuristic-based extraction.
        """
        self.llm_client = llm_client
    
    def extract_from_conversation(
        self,
        messages: list[dict],
        use_llm: bool = True,
    ) -> list[ExtractedFact]:
        """
        Extract important facts from a conversation.
        
        Args:
            messages: List of message dicts with 'role', 'content', and optionally 'id'
            use_llm: Whether to use LLM for extraction (falls back to heuristics if False)
            
        Returns:
            List of ExtractedFact objects
        """
        # Bot-origin deliveries are another bot's speech, never user facts
        # (TASK-1724). With no human turn left there is nothing to extract.
        messages = extraction_input(messages or [])
        if not human_user_turns_present(messages):
            return []

        # Collect message IDs for source linking
        message_ids = [msg.get("id", str(uuid.uuid4())) for msg in messages]
        
        # Format conversation for the prompt
        conversation_text = self._format_conversation(messages)
        
        if use_llm and self.llm_client:
            try:
                return self._extract_with_llm(conversation_text, message_ids)
            except Exception as e:
                logger.warning(f"LLM extraction failed, falling back to heuristics: {e}")
                return self._extract_with_heuristics(messages, message_ids)
        else:
            return self._extract_with_heuristics(messages, message_ids)
    
    def extract_from_summary(
        self,
        summary_text: str,
        session_start: float,
        session_end: float,
        summary_id: str,
        use_llm: bool = True,
    ) -> list[ExtractedFact]:
        """Extract facts from a conversation summary.

        Used by the scheduler to extract memories from pre-computed summaries
        rather than raw message pairs. Designed for use with capable API models
        (e.g., Grok) that produce higher quality extractions.

        Args:
            summary_text: The summary text (third-person)
            session_start: Session start timestamp
            session_end: Session end timestamp
            summary_id: UUID of the summary message (used as source reference)
            use_llm: Whether to use LLM for extraction

        Returns:
            List of ExtractedFact objects
        """
        if not self.summary_has_extractable_content(summary_text):
            return []

        if use_llm and self.llm_client:
            try:
                return self._extract_from_summary_with_llm(
                    summary_text, session_start, session_end, summary_id
                )
            except Exception as e:
                logger.warning(f"LLM summary extraction failed, falling back to heuristics: {e}")

        # Heuristic fallback: treat summary as a single user message
        fake_messages = [{"role": "user", "content": summary_text, "id": summary_id}]
        return self._extract_with_heuristics(fake_messages, [summary_id])

    @staticmethod
    def summary_has_extractable_content(summary_text: str | None) -> bool:
        """Return whether a structured summary contains any substantive text.

        Summarizers can legitimately return an all-placeholder summary for
        reset commands, smoke tests, or empty/internal sessions.  Sending that
        output to a second model can never produce a grounded persistent fact.
        """
        if not summary_text or not summary_text.strip():
            return False

        content = re.sub(
            r"(?im)^\s*(?:\[historical summary\]|summary:|key details:|intent:|tone:|open loops:)\s*",
            "",
            summary_text,
        )
        content = re.sub(
            r"(?i)\b(?:n/?a|none|not applicable|no open loops?)\b[\s,;.!-]*",
            "",
            content,
        )
        return bool(re.search(r"[A-Za-z0-9]", content))

    def _extract_from_summary_with_llm(
        self,
        summary_text: str,
        session_start: float,
        session_end: float,
        summary_id: str,
    ) -> list[ExtractedFact]:
        """Use LLM to extract facts from a conversation summary."""
        from ...models.message import Message

        start_date = datetime.fromtimestamp(session_start).strftime("%Y-%m-%d %H:%M")
        end_date = datetime.fromtimestamp(session_end).strftime("%Y-%m-%d %H:%M")

        prompt = get_summary_extraction_prompt(summary_text, start_date, end_date)

        system_msg = Message(
            role="system",
            content="You are a memory extraction assistant. Extract persistent user facts from conversation summaries and return them as JSON.",
        )
        user_msg = Message(role="user", content=prompt)

        response = self.llm_client.query(
            messages=[system_msg, user_msg],
            plaintext_output=True,
            stream=False,
        )

        facts = self._parse_extraction_response(response, [summary_id])
        return [self._enrich_meaning(f) for f in facts]

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format messages into a conversation string for the prompt.
        
        Uses clear labels to help the LLM distinguish user vs assistant content.
        """
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").lower()
            content = msg.get("content", "")
            # Use very explicit labels
            if role == "user":
                lines.append(f"[USER SAID]: {content}")
            elif role == "assistant":
                lines.append(f"[ASSISTANT/BOT SAID - IGNORE FOR EXTRACTION]: {content}")
            else:
                lines.append(f"[{role.upper()}]: {content}")
        return "\n".join(lines)
    
    def _extract_with_llm(
        self,
        conversation_text: str,
        message_ids: list[str],
    ) -> list[ExtractedFact]:
        """Use LLM to extract facts from the conversation."""
        from ...models.message import Message
        
        prompt = get_fact_extraction_prompt(conversation_text)
        
        # Include a system message to prevent the client from injecting the bot's
        # personality system prompt. The extraction task needs a clean context.
        system_msg = Message(
            role="system",
            content="You are a memory extraction assistant. Extract facts from conversations and return them as JSON."
        )
        user_msg = Message(role="user", content=prompt)
        
        # Use stream=False to avoid any output printing
        response = self.llm_client.query(
            messages=[system_msg, user_msg],
            plaintext_output=True,
            stream=False,
        )
        
        # Parse JSON response
        facts = self._parse_extraction_response(response, message_ids)
        return [self._enrich_meaning(f) for f in facts]
    
    def _parse_extraction_response(
        self,
        response: str,
        message_ids: list[str],
    ) -> list[ExtractedFact]:
        """Parse the LLM response into ExtractedFact objects."""
        facts = []
        
        # Try to extract JSON from the response (look for code blocks first)
        # Match ```json ... ``` or ``` ... ``` blocks - handle both {} and []
        code_block_match = re.search(r'```(?:json)?\s*([\[{][\s\S]*?[\]}])\s*```', response)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # Fall back to finding raw JSON - try object first, then array
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                # Maybe it's an array?
                json_match = re.search(r'\[[\s\S]*\]', response)
            if not json_match:
                logger.debug(f"No JSON found in extraction response. Response was: {response[:200]}...")
                return facts
            json_str = json_match.group()
        
        try:
            # Try to fix common LLM JSON issues
            fixed_json = self._fix_json(json_str)
            data = json.loads(fixed_json)
            
            # Handle both {"facts": [...]} and direct [...] array
            if isinstance(data, list):
                raw_facts = data
            else:
                raw_facts = data.get("facts", [])
            
            for raw_fact in raw_facts:
                content = raw_fact.get("content", "").strip()
                if not content:
                    continue
                
                raw_tags = raw_fact.get("tags") or []
                if isinstance(raw_tags, str):
                    raw_tags = [raw_tags]
                tags = [t.strip().lower() for t in raw_tags if t and isinstance(t, str)]
                if not tags:
                    tags = ["misc"]
                # Normalize known tags
                normalized = []
                for tag in tags:
                    normalized.append(tag if tag in MEMORY_TAGS else tag)
                tags = normalized
                
                importance = float(raw_fact.get("importance", 0.5))
                importance = max(0.0, min(1.0, importance))  # Clamp to 0-1
                
                # Capture LLM-inferred intent if provided
                llm_intent = raw_fact.get("intent")
                meaning = None
                if llm_intent:
                    meaning = MeaningAssociation(intent=llm_intent)
                
                # Capture profile_attribute if provided by LLM
                profile_attr = raw_fact.get("profile_attribute")
                if profile_attr and isinstance(profile_attr, dict):
                    # Validate it has required fields
                    if "category" in profile_attr and "key" in profile_attr:
                        logger.debug(f"LLM identified profile attribute: {profile_attr}")
                    else:
                        profile_attr = None  # Invalid format
                
                facts.append(ExtractedFact(
                    content=content,
                    tags=tags,
                    importance=importance,
                    source_message_ids=message_ids.copy(),
                    meaning=meaning,
                    profile_attribute=profile_attr,
                ))
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction JSON: {e}")
            logger.debug(f"JSON string was: {json_str[:500] if json_str else 'empty'}...")
            logger.debug(f"Fixed JSON was: {fixed_json[:500] if fixed_json else 'empty'}...")
        except Exception as e:
            logger.warning(f"Error processing extraction response: {e}")
        
        return facts
    
    def _fix_json(self, json_str: str) -> str:
        """Attempt to fix common JSON issues from LLM output."""
        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Try to add quotes around unquoted property names
        # Match word: at the start of a line or after { or ,
        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
        
        # Replace single quotes with double quotes for property names
        # This is a simplified approach - matches 'key': patterns
        fixed = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'\s*:", r'"\1":', fixed)
        
        # Replace single-quoted string values with double quotes
        # This is tricky - we try to match ': 'value' patterns
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        
        return fixed
    
    def _extract_with_heuristics(
        self,
        messages: list[dict],
        message_ids: list[str],
    ) -> list[ExtractedFact]:
        """
        Extract facts using simple heuristics (fallback when LLM unavailable).
        
        This is a simplified extraction that looks for certain patterns
        indicating important information.
        """
        facts = []
        
        # Patterns that might indicate important information
        patterns = {
            "preference": [
                r"(?:i|I)\s+(?:prefer|like|love|hate|dislike)\s+(.+)",
                r"(?:my|I)\s+favorite\s+(.+)",
            ],
            "fact": [
                r"(?:i|I)\s+(?:am|'m)\s+(?:a|an)?\s*(.+)",
                r"(?:my|I)\s+(?:name|age|location)\s+(?:is)?\s*(.+)",
            ],
            "professional": [
                r"(?:i|I)\s+work\s+(?:at|for|on|as)\s+(.+)",
                r"(?:my|I)\s+(?:job|career|profession)\s+(?:is)?\s*(.+)",
            ],
            "health": [
                r"(?:i|I)\s+(?:have|had|suffer from)\s+(.+?)(?:\.|$)",
                r"(?:my|I)\s+(?:doctor|medication|condition)\s+(.+)",
            ],
        }
        
        for msg in messages:
            if msg.get("role") != "user":
                continue
            
            content = msg.get("content", "")
            
            for tag, type_patterns in patterns.items():
                for pattern in type_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]
                        
                        fact_content = f"User {match.strip()}"
                        importance = estimate_importance(content)
                        
                        facts.append(ExtractedFact(
                            content=fact_content,
                            tags=[tag],
                            importance=importance,
                            source_message_ids=message_ids.copy(),
                        ))
        # Attach lightweight meaning heuristics
        return [self._enrich_meaning(f) for f in facts]
    
    def determine_memory_actions(
        self,
        new_facts: list[ExtractedFact],
        existing_memories: list[dict],
    ) -> list[MemoryAction]:
        """
        Determine what actions to take for new facts against existing memories.
        
        Args:
            new_facts: Facts extracted from recent conversation
            existing_memories: Current memories from the database
            
        Returns:
            List of MemoryAction objects describing what to do
        """
        if not new_facts:
            return []
        
        if not existing_memories:
            # All facts are new, just add them
            return [
                MemoryAction(action="ADD", fact=fact, reason="New memory")
                for fact in new_facts
            ]
        
        if self.llm_client:
            try:
                return self._determine_actions_with_llm(new_facts, existing_memories)
            except Exception as e:
                logger.warning(f"LLM action determination failed: {e}")
                return self._determine_actions_with_heuristics(new_facts, existing_memories)
        else:
            return self._determine_actions_with_heuristics(new_facts, existing_memories)

    # ------------------------------------------------------------------
    # Meaning enrichment (heuristic, inline)
    # ------------------------------------------------------------------
    def _enrich_meaning(self, fact: ExtractedFact) -> ExtractedFact:
        """Attach heuristic meaning metadata to a fact.
        
        Preserves LLM-inferred intent if present, fills in other fields with heuristics.
        """
        # Preserve LLM-inferred intent if we already have meaning
        existing_intent = fact.meaning.intent if fact.meaning else None
        
        meaning = MeaningAssociation(
            intent=existing_intent or self._infer_intent(fact),
            stakes=self._infer_stakes(fact),
            emotional_charge=self._infer_emotion(fact),
            recurrence_keywords=self._infer_recurrence(fact),
        )
        fact.meaning = meaning
        return fact

    def _infer_intent(self, fact: ExtractedFact) -> str:
        tags = set(fact.tags)
        if "plan" in tags:
            return "planning or future goal"
        if "event" in tags:
            return "share time-bound event"
        if "preference" in tags:
            return "express preference"
        if "professional" in tags:
            return "share work/professional context"
        if "health" in tags:
            return "share health context"
        return "share personal context"

    def _infer_stakes(self, fact: ExtractedFact) -> str:
        if fact.importance >= 0.8:
            return "critical to remember; forgetting would harm trust or outcomes"
        if fact.importance >= 0.6:
            return "important context for smooth interactions"
        return "nice-to-know; low stakes"

    def _infer_emotion(self, fact: ExtractedFact) -> float:
        text = fact.content.lower()
        charge = 0.2 + 0.6 * fact.importance
        keywords_high = ["love", "hate", "angry", "excited", "anxious", "worried", "sad"]
        if any(k in text for k in keywords_high):
            charge = max(charge, 0.8)
        return min(1.0, max(0.0, charge))

    def _infer_recurrence(self, fact: ExtractedFact) -> list[str]:
        tags = [t for t in fact.tags if t]
        words = re.findall(r"[a-zA-Z]{4,}", fact.content.lower())
        return list(set(tags + words[:3]))
    
    def _determine_actions_with_llm(
        self,
        new_facts: list[ExtractedFact],
        existing_memories: list[dict],
    ) -> list[MemoryAction]:
        """Use LLM to determine memory actions."""
        from ...models.message import Message
        
        # Format memories and facts for the prompt
        existing_json = json.dumps([
            {"id": m.get("id"), "content": m.get("content"), "importance": m.get("importance", 0.5)}
            for m in existing_memories
        ], indent=2)
        
        new_facts_json = json.dumps([f.to_dict() for f in new_facts], indent=2)
        
        prompt = get_memory_update_prompt(existing_json, new_facts_json)
        
        # Include a system message to prevent the client from injecting the bot's
        # personality system prompt. The memory management task needs a clean context.
        system_msg = Message(
            role="system",
            content="You are a memory management assistant. Compare facts and determine actions (ADD, UPDATE, DELETE, NONE) as JSON."
        )
        user_msg = Message(role="user", content=prompt)
        
        # Use stream=False to avoid any output printing
        response = self.llm_client.query(
            messages=[system_msg, user_msg],
            plaintext_output=True,
            stream=False,
        )
        
        return self._parse_action_response(response, new_facts)
    
    def _parse_action_response(
        self,
        response: str,
        new_facts: list[ExtractedFact],
    ) -> list[MemoryAction]:
        """Parse the LLM response into MemoryAction objects.
        
        When possible, matches parsed facts back to original facts to preserve
        meaning fields (intent, etc.) that were extracted with conversation context.
        """
        actions = []
        
        # Build lookup for matching back to original facts
        def find_original_fact(content: str) -> ExtractedFact | None:
            """Find original fact by content similarity."""
            content_lower = content.lower().strip()
            for fact in new_facts:
                if fact.content.lower().strip() == content_lower:
                    return fact
                # Also try partial match for slight LLM rewording
                if len(content_lower) > 20 and content_lower[:20] in fact.content.lower():
                    return fact
            return None
        
        # Try to extract JSON from the response (look for code blocks first)
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.debug("No JSON found in action response")
                # Default to adding all facts
                return [MemoryAction(action="ADD", fact=f) for f in new_facts]
            json_str = json_match.group()
        
        try:
            fixed_json = self._fix_json(json_str)
            data = json.loads(fixed_json)
            raw_actions = data.get("actions", [])
            
            for raw_action in raw_actions:
                action_type = raw_action.get("action", "NONE").upper()
                
                if action_type == "ADD":
                    fact_data = raw_action.get("fact", {})
                    content = fact_data.get("content", "")
                    
                    # Try to match back to original fact (preserves intent from extraction)
                    original = find_original_fact(content)
                    if original:
                        fact = original
                    else:
                        # Fallback: create new fact with heuristic meaning
                        tags = fact_data.get("tags") or fact_data.get("memory_type") or []
                        if isinstance(tags, str):
                            tags = [tags]
                        tags = [t.strip().lower() for t in tags if t]
                        if not tags:
                            tags = ["misc"]
                        fact = ExtractedFact(
                            content=content,
                            tags=tags,
                            importance=float(fact_data.get("importance", 0.5)),
                            source_message_ids=fact_data.get("source_message_ids", []),
                        )
                        fact = self._enrich_meaning(fact)
                    
                    actions.append(MemoryAction(
                        action="ADD",
                        fact=fact,
                        reason=raw_action.get("reason", ""),
                    ))
                    
                elif action_type == "UPDATE":
                    fact_data = raw_action.get("fact", {})
                    content = fact_data.get("content", "")
                    
                    # Try to match back to original fact (preserves intent from extraction)
                    original = find_original_fact(content)
                    if original:
                        fact = original
                    else:
                        # Fallback: create new fact with heuristic meaning
                        tags = fact_data.get("tags") or fact_data.get("memory_type") or []
                        if isinstance(tags, str):
                            tags = [tags]
                        tags = [t.strip().lower() for t in tags if t]
                        if not tags:
                            tags = ["misc"]
                        fact = ExtractedFact(
                            content=content,
                            tags=tags,
                            importance=float(fact_data.get("importance", 0.5)),
                            source_message_ids=fact_data.get("source_message_ids", []),
                        )
                        fact = self._enrich_meaning(fact)
                    
                    actions.append(MemoryAction(
                        action="UPDATE",
                        fact=fact,
                        target_memory_id=raw_action.get("target_memory_id"),
                        reason=raw_action.get("reason", ""),
                    ))
                    
                elif action_type == "DELETE":
                    actions.append(MemoryAction(
                        action="DELETE",
                        target_memory_id=raw_action.get("target_memory_id"),
                        reason=raw_action.get("reason", ""),
                    ))
                # NONE actions are ignored
                
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse action JSON: {e}")
            return [MemoryAction(action="ADD", fact=f) for f in new_facts]
        
        return actions
    
    def _determine_actions_with_heuristics(
        self,
        new_facts: list[ExtractedFact],
        existing_memories: list[dict],
    ) -> list[MemoryAction]:
        """
        Simple heuristic-based action determination.
        
        Checks for exact and near-duplicate content to avoid adding duplicates.
        """
        actions = []
        
        existing_contents = {
            m.get("id"): m.get("content", "").lower()
            for m in existing_memories
        }
        
        for fact in new_facts:
            fact_lower = fact.content.lower()
            is_duplicate = False
            
            for mem_id, mem_content in existing_contents.items():
                # Check for exact match or high similarity
                if fact_lower == mem_content:
                    is_duplicate = True
                    break
                
                # Simple similarity check (could use proper similarity metric)
                if self._simple_similarity(fact_lower, mem_content) > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                actions.append(MemoryAction(
                    action="ADD",
                    fact=fact,
                    reason="New information",
                ))
        
        return actions
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-based similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
