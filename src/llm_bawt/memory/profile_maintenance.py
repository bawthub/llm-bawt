"""LLM-based profile attribute consolidation and summarization."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from llm_bawt.profiles import ProfileManager, EntityType, AttributeCategory
from llm_bawt.clients.base import LLMClient
from llm_bawt.models.message import Message
from llm_bawt.memory.extraction.prompts import get_profile_consolidation_prompt

logger = logging.getLogger(__name__)


@dataclass
class ProfileMaintenanceResult:
    """Result of profile maintenance run."""
    entity_id: str
    attributes_before: int
    attributes_after: int
    categories_updated: list[str]
    error: Optional[str] = None


class ProfileMaintenanceService:
    """Service to consolidate and clean up user profile attributes."""
    
    def __init__(self, profile_manager: ProfileManager, llm_client: LLMClient):
        self.profile_manager = profile_manager
        self.llm_client = llm_client
    
    def run(self, entity_id: str, entity_type: str = "user", dry_run: bool = False) -> ProfileMaintenanceResult:
        """
        Consolidate profile attributes for an entity.
        
        Args:
            entity_id: User or bot ID
            entity_type: "user" or "bot"
            dry_run: If True, don't save changes
            
        Returns:
            ProfileMaintenanceResult with before/after counts
        """
        # Convert string to enum
        entity_type_enum = EntityType.USER if entity_type == "user" else EntityType.BOT
        
        # Get current attributes
        attributes = self.profile_manager.get_all_attributes(entity_type_enum, entity_id)
        if not attributes:
            return ProfileMaintenanceResult(
                entity_id=entity_id,
                attributes_before=0,
                attributes_after=0,
                categories_updated=[]
            )
        
        attributes_before = len(attributes)
        
        # Format for LLM
        attr_lines = []
        for attr in attributes:
            attr_lines.append(f"{attr.category}: {attr.key} = {attr.value}")
        
        prompt = get_profile_consolidation_prompt("\n".join(attr_lines))
        
        try:
            # Call LLM
            response = self.llm_client.query([Message(role="user", content=prompt)], plaintext_output=True)
            
            # Parse JSON from response
            consolidated = self._parse_json_response(response)
            if not consolidated:
                return ProfileMaintenanceResult(
                    entity_id=entity_id,
                    attributes_before=attributes_before,
                    attributes_after=attributes_before,
                    categories_updated=[],
                    error="Failed to parse LLM response as JSON"
                )
            
            if dry_run:
                return ProfileMaintenanceResult(
                    entity_id=entity_id,
                    attributes_before=attributes_before,
                    attributes_after=0,
                    categories_updated=list(consolidated.keys())
                )
            
            # Build the final summary string from LLM output
            summary_lines = []
            
            logger.debug(f"Consolidated keys for {entity_id}: {list(consolidated.keys())}")
            
            # Handle "name" / "identity" field - extract name for display_name
            name_value = None
            identity_text = consolidated.get("identity", "")
            # Coerce non-string values (lists, dicts) to string
            if identity_text and not isinstance(identity_text, str):
                identity_text = ", ".join(str(v) for v in identity_text) if isinstance(identity_text, list) else str(identity_text)
            if identity_text and isinstance(identity_text, str):
                # Try to extract name from identity (e.g., "Alex is a software developer")
                import re
                name_match = re.match(r'^(\w+)\s+is\s+', identity_text)
                if name_match:
                    name_value = name_match.group(1)
                summary_lines.append(f"User's identity: {identity_text}")
            
            # Also check for explicit "name" field
            if "name" in consolidated and consolidated["name"]:
                explicit_name = consolidated["name"]
                if isinstance(explicit_name, str) and explicit_name.strip() and explicit_name.lower() != "null":
                    name_value = explicit_name.strip()
            
            # Set display_name if we found a name
            if name_value:
                self.profile_manager.set_display_name(entity_type_enum, entity_id, name_value)
                logger.info(f"Set display_name to '{name_value}' for {entity_id}")
            
            # Add other categories to summary
            category_order = ["preferences", "interests", "context", "notes"]
            category_labels = {
                "preferences": "User's preferences",
                "interests": "User's interests", 
                "context": "User's current context",
                "notes": "Additional notes",
            }
            
            for cat_key in category_order:
                value = consolidated.get(cat_key, "")
                # Coerce non-string values (lists, dicts) to string
                if value and not isinstance(value, str):
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)
                    else:
                        value = str(value)
                if value and isinstance(value, str) and value.strip() and value.lower() != "null":
                    label = category_labels.get(cat_key, cat_key.title())
                    summary_lines.append(f"{label}: {value.strip()}")
            
            # Build final summary
            final_summary = "\n".join(summary_lines) if summary_lines else None
            
            if not final_summary:
                logger.warning(
                    f"Profile maintenance for {entity_id} produced no summary. "
                    f"Consolidated types: {
                        {k: type(v).__name__ for k, v in consolidated.items()}
                    }"
                )
            
            # Save summary to profile (attributes are KEPT for future consolidation runs)
            if final_summary:
                self.profile_manager.set_profile_summary(entity_type_enum, entity_id, final_summary)
                logger.info(f"Saved profile summary for {entity_id} ({len(final_summary)} chars)")
            
            # NOTE: We intentionally do NOT delete individual attributes.
            # They serve as the source of truth for future consolidation runs.
            # The summary is a cached/precomputed view for fast system prompt injection.
            
            return ProfileMaintenanceResult(
                entity_id=entity_id,
                attributes_before=attributes_before,
                attributes_after=attributes_before,  # Unchanged - we keep them
                categories_updated=list(consolidated.keys())
            )
            
        except Exception as e:
            logger.exception(f"Profile maintenance failed for {entity_id}")
            return ProfileMaintenanceResult(
                entity_id=entity_id,
                attributes_before=attributes_before,
                attributes_after=attributes_before,
                categories_updated=[],
                error=str(e)
            )
    
    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        # Look for ```json ... ``` block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
