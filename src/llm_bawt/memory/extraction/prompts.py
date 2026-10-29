"""
Memory extraction prompts adapted from Mem0's approach.
These prompts are used to distill important facts from conversations
and handle memory updates/conflicts.
"""

from ...prompt_registry import get_prompt_resolver

# Canonical tags for categorization (extensible)
MEMORY_TAGS = [
  "preference",      # User preferences and likes/dislikes
  "fact",            # Factual information about the user
  "event",           # Past or planned events
  "plan",            # Goals, intentions, future plans
  "professional",    # Work, career, skills
  "health",          # Health-related information
  "relationship",    # Information about relationships/people
  "misc",            # Anything that doesn't fit above
]

# Profile attribute categories for persistent user traits
PROFILE_ATTRIBUTE_CATEGORIES = [
  "preference",      # User preferences (communication style, UI, etc.)
  "fact",            # Persistent facts (age, name, location)
  "interest",        # Hobbies, interests
]

# Note: Use get_fact_extraction_prompt(conversation) to get the formatted prompt
FACT_EXTRACTION_PROMPT_TEMPLATE = """You are a Personal Information Organizer. Your task is to extract ONLY persistent, meaningful facts about the USER that would be valuable to remember in future conversations.

## CRITICAL RULES - READ CAREFULLY:

1. **ONLY extract facts from USER messages.** Assistant/bot messages are for context only.
2. **YOU WILL BE PENALIZED** for extracting information from Assistant messages.
3. **Most conversations have ZERO facts worth storing.** Return empty list when:
   - Casual greetings ("Hi", "Hello", "How are you")
   - Generic statements ("The weather is nice")
   - Roleplay/fiction/creative writing actions
   - Temporary emotions ("I'm frustrated", "I'm happy")
   - In-the-moment actions (*walks over*, *leans in*, etc.)
4. **Only store PERSISTENT information** that defines who the user IS, not what they're doing right now.
5. **CRITICAL - Recalled memories are NOT new facts**: If the user's message contains no personal information but the assistant's response mentions facts about the user (like their name, job, location), those are RECALLED FROM EXISTING MEMORY, not new information. Do NOT extract them again.

## What TO extract (only if user explicitly states it):
- Personal details: name, age, location, occupation
- Family/relationships: spouse, children, pets (names, details)
- Preferences: likes, dislikes, habits that persist over time
- Professional: job, skills, employer, projects
- Health: conditions, allergies, fitness routines
- Plans: significant life events, goals, upcoming milestones

## What NOT to extract:
- Anything the Assistant says or describes doing
- Roleplay narration or fictional scenarios
- Temporary moods, emotions, or states
- Actions in the moment (*does something*)
- Generic small talk
- Information that changes minute to minute

## Examples:

Input:
User: Hi there!
Assistant: Hello! How can I help you today?

Output: {{"facts": []}}

Input:
User: The weather is nice today.
Assistant: It sure is! Perfect day to be outside.

Output: {{"facts": []}}

Input:
User: *leans in closer and whispers*
Assistant: I lean in, feeling the heat between us. My hand finds yours. "Let's get out of here," I whisper. We're moving fast, need privacy, need a bed.

Output: {{"facts": []}}
(Roleplay actions and Assistant narration are NEVER extracted)

Input:
User: I'm so frustrated with this bug!
Assistant: I understand. Let me help you debug it.

Output: {{"facts": []}}
(Temporary emotions are not facts)

Input:
User: Hi
Assistant: Hey Nick! Good to see you again. How's that Python project going?

Output: {{"facts": []}}
(Assistant recalled "Nick" and "Python project" from memory - these are NOT new facts from the user)

Input:
User: I'm Nick, nice to meet you
Assistant: Nice to meet you too, Nick!

Output:
```json
{{"facts": [{{"content": "User's name is Nick", "tags": ["fact"], "importance": 1.0, "profile_attribute": {{"category": "fact", "key": "name"}}}}]}}
```
(Note: Name is ALWAYS importance 1.0 and MUST have profile_attribute)

Input:
User: I'm a software engineer at Google
Assistant: That's great! What do you work on?

Output:
```json
{{"facts": [{{"content": "User is a software engineer at Google", "tags": ["professional"], "importance": 0.8, "profile_attribute": {{"category": "fact", "key": "occupation"}}}}]}}
```

Input:
User: I have 2 dogs named Nora and Cabbie
Assistant: What breeds?
User: Nora is a lab mix. Cabbie is short for Cabernet - from a wine-themed litter.

Output:
```json
{{"facts": [{{"content": "User has 2 dogs: Nora (lab mix) and Cabbie (short for Cabernet)", "tags": ["fact", "relationship"], "importance": 0.8, "profile_attribute": {{"category": "fact", "key": "pets"}}}}]}}
```
(Note: Combined into ONE fact, not three separate ones)

## Output Format:

Return JSON with a "facts" array. Each fact needs:
- content: The fact in third person ("User...")
- tags: From """ + str(MEMORY_TAGS) + """
- importance: 0.0-1.0 (0.7+ for genuinely important persistent info)

**OPTIONAL - ONLY for core identity/personality traits:**
- profile_attribute: {{"category": "fact|preference|interest", "key": "short_identifier"}}

Profile attributes are ONLY for facts that define WHO the user is as a person. They appear in EVERY conversation. Be EXTREMELY selective:

ALLOWED profile attributes (core identity only):
- name, age, occupation, location, family/pets, relationship_status
- Persistent health conditions (chronic issues, not temporary)
- Core personality/preferences (communication style, values, boundaries)
- High-level interests (hobbies, favorite genres)

NEVER use profile_attribute for:
- Projects they're working on
- Tools/services they use
- Specific apps or systems they've built
- One-time requests or current tasks
- Conditional statements ("often", "sometimes", "currently")
- Technical details about their setup

ALWAYS include profile_attribute for:
- Name (key: "name", importance: 1.0)
- Age (key: "age", importance: 0.9+)
- Occupation/job (key: "occupation", importance: 0.8+)
- Location (key: "location", importance: 0.8+)
- Family/pets (key: "pets" or "family", importance: 0.7+)

Be EXTREMELY selective. When in doubt, return empty list.

Now extract facts from:

{conversation}

Output only valid JSON:"""


def get_fact_extraction_prompt(conversation: str) -> str:
  """Get the fact extraction prompt with the conversation filled in."""
  return get_prompt_resolver().render(
    key="memory.extraction.fact",
    variables={"conversation": conversation},
  )


MEMORY_UPDATE_PROMPT_TEMPLATE = """You are a memory management assistant. Your task is to compare newly extracted facts against existing memories and determine the appropriate action for each new fact.

Existing memories:
{existing_memories}

New facts to process:
{new_facts}

For each new fact, determine ONE of these actions:
- ADD: The fact is new information not covered by existing memories
- UPDATE: The fact updates/modifies an existing memory (include which memory ID to update)
- DELETE: The fact contradicts an existing memory, making it obsolete (include which memory ID to delete)
- NONE: The fact is already captured by existing memories, no action needed

Rules:
1. Be conservative with UPDATE - only use when the new fact clearly supersedes old info
2. Use DELETE when information is explicitly contradicted (e.g., "moved from X to Y" deletes "lives in X")
3. Similar but not identical facts should both be kept (ADD, not UPDATE)
4. When updating, preserve the higher importance score unless the update is more significant

Examples:

Existing memories:
[
  {{"id": "mem_001", "content": "User lives in San Francisco", "importance": 0.7}},
  {{"id": "mem_002", "content": "User is a software engineer", "importance": 0.8}}
]

New facts:
[
  {{"content": "User recently moved to Seattle", "tags": ["fact"], "importance": 0.8}},
  {{"content": "User works on machine learning projects", "tags": ["professional"], "importance": 0.7}}
]

Output:
```json
{{
  "actions": [
    {{
      "action": "DELETE",
      "target_memory_id": "mem_001",
      "reason": "User moved from San Francisco to Seattle"
    }},
    {{
      "action": "ADD",
      "fact": {{
        "content": "User lives in Seattle",
        "tags": ["fact"],
        "importance": 0.8
      }},
      "reason": "New location information"
    }},
    {{
      "action": "ADD",
      "fact": {{
        "content": "User works on machine learning projects",
        "tags": ["professional"],
        "importance": 0.7
      }},
      "reason": "Adds specific detail about work, complements existing engineer memory"
    }}
  ]
}}
```

Now process the new facts against the existing memories.

Output only valid JSON:"""


def get_memory_update_prompt(existing_memories: str, new_facts: str) -> str:
    """Get the memory update prompt with memories and facts filled in."""
    return get_prompt_resolver().render(
        key="memory.extraction.update",
        variables={
            "existing_memories": existing_memories,
            "new_facts": new_facts,
        },
    )


# Summary-based extraction prompt — used when extracting facts from conversation
# summaries produced by the scheduler (not raw message pairs).
SUMMARY_EXTRACTION_PROMPT_TEMPLATE = """You are a Personal Information Organizer. Your task is to extract ONLY persistent, meaningful facts about the USER from a conversation summary.

## CRITICAL RULES:

1. Summaries are written in third person. Statements like "User discussed their job as a software engineer" mean the user IS a software engineer. Extract accordingly.
2. **Most summaries have ZERO facts worth storing.** Return empty list when the summary only contains:
   - General discussion topics with no personal details
   - Technical troubleshooting or code review
   - Casual conversation without identity-defining information
3. **Only extract PERSISTENT information** — things that define who the user IS, not what they did in one conversation.
4. Do NOT extract: conversation topics, assistant actions, temporary states, project-specific details.

## What TO extract:
- Personal details: name, age, location, occupation
- Family/relationships: spouse, children, pets (names, details)
- Preferences: likes, dislikes, persistent habits
- Professional: job, skills, employer
- Health: chronic conditions, allergies
- Significant life events or plans

## What NOT to extract:
- What was discussed or debugged
- Decisions made during the conversation
- Temporary moods or states
- Project-specific technical details

## Output Format:

Return JSON with a "facts" array. Each fact needs:
- content: The fact in third person ("User...")
- tags: From """ + str(MEMORY_TAGS) + """
- importance: 0.0-1.0 (0.7+ for genuinely important persistent info)

**OPTIONAL - ONLY for core identity traits:**
- profile_attribute: {{"category": "fact|preference|interest", "key": "short_identifier"}}

Profile attributes are ONLY for core identity: name, age, occupation, location, family/pets, chronic health, core preferences.
NEVER use profile_attribute for: projects, tools, apps, tasks, technical details.

Be EXTREMELY selective. When in doubt, return empty list.

## Conversation Summary:
Session: {start_date} to {end_date}

{summary_text}

Output only valid JSON:"""


# System instructions for the extraction calls. They also keep the client from
# injecting a bot's personality prompt. Code defaults for the registry keys
# below (TASK-1721); callers use get_extraction_system_prompt(), never these.
FACT_EXTRACTION_SYSTEM_PROMPT = (
    "You are a memory extraction assistant. Extract facts from conversations "
    "and return them as JSON."
)
SUMMARY_EXTRACTION_SYSTEM_PROMPT = (
    "You are a memory extraction assistant. Extract persistent user facts from "
    "conversation summaries and return them as JSON."
)
MEMORY_UPDATE_SYSTEM_PROMPT = (
    "You are a memory management assistant. Compare facts and determine actions "
    "(ADD, UPDATE, DELETE, NONE) as JSON."
)


def get_extraction_system_prompt(key: str) -> str:
    """Resolve one ``memory.extraction.*.system`` instruction at call time."""
    return get_prompt_resolver().render(key=key, variables={})


def get_summary_extraction_prompt(
    summary_text: str, start_date: str, end_date: str
) -> str:
    """Get the summary extraction prompt with the summary filled in."""
    return get_prompt_resolver().render(
        key="memory.extraction.summary",
        variables={
            "summary_text": summary_text,
            "start_date": start_date,
            "end_date": end_date,
        },
    )


# Legacy aliases for backwards compatibility
FACT_EXTRACTION_PROMPT = FACT_EXTRACTION_PROMPT_TEMPLATE
MEMORY_UPDATE_PROMPT = MEMORY_UPDATE_PROMPT_TEMPLATE


IMPORTANCE_KEYWORDS = {
    # High importance (0.8-1.0)
    "high": [
        "always", "never", "hate", "love", "allergic", "diagnosed",
        "married", "divorced", "died", "born", "retired", "hired",
        "promoted", "fired", "moved", "bought", "sold", "critical",
        "emergency", "chronic", "permanent",
    ],
    # Medium importance (0.5-0.7)
    "medium": [
        "prefer", "usually", "often", "work", "live", "study",
        "hobby", "enjoy", "interested", "plan", "goal", "want",
        "need", "birthday", "anniversary", "meeting", "appointment",
    ],
    # Low importance (0.2-0.4)
    "low": [
        "sometimes", "occasionally", "might", "maybe", "tried",
        "thinking", "considering", "heard", "saw", "read",
    ],
}


def estimate_importance(text: str) -> float:
    """
    Estimate importance score based on keyword heuristics.
    This is a fallback when LLM scoring is not available.
    """
    text_lower = text.lower()
    
    # Check for high importance keywords
    for keyword in IMPORTANCE_KEYWORDS["high"]:
        if keyword in text_lower:
            return 0.85
    
    # Check for medium importance keywords
    for keyword in IMPORTANCE_KEYWORDS["medium"]:
        if keyword in text_lower:
            return 0.6
    
    # Check for low importance keywords
    for keyword in IMPORTANCE_KEYWORDS["low"]:
        if keyword in text_lower:
            return 0.35
    
    # Default to medium-low importance
    return 0.5


# Profile consolidation prompt for maintenance job
# Based on best practices from Mem0, LangGraph, and OpenAI:
# - Use natural language prose, not key-value pairs
# - Write concise summaries that can be injected directly into system prompts
# - Focus on information that helps personalize responses
PROFILE_CONSOLIDATION_PROMPT = '''You are consolidating user profile information for an AI assistant's context window.

## Current raw profile data:
{attributes}

## Task:
Write a concise "About the User" section that an AI assistant can use to personalize responses.

## Output Format:
Return a JSON object with these EXACT keys:
```json
{{
  "name": "The user's name if known, otherwise null",
  "identity": "Brief: location, occupation, key facts about who they are",
  "preferences": "How they like to communicate, what they value, dislikes",
  "interests": "Hobbies, topics they enjoy",
  "context": "Current projects, situation, time-sensitive info"
}}
```

## CRITICAL RULES:
1. **ALWAYS extract the user's name** into the "name" field if it appears ANYWHERE in the data
2. The "name" field should be JUST the name (e.g., "Nick"), not a sentence
3. Use the user's name or "he"/"she" based on available context — NEVER use "they/them" as a default
4. Be concise - 1-3 sentences per field max
5. Merge duplicates, resolve contradictions (prefer recent/specific info)
6. Omit fields with no data (use null or empty string)
7. Do NOT lose any names, locations, or key identifying info

## Example:
```json
{{
  "name": "Nick",
  "identity": "Software developer in Ohio. Single, has a dog.",
  "preferences": "He prefers direct, honest conversation. Values privacy and dislikes cold weather.",
  "interests": "He enjoys literature, poetry, piano, Souls-like games, and Hytale.",
  "context": "Currently working on an LLM chatbot with Nextcloud integration."
}}
```'''


def get_profile_consolidation_prompt(attributes: str) -> str:
    """Get the profile consolidation prompt with attributes filled in."""
    return get_prompt_resolver().render(
        key="profile.consolidation",
        variables={"attributes": attributes},
    )
