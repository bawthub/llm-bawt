"""Default registry body for human-message-grounded fact extraction."""

SOURCED_FACT_EXTRACTION_PROMPT = """Extract only persistent, meaningful facts about the human user from the conversation below.
Assistant turns are context, never evidence. Commands such as /new are not facts. Do not infer a fact from an assistant reply or recall of previous memories. Most conversations have no new persistent facts; return an empty facts list when unsure.

Each human turn is labeled [USER ID: ...]. For every fact include source_message_ids containing ONLY the ID(s) of human turns that directly state and support that fact. A fact combining details from two human turns cites both. Never cite an assistant turn, a command, or a message that does not support the fact. If you cannot identify supporting human IDs, omit the fact. Do not invent IDs.

Remember enduring identity, family and pets, preferences, professional context, health conditions and significant plans the user explicitly describes. Do not store greetings, small talk, passing emotions, roleplay, in-the-moment actions or facts recalled by the assistant. Merge details about the same subject into one supported fact when appropriate. The assistant's mention of a detail never establishes it as the user's statement.

Return only JSON: {{"facts": [{{"content": "User ...", "tags": ["fact"], "importance": 0.8, "source_message_ids": ["id-from-human-turn"]}}]}}. Tags may include preference, fact, event, plan, professional, health, relationship, misc. importance is 0.0-1.0. Use 0.7+ only for important persistent information. profile_attribute is for core identity only: include it for a stated name, age, occupation, location or family/pets (category fact, key name/age/occupation/location/pets/family); it may also cover persistent health or core values/preferences. Never use it for projects, tools, technical setup or temporary details.

Example: [USER ID: a] I have two dogs. [ASSISTANT]: What kind? [USER ID: b] One is a lab mix. Output {{"facts": [{{"content": "User has two dogs, one a lab mix", "tags": ["fact", "relationship"], "importance": 0.8, "source_message_ids": ["a", "b"], "profile_attribute": {{"category": "fact", "key": "pets"}}}}]}}.

Conversation:
{conversation}
"""
