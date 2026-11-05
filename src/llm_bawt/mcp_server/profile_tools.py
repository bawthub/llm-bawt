"""MCP tool for structured user and bot profiles."""

from __future__ import annotations

from .server import mcp

_PROFILE_MANAGER_SINGLETON = None


def _get_profile_manager():
    global _PROFILE_MANAGER_SINGLETON
    if _PROFILE_MANAGER_SINGLETON is None:
        from llm_bawt.profiles import ProfileManager
        from llm_bawt.utils.config import Config
        _PROFILE_MANAGER_SINGLETON = ProfileManager(Config())
    return _PROFILE_MANAGER_SINGLETON


@mcp.tool(name="profile")
async def profile(
    action: str,
    entity_type: str = "user",
    entity_id: str | None = None,
    category: str | None = None,
    key: str | None = None,
    value: str | None = None,
) -> str:
    """Read or mutate structured user/bot profile attributes.

    Actions: ``summary``, ``list``, ``get``, ``set``, ``delete``. This is not
    semantic memory; use memory tools for free-text facts. A user
    ``entity_id`` defaults to the configured default user.
    """
    from llm_bawt.bots import BotManager
    from llm_bawt.default_user import default_user_id
    from llm_bawt.profiles import AttributeCategory, EntityType
    from llm_bawt.utils.config import Config

    entity_id = (entity_id or "").strip() or (default_user_id() if entity_type == "user" else "")
    if not entity_id:
        return f"Error: entity_id is required for entity_type '{entity_type}'."

    if action in ("set", "delete", "list", "get"):
        manager = _get_profile_manager()
        if entity_type == "bot" and not BotManager(Config()).get_bot(entity_id):
            return f"Error: '{entity_id}' is not a valid bot. Check the bot slug."
        if entity_type == "user":
            existing = manager.get_profile(EntityType.USER, entity_id)
            if not existing and BotManager(Config()).get_bot(entity_id):
                return f"Error: '{entity_id}' is a bot, not a user. Use entity_type=\"bot\"."

    categories = [
        AttributeCategory.FACT,
        AttributeCategory.PREFERENCE,
        AttributeCategory.PERSONALITY,
        AttributeCategory.INTEREST,
        AttributeCategory.CONTEXT,
        AttributeCategory.COMMUNICATION,
    ]
    manager = _get_profile_manager()
    entity = EntityType.USER if entity_type == "user" else EntityType.BOT
    if action == "summary":
        result = (
            manager.get_user_profile_summary(entity_id)
            if entity_type == "user"
            else manager.get_bot_profile_summary(entity_id)
        )
        return result or f"No profile data found for {entity_type} '{entity_id}'."
    if action == "list":
        attrs = (
            manager.get_attributes_by_category(entity, entity_id, category)
            if category
            else manager.get_all_attributes(entity, entity_id)
        )
        if not attrs:
            return f"No attributes found for {entity_type} '{entity_id}'."
        return "\n".join(
            f"[{attr.category}] {attr.key}: {attr.value} (confidence={attr.confidence})"
            for attr in attrs
        )
    if action == "get":
        if not key:
            return "Error: 'key' is required for action='get'."
        attr = manager.get_attribute(entity, entity_id, category, key) if category else None
        if attr is None:
            for candidate in categories:
                attr = manager.get_attribute(entity, entity_id, candidate, key)
                if attr:
                    break
        return (
            f"[{attr.category}] {attr.key}: {attr.value} (confidence={attr.confidence})"
            if attr else f"Attribute '{key}' not found."
        )
    if action == "set":
        if not key or value is None:
            return "Error: 'key' and 'value' are required for action='set'."
        selected = category or AttributeCategory.FACT
        manager.set_attribute(entity, entity_id, selected, key, value)
        return f"Set [{selected}] {key} = {value}"
    if action == "delete":
        if not key:
            return "Error: 'key' is required for action='delete'."
        deleted = manager.delete_attribute(entity, entity_id, category, key) if category else False
        if not category:
            for candidate in categories:
                if manager.delete_attribute(entity, entity_id, candidate, key):
                    deleted = True
                    break
        return f"Deleted: {deleted}"
    return f"Unknown action: {action}. Valid: summary, list, get, set, delete."
