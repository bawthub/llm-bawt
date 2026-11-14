import time
from typing import Any
try:
    import tiktoken
    _tiktoken_present = True
except ImportError:
    tiktoken = None
    _tiktoken_present = False

class Message:
    """Standard message format for all LLM clients"""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: float | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        db_id: str | None = None,
        # OpenAI content-array parts (images etc.) attached to this message.
        # When set, to_api_format() emits a multimodal content array instead
        # of a plain string.
        content_parts: list[dict[str, Any]] | None = None,
        # TASK-284: the durable session/thread this message belongs to. Carried
        # through hydration so the read path can scope context to one thread.
        # Internal metadata only — deliberately NOT emitted by to_api_format().
        session_id: str | None = None,
        author_entity_type: str | None = None,
        author_entity_id: str | None = None,
    ):
        self.role = role
        self.content = content if content is not None else ""
        self.timestamp = timestamp if timestamp else time.time()
        self.tool_calls = tool_calls  # For assistant messages that make tool calls
        self.tool_call_id = tool_call_id  # For tool result messages
        self.db_id = db_id  # Primary key from the database (when loaded from PostgreSQL)
        self.content_parts = content_parts  # Extra multimodal parts (image_url, etc.)
        self.session_id = session_id  # TASK-284 durable thread id (internal only)
        self.author_entity_type = author_entity_type
        self.author_entity_id = author_entity_id

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        """Create a Message from a dictionary"""
        role = data.get("role", "user")
        content = cls._extract_content(data)
        timestamp = data.get("timestamp", time.time())
        tool_calls = data.get("tool_calls")
        tool_call_id = data.get("tool_call_id")
        db_id = data.get("db_id")
        session_id = data.get("session_id")
        content_parts = data.get("content_parts")
        if content_parts is None and isinstance(data.get("content"), list):
            content_parts = [
                item for item in data["content"]
                if isinstance(item, dict) and item.get("type") != "text"
            ] or None
        return cls(
            role, content, timestamp, tool_calls, tool_call_id, db_id=db_id,
            session_id=session_id,
            author_entity_type=data.get("author_entity_type"),
            author_entity_id=data.get("author_entity_id"),
            content_parts=content_parts,
        )

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Extract content from the data dictionary"""
        content = data.get("content", "")
        if isinstance(content, list):
            return "\n\n".join(item.get("text", "") for item in content if item.get("type") == "text")
        return str(content) # Ensure content is always a string

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        d = {"role": self.role, "content": self.content, "timestamp": self.timestamp}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.db_id:
            d["db_id"] = self.db_id
        if self.session_id:
            d["session_id"] = self.session_id
        if self.author_entity_type:
            d["author_entity_type"] = self.author_entity_type
        if self.author_entity_id:
            d["author_entity_id"] = self.author_entity_id
        return d

    def to_api_format(self) -> dict:
        """Convert to API-compatible format (without timestamp)"""
        # Map internal roles to API-compatible roles
        # 'summary' is stored internally but should be sent as 'system' to the LLM
        api_role = self.role
        if api_role == "summary":
            api_role = "system"

        # Build content: use multimodal content array when extra parts exist
        if self.content_parts:
            content_array: list[dict[str, Any]] = [{"type": "text", "text": self.content or ""}]
            content_array.extend(self.content_parts)
            d = {"role": api_role, "content": content_array}
        else:
            # Ensure content is never None (OpenAI API rejects null content)
            d = {"role": api_role, "content": self.content or ""}
        
        # Include tool_calls for assistant messages that made tool calls
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", "{}") if isinstance(tc.get("arguments"), str) else __import__("json").dumps(tc.get("arguments", {})),
                    }
                }
                for tc in self.tool_calls
            ]
        
        # Include tool_call_id for tool result messages
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
            
        return d
