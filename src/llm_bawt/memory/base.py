"""Abstract base class for memory backends."""

from ..utils.bot_identity import require_bot_id
from abc import ABC, abstractmethod
from typing import Any


class MemoryBackend(ABC):
    """Abstract base class for memory backends.
    
    Memory backends provide persistent storage and semantic retrieval of
    conversation history. Implementations can use various storage engines
    (PostgreSQL, SQLite, etc.) and retrieval methods (fulltext search,
    vector similarity, etc.).
    
    All backends must implement the core methods: add, search, clear, list, and stats.
    
    Memory is isolated per bot - each bot has its own separate memory space.
    
    Attributes:
        config: The application Config object containing settings.
        bot_id: The bot identifier for memory isolation.
    """
    
    def __init__(self, config: Any, bot_id: str = ""):
        """Initialize the memory backend.
        
        Args:
            config: The application Config object. Backends can define their
                   own config fields with the LLM_BAWT_ prefix.
            bot_id: The bot identifier for memory isolation. Each bot maintains
                   its own separate memory space.
        """
        self.config = config
        self.bot_id = require_bot_id(bot_id, type(self).__name__)
    
    @abstractmethod
    def add(self, message_id: str, role: str, content: str, timestamp: float) -> None:
        """Add a message to memory storage.
        
        Args:
            message_id: A unique identifier for the message (UUID string).
            role: The role of the message sender ('user' or 'assistant').
            content: The text content of the message.
            timestamp: Unix timestamp when the message was created.
        """
        pass
    
    @abstractmethod
    def search(self, query: str, n_results: int = 5, min_relevance: float = 0.0) -> list[dict] | None:
        """Search for memories relevant to the query.
        
        Args:
            query: The text to search for relevant memories.
            n_results: Maximum number of results to return.
            min_relevance: Minimum relevance score (0.0-1.0) for results.
                          Results below this threshold are filtered out.
            
        Returns:
            A list of dictionaries containing retrieved memories, each with:
                - id: The message ID
                - document: The message content
                - metadata: Dict with 'role' and 'timestamp'
                - relevance: Relevance score (0.0-1.0, higher is more relevant)
            Returns None if search fails or no results found.
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all memories from storage.
        
        Returns:
            True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def list_recent(self, n: int = 10) -> list[dict]:
        """List the most recent memories.
        
        Args:
            n: Number of recent memories to return.
            
        Returns:
            A list of memory dictionaries, most recent first.
        """
        pass
    
    @abstractmethod
    def stats(self) -> dict:
        """Get statistics about the memory storage.
        
        Returns:
            A dictionary with storage statistics, e.g.:
                - total_count: Total number of stored memories
                - oldest_timestamp: Timestamp of oldest memory
                - newest_timestamp: Timestamp of newest memory
                - storage_size: Size of storage in bytes (if applicable)
        """
        pass
    
    def delete(self, message_id: str) -> bool:
        """Delete a specific memory by ID.
        
        Args:
            message_id: The ID of the memory to delete.
            
        Returns:
            True if successfully deleted, False otherwise.
            
        Note:
            This method has a default implementation that returns False.
            Backends should override if they support deletion.
        """
        return False
    
    def prune_older_than(self, days: int) -> int:
        """Delete memories older than a specified number of days.
        
        Args:
            days: Delete memories older than this many days.
            
        Returns:
            Number of memories deleted.
            
        Note:
            This method has a default implementation that returns 0.
            Backends should override if they support pruning.
        """
        return 0
