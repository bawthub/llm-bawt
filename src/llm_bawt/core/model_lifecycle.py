"""Model lifecycle management - ensures only one primary model is loaded at a time.

This module provides a singleton ModelLifecycleManager that:
1. Tracks the currently loaded model
2. Properly unloads models before loading new ones
3. Provides model switching capabilities for the LLM via tools
4. Clears GPU memory when switching between GGUF models
"""

import gc
import logging
import threading
from typing import TYPE_CHECKING, Any

from rich.console import Console

from ..utils.config import Config

if TYPE_CHECKING:
    from ..clients.base import LLMClient

logger = logging.getLogger(__name__)
console = Console()


def _owner_key(owner: tuple[str, str] | None) -> tuple[str, str] | None:
    if owner is None:
        return None
    bot_id, user_id = owner
    return (str(bot_id or "").lower().strip(), str(user_id or ""))


class ModelLifecycleManager:
    """Singleton manager for model lifecycle.
    
    Ensures only one primary model is loaded at a time and provides
    proper cleanup when switching models.
    """
    
    _instance: "ModelLifecycleManager | None" = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Config | None = None) -> "ModelLifecycleManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Config | None = None):
        if self._initialized:
            return
        
        self._initialized = True
        self.config = config
        self._current_model_alias: str | None = None
        self._current_client: "LLMClient | None" = None
        self._model_lock = threading.RLock()
        # Pending switches keyed by owner (bot_id, user_id); None = unscoped admin request.
        self._pending_model_switches: dict[tuple[str, str] | None, str] = {}
        
        # Callbacks for model switch events
        self._on_model_unloaded: list = []
        self._on_model_loaded: list = []
    
    @classmethod
    def get_instance(cls, config: Config | None = None) -> "ModelLifecycleManager":
        """Get the singleton instance."""
        return cls(config)
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.unload_current_model()
            cls._instance = None
    
    @property
    def current_model(self) -> str | None:
        """Get the currently loaded model alias."""
        return self._current_model_alias
    
    @property
    def current_client(self) -> "LLMClient | None":
        """Get the currently loaded client."""
        return self._current_client
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._current_client is not None
    
    def get_available_models(self) -> list[str]:
        """Get list of available model aliases (shortcuts).
        
        Returns only the shortcut names, not full paths or model IDs.
        """
        if not self.config:
            return []
        return self.config.get_model_options()
    
    def _normalize_alias(self, alias: str) -> str | None:
        """Normalize a model alias to match available models (case-insensitive).
        
        Returns the correctly-cased alias if found, None otherwise.
        """
        if not self.config:
            return None
        models = self.config.defined_models.get("models", {})
        # Direct match first
        if alias in models:
            return alias
        # Case-insensitive match
        alias_lower = alias.lower()
        for model_name in models:
            if model_name.lower() == alias_lower:
                return model_name
        return None
    
    def get_model_info(self, alias: str) -> dict[str, Any] | None:
        """Get model definition by alias (case-insensitive)."""
        if not self.config:
            return None
        # Try case-insensitive match
        normalized = self._normalize_alias(alias)
        if normalized:
            return self.config.resolve_model(normalized)
        return None
    
    def unload_current_model(self) -> bool:
        """Unload the currently loaded model and free memory.
        
        Returns True if a model was unloaded, False if no model was loaded.
        """
        with self._model_lock:
            if self._current_client is None:
                logger.debug("No model currently loaded to unload")
                return False
            
            previous_alias = self._current_model_alias
            logger.info(f"Unloading model: {previous_alias}")
            
            # Call client cleanup if available
            if hasattr(self._current_client, 'unload'):
                try:
                    self._current_client.unload()
                except Exception as e:
                    logger.warning(f"Error during client unload: {e}")
            
            # Clear references
            self._current_client = None
            self._current_model_alias = None
            
            # Force garbage collection and clear CUDA cache if available
            self._force_memory_cleanup()
            
            # Notify callbacks
            for callback in self._on_model_unloaded:
                try:
                    callback(previous_alias)
                except Exception as e:
                    logger.warning(f"Error in unload callback: {e}")
            
            logger.info(f"Model unloaded successfully: {previous_alias}")
            return True
    
    def _force_memory_cleanup(self):
        """Force garbage collection and clear GPU memory."""
        # Python garbage collection
        gc.collect()
        
        # Clear CUDA cache if available (for GGUF models using GPU)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("CUDA cache cleared")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not clear CUDA cache: {e}")
    
    def register_client(self, alias: str, client: "LLMClient") -> bool:
        """Register a newly loaded client as the current primary model.
        
        If another model is already loaded, it will be unloaded first.
        
        Args:
            alias: The model alias (shortcut name)
            client: The loaded LLM client
            
        Returns:
            True if registration successful
        """
        with self._model_lock:
            # Only unload if the previous model holds local resources (GGUF/vLLM).
            # API clients (openai, grok) and agent backends (claude-code, codex)
            # are stateless — no need to unload/reload when switching between them.
            if self._current_client is not None and self._current_model_alias != alias:
                prev_info = self.get_model_info(self._current_model_alias) or {}
                prev_type = prev_info.get("type", "")
                needs_unload = prev_type in ("gguf", "vllm", "llamacpp")
                if needs_unload:
                    logger.info(f"Switching model: {self._current_model_alias} -> {alias}")
                    self.unload_current_model()
                else:
                    logger.debug(f"Model switch: {self._current_model_alias} -> {alias} (no unload needed)")

            self._current_model_alias = alias
            self._current_client = client
            
            # Notify callbacks
            for callback in self._on_model_loaded:
                try:
                    callback(alias, client)
                except Exception as e:
                    logger.warning(f"Error in load callback: {e}")
            
            logger.info(f"Model registered as primary: {alias}")
            return True
    
    def switch_model(
        self, new_alias: str, owner: tuple[str, str] | None = None
    ) -> tuple[bool, str]:
        """Request a switch to a different model.
        
        This does NOT unload the current model immediately - that would crash
        any in-progress streaming. Instead, it marks the switch as pending,
        and the actual switch happens on the next request.
        
        Args:
            new_alias: The alias of the model to switch to (case-insensitive)
            
        Returns:
            Tuple of (success, message)
        """
        owner = _owner_key(owner)
        with self._model_lock:
            # Normalize the alias (case-insensitive match)
            normalized_alias = self._normalize_alias(new_alias)
            if not normalized_alias:
                available = self.get_available_models()
                return False, f"Model '{new_alias}' not found. Available: {', '.join(available)}"
            
            # Already using this model? The global alias is only the last
            # registered client, not a scoped owner's cached model, so the
            # shortcut applies to unscoped (admin) requests only; a scoped
            # request is always recorded and resolved on the owner's next turn.
            if owner is None and self._current_model_alias == normalized_alias:
                # Choosing the current model cancels this owner's pending switch.
                self._pending_model_switches.pop(owner, None)
                return True, f"Already using model '{normalized_alias}'"
            
            previous = self._current_model_alias
            
            # Mark the pending switch - do NOT unload yet!
            # The actual switch will happen when _get_llm_bawt is called next
            self._pending_model_switches[owner] = normalized_alias
            
            if previous:
                return True, f"Got it! I'll switch from '{previous}' to '{normalized_alias}' for your next message."
            else:
                return True, f"Got it! I'll use '{normalized_alias}' for your next message."
    
    def has_pending_switch(self, owner: tuple[str, str] | None = None) -> str | None:
        """Return the pending switch for this owner (or the unscoped one), if any."""
        owner = _owner_key(owner)
        with self._model_lock:
            pending = self._pending_model_switches.get(owner)
            if pending is None and owner is not None:
                pending = self._pending_model_switches.get(None)
            return pending
    
    def clear_pending_switch(self, owner: tuple[str, str] | None = None) -> str | None:
        """Clear and return the pending switch for this owner.

        Only the owner's own request (or an unscoped admin request) is consumed,
        so one bot/user never receives another's switch.
        """
        owner = _owner_key(owner)
        with self._model_lock:
            pending = self._pending_model_switches.pop(owner, None)
            if pending is None and owner is not None:
                pending = self._pending_model_switches.pop(None, None)
            return pending
    
    def on_model_unloaded(self, callback):
        """Register a callback for when a model is unloaded."""
        self._on_model_unloaded.append(callback)
    
    def on_model_loaded(self, callback):
        """Register a callback for when a model is loaded."""
        self._on_model_loaded.append(callback)
    
    def get_status(self) -> dict[str, Any]:
        """Get current model status."""
        return {
            "current_model": self._current_model_alias,
            "is_loaded": self.is_model_loaded,
            "available_models": self.get_available_models(),
        }


# Module-level convenience function
def get_model_lifecycle(config: Config | None = None) -> ModelLifecycleManager:
    """Get the singleton ModelLifecycleManager instance."""
    return ModelLifecycleManager.get_instance(config)
