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
        self._pending_model_switch: str | None = None
        
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
    
    def switch_model(self, new_alias: str) -> tuple[bool, str]:
        """Request a switch to a different model.
        
        This does NOT unload the current model immediately - that would crash
        any in-progress streaming. Instead, it marks the switch as pending,
        and the actual switch happens on the next request.
        
        Args:
            new_alias: The alias of the model to switch to (case-insensitive)
            
        Returns:
            Tuple of (success, message)
        """
        with self._model_lock:
            # Normalize the alias (case-insensitive match)
            normalized_alias = self._normalize_alias(new_alias)
            if not normalized_alias:
                available = self.get_available_models()
                return False, f"Model '{new_alias}' not found. Available: {', '.join(available)}"
            
            # Already using this model?
            if self._current_model_alias == normalized_alias:
                return True, f"Already using model '{normalized_alias}'"
            
            previous = self._current_model_alias
            
            # Mark the pending switch - do NOT unload yet!
            # The actual switch will happen when _get_llm_bawt is called next
            self._pending_model_switch = normalized_alias
            
            if previous:
                return True, f"Got it! I'll switch from '{previous}' to '{normalized_alias}' for your next message."
            else:
                return True, f"Got it! I'll use '{normalized_alias}' for your next message."
    
    def has_pending_switch(self) -> str | None:
        """Check if there's a pending model switch.
        
        Returns the model alias to switch to, or None if no switch pending.
        """
        return getattr(self, '_pending_model_switch', None)
    
    def clear_pending_switch(self) -> str | None:
        """Clear and return the pending switch.
        
        Returns the model alias that was pending, or None.
        """
        pending = getattr(self, '_pending_model_switch', None)
        self._pending_model_switch = None
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
