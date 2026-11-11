import time
import logging
import os
import contextlib
import json
from typing import List, Dict, Any, Iterator, Optional, Union

from rich.rule import Rule
from rich.json import JSON

from llm_bawt.clients.base import LLMClient
from llm_bawt.utils.config import Config # Keep for type hinting
from llm_bawt.models.message import Message

try:
    from llama_cpp import Llama
    _llama_cpp_available = True
except ImportError:
    Llama = None
    _llama_cpp_available = False

logger = logging.getLogger(__name__)

class LlamaCppClient(LLMClient):
    """Client for running GGUF models using llama-cpp-python."""

    def __init__(self, model_path: str, config: Config, chat_format: str | None = None, model_definition: dict | None = None):
        if not _llama_cpp_available:
            raise ImportError(
                "`llama-cpp-python` not found. Install it following instructions: "
                "https://github.com/abetlen/llama-cpp-python#installation"
            )
        super().__init__(model_path, config, model_definition=model_definition)
        self.model_path = model_path
        self.chat_format = chat_format  # Allow explicit chat format override
        self.llm_model = None  # The loaded Llama instance
        self._context_sizing_result = None  # Set during _load_model
        self._load_model()

    def _load_model(self):
        """Loads the GGUF model.

        Native llama.cpp stderr is NOT suppressed — it flows to the container's
        stderr (Docker logs) so that a CUDA abort() or segfault leaves
        diagnosable output instead of a silent restart.
        """
        import sys
        from llm_bawt.utils.vram import auto_size_context_window, detect_vram

        # Per-model overrides from model definition, falling back to global config
        n_gpu_layers = self.model_definition.get("n_gpu_layers", self.config.LLAMA_CPP_N_GPU_LAYERS)
        n_batch = getattr(self.config, 'LLAMA_CPP_N_BATCH', 2048)
        flash_attn = getattr(self.config, 'LLAMA_CPP_FLASH_ATTN', True)

        # Pre-flight VRAM check — always logged so crashes have context
        vram = detect_vram()
        file_size = os.path.getsize(self.model_path) if os.path.exists(self.model_path) else 0
        if vram:
            logger.info(
                "VRAM pre-flight: %s — %.1f GiB free, model weights ~%.1f GiB",
                vram.gpu_name, vram.free_gb, file_size / (1024 ** 3),
            )
            if file_size > 0 and vram.free_bytes < file_size:
                logger.error(
                    "VRAM insufficient: %.1f GiB free < %.1f GiB model weights. "
                    "Aborting load to prevent CUDA crash.",
                    vram.free_gb, file_size / (1024 ** 3),
                )
                raise RuntimeError(
                    f"Insufficient VRAM ({vram.free_gb:.1f} GiB free) to load "
                    f"model weights ({file_size / (1024 ** 3):.1f} GiB)"
                )
        else:
            logger.warning("VRAM detection failed — loading blind (crash risk)")

        # Auto-size context window based on VRAM
        sizing = auto_size_context_window(
            model_definition=self.model_definition,
            global_n_ctx=self.config.LLAMA_CPP_N_CTX,
            global_max_tokens=self.effective_max_tokens,
            model_path=self.model_path,
        )
        self._context_sizing_result = sizing
        n_ctx = sizing.context_window

        # Always log load parameters so container logs have crash context
        logger.info(
            "Loading GGUF: %s — n_ctx=%d (source=%s), n_gpu_layers=%s, "
            "n_batch=%d, flash_attn=%s, chat_format=%s",
            self.model_path, n_ctx, sizing.source,
            "all" if n_gpu_layers == -1 else n_gpu_layers,
            n_batch, flash_attn, self.chat_format or "auto",
        )
        if sizing.vram_info:
            logger.info(
                "VRAM sizing: %s — weights %.1f GiB, KV budget %.1f GiB",
                sizing.vram_info, sizing.model_file_size_gb,
                sizing.estimated_kv_budget_gb,
            )

        model_load_params = {
            "model_path": self.model_path,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "flash_attn": flash_attn,
            "chat_format": self.chat_format,  # None = auto-detect
            "verbose": False,
        }

        try:
            # Flush before load so pre-flight info survives a CUDA abort()
            sys.stdout.flush()
            sys.stderr.flush()

            if Llama is None:
                raise ImportError("Llama class is not available from llama_cpp.")
            self.llm_model = Llama(**model_load_params)

            ctx_size = getattr(self.llm_model, 'n_ctx', 'N/A')
            gpu_layers = model_load_params['n_gpu_layers']
            logger.info(
                "Model loaded: Context=%s, Batch=%d, FlashAttn=%s, GPU Layers=%s",
                ctx_size, n_batch, flash_attn,
                "All" if gpu_layers == -1 else gpu_layers,
            )
        except (ValueError, RuntimeError) as e:
            # Context window too large for available VRAM — retry with smaller values
            if "Failed to create llama_context" in str(e):
                fallback_sizes = []
                global_n_ctx = self.config.LLAMA_CPP_N_CTX
                if n_ctx != global_n_ctx and global_n_ctx > 0:
                    fallback_sizes.append(global_n_ctx)
                if n_ctx > 8192:
                    fallback_sizes.append(8192)
                if n_ctx > 4096:
                    fallback_sizes.append(4096)
                fallback_sizes = sorted(set(s for s in fallback_sizes if s < n_ctx), reverse=True)

                for fallback_ctx in fallback_sizes:
                    logger.warning(
                        "Context window %d failed. Retrying with %d...",
                        n_ctx, fallback_ctx,
                    )
                    model_load_params["n_ctx"] = fallback_ctx
                    try:
                        sys.stdout.flush()
                        sys.stderr.flush()
                        self.llm_model = Llama(**model_load_params)
                        logger.info("Model loaded with fallback context: %d", fallback_ctx)
                        if self._context_sizing_result:
                            self._context_sizing_result.context_window = fallback_ctx
                            self._context_sizing_result.source = "fallback-retry"
                        return  # Success
                    except (ValueError, RuntimeError):
                        continue

                logger.error("All context window sizes failed for %s: %s", self.model_path, e)
                raise
            else:
                logger.error("GGUF model load failed for %s: %s", self.model_path, e)
                raise
        except Exception as e:
            logger.error("GGUF model load failed for %s: %s", self.model_path, e)
            raise

    def stream_raw(self, messages: List[Message], stop: list[str] | str | None = None, **kwargs: Any) -> Iterator[str]:
        """
        Stream raw text chunks from llama.cpp without console formatting.

        Used by the API service for SSE streaming.
        """
        if not self.llm_model:
            raise RuntimeError("Llama.cpp model not properly initialized.")

        api_messages = [msg.to_api_format() for msg in messages]
        generation_params = {
            "messages": api_messages,
            "max_tokens": self.effective_max_tokens,
            "temperature": self.config.TEMPERATURE,
            "top_p": self.config.TOP_P,
            "stream": True,
        }
        if stop:
            generation_params["stop"] = stop

        logger.debug(f"stream_raw: calling create_chat_completion with {len(stop) if stop else 0} stop sequences")
        raw_stream = self.llm_model.create_chat_completion(**generation_params)
        logger.debug("stream_raw: got raw_stream, iterating chunks")
        yield from self._iterate_llama_cpp_chunks(raw_stream)

    def query(self, messages: List[Message], plaintext_output: bool = False, stream: bool = True, stop: list[str] | str | None = None, **kwargs: Any) -> str:
        """Query the loaded GGUF model.

        Args:
            messages: List of Message objects.
            plaintext_output: If True, return raw text.
            stream: If True, stream the output token by token.
            **kwargs: Additional arguments (ignored).

        Returns:
            The model's response as a string.
        """
        if not self.llm_model:
            error_msg = "Error: Llama.cpp model not properly initialized."
            self.console.print(f"[bold red]{error_msg}[/bold red]")
            return error_msg

        api_messages = [msg.to_api_format() for msg in messages]

        generation_params = {
            "messages": api_messages,
            "max_tokens": self.effective_max_tokens,
            "temperature": self.config.TEMPERATURE,
            "top_p": self.config.TOP_P,
            "stream": stream and not self.config.NO_STREAM,
        }
        if stop:
            generation_params["stop"] = stop

        if self.config.VERBOSE:
             log_params = {k: v for k, v in generation_params.items() if k != 'messages'}
             logger.debug(f"Llama.cpp Request Parameters: {log_params}")

             self.console.print(Rule("Querying Llama.cpp Model", style="yellow"))
             self.console.print(f"[dim]Params:[/dim] [italic]max_tokens={generation_params['max_tokens']}, temp={generation_params['temperature']}, top_p={generation_params['top_p']}, stream={generation_params['stream']}[/italic]")
             self.console.print(Rule("Request Payload", style="dim blue"))
             try:
                 payload_str = json.dumps(generation_params, indent=2) # Pretty print for structure
                 self.console.print(JSON(payload_str))
             except TypeError as e:
                 logger.error(f"Could not serialize payload for Rich JSON printing: {e}")
                 self.console.print(f"[red]Error printing payload:[/red] {e}")
                 import pprint
                 self.console.print(pprint.pformat(generation_params))

             self.console.print(Rule(style="yellow"))

        response_text_final = ""
        # Let base class handle panel title using bot_name
        panel_title, panel_border_style = self.get_styling()
        try:
            if generation_params["stream"]:
                raw_stream = self.llm_model.create_chat_completion(**generation_params)
                response_text_final = self._handle_streaming_output(
                    stream_iterator=self._iterate_llama_cpp_chunks(raw_stream),
                    plaintext_output=plaintext_output,
                    panel_title=panel_title,
                    panel_border_style=panel_border_style,
                )
            else:
                start_time = time.time()
                completion = self.llm_model.create_chat_completion(**generation_params)
                end_time = time.time()

                if completion and 'choices' in completion and completion['choices']:
                    response_text_final = completion['choices'][0].get('message', {}).get('content', '').strip()

                    if self.config.VERBOSE:
                        usage = completion.get('usage', {})
                        prompt_tokens = usage.get('prompt_tokens', 'N/A')
                        completion_tokens = usage.get('completion_tokens', 'N/A')
                        elapsed_time = end_time - start_time
                        if isinstance(completion_tokens, int) and elapsed_time > 0:
                            tokens_per_sec = completion_tokens / elapsed_time
                            self.console.print(f"[dim]Generated {completion_tokens} tokens (prompt: {prompt_tokens}) in {elapsed_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)[/dim]")
                        else:
                            self.console.print(f"[dim]Generated {completion_tokens} tokens (prompt: {prompt_tokens})[/dim]")

                    if not plaintext_output:
                        parts = response_text_final.split("\n\n", 1)
                        first_part = parts[0]
                        second_part = parts[1] if len(parts) > 1 else None
                        self._print_assistant_message(first_part, second_part=second_part, panel_title=panel_title, panel_border_style=panel_border_style)
                else:
                    self.console.print("[bold red]Error: No response generated by llama.cpp model.[/bold red]")
                    response_text_final = "Error: Failed to get response from llama.cpp model."

        except Exception as e:
            self.console.print(f"[bold red]Error during llama.cpp generation:[/bold red] {e}")
            logger.exception("Error during llama.cpp generation")
            response_text_final = f"Error: An exception occurred during generation: {e}"
        finally:
            if self.config.VERBOSE:
                self.console.print(Rule(style="yellow"))

        return response_text_final.strip()

    def _iterate_llama_cpp_chunks(self, stream: Iterator[Dict[str, Any]]) -> Iterator[str]:
        """Extracts content delta from llama.cpp stream chunks."""
        try:
            for chunk in stream:
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                content = delta.get('content')
                if content:
                    yield content
        except Exception as e:
            self.console.print(f"\n[bold red]Error processing llama.cpp stream:[/bold red] {e}")
            logger.exception("Error processing llama.cpp stream")
            yield f"\nERROR: {e}"

    def get_styling(self) -> tuple[str | None, str]:
        """Return Llama.cpp specific styling."""
        # Return None for title to let base class use bot_name, only specify border style
        return None, "yellow"

    def supports_native_tools(self) -> bool:
        """Check if this client supports native tool calling.
        
        llama-cpp-python supports native function calling via chat_format.
        We return True here, but actual support depends on model compatibility.
        """
        return True

    def query_with_tools(
        self,
        messages: List[Message],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Union[str, Dict[str, Any], None] = "auto",
        stop: Optional[Union[List[str], str]] = None,
        **kwargs: Any,
    ) -> tuple[Any, Optional[List[Dict[str, Any]]]]:
        """Query with native function calling support.
        
        Uses llama-cpp-python's built-in chatml-function-calling format
        which constrains output with grammar to ensure valid JSON.
        
        Args:
            messages: List of Message objects
            tools_schema: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", or {"type": "function", "function": {"name": "..."}}
            stop: Stop sequences (ignored for native tool calling)
            
        Returns:
            Tuple of (response_content, tool_calls_or_none)
            - If tool_calls is not None, it contains OpenAI-format tool calls
            - Otherwise response_content is the text response
        """
        if not self.llm_model:
            raise RuntimeError("Llama.cpp model not properly initialized.")
        
        # If no tools, fall back to regular query
        if not tools_schema:
            response = self.query(messages, plaintext_output=True, stream=False, stop=stop, **kwargs)
            return response, None
        
        api_messages = [msg.to_api_format() for msg in messages]
        
        generation_params = {
            "messages": api_messages,
            "tools": tools_schema,
            "tool_choice": tool_choice or "auto",
            "max_tokens": self.effective_max_tokens,
            "temperature": self.config.TEMPERATURE,
            "top_p": self.config.TOP_P,
            "stream": False,  # Tool calls don't stream well
        }
        
        if self.config.VERBOSE:
            logger.debug(f"Native tool call with {len(tools_schema)} tools, tool_choice={tool_choice}")
        
        try:
            with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
                completion = self.llm_model.create_chat_completion(**generation_params)
            
            if not completion or 'choices' not in completion:
                return "Error: No response from model", None
            
            choice = completion['choices'][0]
            message = choice.get('message', {})
            
            # Check for tool calls
            tool_calls = message.get('tool_calls')
            if tool_calls:
                # Return the raw text (if any) and the tool calls
                content = message.get('content') or ""
                return content, tool_calls
            
            # No tool calls, return content
            content = message.get('content', '')
            return content, None
            
        except Exception as e:
            logger.warning(f"Native tool calling failed: {e}, falling back to text-based")
            # Fall back to regular query without tools
            response = self.query(messages, plaintext_output=True, stream=False, stop=stop, **kwargs)
            return response, None

    def stream_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        **kwargs: Any,
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """Stream responses with native tool calling support.
        
        Yields:
            - String chunks for regular content
            - Dict with tool_calls when a tool call is detected
        """
        if not self.llm_model:
            raise RuntimeError("Llama.cpp model not properly initialized.")
        
        api_messages = [msg.to_api_format() for msg in messages]
        
        generation_params = {
            "messages": api_messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": self.effective_max_tokens,
            "temperature": self.config.TEMPERATURE,
            "top_p": self.config.TOP_P,
            "stream": True,
        }
        
        try:
            with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
                stream = self.llm_model.create_chat_completion(**generation_params)
            
            accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
            
            for chunk in stream:
                choice = chunk.get('choices', [{}])[0]
                delta = choice.get('delta', {})
                
                # Check for content
                content = delta.get('content')
                if content:
                    yield content
                
                # Check for tool calls
                tool_calls = delta.get('tool_calls')
                if tool_calls:
                    for tc in tool_calls:
                        idx = tc.get('index', 0)
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc.get('id', f"call_{idx}"),
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": "",
                                }
                            }
                        
                        if 'function' in tc:
                            if 'name' in tc['function']:
                                accumulated_tool_calls[idx]['function']['name'] += tc['function']['name']
                            if 'arguments' in tc['function']:
                                accumulated_tool_calls[idx]['function']['arguments'] += tc['function']['arguments']
                
                # Check for finish reason
                if choice.get('finish_reason') == 'tool_calls':
                    # Yield the accumulated tool calls
                    yield {"tool_calls": list(accumulated_tool_calls.values())}
                    return
                    
        except Exception as e:
            logger.exception(f"Error in stream_with_tools: {e}")
            yield f"\nERROR: {e}"

    def unload(self) -> None:
        """Unload the GGUF model and free GPU/CPU memory."""
        if self.llm_model is None:
            return
        
        logger.info(f"Unloading GGUF model: {self.model_path}")
        
        try:
            # Delete the model instance
            del self.llm_model
            self.llm_model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Try to clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            
            if self.config.VERBOSE:
                self.console.print("[green]Model unloaded and memory freed[/green]")
                
        except Exception as e:
            logger.warning(f"Error during model unload: {e}") 
