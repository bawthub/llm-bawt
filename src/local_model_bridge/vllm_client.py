"""vLLM client for high-throughput inference with HuggingFace models."""

from __future__ import annotations

import json
import logging
import os
import string
import time
from collections.abc import Iterator
from random import choices
from typing import Any

from rich.json import JSON
from rich.rule import Rule

from llm_bawt.clients.base import LLMClient
from llm_bawt.models.message import Message
from llm_bawt.utils.config import Config

# Disable vLLM's multiprocess engine before import — it spawns engine cores in
# subprocesses via ZMQ/shared memory, which fails in Docker (especially WSL2).
# With this off, vLLM uses InprocClient and runs the engine in the same process.
# Must be set before `import vllm` since vllm.envs caches env vars at import time.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

try:
    from vllm import LLM, SamplingParams
    _vllm_available = True
except ImportError:
    LLM = None
    SamplingParams = None
    _vllm_available = False

logger = logging.getLogger(__name__)

# Mistral tool call IDs are 9-character alphanumeric strings
_ALPHANUMERIC = string.ascii_letters + string.digits


def _generate_mistral_tool_id() -> str:
    """Generate a 9-character alphanumeric tool call ID (Mistral format)."""
    return "".join(choices(_ALPHANUMERIC, k=9))


class VLLMClient(LLMClient):
    """Client for running HuggingFace models using vLLM for high-throughput inference.
    
    vLLM is optimized for serving LLMs with features like:
    - PagedAttention for efficient KV cache management
    - Continuous batching for high throughput
    - Quantization support (AWQ, GPTQ, FP8)
    - Fast model loading with tensor parallelism
    
    Note: Model loading takes 30-150s due to CUDA graph compilation.
    """

    SUPPORTS_STREAMING = True

    def __init__(
        self,
        model_id: str,
        config: Config,
        model_definition: dict | None = None
    ):
        """Initialize vLLM client with lazy model loading.
        
        Args:
            model_id: HuggingFace model ID or local path to model
            config: Global configuration object
            model_definition: Model-specific configuration from models.yaml
        """
        if not _vllm_available:
            raise ImportError(
                "vLLM not found. Install with: pip install vllm\n"
                "For CUDA 12.1+: pip install vllm\n"
                "See https://docs.vllm.ai/en/latest/getting_started/installation.html"
            )
        
        super().__init__(model_id, config, model_definition=model_definition)
        self.model_id = model_id
        self.llm_engine: LLM | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the vLLM model with optimized settings.
        
        This takes 30-150 seconds due to CUDA graph compilation.
        Suppresses vLLM's verbose initialization output unless VERBOSE is enabled.
        """
        # Guard against double-loading
        if self.llm_engine is not None:
            logger.warning(f"Model {self.model_id} already loaded, skipping reload")
            return
        
        if self.config.VERBOSE:
            self.console.print(
                f"[bold bright_cyan]Loading vLLM model:[/bold bright_cyan] {self.model_id}"
            )
            self.console.print(
                "[dim]⚠ Model loading takes 30-150s (CUDA graph compilation)[/dim]"
            )

        # Extract vLLM-specific config from model definition
        quantization = self.model_definition.get("quantization")  # awq|gptq|fp8|null
        dtype = self.model_definition.get("dtype", "auto")  # float16|bfloat16|auto
        max_model_len = self.model_definition.get("max_model_len")
        gpu_memory_utilization = self.model_definition.get("gpu_memory_utilization", 0.85)
        enforce_eager = self.model_definition.get("enforce_eager", True)
        enable_prefix_caching = self.model_definition.get("enable_prefix_caching", True)
        # Cap concurrent sequences. vLLM defaults to 256, which on large-vocab
        # models (e.g. Gemma 4 has a 256K vocab) blows up the sampler-warmup
        # logits buffer and OOMs on a 16GB card. We serve one user at a time, so
        # a small value is plenty. None => let vLLM pick its default.
        max_num_seqs = self.model_definition.get("max_num_seqs")

        # Build vLLM engine parameters
        engine_params = {
            "model": self.model_id,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": enable_prefix_caching,
            "trust_remote_code": True,  # Always True for community models
        }
        if max_num_seqs:
            engine_params["max_num_seqs"] = int(max_num_seqs)
        
        # Add optional parameters if specified
        # Note: Don't pass quantization parameter - let vLLM auto-detect from model config.
        # Only override if explicitly needed (e.g., forcing a different quantization method).
        # Passing quantization when it's already in the model's config.json causes validation errors.
        if quantization and self.config.VERBOSE:
            logger.info(f"Model specifies quantization: {quantization} (vLLM will auto-detect from config)")
        # Cap context length to what's configured (or 32768 default) so vLLM
        # doesn't try to allocate KV cache for the model's full context (e.g. 256K).
        engine_params["max_model_len"] = max_model_len or self.effective_context_window

        if self.config.VERBOSE:
            log_params = {k: v for k, v in engine_params.items() if k != "model"}
            logger.debug(f"vLLM engine parameters: {log_params}")
            if quantization:
                self.console.print(f"[dim]Model quantization: {quantization} (auto-detected)[/dim]")
            self.console.print(f"[dim]Dtype: {dtype}[/dim]")
            self.console.print(f"[dim]GPU Memory: {gpu_memory_utilization*100:.0f}%[/dim]")
            if max_model_len:
                self.console.print(f"[dim]Max Model Length: {max_model_len}[/dim]")

        start_time = time.time()
        
        try:
            # Always show vLLM initialization output - it includes useful progress info
            # and prevents confusion during the 30-150s CUDA graph compilation.
            # vLLM's output is informative and shows the model is actually loading.
            if LLM is None:
                raise ImportError("LLM class not available from vllm")
            
            if not self.config.VERBOSE:
                # Non-verbose: at least show we're starting
                self.console.print(
                    "[yellow]⏳ Compiling CUDA graphs (30-150s)...[/yellow]"
                )
            
            self.llm_engine = LLM(**engine_params)

            # Pre-warm with a 1-token inference to compile CUDA kernels.
            # First inference is extremely slow (~60-90s) due to kernel JIT;
            # paying this cost at load time keeps real requests fast.
            self.console.print(
                "[yellow]⏳ Warming up inference engine...[/yellow]"
            )
            warmup_params = SamplingParams(max_tokens=1, temperature=0)
            self.llm_engine.chat(
                messages=[{"role": "user", "content": "hi"}],
                sampling_params=warmup_params,
                use_tqdm=False,
            )

            load_time = time.time() - start_time

            if self.config.VERBOSE:
                self.console.print(
                    f"[green]✓ vLLM model loaded in {load_time:.1f}s[/green]"
                )
                # Show actual max model length if available
                if hasattr(self.llm_engine.llm_engine, 'model_config'):
                    actual_max_len = getattr(
                        self.llm_engine.llm_engine.model_config,
                        'max_model_len',
                        None
                    )
                    if actual_max_len:
                        self.console.print(f"[dim]Context window: {actual_max_len} tokens[/dim]")
            else:
                self.console.print(
                    f"[green]✓ Model ready[/green] [dim]({load_time:.1f}s)[/dim]"
                )

        except Exception as e:
            self.console.print(
                f"[bold red]Error loading vLLM model {self.model_id}:[/bold red] {e}"
            )
            self.console.print(
                "[yellow]Troubleshooting tips:[/yellow]\n"
                "  1. Ensure CUDA is available and compatible\n"
                "  2. Check GPU has sufficient VRAM for model\n"
                "  3. Try reducing gpu_memory_utilization\n"
                "  4. For quantized models, verify quantization format matches model"
            )
            logger.exception("vLLM model loading failed")
            raise

    def _prepare_messages_for_vllm(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects to vLLM chat format."""
        return [msg.to_api_format() for msg in messages]

    def _create_sampling_params(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
    ) -> "SamplingParams":
        """Create vLLM SamplingParams from config and overrides.
        
        Args:
            max_tokens: Override for max output tokens
            temperature: Override for temperature
            top_p: Override for top_p
            stop: Stop sequences (string or list)
            
        Returns:
            SamplingParams object for vLLM generation
        """
        if SamplingParams is None:
            raise ImportError("SamplingParams not available from vllm")
        
        params = {
            "max_tokens": max_tokens or self.effective_max_tokens,
            "temperature": temperature if temperature is not None else self.config.TEMPERATURE,
            "top_p": top_p if top_p is not None else self.config.TOP_P,
        }
        
        # Handle stop sequences
        if stop:
            if isinstance(stop, str):
                params["stop"] = [stop]
            else:
                params["stop"] = stop
        
        return SamplingParams(**params)

    def _generate_chat(
        self,
        vllm_messages: list[dict[str, str]],
        sampling_params: "SamplingParams",
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a complete response via LLM.chat().

        Uses the standard vLLM offline API which handles the full engine
        lifecycle correctly (request submission, stepping, cleanup).

        Args:
            vllm_messages: Chat messages in vLLM dict format.
            sampling_params: Sampling parameters.
            tools: Optional tool schemas for chat template.

        Returns:
            The generated response text.
        """
        assert self.llm_engine is not None

        chat_kwargs: dict[str, Any] = {"use_tqdm": False}
        if tools:
            chat_kwargs["tools"] = tools

        outputs = self.llm_engine.chat(
            messages=vllm_messages,
            sampling_params=sampling_params,
            **chat_kwargs,
        )

        if not outputs:
            raise RuntimeError("No response from vLLM model.")

        return outputs[0].outputs[0].text

    def _stream_chunks(self, text: str) -> Iterator[str]:
        """Break a complete response into word-sized chunks for streaming.

        Yields:
            Small text chunks (word + trailing whitespace).
        """
        import re
        # \s*\S+\s* keeps leading/indent whitespace; the trailing \s+ covers
        # whitespace-only text so concatenated chunks equal the input.
        for chunk in re.findall(r'\s*\S+\s*|\s+', text):
            yield chunk

    def query(
        self,
        messages: list[Message],
        plaintext_output: bool = False,
        stream: bool = True,
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ) -> str:
        """Query the vLLM model using chat template."""
        if not self.llm_engine:
            raise RuntimeError("vLLM model not properly initialized.")

        vllm_messages = self._prepare_messages_for_vllm(messages)
        sampling_params = self._create_sampling_params(stop=stop)
        should_stream = stream and not self.config.NO_STREAM

        if self.config.VERBOSE:
            self.console.print(Rule("Querying vLLM Model", style="bright_cyan"))
            self.console.print(
                f"[dim]Params:[/dim] [italic]max_tokens={sampling_params.max_tokens}, "
                f"temp={sampling_params.temperature}, top_p={sampling_params.top_p}, "
                f"stream={should_stream}[/italic]"
            )
            if sampling_params.stop:
                self.console.print(f"[dim]Stop sequences:[/dim] {sampling_params.stop}")

            self.console.print(Rule("Request Messages", style="dim bright_cyan"))
            try:
                self.console.print(JSON(json.dumps(vllm_messages, indent=2)))
            except TypeError:
                logger.exception("Could not serialize messages for display")

            self.console.print(Rule(style="bright_cyan"))

        panel_title, panel_border_style = self.get_styling()

        if should_stream:
            full_text = self._generate_chat(vllm_messages, sampling_params)
            response_text = self._handle_streaming_output(
                stream_iterator=self._stream_chunks(full_text),
                plaintext_output=plaintext_output,
                panel_title=panel_title,
                panel_border_style=panel_border_style,
            )
        else:
            start_time = time.time()

            outputs = self.llm_engine.chat(
                messages=vllm_messages,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            elapsed = time.time() - start_time

            if not outputs:
                raise RuntimeError("No response generated by vLLM model.")

            output = outputs[0]
            response_text = output.outputs[0].text.strip()

            if self.config.VERBOSE:
                num_tokens = len(output.outputs[0].token_ids)
                prompt_tokens = len(output.prompt_token_ids)

                if elapsed > 0:
                    tokens_per_sec = num_tokens / elapsed
                    self.console.print(
                        f"[dim]Generated {num_tokens} tokens "
                        f"(prompt: {prompt_tokens}) in {elapsed:.2f}s "
                        f"({tokens_per_sec:.2f} tokens/sec)[/dim]"
                    )
                else:
                    self.console.print(
                        f"[dim]Generated {num_tokens} tokens (prompt: {prompt_tokens})[/dim]"
                    )
                self.console.print(Rule(style="bright_cyan"))

            if not plaintext_output:
                parts = response_text.split("\n\n", 1)
                first_part = parts[0]
                second_part = parts[1] if len(parts) > 1 else None
                self._print_assistant_message(
                    first_part,
                    second_part=second_part,
                    panel_title=panel_title,
                    panel_border_style=panel_border_style,
                )

        if self.config.VERBOSE:
            self.console.print(Rule(style="bright_cyan"))

        return response_text

    def stream_raw(
        self,
        messages: list[Message],
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text chunks from vLLM.

        Generates full response via LLM.chat() then yields word-sized
        chunks. vLLM V1's offline engine doesn't support true per-token
        streaming, but chunks arrive rapidly after generation completes.
        """
        if not self.llm_engine:
            raise RuntimeError("vLLM model not properly initialized.")

        vllm_messages = self._prepare_messages_for_vllm(messages)
        sampling_params = self._create_sampling_params(stop=stop)

        logger.debug("stream_raw: generating response")

        full_text = self._generate_chat(vllm_messages, sampling_params)
        yield from self._stream_chunks(full_text)

    def supports_native_tools(self) -> bool:
        """vLLM supports native tool calling via model chat templates."""
        return True

    def stream_with_tools(
        self,
        messages: list[Message],
        tools_schema: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = "auto",
        stop: list[str] | str | None = None,
    ) -> Iterator[str | dict]:
        """Stream response with native tool support via vLLM chat API.

        Generates the full response via LLM.chat(), parses tool calls,
        then yields content as word-sized chunks. If the response contains
        tool calls, a tool_calls dict is yielded at the end.

        Yields:
            str: Content text chunks
            dict: {"tool_calls": [...], "content": "..."} when model calls tools
        """
        if not self.llm_engine:
            raise RuntimeError("vLLM model not properly initialized.")

        vllm_messages = self._prepare_messages_for_vllm(messages)
        sampling_params = self._create_sampling_params(stop=stop)

        logger.debug("stream_with_tools: generating with %d tools", len(tools_schema or []))

        response_text = self._generate_chat(
            vllm_messages, sampling_params, tools=tools_schema
        ).strip()

        logger.debug("stream_with_tools raw output: %r", response_text[:200])

        # Try to parse tool calls from the complete output
        tool_calls = self._parse_tool_calls(response_text)

        if tool_calls:
            content_before = response_text.split("[TOOL_CALLS]", 1)[0].strip()
            if content_before:
                yield from self._stream_chunks(content_before)

            logger.info(
                "Parsed %d tool call(s): %s",
                len(tool_calls),
                ", ".join(tc["function"]["name"] for tc in tool_calls),
            )
            yield {"tool_calls": tool_calls, "content": content_before}
        else:
            yield from self._stream_chunks(response_text)

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]] | None:
        """Parse tool calls from model output text.

        Supports two Mistral tokenizer formats:
        - Modern (v1.1+): [TOOL_CALLS]function_name{"arg": "value"}
        - Legacy (pre-v1.1): [TOOL_CALLS] [{"name": "fn", "arguments": {"arg": "val"}}]
        """
        if "[TOOL_CALLS]" not in text:
            return None

        raw = text.split("[TOOL_CALLS]", 1)[1].strip()
        if not raw:
            return None

        # Try legacy JSON array format first: [{"name": ..., "arguments": ...}]
        if raw.startswith("["):
            try:
                calls = json.loads(raw)
                if isinstance(calls, list):
                    return [
                        {
                            "id": call.get("id", _generate_mistral_tool_id()),
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call.get("arguments", {})),
                            },
                        }
                        for call in calls
                    ]
            except (json.JSONDecodeError, KeyError):
                pass  # Fall through to modern format parsing

        # Modern format: function_name{"arg": "value"}
        # Multiple calls are concatenated: func1{"a":1}func2{"b":2}
        return self._parse_modern_tool_calls(raw)

    def _parse_modern_tool_calls(self, raw: str) -> list[dict[str, Any]] | None:
        """Parse modern Mistral tool call format: name{args}[name{args}...].

        The function name is everything before the first '{', and arguments
        are the JSON object that follows. Multiple tool calls are concatenated
        with no delimiter — we track brace depth to find boundaries.
        """
        calls: list[dict[str, Any]] = []
        pos = 0
        length = len(raw)

        while pos < length:
            # Skip whitespace between tool calls
            while pos < length and raw[pos] in " \t\n\r":
                pos += 1
            if pos >= length:
                break

            # Find the opening brace — everything before it is the function name
            brace_start = raw.find("{", pos)
            if brace_start < 0:
                logger.warning("Modern tool call parse: no '{' found at pos %d in %r", pos, raw[pos:pos+50])
                break

            func_name = raw[pos:brace_start].strip()
            if not func_name:
                logger.warning("Modern tool call parse: empty function name at pos %d", pos)
                break

            # Track brace depth to find the matching closing brace
            depth = 0
            json_start = brace_start
            json_end = -1
            for i in range(brace_start, length):
                ch = raw[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_end = i + 1
                        break

            if json_end < 0:
                # Unbalanced braces — try to parse what we have
                logger.warning("Modern tool call parse: unbalanced braces for %s", func_name)
                json_str = raw[json_start:]
                json_end = length
            else:
                json_str = raw[json_start:json_end]

            # Parse the JSON arguments
            try:
                args = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Modern tool call parse: invalid JSON for %s: %r", func_name, json_str[:100])
                args = {}

            calls.append({
                "id": _generate_mistral_tool_id(),
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else "{}",
                },
            })
            logger.debug("Parsed tool call: %s(%s)", func_name, json_str[:80])

            pos = json_end

        if not calls:
            logger.warning("Failed to parse any tool calls from modern format: %r", raw[:100])
            return None

        return calls

    def get_styling(self) -> tuple[str | None, str]:
        """Return vLLM-specific styling for output formatting."""
        return None, "bright_cyan"

    def unload(self) -> None:
        """Unload the vLLM model and free GPU memory."""
        if self.llm_engine is None:
            return
        
        logger.info(f"Unloading vLLM model: {self.model_id}")
        
        try:
            # Delete engine instance
            del self.llm_engine
            self.llm_engine = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    if self.config.VERBOSE:
                        self.console.print("[green]✓ CUDA cache cleared[/green]")
            except ImportError:
                logger.warning("PyTorch not available, cannot clear CUDA cache")
            
            if self.config.VERBOSE:
                self.console.print("[green]✓ vLLM model unloaded and memory freed[/green]")
        
        except Exception as e:
            logger.warning(f"Error during vLLM model unload: {e}")
            # Don't re-raise - unload is best-effort cleanup
