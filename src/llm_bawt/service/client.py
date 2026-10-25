"""
Client for communicating with the background service.

The client gracefully handles the case where the service is unavailable,
allowing the main CLI to work standalone.

Uses httpx for async operations when available, falls back to urllib.
"""

import json
import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .delivery_reply import is_accepted_receipt
from .tasks import Task, TaskResult, TaskStatus

logger = logging.getLogger(__name__)

# Wire prefix of a negotiated extension frame (chat_response_adapter's
# EXTENSION_COMMENT). Kept local so the CLI client does not import FastAPI.
_EXTENSION_EVENT = ": llm-bawt-event "

# Check for optional httpx
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Default socket path for Unix domain socket communication
DEFAULT_SOCKET_PATH = Path.home() / ".cache" / "llm-bawt" / "service.sock"
DEFAULT_HTTP_PORT = 8642  # Fallback to HTTP if Unix socket unavailable


@dataclass
class ServiceStatus:
    """Status of the background service."""
    available: bool
    version: str | None = None
    uptime_seconds: float | None = None
    tasks_processed: int = 0
    tasks_pending: int = 0
    models_loaded: list[str] | None = None
    current_model: str | None = None
    available_models: list[str] | None = None

    @property
    def healthy(self) -> bool:
        return self.available


class ServiceClient:
    """
    Client for communicating with the llm_bawt background service.
    
    Supports both Unix domain sockets (preferred for local) and HTTP.
    Gracefully handles service unavailability.
    """
    
    def __init__(
        self,
        socket_path: Path | str | None = None,
        http_url: str | None = None,
        timeout: float = 5.0,
    ):
        self.socket_path = Path(socket_path) if socket_path else DEFAULT_SOCKET_PATH
        self.http_url = http_url or f"http://localhost:{DEFAULT_HTTP_PORT}"
        self.timeout = timeout
        self._availability_timeout = 0.5  # Fast timeout for availability checks
        self._available: bool | None = None
        self._last_check: float = 0
        self._check_interval = 30.0  # Re-check availability every 30 seconds
        self.last_error: str | None = None  # Track last error for better user feedback
    
    def is_available(self, force_check: bool = False) -> bool:
        """Check if the background service is available."""
        now = time.time()
        
        # Use cached result if recent
        if not force_check and self._available is not None:
            if now - self._last_check < self._check_interval:
                return self._available
        
        self._available = self._check_availability()
        self._last_check = now
        return self._available
    
    def _check_availability(self) -> bool:
        """Actually check if service is reachable.
        
        Uses raw TCP socket connect which fails immediately on connection refused,
        avoiding HTTP timeout delays. Also supports HTTP health check for remote hosts.
        """
        # Try Unix socket first (Linux/macOS) - only for local connections
        if self.socket_path.exists() and self._is_local_url():
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.setblocking(False)
                sock.connect(str(self.socket_path))
                sock.close()
                logger.debug(f"Background service available via Unix socket: {self.socket_path}")
                return True
            except (socket.error, OSError, BlockingIOError) as e:
                logger.debug(f"Unix socket unavailable: {e}")
        
        # Parse host from http_url for TCP check
        host, port = self._parse_url_host_port()
        
        # Try raw TCP connect (fast - fails immediately if port not listening)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)  # 500ms timeout for remote hosts
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                logger.debug(f"Background service available at {host}:{port}")
                return True
            else:
                logger.debug(f"TCP {host}:{port} not listening (errno={result})")
        except Exception as e:
            logger.debug(f"TCP check failed for {host}:{port}: {e}")
        
        # For non-local hosts, also try HTTP health endpoint as fallback
        if not self._is_local_url():
            try:
                import urllib.request
                import urllib.error
                req = urllib.request.Request(f"{self.http_url}/health", method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        logger.debug(f"Background service available via HTTP health check")
                        return True
            except Exception as e:
                logger.debug(f"HTTP health check failed: {e}")
        
        return False
    
    def _parse_url_host_port(self) -> tuple[str, int]:
        """Parse host and port from http_url."""
        url = self.http_url
        # Remove protocol prefix
        if "://" in url:
            url = url.split("://", 1)[1]
        # Remove path
        if "/" in url:
            url = url.split("/", 1)[0]
        # Parse host:port
        if ":" in url:
            host, port_str = url.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = DEFAULT_HTTP_PORT
        else:
            host = url
            port = DEFAULT_HTTP_PORT
        return host, port

    def _format_http_error(self, method: str, url: str, status: int | None = None, body: str | None = None) -> str:
        parts = [f"{method} {url}"]
        if status is not None:
            parts.append(f"HTTP {status}")
        if body:
            parts.append(body)
        return " | ".join(parts)
    
    def _is_local_url(self) -> bool:
        """Check if URL points to localhost."""
        host, _ = self._parse_url_host_port()
        return host in ("localhost", "127.0.0.1", "::1")
    
    def get_status(self, silent: bool = False) -> ServiceStatus:
        """Get detailed status of the background service."""
        if not self.is_available():
            return ServiceStatus(available=False)
        
        try:
            response = self._request("GET", "/status")
            return ServiceStatus(
                available=True,
                version=response.get("version"),
                uptime_seconds=response.get("uptime_seconds"),
                tasks_processed=response.get("tasks_processed", 0),
                tasks_pending=response.get("tasks_pending", 0),
                models_loaded=response.get("models_loaded"),
                current_model=response.get("current_model"),
                available_models=response.get("available_models"),
            )
        except Exception as e:
            if not silent:
                logger.warning(f"Failed to get service status: {e}")
            return ServiceStatus(available=False)
    
    def submit_task(self, task: Task) -> bool:
        """
        Submit a task to the background service.
        
        Returns True if task was accepted, False if service unavailable.
        This is fire-and-forget - use get_task_result for results.
        """
        if not self.is_available():
            logger.debug("Background service unavailable, task not submitted")
            return False
        
        try:
            self._request("POST", "/v1/tasks", data=task.to_dict())
            logger.debug(f"Submitted task {task.task_id} ({task.task_type.value})")
            return True
        except Exception as e:
            logger.warning(f"Failed to submit task: {e}")
            return False
    
    def get_task_result(self, task_id: str, wait: bool = False, timeout: float = 30.0) -> TaskResult | None:
        """
        Get the result of a submitted task.
        
        Args:
            task_id: The task ID to check
            wait: If True, block until result is ready (up to timeout)
            timeout: Maximum seconds to wait if wait=True
        
        Returns:
            TaskResult if available, None if not found or service unavailable
        """
        if not self.is_available():
            return None
        
        try:
            params = {"wait": str(wait).lower(), "timeout": str(timeout)}
            response = self._request("GET", f"/v1/tasks/{task_id}", params=params)
            
            if response:
                return TaskResult(
                    task_id=response["task_id"],
                    status=TaskStatus(response["status"]),
                    result=response.get("result"),
                    error=response.get("error"),
                    processing_time_ms=response.get("processing_time_ms", 0),
                )
        except Exception as e:
            logger.debug(f"Failed to get task result: {e}")
        
        return None
    
    def chat_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        bot_id: str | None = None,
        user_id: str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict | None:
        """
        Send a chat completion request to the background service.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model alias to use (uses service default if not specified)
            bot_id: Bot personality to use
            user_id: User ID for memory context
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            The completion response dict, or None if service unavailable
        """
        if not self.is_available():
            return None
        
        payload = {
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        if model:
            payload["model"] = model
        if bot_id:
            payload["bot_id"] = bot_id
        if user_id:
            payload["user"] = user_id
        
        try:
            if stream:
                return self._stream_chat_completion(payload)
            else:
                # Use a long timeout for chat completions — agent-backend
                # turns (claude-code, openclaw) routinely take 30–90s.  The
                # default self.timeout (5s) is only appropriate for cheap
                # admin/list endpoints.
                response = self._request(
                    "POST", "/v1/chat/completions", data=payload, timeout=600.0,
                )
                self.last_error = None  # Clear on success
                return response
        except Exception as e:
            if not self.last_error:
                self.last_error = str(e)
            logger.warning(f"Chat completion via service failed: {e}")
            return None
    
    def _stream_chat_completion(self, payload: dict):
        """Stream chat completion from the service.

        Yields content chunks. If the request fails, yields nothing and logs error.
        The first yielded item is a dict with metadata: {"model": "actual_model_used"}.
        Subsequent yields are content strings.
        """
        import urllib.request
        import urllib.error

        url = f"{self.http_url}/v1/chat/completions"
        # The route sends a stock OpenAI stream. Warnings and server-run tool
        # calls arrive only when negotiated, as ": llm-bawt-event " comments
        # carrying the same JSON a data frame would.
        headers = {"Content-Type": "application/json",
                   "X-LLM-Bawt-Extension-Events": "comment"}
        body = json.dumps(payload).encode()

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=600.0) as resp:
                first_chunk = True
                # Use readline() instead of iterating the response directly.
                # `for line in resp` uses BufferedIOBase iteration which reads
                # in 8KB chunks before yielding, defeating streaming.
                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode().strip()
                    if line.startswith(_EXTENSION_EVENT):
                        line = "data: " + line[len(_EXTENSION_EVENT):]
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if not isinstance(chunk, dict):
                                continue
                            if is_accepted_receipt(chunk):
                                # Only an older service still sends receipts.
                                yield chunk
                                continue
                            if isinstance(chunk.get("error"), dict):
                                # Terminal in-band failure: the turn was admitted,
                                # so the caller must not fall back and resubmit.
                                message = str(chunk["error"].get("message") or "stream failed")
                                self.last_error = message
                                yield {"stream_error": message}
                                return
                            # Service warning event (e.g. model fallback)
                            if chunk.get("object") == "service.warning":
                                yield {"warnings": chunk.get("warnings", []), "model": chunk.get("model")}
                                continue
                            # On first chunk, yield metadata with actual model used
                            if first_chunk:
                                actual_model = chunk.get("model")
                                if actual_model:
                                    yield {"model": actual_model}
                                first_chunk = False
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            # OpenAI-compat tool_calls in delta
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                for tc in tool_calls:
                                    func = tc.get("function", {})
                                    yield {
                                        "tool_call": func.get("name") or "tool",
                                        "tool_args": func.get("arguments", ""),
                                        "model": chunk.get("model"),
                                    }
                                continue
                            # Check finish_reason for tool_calls
                            finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                            if finish_reason == "tool_calls":
                                continue  # Server executing tools, follow-up content coming
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()[:500]  # Read error body, limit size
            except Exception:
                pass
            self.last_error = self._format_http_error("POST", url, e.code, error_body)
            logger.warning(f"Service streaming failed: {self.last_error}")
            return
        except Exception as e:
            self.last_error = self._format_http_error("POST", url, None, str(e))
            logger.warning(f"Service streaming failed: {e}")
            return
    
    def wait_for_queued_reply(
        self, receipt: dict[str, Any], *, bot_id: str | None,
        user_id: str | None, timeout_seconds: float = 120,
    ) -> dict[str, Any] | None:
        """Wait for one accepted delivery's exact reply, never resubmit its prompt."""
        from urllib.parse import quote

        delivery_id = receipt.get("delivery_id")
        message_id = receipt.get("user_message_id")
        if not isinstance(delivery_id, str) or not delivery_id or not isinstance(message_id, str) or not message_id:
            raise ValueError("queued receipt lacks delivery or user-message identity")
        base = f"/v1/inter-bot-deliveries/{quote(delivery_id, safe='')}"
        resolved_bot = receipt.get("bot_id") or bot_id
        resolved_user = receipt.get("user_id") or user_id
        if bot_id and resolved_bot != bot_id or user_id and resolved_user != user_id:
            raise ValueError("queued receipt changed the requested bot/user")
        if not resolved_bot or not resolved_user:
            delivery = self._request("GET", base)
            author = delivery.get("author") or {}
            if (delivery.get("delivery_id") != delivery_id
                    or delivery.get("user_message_id") != message_id
                    or author.get("entity_type") != "user"):
                raise ValueError("queued delivery lookup changed identity")
            if resolved_bot and resolved_bot != delivery.get("target_bot_id") or resolved_user and resolved_user != author.get("entity_id"):
                raise ValueError("queued delivery lookup changed scope")
            resolved_bot = delivery.get("target_bot_id")
            resolved_user = author.get("entity_id")
        if not isinstance(resolved_bot, str) or not resolved_bot or not isinstance(resolved_user, str) or not resolved_user:
            raise ValueError("queued delivery lacks resolved bot/user identity")
        deadline = time.monotonic() + timeout_seconds
        while True:
            reply = self._request("GET", f"{base}/reply", params={
                "bot_id": resolved_bot, "user_id": resolved_user,
                "user_message_id": message_id,
            })
            if reply.get("delivery_id") != delivery_id:
                raise ValueError("queued reply changed delivery identity")
            if reply.get("status") in {"DELIVERED", "FAILED", "CANCELLED", "UNRESOLVED"}:
                return reply
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(1, max(0, deadline - time.monotonic())))

    def chat_completion_full(
        self,
        messages: list[dict],
        model: str | None = None,
        bot_id: str | None = None,
        user_id: str | None = None,
        **kwargs,
    ) -> str | None:
        """
        Get a complete chat response as a string.
        
        Convenience method that handles streaming internally.
        
        Returns:
            The assistant's response content, or None if unavailable.
        """
        response = self.chat_completion(
            messages=messages,
            model=model,
            bot_id=bot_id,
            user_id=user_id,
            stream=False,
            **kwargs,
        )
        
        if response and "choices" in response:
            return response["choices"][0].get("message", {}).get("content")
        return None

    def _request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        params: dict | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Make a request to the service.

        ``timeout`` overrides ``self.timeout`` for slow endpoints (e.g.
        non-streaming chat completions that wait for the full response).
        """
        import urllib.request
        import urllib.parse
        import urllib.error
        
        url = f"{self.http_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        
        headers = {"Content-Type": "application/json"}
        body = json.dumps(data).encode() if data else None
        
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        effective_timeout = timeout if timeout is not None else self.timeout
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()[:500]
            except Exception:
                pass
            self.last_error = self._format_http_error(method, url, e.code, error_body)
            raise
        except Exception as e:
            self.last_error = self._format_http_error(method, url, None, str(e))
            raise
    
    async def _request_async(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make an async request to the service (requires httpx)."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not available. Install with: pip install llm-bawt[service]")
        
        url = f"{self.http_url}{path}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if method == "GET":
                resp = await client.get(url, params=params)
            elif method == "POST":
                resp = await client.post(url, json=data, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = e.response.text[:500] if e.response is not None else ""
                self.last_error = self._format_http_error(method, url, e.response.status_code if e.response else None, body)
                raise
            except Exception as e:
                self.last_error = self._format_http_error(method, url, None, str(e))
                raise
            return resp.json()
    
    async def submit_task_async(self, task: Task) -> bool:
        """
        Submit a task to the background service asynchronously.
        
        Returns True if task was accepted, False if service unavailable.
        Requires httpx to be installed.
        """
        if not self.is_available():
            logger.debug("Background service unavailable, task not submitted")
            return False
        
        if not HTTPX_AVAILABLE:
            # Fall back to sync method
            return self.submit_task(task)
        
        try:
            await self._request_async("POST", "/v1/tasks", data=task.to_dict())
            logger.debug(f"Submitted task {task.task_id} ({task.task_type.value})")
            return True
        except Exception as e:
            logger.warning(f"Failed to submit task: {e}")
            return False
    
    async def get_task_result_async(
        self, task_id: str, wait: bool = False, timeout: float = 30.0
    ) -> TaskResult | None:
        """
        Get the result of a submitted task asynchronously.
        
        Requires httpx to be installed.
        """
        if not self.is_available():
            return None
        
        if not HTTPX_AVAILABLE:
            # Fall back to sync method
            return self.get_task_result(task_id, wait=wait, timeout=timeout)
        
        try:
            params = {"wait": str(wait).lower(), "timeout": str(timeout)}
            response = await self._request_async("GET", f"/v1/tasks/{task_id}", params=params)
            
            if response:
                return TaskResult(
                    task_id=response["task_id"],
                    status=TaskStatus(response["status"]),
                    result=response.get("result"),
                    error=response.get("error"),
                    processing_time_ms=response.get("processing_time_ms", 0),
                )
        except Exception as e:
            logger.debug(f"Failed to get task result: {e}")
        
        return None

    def search_memories(
        self,
        query: str,
        bot_id: str = "nova",
        n_results: int = 5,
        min_relevance: float = 0.0,
    ) -> list[dict] | None:
        """
        Search memories using the service's semantic search.
        
        The service keeps the embedding model loaded, so this is fast.
        
        Args:
            query: Search query text
            bot_id: Bot to search memories for
            n_results: Maximum number of results
            min_relevance: Minimum relevance score (0.0-1.0)
        
        Returns:
            List of memory dicts, or None if service unavailable
        """
        if not self.is_available():
            return None
        
        try:
            response = self._request("POST", "/v1/memory/search", data={
                "query": query,
                "bot_id": bot_id,
                "n_results": n_results,
                "min_relevance": min_relevance,
            })
            
            if response and "results" in response:
                # Format results for compatibility with memory backend
                return [
                    {
                        "id": r["id"],
                        "document": r["content"],
                        "metadata": {
                            "tags": r.get("tags", ["misc"]),
                            "importance": r.get("importance", 0.5),
                            "source_message_ids": r.get("source_message_ids", []),
                        },
                        "relevance": r.get("relevance", 0.0),
                    }
                    for r in response["results"]
                ]
        except Exception as e:
            logger.warning(f"Memory search via service failed: {e}")
        
        return None

    def generate_embedding(self, text: str) -> list[float] | None:
        """
        Generate an embedding using the service's loaded model.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector, or None if service unavailable
        """
        if not self.is_available():
            return None

        try:
            response = self._request("POST", "/v1/memory/embedding", data={
                "text": text,
            })

            if response and "embedding" in response:
                return response["embedding"]
        except Exception as e:
            logger.warning(f"Embedding generation via service failed: {e}")

        return None

    def list_models(self) -> list[dict[str, Any]] | None:
        """List available models from the service.

        Returns:
            List of model dicts (id, type, model_id, description),
            or None if service unavailable.
        """
        if not self.is_available():
            return None

        try:
            response = self._request("GET", "/v1/models")
            if response and "data" in response:
                return [m for m in response["data"] if m.get("id")]
        except Exception as e:
            logger.warning(f"List models via service failed: {e}")

        return None

    def reload_models(self) -> dict[str, Any] | None:
        """Trigger a service-side model catalog reload.

        Returns raw JSON response when successful, otherwise None.
        """
        if not self.is_available():
            return None

        try:
            response = self._request("POST", "/v1/models/reload")
            if isinstance(response, dict):
                return response
        except Exception as e:
            logger.warning(f"Reload models via service failed: {e}")

        return None

    def upsert_catalog_model(self, alias: str, model_data: dict[str, Any]) -> dict[str, Any] | None:
        """Translate one CLI model entry into normalized model + endpoint writes."""
        if not self.is_available():
            return None

        model_type = str(model_data.get("type") or "openai").strip().lower()
        backend = str(model_data.get("backend") or "").strip().lower()
        raw_model_id = str(model_data.get("model_id") or "").strip()
        provider = str(model_data.get("provider") or "").strip().lower()
        if model_type == "codex" or (model_type == "agent_backend" and backend == "codex"):
            vendor, access_key = "openai", "openai-oauth"
        elif model_type == "claude-code" and (provider == "openai_chatgpt" or raw_model_id.startswith("openai_chatgpt/")):
            vendor, access_key = "openai", "openai-oauth"
            raw_model_id = raw_model_id.removeprefix("openai_chatgpt/")
        elif model_type == "claude-code" and (provider == "xai" or raw_model_id.startswith("xai/")):
            vendor, access_key = "xai", "xai-responses"
            raw_model_id = raw_model_id.removeprefix("xai/")
        elif model_type == "claude-code" and (provider == "zai" or raw_model_id.startswith("zai/")):
            vendor, access_key = "zai", "zai-anthropic"
            raw_model_id = raw_model_id.removeprefix("zai/")
        elif model_type == "claude-code":
            vendor, access_key = "anthropic", "anthropic-oauth"
        elif model_type in {"grok", "xai"}:
            vendor, access_key = "xai", "xai-chat"
        elif model_type == "gguf":
            vendor, access_key = "community", "local-llamacpp"
        elif model_type == "vllm":
            vendor, access_key = "community", "local-vllm"
        elif model_type == "ollama":
            vendor, access_key = "community", "ollama"
        else:
            vendor, access_key = "openai", "openai-api"

        context_window = model_data.get("context_window")
        if not isinstance(context_window, int) or context_window <= 0:
            logger.warning("Cannot add '%s': normalized catalog requires context_window", alias)
            return None
        serving_config = {
            key: value
            for key, value in model_data.items()
            if key not in {
                "type", "model_id", "description", "context_window",
                "tool_support", "pricing", "provider", "upstream_model",
            }
        }
        if model_data.get("repo_id") is not None:
            serving_config["repo_id"] = model_data["repo_id"]
        if model_data.get("filename") is not None:
            serving_config["filename"] = model_data["filename"]

        try:
            model = self._request(
                "PUT",
                f"/v1/models/catalog/models/{alias}",
                data={
                    "vendor": vendor,
                    "display_name": model_data.get("description") or alias,
                    "description": model_data.get("description"),
                    "default_context_window": context_window,
                    "default_tool_support": model_data.get("tool_support"),
                },
            )
            endpoint = self._request(
                "PUT",
                f"/v1/models/catalog/endpoints/{alias}/{access_key}",
                data={
                    "upstream_model_id": model_data.get("upstream_model") or raw_model_id or None,
                    "serving_config": serving_config,
                    "pricing": model_data.get("pricing"),
                },
            )
            return {"model": model, "endpoint": endpoint}
        except Exception as exc:
            logger.warning("Upsert normalized model '%s' via service failed: %s", alias, exc)
            return None

    def delete_catalog_model(self, alias: str) -> dict[str, Any] | None:
        """Delete a normalized model and its unassigned endpoint rows."""
        if not self.is_available():
            return None
        try:
            response = self._request("DELETE", f"/v1/models/catalog/models/{alias}")
            return response if isinstance(response, dict) else None
        except Exception as exc:
            logger.warning("Delete normalized model '%s' via service failed: %s", alias, exc)
            return None

    def get_history(self, bot_id: str | None = None, limit: int = 50, before: str | None = None) -> dict[str, Any] | None:
        """Fetch conversation history from the service history endpoint.

        Returns the raw HistoryResponse payload as a dict, or None if unavailable.
        """
        if not self.is_available():
            return None

        params: dict[str, Any] = {"limit": limit}
        if bot_id:
            params["bot_id"] = bot_id
        if before:
            params["before"] = before

        try:
            return self._request("GET", "/v1/history", params=params)
        except Exception as e:
            logger.warning(f"Get history via service failed: {e}")
            return None

    def get_tool_call_events(
        self,
        bot_id: str | None = None,
        user_id: str | None = None,
        message_id: str | None = None,
        message_ids: list[str] | None = None,
        since_hours: int = 24,
        limit: int = 200,
    ) -> dict[str, Any] | None:
        """Fetch tool-call events for history annotation."""
        if not self.is_available():
            return None

        params: dict[str, Any] = {"since_hours": since_hours, "limit": limit}
        if bot_id:
            params["bot_id"] = bot_id
        if user_id:
            params["user_id"] = user_id
        if message_id:
            params["message_id"] = message_id
        if message_ids:
            params["message_ids"] = list(message_ids)

        try:
            return self._request("GET", "/v1/tool-calls", params=params)
        except Exception as e:
            logger.warning(f"Get tool-call events via service failed: {e}")
            return None

    def clear_history(self, bot_id: str | None = None) -> dict[str, Any] | None:
        """Clear conversation history through the service history endpoint.

        Returns the raw HistoryClearResponse payload as a dict, or None if unavailable.
        """
        if not self.is_available():
            return None

        params: dict[str, Any] = {}
        if bot_id:
            params["bot_id"] = bot_id

        try:
            return self._request("DELETE", "/v1/history", params=params)
        except Exception as e:
            logger.warning(f"Clear history via service failed: {e}")
            return None

    # -----------------------------------------------------------------
    # Full system status (GET /v1/status)
    # -----------------------------------------------------------------

    def get_full_status(self) -> dict[str, Any] | None:
        """Fetch the full system status from the service.

        Returns the raw SystemStatusResponse payload, or None if unavailable.
        """
        if not self.is_available():
            return None
        try:
            return self._request("GET", "/v1/status")
        except Exception as e:
            logger.warning(f"Get full status via service failed: {e}")
            return None

    # -----------------------------------------------------------------
    # Bots (GET /v1/bots)
    # -----------------------------------------------------------------

    def list_bots(self) -> list[dict[str, Any]] | None:
        """List bots from the service.

        Returns list of BotInfo dicts, or None if unavailable.
        """
        if not self.is_available():
            return None
        try:
            response = self._request("GET", "/v1/bots")
            if response and "data" in response:
                return response["data"]
        except Exception as e:
            logger.warning(f"List bots via service failed: {e}")
        return None

    def create_bot(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Create a new bot profile via the service."""
        if not self.is_available():
            return None
        import urllib.error

        slug = str(data.get("slug") or "").strip().lower()
        try:
            return self._request("POST", "/v1/bots", data=data)
        except urllib.error.HTTPError as e:
            if e.code in (404, 405) and slug:
                existing = self.get_bot_profile(slug)
                if existing is not None:
                    self.last_error = (
                        f"Bot '{slug}' already exists. "
                        "Legacy service fallback refused to overwrite it."
                    )
                    logger.warning(self.last_error)
                    return None

                legacy_payload = dict(data)
                legacy_payload.pop("slug", None)
                result = self.upsert_bot_profile(slug, legacy_payload)
                if result is not None:
                    self.last_error = (
                        "Connected service does not support POST /v1/bots yet; "
                        f"used legacy PUT /v1/bots/{slug}/profile fallback."
                    )
                    logger.warning(self.last_error)
                    return result
        except Exception as e:
            logger.warning(f"Create bot via service failed: {e}")
            return None

    # -----------------------------------------------------------------
    # Jobs (GET /v1/jobs, GET /v1/jobs/runs)
    # -----------------------------------------------------------------

    def get_jobs(self, **params) -> dict[str, Any] | None:
        """Fetch scheduled jobs from the service."""
        if not self.is_available():
            return None
        try:
            return self._request("GET", "/v1/jobs", params=params or None)
        except Exception as e:
            logger.warning(f"Get jobs via service failed: {e}")
            return None

    def get_job_runs(self, limit: int = 10, **params) -> dict[str, Any] | None:
        """Fetch recent job runs from the service."""
        if not self.is_available():
            return None
        try:
            p: dict[str, Any] = {"limit": limit, **params}
            return self._request("GET", "/v1/jobs/runs", params=p)
        except Exception as e:
            logger.warning(f"Get job runs via service failed: {e}")
            return None

    # -----------------------------------------------------------------
    # Runtime settings
    # -----------------------------------------------------------------

    def get_settings(self, scope_type: str = "bot", scope_id: str | None = None) -> dict[str, Any] | None:
        """Fetch runtime settings for a scope."""
        if not self.is_available():
            return None
        try:
            params: dict[str, Any] = {"scope_type": scope_type}
            if scope_id:
                params["scope_id"] = scope_id
            return self._request("GET", "/v1/settings", params=params)
        except Exception as e:
            logger.warning(f"Get settings via service failed: {e}")
            return None

    def set_setting(self, scope_type: str, scope_id: str | None, key: str, value: Any) -> dict[str, Any] | None:
        """Upsert one runtime setting."""
        if not self.is_available():
            return None
        try:
            return self._request("PUT", "/v1/settings", data={
                "scope_type": scope_type,
                "scope_id": scope_id,
                "key": key,
                "value": value,
            })
        except Exception as e:
            logger.warning(f"Set setting via service failed: {e}")
            return None

    def delete_setting(self, scope_type: str, key: str, scope_id: str | None = None) -> dict[str, Any] | None:
        """Delete one runtime setting."""
        if not self.is_available():
            return None
        try:
            params: dict[str, Any] = {"scope_type": scope_type, "key": key}
            if scope_id:
                params["scope_id"] = scope_id
            return self._request("DELETE", "/v1/settings", params=params)
        except Exception as e:
            logger.warning(f"Delete setting via service failed: {e}")
            return None

    def batch_set_settings(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Batch upsert runtime settings."""
        if not self.is_available():
            return None
        try:
            return self._request("POST", "/v1/settings/batch", data={"items": items})
        except Exception as e:
            logger.warning(f"Batch set settings via service failed: {e}")
            return None

    # -----------------------------------------------------------------
    # Bot profiles
    # -----------------------------------------------------------------

    def get_bot_profile(self, slug: str) -> dict[str, Any] | None:
        """Fetch a bot profile from the service."""
        if not self.is_available():
            return None
        try:
            return self._request("GET", f"/v1/bots/{slug}/profile")
        except Exception as e:
            logger.debug(f"Get bot profile via service failed: {e}")
            return None

    def upsert_bot_profile(self, slug: str, data: dict[str, Any]) -> dict[str, Any] | None:
        """Create or update a bot profile via the service."""
        if not self.is_available():
            return None
        try:
            return self._request("PUT", f"/v1/bots/{slug}/profile", data=data)
        except Exception as e:
            logger.warning(f"Upsert bot profile via service failed: {e}")
            return None

    def delete_bot_profile(self, slug: str, purge: bool = False) -> bool:
        """Delete a bot profile via the service. Pass purge=True to also wipe all data."""
        if not self.is_available():
            return False
        try:
            params = "?purge=true" if purge else ""
            self._request("DELETE", f"/v1/bots/{slug}/profile{params}")
            return True
        except Exception as e:
            logger.warning(f"Delete bot profile via service failed: {e}")
            return False

    def purge_bot_data(self, slug: str) -> dict | None:
        """Purge all bot data without deleting the profile."""
        if not self.is_available():
            return None
        try:
            return self._request("POST", f"/v1/bots/{slug}/purge-data")
        except Exception as e:
            logger.warning(f"Purge bot data via service failed: {e}")
            return None

    def cleanup_orphaned_bot_data(self, dry_run: bool = True,
                                 manifest_digest: str | None = None) -> dict | None:
        """Preview orphan partitions or confirm the digest returned by preview."""
        if not self.is_available():
            return None
        if not dry_run and not manifest_digest:
            raise ValueError("A preview manifest digest is required to execute cleanup")
        try:
            params = f"?dry_run={'true' if dry_run else 'false'}"
            if manifest_digest:
                from urllib.parse import quote
                params += f"&manifest_digest={quote(manifest_digest, safe='')}"
            return self._request("POST", f"/v1/bots/cleanup-orphans{params}")
        except Exception as e:
            logger.warning(f"Cleanup orphaned bot data via service failed: {e}")
            return None

    # -----------------------------------------------------------------
    # User/bot profiles
    # -----------------------------------------------------------------

    def get_profile(self, entity_id: str, entity_type: str | None = None) -> dict[str, Any] | None:
        """Fetch a profile by entity ID."""
        if not self.is_available():
            return None
        try:
            if entity_type:
                return self._request("GET", f"/v1/profiles/{entity_type}/{entity_id}")
            return self._request("GET", f"/v1/profiles/{entity_id}")
        except Exception as e:
            logger.debug(f"Get profile via service failed: {e}")
            return None

    def list_profiles(self, entity_type: str | None = None) -> dict[str, Any] | None:
        """List profiles, optionally filtered by type."""
        if not self.is_available():
            return None
        try:
            params = {"entity_type": entity_type} if entity_type else None
            return self._request("GET", "/v1/profiles", params=params)
        except Exception as e:
            logger.warning(f"List profiles via service failed: {e}")
            return None

    def upsert_profile_attribute(self, entity_type: str, entity_id: str, category: str, key: str, value: Any, confidence: float = 1.0, source: str = "explicit") -> dict[str, Any] | None:
        """Create or update a profile attribute."""
        if not self.is_available():
            return None
        try:
            return self._request("POST", "/v1/profiles/attribute", data={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "category": category,
                "key": key,
                "value": value,
                "confidence": confidence,
                "source": source,
            })
        except Exception as e:
            logger.warning(f"Upsert profile attribute via service failed: {e}")
            return None

    # -----------------------------------------------------------------
    # Jobs - trigger
    # -----------------------------------------------------------------

    def trigger_job(self, job_type: str) -> dict[str, Any] | None:
        """Trigger a scheduled job to run immediately.

        Args:
            job_type: The job type to trigger (e.g., "profile_maintenance")

        Returns:
            Response dict with success status, or None if service unavailable
        """
        if not self.is_available():
            return None
        try:
            return self._request("POST", f"/v1/jobs/{job_type}/trigger")
        except Exception as e:
            logger.warning(f"Trigger job via service failed: {e}")
            return None

    # -----------------------------------------------------------------
    # History (existing methods follow)
    # -----------------------------------------------------------------

    def get_all_history(self, bot_id: str | None = None, page_limit: int = 200, max_pages: int = 100) -> list[dict[str, Any]] | None:
        """Fetch all visible history messages via paginated /v1/history calls.

        Returns messages in chronological order, or None if service unavailable.
        """
        if not self.is_available():
            return None

        all_messages: list[dict[str, Any]] = []
        before: str | None = None
        seen_oldest: set[str] = set()

        for _ in range(max_pages):
            page = self.get_history(bot_id=bot_id, limit=page_limit, before=before)
            if page is None:
                return None

            messages = page.get("messages", [])
            if not messages:
                break
            all_messages = messages + all_messages

            if not page.get("has_more", False):
                break

            oldest = page.get("oldest_timestamp")
            if oldest is None:
                break
            before = str(oldest)
            if before in seen_oldest:
                break
            seen_oldest.add(before)

        return all_messages


# Singleton instance for easy access
_service_client: ServiceClient | None = None


def _build_service_http_url(config: Any | None) -> str | None:
    """Build service URL from config, normalizing bind-all hosts for client use."""
    if config is None:
        return None

    explicit_url = getattr(config, "SERVICE_URL", None)
    if explicit_url:
        return explicit_url

    host = getattr(config, "SERVICE_HOST", None)
    port = getattr(config, "SERVICE_PORT", None)
    if host and port:
        connect_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
        return f"http://{connect_host}:{port}"

    return None


def get_service_client(config: Any | None = None) -> ServiceClient:
    """Get the global service client instance.

    If config is provided, uses its service host/port to build the client URL.
    """
    global _service_client
    desired_url = _build_service_http_url(config)
    if _service_client is None:
        _service_client = ServiceClient(http_url=desired_url)
    elif desired_url and _service_client.http_url != desired_url:
        # Recreate with updated endpoint when config differs.
        _service_client = ServiceClient(http_url=desired_url)
    return _service_client
