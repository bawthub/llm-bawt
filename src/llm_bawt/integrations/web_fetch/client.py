"""Web fetch client for retrieving web page content via Crawl4AI.

Uses the Crawl4AI Docker service to fetch and extract markdown content
from web pages, including JS-rendered content.
"""

import logging
import os
from typing import Any

import httpx

from agent_bridge import crawl4ai_token

logger = logging.getLogger(__name__)

DEFAULT_CRAWL4AI_URL = "http://localhost:11235"
DEFAULT_MAX_CONTENT_CHARS = 50_000
DEFAULT_TIMEOUT = 30


class WebFetchClient:
    """Fetches web pages via Crawl4AI and returns markdown content."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ):
        self._base_url = (
            base_url
            or os.environ.get("LLM_BAWT_CRAWL4AI_URL", "")
            or DEFAULT_CRAWL4AI_URL
        )
        self._timeout = timeout
        self._max_content_chars = int(
            os.environ.get("LLM_BAWT_WEB_FETCH_MAX_CHARS", "")
            or max_content_chars
        )
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Lazy-initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            )
        return self._client

    def fetch(self, url: str) -> str:
        """Fetch a URL and return its content as markdown.

        Args:
            url: Full URL to fetch (must start with http:// or https://).

        Returns:
            Formatted string with metadata header and markdown content.
        """
        if not url or not url.startswith(("http://", "https://")):
            return f"Error: Invalid URL: {url}"

        try:
            client = self._get_client()
            # Derived token file first, legacy env second (TASK-1476).
            headers = crawl4ai_token.auth_headers()

            response = client.post(
                "/crawl",
                json={
                    "urls": [url],
                    "browser_config": {
                        "type": "BrowserConfig",
                        "params": {"headless": True},
                    },
                    "crawler_config": {
                        "type": "CrawlerRunConfig",
                        "params": {"cache_mode": "bypass"},
                    },
                },
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            result = data["results"][0]
            if not result.get("success"):
                error_msg = result.get("error_message", "Unknown crawl error")
                return f"Error fetching {url}: {error_msg}"

            # Extract markdown - can be a dict (v0.8+) or string (older)
            md_data = result.get("markdown", "")
            if isinstance(md_data, dict):
                markdown = md_data.get("raw_markdown", "")
            else:
                markdown = str(md_data)

            # Extract metadata
            metadata = result.get("metadata") or {}
            title = metadata.get("title", "")
            description = metadata.get("description", "")

            # Build formatted output
            lines = []
            if title:
                lines.append(f"Title: {title}")
            lines.append(f"URL: {result.get('url') or url}")
            if description:
                lines.append(f"Description: {description}")
            lines.append("")
            lines.append("---")
            lines.append("")

            header = "\n".join(lines)
            remaining_chars = self._max_content_chars - len(header)

            if len(markdown) > remaining_chars:
                markdown = markdown[:remaining_chars] + "\n\n[Content truncated]"

            return header + markdown

        except httpx.HTTPStatusError as e:
            return self._handle_http_error(e)
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Crawl4AI at {self._base_url}")
            return f"Error: Cannot connect to web fetch service at {self._base_url}. Is the crawl4ai container running?"
        except httpx.RequestError as e:
            logger.error(f"Web fetch request failed: {e}")
            return f"Error: Web fetch request failed: {e}"
        except Exception as e:
            logger.error(f"Web fetch failed: {e}")
            return f"Error: {e}"

    def _handle_http_error(self, e: httpx.HTTPStatusError) -> str:
        """Handle HTTP errors from Crawl4AI."""
        status = e.response.status_code
        if status == 401:
            return "Error: Crawl4AI API token is invalid."
        elif status == 429:
            return "Error: Crawl4AI rate limit exceeded. Try again later."
        else:
            logger.error(f"Crawl4AI HTTP error: {e}")
            return f"Error: Crawl4AI HTTP {status}"

    def is_available(self) -> bool:
        """Check if the Crawl4AI service is reachable."""
        try:
            client = self._get_client()
            resp = client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    def __del__(self) -> None:
        if self._client:
            self._client.close()
