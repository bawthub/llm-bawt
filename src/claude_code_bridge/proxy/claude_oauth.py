"""Native Claude SDK HTTP gateway with request-scoped OAuth recovery (TASK-845).

This is a transport, not a model adapter: preserve the CLI's request bytes,
beta headers, model IDs and response bytes. Only replace the bearer. Every
request resolves the app-owned token; one HTTP 401 may be retried BEFORE
sending response headers/body downstream. Never restart the SDK turn, replay
tools, retry a partial stream, or refresh the OAuth chain in this process.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator, Mapping

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .._bridge_helpers import _fetch_broker_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/claude-oauth")

# Hop-by-hop headers must not cross either connection. httpx will compute the
# outgoing request length; response compression is preserved via aiter_raw().
_HOP_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host",
})


def _headers(headers: Mapping[str, str]) -> dict[str, str]:
    items = {k.lower(): v for k, v in headers.items()}
    connection = {v.strip().lower() for v in items.get("connection", "").split(",")}
    return {k: v for k, v in items.items() if k.lower() not in _HOP_HEADERS | connection}


class ClaudeOAuthGateway:
    """Own upstream connections; leave the SDK's native agent loop untouched."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=20.0), follow_redirects=False,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def forward(self, request: Request, path: str) -> StreamingResponse | JSONResponse:
        body = await request.body()
        headers = _headers(request.headers)
        for key in ("authorization", "x-api-key", "content-length", "x-bridge-token"):
            headers.pop(key, None)
        # Deliberately fixed origin. A model/request cannot select an upstream
        # host or forward subscription credentials to a third-party adapter.
        url = httpx.URL(f"https://api.anthropic.com/v1/{path}").copy_with(
            query=request.url.query.encode(),
        )
        token, _ = await asyncio.to_thread(_fetch_broker_token)
        if not token:
            return JSONResponse(status_code=503, content={
                "type": "error", "error": {
                    "type": "api_error", "message": "Claude token broker unavailable",
                },
            })

        response: httpx.Response | None = None
        try:
            for attempt in range(2):
                headers["authorization"] = f"Bearer {token}"
                outgoing = self.client.build_request(
                    request.method, url, content=body, headers=headers,
                )
                response = await self.client.send(outgoing, stream=True)
                if response.status_code != 401 or attempt:
                    break
                rejected_hash = hashlib.sha256(token.encode()).hexdigest()
                current, _ = await asyncio.to_thread(
                    _fetch_broker_token, force=True, rejected_token_sha256=rejected_hash,
                )
                if not current or current == token:
                    break  # no new credential; preserve the real upstream 401
                await response.aclose()
                response = None
                token = current
                logger.info("Claude request rejected rotated token; retrying once with current broker token")
        except BaseException:
            if response is not None:
                await response.aclose()
            raise

        assert response is not None

        async def stream() -> AsyncIterator[bytes]:
            try:
                # No parsing/splicing/retry after a response has started. This
                # preserves signed thinking, tools and all native SSE semantics.
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(
            stream(), status_code=response.status_code, headers=_headers(response.headers),
            background=BackgroundTask(response.aclose),
        )


@router.post("/v1/messages")
async def messages(request: Request):
    return await request.app.state.claude_oauth_gateway.forward(request, "messages")


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    return await request.app.state.claude_oauth_gateway.forward(request, "messages/count_tokens")
