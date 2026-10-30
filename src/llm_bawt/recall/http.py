"""Streaming byte limits applied before parsing external JSON."""
from __future__ import annotations

import json

import httpx

from .contracts import RecallError


class BoundedHTTP:
    def __init__(self, client: httpx.AsyncClient, *, max_calls: int, max_bytes: int):
        self.client = client
        self.max_calls = max_calls
        self.remaining_bytes = max_bytes
        self.calls = 0

    async def json(self, method: str, url: str, *, byte_limit: int = 65_536,
                   status_codes: dict[int, str] | None = None,
                   error_details: dict[str, str] | None = None, **kwargs):
        """``status_codes`` maps an HTTP error status to the caller's own error code;
        ``error_details`` maps a known JSON ``detail`` of an error body to one,
        and wins over the status. Only a bounded error body is read."""
        if self.calls >= self.max_calls or self.remaining_bytes <= 0:
            raise RecallError("retrieval_budget")
        self.calls += 1
        chunks = bytearray()
        try:
            async with self.client.stream(method, url, **kwargs) as response:
                if response.status_code == 504:
                    raise RecallError("timeout")
                if response.status_code == 429:
                    raise RecallError("busy")
                if response.status_code >= 400:
                    code = await _detail_code(response, error_details) if error_details else None
                    raise RecallError(code or (status_codes or {}).get(response.status_code,
                                                                       "source_unavailable"))
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    self.remaining_bytes -= len(chunk)
                    if len(chunks) + len(chunk) > byte_limit or self.remaining_bytes < 0:
                        raise RecallError("retrieval_budget")
                    chunks.extend(chunk)
            return json.loads(chunks)
        except httpx.TimeoutException as error:
            raise RecallError("timeout") from error
        except (ValueError, httpx.HTTPError) as error:
            raise RecallError("source_unavailable") from error


_ERROR_BODY_LIMIT = 1024


async def _detail_code(response: httpx.Response, error_details: dict[str, str]) -> str | None:
    """The caller's code for a known error ``detail``; None for anything else."""
    body = bytearray()
    try:
        # One byte past the limit is enough to know the body is too big; a larger
        # chunk would pull up to twice the limit off the wire first.
        async for chunk in response.aiter_bytes(chunk_size=_ERROR_BODY_LIMIT + 1):
            body.extend(chunk)
            if len(body) > _ERROR_BODY_LIMIT:
                return None
        detail = json.loads(body).get("detail")
    except (ValueError, AttributeError, httpx.HTTPError):
        return None
    return error_details.get(detail) if isinstance(detail, str) else None
