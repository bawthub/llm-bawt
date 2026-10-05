"""Synchronous inter-bot visibility trust-boundary regressions."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from llm_bawt.service.routes.chat import _internal_inter_bot_sender


def _request(host: str, sender: str | None) -> Request:
    headers = []
    if sender is not None:
        headers.append((b"x-llm-bawt-inter-bot-sender", sender.encode()))
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": headers,
        "client": (host, 1234),
    })


def test_internal_sender_header_requires_loopback() -> None:
    assert _internal_inter_bot_sender(_request("127.0.0.1", "Vex")) == "Vex"
    assert _internal_inter_bot_sender(_request("::1", "Vex")) == "Vex"
    assert _internal_inter_bot_sender(_request("10.0.0.50", None)) is None

    with pytest.raises(HTTPException) as caught:
        _internal_inter_bot_sender(_request("10.0.0.50", "Vex"))
    assert caught.value.status_code == 403
