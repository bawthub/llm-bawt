import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from llm_bawt.api_auth.middleware import Principal
from llm_bawt import inspection_admission
from llm_bawt.inspection_admission import INSPECTION_ADMISSION_HEADER, WebInspectionAdmission
from llm_bawt.service import chat_response_adapter
from llm_bawt.service.routes import chat
from llm_bawt.service.schemas import ChatCompletionRequest

ORIGIN = "https://tenant.invalid:8443"
ACCOUNT = "owner-account"
BOT = "installed-bot"
USER = "owner-profile"
NOW = 1_700_000_000
PROOF = WebInspectionAdmission(BOT, USER, ACCOUNT, ORIGIN, NOW)
WEB = Principal("bawthub-web", "key")


@pytest.mark.parametrize(
    "principal,origins,expected",
    [
        (WEB, [ORIGIN], PROOF),
        (None, [ORIGIN], None),
        (Principal("bawthub-web", "service"), [ORIGIN], None),
        (Principal("bawthub-web", "bot"), [ORIGIN], None),
        (Principal("another-client", "key"), [ORIGIN], None),
        (WEB, ["https://other.invalid"], None),
        (WEB, [ORIGIN, ORIGIN], None),
        (WEB, [], None),
    ],
)
def test_streaming_route_only_marks_verified_web_admission(monkeypatch, principal, origins, expected):
    monkeypatch.setenv("BAWTHUB_INSPECTION_ORIGIN", ORIGIN)
    monkeypatch.setattr(inspection_admission.time, "time", lambda: NOW)
    monkeypatch.setattr(chat, "get_service", object)
    monkeypatch.setattr(chat_response_adapter, "open_stream", AsyncMock(return_value=iter(())))
    request = ChatCompletionRequest(model="auto", messages=[], bot_id=BOT, user=USER, stream=True)
    http = Request({
        "type": "http",
        "headers": [
            (INSPECTION_ADMISSION_HEADER.lower().encode(), json.dumps([ACCOUNT, origin]).encode())
            for origin in origins
        ],
        "state": {"api_principal": principal},
    })
    assert asyncio.run(chat.chat_completions(request, http)).status_code == 200
    assert request._inspection_web_admission == expected


def test_web_proof_cannot_enter_from_json_or_survive_a_request_copy():
    request = ChatCompletionRequest.model_validate({
        "model": "auto", "messages": [], "_inspection_web_admission": PROOF,
    })
    assert request._inspection_web_admission is None
    request._inspection_web_admission = PROOF
    assert "_inspection_web_admission" not in request.model_dump()
    for deep in (False, True):
        assert request.model_copy(deep=deep)._inspection_web_admission is None
        assert request.model_copy(update={"_inspection_web_admission": PROOF}, deep=deep)._inspection_web_admission is None
    assert request._inspection_web_admission == PROOF
