"""A fake ``codex app-server`` for the TASK-1243 transport tests.

:func:`write_fake` writes an executable script that speaks the app-server
JSON-RPC subset the bridge uses, over real stdin/stdout pipes, so the client
is tested against a real child process: its pipes, EOF and exit. Everything
the bridge writes is logged to ``FAKE_LOG`` and everything the fake writes to
``FAKE_SENT``; :func:`assert_matches_schema` checks both against the pinned
0.157.1 schema in ``tests/fixtures/codex_app_server/``.

Behaviour comes from the environment:

``FAKE_VERSION``  version in the ``initialize`` userAgent (default 0.157.1)
``FAKE_ARGV``     file the fake writes its argv to
``FAKE_MODE``     ``complete``: stream a reply, ask for an approval and wait
                  for the refusal, then complete. ``hold``: stream a delta and
                  wait for a steer or an interrupt. ``fail``: fail the turn
                  with ``FAKE_ERROR``.
``FAKE_STEER``    ``ok`` (default): accept and show the user message, then
                  keep holding. ``lose``: accept, never show it, and complete.
``FAKE_INTERRUPT`` ``ok`` (default): reply and end the turn interrupted.
                  ``ignore``: never answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "codex_app_server" / "0.157.1"

THREAD_ID = "thr-1"
TURN_ID = "turn-1"

_SCRIPT = r'''
import json, os, sys

LOG = os.environ.get("FAKE_LOG")
SENT = os.environ.get("FAKE_SENT")
MODE = os.environ.get("FAKE_MODE", "complete")
STEER = os.environ.get("FAKE_STEER", "ok")
INTERRUPT = os.environ.get("FAKE_INTERRUPT", "ok")
THREAD, TURN = "thr-1", "turn-1"

if os.environ.get("FAKE_ARGV"):
    with open(os.environ["FAKE_ARGV"], "w") as fh:
        json.dump(sys.argv[1:], fh)


def record(path, message):
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(message) + "\n")


def send(message):
    record(SENT, message)
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def note(method, params):
    send({"method": method, "params": params})


def item(method, body):
    params = {"threadId": THREAD, "turnId": TURN, "item": body}
    params["startedAtMs" if method == "item/started" else "completedAtMs"] = 1
    note(method, params)


def turn_completed(status, error=None):
    turn = {"id": TURN, "items": [], "status": status}
    if error is not None:
        turn["error"] = {"message": error}
    note("turn/completed", {"threadId": THREAD, "turn": turn})


def agent_text(item_id, chunks):
    item("item/started", {"type": "agentMessage", "id": item_id, "text": ""})
    for chunk in chunks:
        note("item/agentMessage/delta",
             {"threadId": THREAD, "turnId": TURN, "itemId": item_id, "delta": chunk})
    item("item/completed", {"type": "agentMessage", "id": item_id, "text": "".join(chunks)})


def usage():
    total = {"inputTokens": 10, "cachedInputTokens": 4, "outputTokens": 3,
             "reasoningOutputTokens": 1, "totalTokens": 13}
    note("thread/tokenUsage/updated", {"threadId": THREAD, "turnId": TURN,
                                       "tokenUsage": {"last": total, "total": total}})


for line in sys.stdin:
    message = json.loads(line)
    record(LOG, message)
    method, rid = message.get("method"), message.get("id")
    if method == "initialize":
        version = os.environ.get("FAKE_VERSION", "0.157.1")
        send({"id": rid, "result": {"userAgent": "llm_bawt_codex_bridge/" + version + " (Linux)"}})
    elif method in ("thread/start", "thread/resume"):
        send({"id": rid, "result": {"thread": {"id": THREAD}}})
    elif method == "turn/start":
        send({"id": rid, "result": {"turn": {"id": TURN, "items": [], "status": "inProgress"}}})
        if MODE == "complete":
            agent_text("msg-1", ["Hel", "lo"])
            send({"id": "srv-1", "method": "item/commandExecution/requestApproval",
                  "params": {"itemId": "cmd-1", "threadId": THREAD, "turnId": TURN, "startedAtMs": 1}})
        elif MODE == "fail":
            turn_completed("failed", os.environ.get("FAKE_ERROR", "boom"))
        else:
            item("item/started", {"type": "agentMessage", "id": "msg-1", "text": ""})
            note("item/agentMessage/delta",
                 {"threadId": THREAD, "turnId": TURN, "itemId": "msg-1", "delta": "work"})
    elif rid == "srv-1" and method is None:
        usage()
        turn_completed("completed")
    elif method == "turn/steer":
        params = message["params"]
        if params.get("expectedTurnId") != TURN:
            send({"id": rid, "error": {"code": -32600, "message": "no active turn to steer"}})
            continue
        send({"id": rid, "result": {"turnId": TURN}})
        if STEER == "lose":
            usage()
            turn_completed("completed")
        else:
            body = {"type": "userMessage", "id": "user-2", "content": params["input"]}
            if params.get("clientUserMessageId"):
                body["clientId"] = params["clientUserMessageId"]
            item("item/started", body)
            item("item/completed", body)
    elif method == "turn/interrupt":
        if INTERRUPT == "ignore":
            continue
        send({"id": rid, "result": {}})
        turn_completed("interrupted")
'''


def write_fake(directory: Path) -> Path:
    """The fake as an executable in ``directory``, run by this interpreter."""
    path = directory / "codex"
    path.write_text(f"#!{sys.executable}\n{_SCRIPT}")
    path.chmod(0o755)
    return path


def fake_env(directory: Path, **behaviour: str) -> dict[str, str]:
    """Environment for one fake run: its logs in ``directory`` plus ``behaviour``."""
    env = {"FAKE_LOG": str(directory / "received.jsonl"), "FAKE_SENT": str(directory / "sent.jsonl"),
           "PATH": "/usr/bin:/bin"}
    env.update({f"FAKE_{key.upper()}": value for key, value in behaviour.items()})
    return env


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _validator(name: str):
    import jsonschema

    schema = json.loads((FIXTURES / name).read_text())
    return jsonschema.Draft7Validator(schema)


def assert_matches_schema(directory: Path) -> None:
    """Every message each side wrote is valid for the pinned protocol.

    The bridge's requests and notifications against ``ClientRequest`` and
    ``ClientNotification``; the fake's notifications and requests against
    ``ServerNotification`` and ``ServerRequest``, so the fake stays honest.
    Replies (no ``method``) carry no schema of their own here.
    """
    client = {"request": _validator("ClientRequest.json"), "note": _validator("ClientNotification.json")}
    server = {"request": _validator("ServerRequest.json"), "note": _validator("ServerNotification.json")}
    for log, validators in ((directory / "received.jsonl", client), (directory / "sent.jsonl", server)):
        for message in read_log(log):
            if "method" not in message:
                continue
            kind = "request" if "id" in message else "note"
            errors = sorted(validators[kind].iter_errors(message), key=str)
            assert not errors, f"{log.name}: {message['method']} does not match the schema: {errors[0].message}"
