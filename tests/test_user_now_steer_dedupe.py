"""NOW recovery uses the existing durable RPC identity, not CLI NEXT dedupe."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent_bridge.subscriber import RedisSubscriber


def test_user_now_reuses_rpc_across_lost_ack_and_distinguishes_target():
    async def run():
        # Emulate the Redis script's SET-if-absent + RPC receipt lookup; no DB.
        queued = {}
        async def eval_script(_script, _keys, key, _stream, _limit, *fields):
            queued.setdefault(key, dict(zip(fields[::2], fields[1::2])))
            return [1, "1-0"]
        subscriber = object.__new__(RedisSubscriber)
        subscriber._pub_redis = SimpleNamespace(eval=AsyncMock(side_effect=eval_script))
        subscriber._redis = SimpleNamespace(xread=AsyncMock(return_value=[
            ("rpc", [("1-0", {"payload": json.dumps({"ok": True})})])
        ]))
        args = dict(session_key="bot:user", message="continue", message_id="message-uuid",
                    backend="claude-code", target_request_id="turn-1", priority="now", origin="user")
        # First caller times out after enqueue. A later caller consults the SAME
        # receipt and cannot enqueue another command through the Redis script.
        try:
            await subscriber.send_steer(**args, timeout_s=0)
        except TimeoutError:
            pass
        else:
            raise AssertionError("expected lost acknowledgement")
        assert await subscriber.send_steer(**args) == {"ok": True}
        assert len(queued) == 1
        rpc_id = next(iter(queued.values()))["request_id"]
        assert next(iter(subscriber._redis.xread.call_args.args[0])) == f"agent:rpc:{rpc_id}"
        await subscriber.send_steer(**{**args, "target_request_id": "turn-2"})
        await subscriber.send_steer(**{**args, "message_id": "another-uuid"})
        await subscriber.send_steer(**{**args, "session_key": "other:user"})
        assert len(queued) == 4
        await subscriber.send_steer(**args, request_id="caller-owned-id")
        assert "agent:steer:request:caller-owned-id" in queued
    asyncio.run(run())
