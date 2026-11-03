"""Offline Git identity and per-attempt SDK capability regressions (TASK-1280).

Uses stdlib unittest so the focused cases can also run without pytest. HTTP
requests stay in-process; signing uses a fresh in-memory key, never the keyfile.
"""

from __future__ import annotations

import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
from cryptography.fernet import Fernet
from fastapi import FastAPI

from agent_bridge.mcp_call_context import MCP_TASK_TURN_CONTEXT_ENV
from claude_code_bridge.send_stream import ClaudeStreamMixin
from llm_bawt import task_turn_context as codec
from llm_bawt.service.routes import providers


class GitIdentityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.signer = Fernet(Fernet.generate_key())
        key_patch = patch.object(codec, "_get_fernet", return_value=self.signer)
        key_patch.start()
        self.addCleanup(key_patch.stop)
        self.now = 2_000_000_000
        clock_patch = patch("cryptography.fernet.time.time", return_value=self.now)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.values = {
            "session_id": str(uuid4()),
            "turn_id": "turn-" + uuid4().hex,
            "trigger_message_id": str(uuid4()),
            "bot_id": "test-bot_2",
            "user_id": "test-user",
        }
        self.app = FastAPI()
        self.app.include_router(providers.router)

    def capability(self, *, timestamp: int | None = None, **overrides) -> str:
        token = codec.mint_task_turn_context(**(self.values | overrides))
        if timestamp is not None:
            token = self.signer.encrypt_at_time(
                self.signer.decrypt(token.encode()), current_time=timestamp,
            ).decode()
        return token

    async def get(self, capability: str | None = None, *, query: str = ""):
        headers = {} if capability is None else {codec.TASK_TURN_CONTEXT_HEADER: capability}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test",
        ) as client:
            return await client.get("/v1/git/identity" + query, headers=headers)

    def assert_denied(self, response, status: int = 401) -> None:
        self.assertEqual(response.status_code, status)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(set(response.json()), {"detail"})
        for field in ("session_id", "trigger_message_id", "user_id", "issued_at", "author_email"):
            self.assertNotIn(field, response.json())

    async def test_missing_and_blank_context_cannot_use_bot_hints(self) -> None:
        for capability in (None, "", "   "):
            with self.subTest(context="absent" if capability is None else "blank"):
                self.assert_denied(await self.get(capability, query="?bot=other&bot_id=other"))

    async def test_invalid_context_fails_closed(self) -> None:
        self.assert_denied(await self.get("not-a-valid-token"))
        other_signer = Fernet(Fernet.generate_key())
        foreign = other_signer.encrypt(json.dumps(self.values).encode()).decode()
        self.assert_denied(await self.get(foreign))

    async def test_expired_context_fails_closed(self) -> None:
        token = self.capability(timestamp=self.now - codec.TASK_TURN_CONTEXT_TTL_SECONDS - 1)
        self.assert_denied(await self.get(token))

    async def test_verified_bot_payload_ignores_hints_and_contains_no_secrets(self) -> None:
        response = await self.get(self.capability(), query="?bot=other&bot_id=other")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.json(), {
            "bot_id": "test-bot_2",
            "author_name": "test-bot_2",
            "author_email": "test-bot_2@bots.bawt.local",
            "turn_id": self.values["turn_id"],
        })

    async def test_attribution_within_ttl_requires_no_live_turn_lookup(self) -> None:
        # No service or active-turn fixture exists: the six-hour bearer alone
        # remains sufficient, including for synthetic delivery turns.
        token = self.capability(
            timestamp=self.now - codec.TASK_TURN_CONTEXT_TTL_SECONDS + 1,
            turn_id="turn-delivery-" + uuid4().hex,
        )
        self.assertEqual((await self.get(token)).status_code, 200)

    async def test_signed_actor_ids_that_are_not_safe_slugs_are_rejected(self) -> None:
        # These pass the actor codec but must not become Git/email identities.
        for bot_id in ("bot@example", "bot.name", "bot:name", "Uppercase"):
            with self.subTest(bot_id=bot_id):
                self.assert_denied(await self.get(self.capability(bot_id=bot_id)), status=403)

    async def test_signed_malformed_bot_identity_fails_closed(self) -> None:
        for bot_id in ("bad\nname", "bad<name>", "bad name", "-leading", "a" * 65):
            with self.subTest(identity="malformed"):
                body = self.values | {"bot_id": bot_id, "issued_at": self.now}
                token = self.signer.encrypt(json.dumps(body).encode()).decode()
                self.assert_denied(await self.get(token))


class ClaudeGitCapabilityEnvironmentTests(unittest.TestCase):
    def options(self, bridge, sdk_env, capability=None):
        return ClaudeStreamMixin._build_agent_options(
            bridge,
            model="test-model",
            system_prompt="test prompt",
            disallowed_tools=[],
            resume_id=None,
            sdk_env=sdk_env,
            settings_path=None,
            bot_effort=None,
            bot_max_turns=None,
            can_use_tool_cb=None,
            pre_tool_use_cb=None,
            post_tool_use_cb=None,
            stderr=None,
            task_turn_capability=capability,
        )

    def test_per_attempt_environment_does_not_leak_between_turns_or_retries(self) -> None:
        bridge = SimpleNamespace(
            _mcp_servers={"bawthub": {"type": "http", "url": "http://test/mcp"}},
            _cwd="/tmp", _permission_mode="default", _add_dirs=[],
        )
        mcp_before = copy.deepcopy(bridge._mcp_servers)
        base = {"KEEP": "unchanged", MCP_TASK_TURN_CONTEXT_ENV: "stale-test-context"}
        base_before = dict(base)
        with patch.dict(os.environ, {MCP_TASK_TURN_CONTEXT_ENV: "inherited-test-context"}):
            first = self.options(bridge, base, "first-test-context")
            second = self.options(bridge, base, "second-test-context")
            retry = self.options(bridge, base, "first-test-context")
            absent = self.options(bridge, first.env)
            inherited = self.options(bridge, {})
            self.assertEqual(os.environ[MCP_TASK_TURN_CONTEXT_ENV], "inherited-test-context")

        self.assertEqual(base, base_before)
        self.assertEqual(bridge._mcp_servers, mcp_before)
        for options, expected in (
            (first, "first-test-context"), (second, "second-test-context"),
            (retry, "first-test-context"), (absent, ""), (inherited, ""),
        ):
            self.assertEqual(options.env[MCP_TASK_TURN_CONTEXT_ENV], expected)
            self.assertIsNot(options.env, base)
        self.assertIsNot(first.env, retry.env)
        retry.env["KEEP"] = "retry-only"
        self.assertEqual(first.env["KEEP"], "unchanged")
        self.assertEqual(first.mcp_servers["bawthub"]["headers"], {
            codec.TASK_TURN_CONTEXT_HEADER: "first-test-context",
        })
        self.assertNotIn("headers", absent.mcp_servers["bawthub"])


if __name__ == "__main__":
    unittest.main()
