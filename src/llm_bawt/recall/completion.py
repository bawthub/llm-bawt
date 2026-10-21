"""One isolated utility completion; no bot prompt, transcript, tools or writes."""
from __future__ import annotations

from dataclasses import asdict

import httpx

from .budgets import compact_json, tokens
from .contracts import MAX_INPUT_TOKENS, MAX_MODEL_BYTES, Evidence, RecallError, Synthesizer
from .http import BoundedHTTP
from .prompt import resolve_recall_system_prompt


def synthesis_prompt(query: str, evidence: list[Evidence]) -> str:
    return compact_json({"question": query, "untrusted_evidence": [asdict(item) for item in evidence]})


def input_tokens(system: str, query: str, evidence: list[Evidence]) -> int:
    return tokens(system) + tokens(synthesis_prompt(query, evidence))


class UtilitySynthesizer(Synthesizer):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def system_prompt(self, bot_id: str) -> str:
        return resolve_recall_system_prompt(bot_id)

    async def synthesize(self, system: str, query: str, evidence: list[Evidence],
                         output_tokens: int) -> str:
        if input_tokens(system, query, evidence) > MAX_INPUT_TOKENS:
            raise RecallError("input_budget")
        body = {"system": system, "prompt": synthesis_prompt(query, evidence),
                "max_tokens": min(output_tokens, 1600)}
        async with httpx.AsyncClient(timeout=22.0, follow_redirects=False) as client:
            reader = BoundedHTTP(client, max_calls=1, max_bytes=MAX_MODEL_BYTES)
            try:
                result = await reader.json("POST", f"{self.base_url}/v1/llm/recall-complete",
                                           byte_limit=MAX_MODEL_BYTES, json=body)
            except RecallError as error:
                code = error.code if error.code in {"timeout", "busy"} else "model_unavailable"
                raise RecallError(code) from error
        if not isinstance(result, dict) or not isinstance(result.get("content"), str):
            raise RecallError("invalid_synthesis")
        return result["content"]
