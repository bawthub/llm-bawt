"""Opt-in live smoke: run with the llm-bawt Python environment (requires mcp).

Never runs fixture code. Not collected by pytest and never downloads dependencies.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import subprocess

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parents[1] / "tests/fixtures/rea-javascript-app"


def snapshot() -> dict[str, str]:
    return {str(p.relative_to(FIXTURE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in FIXTURE.rglob("*") if p.is_file()}


def validate(evidence: dict) -> str:
    assert evidence["operation"] == "analyze_javascript_application"
    result = evidence["normalized_result"]
    assert result["statistics"]["parse_failures"] == 0
    assert result["statistics"]["parsed_javascript_files"] == 2
    edges = result["graph"]["edges"]
    assert any(edge["relation"] == "imports"
               and edge["properties"].get("resolved_path") == "feature.js"
               and edge["properties"].get("imported_name") == "describeFeature"
               and edge["evidence"]["location"]["value"].get("source") == "index.js"
               for edge in edges)
    return evidence["evidence_id"]


def payload(result) -> dict:
    return json.loads(next(item.text for item in result.content if item.type == "text"))


async def main() -> None:
    before = snapshot()
    assert "EXECUTED" not in before, "Fixture already executed: investigate before smoke"
    cli = subprocess.run([str(HERE / "rea"), "analyze-javascript-application", str(FIXTURE), "--json"],
                         text=True, capture_output=True, check=True, timeout=60)
    cli_evidence = validate(json.loads(cli.stdout))
    async with asyncio.timeout(90):
        params = StdioServerParameters(command=str(HERE / "rea"), args=["mcp"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                info = await session.initialize()
                catalog = await session.list_tools()
                tool = next(t for t in catalog.tools if t.name == "analyze_javascript_application")
                assert "input_path" in tool.input_schema["required"]
                summary = await session.call_tool(tool.name, {"input_path": str(FIXTURE), "detail": "summary"})
                assert not summary.is_error
                parent = payload(summary)["normalized_result"]["parent_evidence_id"]
                complete = await session.call_tool(tool.name, {"input_path": str(FIXTURE), "detail": "complete"})
                assert not complete.is_error
                mcp_evidence = validate(payload(complete))
                assert parent == mcp_evidence == cli_evidence
                missing_path = FIXTURE / "not-a-real-input"
                assert not missing_path.exists()
                missing = await session.call_tool(tool.name, {"input_path": str(missing_path), "detail": "summary"})
                assert missing.is_error
                assert payload(missing)["error"]["category"] == "invalid_input"
                receipt = {"server": info.server_info.model_dump(), "tools": len(catalog.tools),
                           "cli_evidence": cli_evidence, "mcp_evidence": mcp_evidence,
                           "verified_relation": "index.js imports describeFeature from feature.js",
                           "missing_input": "explicit invalid_input error", "fixture_executed": False}
    assert snapshot() == before, "Static analysis mutated the fixture"
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
