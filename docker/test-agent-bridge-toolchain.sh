#!/bin/sh
# TASK-1061: run at image build and offline in a mount-free candidate container.
# This must not start a bridge, connect to Redis, or use an agent credential.
set -eu

test "$(id -un)" = bridge
test "$(node --version)" = v24.21.0
test "$(npm --version)" = 11.19.0
test "$(npx --version)" = 11.19.0
test "$(corepack --version)" = 0.36.0
# Do not auto-update/download package managers during validation.
export COREPACK_ENABLE_NETWORK=0
test "$(pnpm --version)" = 10.32.1
for tool in node npm npx corepack pnpm; do
    test "$(command -v "$tool")" = "/usr/local/bin/$tool"
done
test ! -e /usr/bin/node
test ! -e /usr/bin/npm
node --input-type=module -e '
import assert from "node:assert/strict";
assert.equal(typeof fetch, "function");
assert.equal(typeof WebSocket, "function");
assert.equal((await import("node:crypto")).createHash("sha256").update("bridge").digest("hex").length, 64);
console.log("Node " + process.version + ": ESM, fetch, WebSocket, crypto OK");
'
uv --version
python - <<'PY'
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

assert sys.version_info[:2] == (3, 12), sys.version
assert sys.executable == "/app/.venv/bin/python", sys.executable
for name in (
    "redis", "sqlalchemy", "psycopg2", "websockets", "httpx", "anyio",
    "claude_agent_sdk", "openai_codex_sdk", "openai_codex", "openai",
    "fastapi", "uvicorn", "pydantic", "PIL", "yaml", "tomlkit",
    "claude_code_bridge.bridge", "codex_bridge.bridge",
):
    importlib.import_module(name)
for package, suffix in (
    ("claude-agent-sdk", "claude_agent_sdk/_bundled/claude"),
    ("openai-codex-cli-bin", "codex_cli_bin/bin/codex"),
):
    dist = importlib.metadata.distribution(package)
    cli = Path(dist.locate_file(suffix))
    assert cli.is_file(), cli
    subprocess.run([str(cli), "--version"], check=True, timeout=30)
    print(f"{package}={dist.version}", flush=True)
for line in Path("/app/agent-bridge-constraints.txt").read_text().splitlines():
    if line and not line.startswith("#"):
        package, version = line.split("==", 1)
        assert importlib.metadata.version(package) == version, line
with tempfile.TemporaryDirectory() as directory:
    Path(directory, "package.json").write_text(json.dumps({
        "private": True,
        "packageManager": "pnpm@10.32.1",
        "scripts": {"smoke": "node --version"},
    }))
    for command in (
        ["npx", "--offline", "--", "node", "--version"],
        ["pnpm", "exec", "node", "--version"],
        ["pnpm", "run", "smoke"],
    ):
        subprocess.run(command, cwd=directory, check=True, timeout=30)
print("Python bridge imports, pinned dependencies, harnesses, npx/pnpm execution OK", flush=True)
PY
llm --help >/dev/null
printf '%s\n' 'Agent-bridge toolchain checks passed.'
