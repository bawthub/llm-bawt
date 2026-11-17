"""TASK-2380: the house preview's llm-bawt key preload, run in real Node.

Two ephemeral local servers stand in for llm-bawt (allowlisted) and an
outside host; the preload must key only the first and never carry a key,
automatic or caller-set, across a redirect that leaves it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_bridge.house_preview_supervisor import LLM_KEY_PRELOAD

PROBE = r"""
import http from "node:http";
const seen = [];
function serve() {
  return new Promise((resolve) => {
    const s = http.createServer((q, res) => {
      const port = s.address().port;
      seen.push({ port, url: q.url, key: q.headers["x-llm-bawt-key"] ?? null });
      if (q.url === "/same") { res.writeHead(307, { location: "/ok" }); return res.end(); }
      if (q.url === "/away") { res.writeHead(302, { location: `http://127.0.0.1:${process.env.OUT}/ok` }); return res.end(); }
      res.end("ok");
    }).listen(Number(process.env.PORT || 0), "127.0.0.1", () => resolve(s));
  });
}
const llmPort = Number(process.env.LLM_PORT);
const llm = await serve.call(null, process.env.PORT = String(llmPort));
const out = await serve.call(null, process.env.PORT = "0");
process.env.OUT = String(out.address().port);
const L = `http://127.0.0.1:${llmPort}`, O = `http://127.0.0.1:${process.env.OUT}`;
async function attempt(fn) { try { const r = await fn(); return r.status; } catch { return "refused"; } }
const r = {};
r.auto = await attempt(() => fetch(`${L}/ok`));
r.explicit = await attempt(() => fetch(`${L}/ok`, { headers: { "X-LLM-Bawt-Key": "caller" } }));
r.other = await attempt(() => fetch(`${O}/ok`));
r.same = await attempt(() => fetch(`${L}/same`));
r.autoAway = await attempt(() => fetch(`${L}/away`));
r.explicitAway = await attempt(() => fetch(`${L}/away`, { headers: { "X-LLM-Bawt-Key": "caller" } }));
r.errorMode = await attempt(() => fetch(`${L}/same`, { redirect: "error" }));
r.manual = await attempt(() => fetch(`${L}/away`, { redirect: "manual" }));
r.post = await attempt(() => fetch(`${L}/ok`, { method: "POST", body: "{}" }));
// A branch wrapper stacked on top: sets its own key, takes hops manually.
r.stacked = await attempt(() => fetch(`${L}/away`, { headers: { "X-LLM-Bawt-Key": "branch" }, redirect: "manual" }));
console.log(JSON.stringify({ r, seen, out: Number(process.env.OUT) }));
llm.close(); out.close();
"""


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(tmp_path: Path, key_file: str | None) -> dict:
    node = shutil.which("node")
    assert node, "node is required for the house preview preload test"
    probe = tmp_path / "probe.mjs"
    probe.write_text(PROBE)
    port = _free_port()
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LLM_PORT": str(port),
           "BAWTHUB_LLM_URL": f"http://127.0.0.1:{port}"}
    if key_file:
        env["LLM_BAWT_API_KEY_FILE"] = key_file
    done = subprocess.run([node, f"--import={LLM_KEY_PRELOAD.as_uri()}", str(probe)], env=env,
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def _keys_at(result: dict, port: int, url: str) -> list:
    return [s["key"] for s in result["seen"] if s["port"] == port and s["url"] == url]


@pytest.mark.parametrize("with_file", [True, False])
def test_preload_keys_llm_bawt_only_and_confines_redirects(tmp_path, with_file):
    key_file = None
    if with_file:
        (tmp_path / "key").write_text("filekey\n")
        key_file = str(tmp_path / "key")
    res = _run(tmp_path, key_file)
    r, out = res["r"], res["out"]
    assert r["explicitAway"] == "refused"  # the blocker: caller key never leaves
    assert r["errorMode"] == "refused"
    assert r["manual"] == 302 and r["stacked"] == 302  # not followed by the preload
    assert r["auto"] == r["explicit"] == r["other"] == r["same"] == r["post"] == 200
    # The outside server never sees any key.
    assert all(s["key"] is None for s in res["seen"] if s["port"] == out)
    if with_file:
        assert r["autoAway"] == "refused"
        assert "filekey" in _keys_at(res, res["seen"][0]["port"], "/ok")
        assert "caller" in _keys_at(res, res["seen"][0]["port"], "/ok")
        assert "branch" in _keys_at(res, res["seen"][0]["port"], "/away")
    else:
        # No key at all: an unkeyed request behaves like plain fetch.
        assert r["autoAway"] == 200
