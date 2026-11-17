// House preview preload (TASK-2380): key every llm-bawt request a preview's
// Node processes make, whatever branch the preview serves. Branches based
// before BawtHub's own wrapper (frontend/server/installLlmApiKey.ts) otherwise
// call llm-bawt anonymously. The supervisor loads this through NODE_OPTIONS.
//
// Same contract as BawtHub's wrapper: the key is re-read from
// LLM_BAWT_API_KEY_FILE on every call, only llm-bawt origins get it, a header
// the caller already set is kept, and a keyed request never follows a
// redirect on its own (fetch would copy the key to the new destination).
import { readFileSync } from "node:fs";

const HEADER = "X-LLM-Bawt-Key";

function readKey() {
  const file = process.env.LLM_BAWT_API_KEY_FILE?.trim();
  if (!file) return null;
  try {
    return readFileSync(file, "utf8").trim() || null;
  } catch {
    return null;
  }
}

function origins() {
  const out = new Set();
  for (const url of [process.env.BAWTHUB_LLM_URL, process.env.BACKEND_URL,
    "http://127.0.0.1:8642", "http://host.docker.internal:8642"]) {
    if (!url) continue;
    try { out.add(new URL(url).origin); } catch { /* unmatched */ }
  }
  return out;
}

function urlOf(input) {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.href;
  return input?.url ?? null;
}

const inner = globalThis.fetch;
if (typeof inner === "function" && !inner.__houseLlmKey) {
  const llm = origins();
  const keyed = async (input, init) => {
    let origin = null;
    try { origin = new URL(urlOf(input)).origin; } catch { /* not llm-bawt */ }
    if (!origin || !llm.has(origin)) return inner(input, init);
    const request = new Request(input, init);
    // A caller-set key is kept but confined exactly like the file's key.
    const key = request.headers.get(HEADER) || readKey();
    if (!key) return inner(request);
    const headers = new Headers(request.headers);
    headers.set(HEADER, key);
    if (request.redirect === "manual") return inner(new Request(request, { headers }));
    // Follow redirects one hop at a time, re-keyed only while they stay on
    // llm-bawt; a bodied request or a hop elsewhere is refused.
    let current = new Request(request, { headers, redirect: "manual" });
    const bodied = request.body !== null;
    for (let hop = 0; hop <= 5; hop += 1) {
      const response = await inner(current);
      if (![301, 302, 303, 307, 308].includes(response.status)) return response;
      if (request.redirect === "error") throw new TypeError("fetch failed: redirect");
      const location = response.headers.get("location");
      const next = location ? new URL(location, current.url) : null;
      if (!next || !llm.has(next.origin) || bodied) throw new TypeError("fetch failed: keyed redirect refused");
      current = new Request(next, { method: request.method, headers, redirect: "manual", signal: request.signal });
    }
    throw new TypeError("fetch failed: too many redirects");
  };
  keyed.__houseLlmKey = true;
  globalThis.fetch = keyed;
}
