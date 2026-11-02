"""Explicit Crawl4AI export/probe boundary; no settings writes or service actions.

Run only in the app process environment after an authorized generation update:
``python -m llm_bawt.service.crawl4ai_rekey export <expected-generation>``
``python -m llm_bawt.service.crawl4ai_rekey probe <previous> <expected>``
Arguments are NONSECRET counters. Credentials stay in memory/existing exports;
output is closed booleans only, including on errors. No automatic rollback.
"""
from __future__ import annotations

import contextlib
import hmac
import json
import logging
import os
import sys
from pathlib import Path

import httpx

from . import crawl4ai_token_export as tokens


def export(expected: int) -> dict[str, bool]:
    expected = tokens.validate_generation(expected)
    if tokens.current_generation() != expected:
        return {"ok": False, "generation_matches": False}
    # Snapshot once, both targets use exactly the same generation/root output.
    token = tokens.derive_for_generation(expected)
    results = tokens.export_crawl4ai_token(derive=lambda: token)
    matches = tokens.current_generation() == expected
    return {"ok": matches and all(results[t.slug] == "written" for t in tokens.TARGETS),
            "generation_matches": matches,
            **{t.slug: results[t.slug] == "written" for t in tokens.TARGETS}}


def probe(previous: int, expected: int, *, client: httpx.Client | None = None) -> dict[str, bool]:
    """Exact authenticated /schema checks, no crawling or response-body output."""
    previous = tokens.validate_generation(previous)
    expected = tokens.validate_generation(expected)
    if previous == expected or tokens.current_generation() != expected:
        return {"ok": False, "generation_matches": False}
    new = tokens.derive_for_generation(expected)
    old = tokens.derive_for_generation(previous)
    files_match = True
    for target in tokens.TARGETS:
        raw = (os.environ.get(target.path_env) or "").strip()
        if not raw:
            files_match = False
        else:
            value = Path(raw).read_text().strip()
            files_match = hmac.compare_digest(value, new) and files_match
    from llm_bawt.integrations.web_fetch.client import DEFAULT_CRAWL4AI_URL  # noqa: PLC0415

    base = (os.environ.get("LLM_BAWT_CRAWL4AI_URL") or DEFAULT_CRAWL4AI_URL).rstrip("/")
    # Never follow redirects with a bearer or inherit proxy environment settings.
    with (httpx.Client(follow_redirects=False, trust_env=False) if client is None
          else contextlib.nullcontext(client)) as http:
        new_status = http.get(base + "/schema", headers={"Authorization": f"Bearer {new}"},
                              timeout=15, follow_redirects=False).status_code
        old_status = http.get(base + "/schema", headers={"Authorization": f"Bearer {old}"},
                              timeout=15, follow_redirects=False).status_code
        anonymous_status = http.get(base + "/schema", timeout=15,
                                    follow_redirects=False).status_code
    matches = tokens.current_generation() == expected
    checks = {"generation_matches": matches, "exports_match": files_match,
              "new_accepted": new_status == 200, "old_rejected": old_status in (401, 403),
              "anonymous_rejected": anonymous_status in (401, 403)}
    return {"ok": all(checks.values()), **checks}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    result = {"ok": False}
    # Discard library output, including exception diagnostics. Do not redirect it
    # to stderr: stderr is also captured in ops/tool logs. No traceback on error.
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            try:
                if len(args) == 2 and args[0] == "export":
                    result = export(int(args[1]))
                elif len(args) == 3 and args[0] == "probe":
                    result = probe(int(args[1]), int(args[2]))
            except Exception:  # noqa: BLE001 - closed boolean failure, no secret exception text
                result = {"ok": False}
    finally:
        logging.disable(previous_logging)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
