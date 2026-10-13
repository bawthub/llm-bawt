"""CLI PreCompact command hook: fetch live guidance only without caller instructions."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable

from .compact_prompt_lookup import fetch_compact_prompt


def guidance_for_hook(
    data: object, *, lookup: Callable[[str | None], str | None] = fetch_compact_prompt,
    api_url: str | None = None,
) -> str:
    if not isinstance(data, dict) or data.get("trigger") not in {"auto", "manual"}:
        return ""
    custom = data.get("custom_instructions")
    if custom is not None and (not isinstance(custom, str) or custom.strip()):
        return ""
    return lookup(api_url if api_url is not None else os.environ.get("LLM_BAWT_API_URL")) or ""


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        data = json.load(sys.stdin)
    except (ValueError, OSError):
        return
    guidance = guidance_for_hook(data)
    if guidance:
        print(guidance)


if __name__ == "__main__":
    main()
