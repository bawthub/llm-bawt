"""Fetch the current global compaction prompt from the app-owned registry.

No bridge-side DB access or cache: a saved edit must affect the next compact in
an existing SDK session. On an outage, compact without *added* instructions,
not with a possibly stale local copy that could contradict an edited prompt.
"""

from __future__ import annotations

import hashlib
import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

COMPACT_PROMPT_KEY = "agents.compaction_summary"
_TIMEOUT_SECONDS = 1.5
_MAX_RESPONSE_BYTES = 256 * 1024


def fetch_compact_prompt(api_url: str | None) -> str | None:
    """Return the latest registry body or None on bounded, logged failure."""
    if not api_url:
        logger.warning("Compaction prompt %s unavailable: app URL not configured", COMPACT_PROMPT_KEY)
        return None
    url = f"{api_url.rstrip('/')}/v1/prompts/{COMPACT_PROMPT_KEY}"
    try:
        request = Request(url, headers={"Accept": "application/json", "Cache-Control": "no-cache"})
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("prompt response too large")
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("key") != COMPACT_PROMPT_KEY:
            raise ValueError("unexpected prompt identity")
        body = data.get("body")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("missing or empty prompt body")
        source = data.get("source")
        if source not in ("code_default", "db_override"):
            raise ValueError("unexpected prompt source")
    except (HTTPError, URLError, OSError, ValueError, UnicodeError) as exc:
        logger.warning("Compaction prompt %s unavailable (%s); using CLI defaults",
                       COMPACT_PROMPT_KEY, type(exc).__name__)
        return None
    fingerprint = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    logger.info("Compaction prompt selected key=%s source=%s updated_at=%s sha256=%s",
                COMPACT_PROMPT_KEY, source, data.get("updated_at"), fingerprint)
    return body
