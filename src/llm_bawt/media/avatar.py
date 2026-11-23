"""Write-time bot-avatar resolution → a self-hosted ``data:`` URL.

The chat UI must never depend on an external CDN at runtime — only LLM
calls may touch the internet. Bot avatars are therefore resolved ONCE,
when the avatar is assigned, and the result is stored inline on the
``bot_profiles.avatar_render`` column as a self-contained ``data:`` URL:

- **emoji**  → the Twemoji SVG, fetched here once, as ``data:image/svg+xml``
- **image**  → normalized + downscaled small WebP, as ``data:image/webp``

Callers that cannot wait for egress (including bot creation) opt out of
remote lookups. A skipped or failed lookup returns ``None`` so the frontend
can use the native emoji glyph, with no CDN dependency.
"""

from __future__ import annotations

import base64
import logging
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

#: Same Twemoji release the frontend used, so the look is unchanged.
TWEMOJI_CDN = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg"

#: Avatars are rendered tiny; cap hard so the inlined blob stays a few KB.
AVATAR_MAX = (96, 96)
AVATAR_WEBP_QUALITY = 82

_FETCH_TIMEOUT = 8  # socket timeout, not a whole-lookup deadline


def _is_emoji(s: str) -> bool:
    """True for an emoji string (mirrors the frontend's isEmoji heuristic)."""
    if s.startswith(("http", "/", "data:")) or "." in s:
        return False
    return any(ord(c) > 0xFF for c in s)


def _emoji_codepoints(emoji: str) -> str:
    """Hex codepoints joined by '-', dropping VS-16 (fe0f).

    Matches ``emojiToTwemojiUrl`` in the frontend so we resolve the exact
    same asset the UI used to fetch.
    """
    return "-".join(f"{ord(c):x}" for c in emoji if f"{ord(c):x}" != "fe0f")


def _resolve_emoji(emoji: str) -> str | None:
    cp = _emoji_codepoints(emoji)
    if not cp:
        return None
    url = f"{TWEMOJI_CDN}/{cp}.svg"
    try:
        with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            svg = resp.read()
    except Exception as e:
        log.warning("avatar: twemoji fetch failed for %r (cp=%s): %s", emoji, cp, e)
        return None
    if not svg or b"<svg" not in svg[:512]:
        log.warning("avatar: twemoji response for %r was not an SVG", emoji)
        return None
    b64 = base64.b64encode(svg).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _read_image_bytes(value: str) -> bytes | None:
    """Fetch/decode raw image bytes from an http(s) URL or a data: URL."""
    try:
        if value.startswith("data:"):
            header, _, payload = value.partition(",")
            if ";base64" in header:
                return base64.b64decode(payload)
            return urllib.parse.unquote_to_bytes(payload)
        if value.startswith("http"):
            with urllib.request.urlopen(value, timeout=_FETCH_TIMEOUT) as resp:
                if getattr(resp, "status", 200) != 200:
                    return None
                return resp.read()
    except Exception as e:
        log.warning("avatar: image fetch failed for %r: %s", value[:80], e)
    return None


def _resolve_image(value: str) -> str | None:
    raw = _read_image_bytes(value)
    if not raw:
        return None
    try:
        # Reuse the chat-upload normalization pipeline: strip metadata,
        # collapse exotic color modes, then downscale to a tiny avatar.
        from .store import _encode_variant, _normalize_to_original

        _, normalized, _, _ = _normalize_to_original(raw)
        webp = _encode_variant(normalized, AVATAR_MAX, AVATAR_WEBP_QUALITY)
    except Exception as e:
        log.warning("avatar: image normalize failed: %s", e)
        return None
    b64 = base64.b64encode(webp).decode("ascii")
    return f"data:image/webp;base64,{b64}"


def resolve_avatar_render(avatar: str | None, *, allow_remote: bool = True) -> str | None:
    """Resolve a raw avatar value to a self-contained ``data:`` URL, or None.

    - Emoji → Twemoji SVG data URL if remote lookups are allowed.
    - data: image (or http(s) image if allowed) → normalized small WebP data URL.
    - Plain text (initials) or unresolvable input → ``None`` (frontend draws
      the native glyph / letter / gradient).
    """
    if not avatar:
        return None
    a = avatar.strip()
    if not a:
        return None
    if a.startswith("data:image/") or (allow_remote and a.startswith("http")):
        return _resolve_image(a)
    if allow_remote and _is_emoji(a):
        return _resolve_emoji(a)
    # Same-origin relative path ("/...") is already self-hosted; leave it to
    # the frontend. Plain short text has no render asset.
    return None
