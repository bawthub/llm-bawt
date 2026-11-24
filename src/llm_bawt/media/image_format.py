"""Pure image-format helpers shared by attachment readers and generators.

Kept free of tool registration so attachment readers never import the image
generation module (which is omitted from the public text profile).
"""

from __future__ import annotations


def sniff_image_format(raw: bytes) -> str:
    """Best-effort image subtype from magic bytes ('png' | 'jpeg' | 'webp' | 'gif').

    Only used for declared MIME hints and the FastMCP ``Image`` format; the
    media store re-derives the real format from the bytes via Pillow regardless,
    so a wrong guess is cosmetic, never corrupting.
    """
    if raw[:8].startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "png"
