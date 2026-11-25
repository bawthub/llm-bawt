"""Persist image-bearing tool results to the media store (shared by bridges).

The Claude bridge (TASK-483/668) and the Codex bridge (TASK-1324) both upload
MCP image results — Playwright screenshots, generated images — through the
app's ``/v1/uploads`` endpoint and attach the resulting ``{asset_id, kind}``
refs to the tool card and the turn's reply.
"""

from __future__ import annotations

import logging

from agent_bridge.user_scope import session_user

logger = logging.getLogger("agent_bridge.tool_images")


class ToolImageMixin:
    """Image extraction and upload helpers; needs ``self._app_api_url``."""

    #: Tool tails whose image-bearing results we persist to the media store so
    #: the app can attach them to the reply (browsable per turn + inline card).
    #: - browser_take_screenshot: Playwright screenshots.
    #: - generate_image: Grok Imagine output (TASK-599). The tool already stored
    #:   the identical raw bytes, so this re-upload dedups to the same asset.
    #: - inspect_bawthub_tasks: read-only Tasks inspection snapshot (TASK-2642);
    #:   without this the user sees only a placeholder, never the picture.
    _IMAGE_RESULT_TOOL_TAILS = frozenset({
        "browser_take_screenshot", "generate_image", "inspect_bawthub_tasks",
    })

    @classmethod
    def _is_image_result_tool(cls, tool_name: str | None) -> bool:
        """True for tools whose result carries an image we should persist
        (namespaced ``mcp__x__name`` or bare)."""
        if not tool_name:
            return False
        return tool_name.split("__")[-1] in cls._IMAGE_RESULT_TOOL_TAILS

    @staticmethod
    def _extract_image_block(block: object) -> tuple[str, str] | None:
        """Pull (base64_data, mime) from a tool-result image content block.

        Handles both the MCP shape ``{type:image, data, mimeType}`` and the
        Anthropic API shape ``{type:image, source:{data, media_type}}``.
        """
        if not isinstance(block, dict) or block.get("type") != "image":
            return None
        data = block.get("data")
        mime = block.get("mimeType") or block.get("mime_type")
        if not data:
            src = block.get("source")
            if isinstance(src, dict):
                data = src.get("data")
                mime = mime or src.get("media_type")
        if not data:
            return None
        return data, (mime or "image/png")

    @staticmethod
    def _artifact_refs(artifacts: list[dict]) -> list[dict]:
        """Reduce canonical upload envelopes to persisted attachment refs."""
        refs: list[dict] = []
        seen: set[str] = set()
        for artifact in artifacts:
            asset_id = artifact.get("asset_id") if isinstance(artifact, dict) else None
            if isinstance(asset_id, str) and asset_id and asset_id not in seen:
                seen.add(asset_id)
                refs.append({"asset_id": asset_id, "kind": "image"})
        return refs

    def _absolute_artifact_url(self, path: object) -> str | None:
        if not isinstance(path, str) or not path:
            return None
        if path.startswith(("http://", "https://")):
            return path
        if not self._app_api_url:
            return path
        return f"{self._app_api_url.rstrip('/')}/{path.lstrip('/')}"

    def _format_screenshot_artifacts(self, artifacts: list[dict]) -> str:
        """Render canonical upload envelopes as model-visible curlable URLs."""
        lines = [
            "[Playwright Screenshot Artifact] The screenshot is stored durably. "
            "Use these URLs with curl if you need the raw artifact; paste the "
            "`public` link if the user should be able to open it:",
        ]
        rendered = 0
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not artifact.get("asset_id"):
                continue
            rendered += 1
            meta = [
                f"asset_id={artifact['asset_id']}",
                f"type={artifact.get('mime_type') or 'image'}",
            ]
            if artifact.get("width") and artifact.get("height"):
                meta.append(f"{artifact['width']}x{artifact['height']}")
            lines.append(f"{rendered}. " + "  ".join(meta))
            urls = artifact.get("urls") if isinstance(artifact.get("urls"), dict) else {}
            for label in ("original", "preview", "thumb"):
                absolute = self._absolute_artifact_url(urls.get(label))
                if absolute:
                    lines.append(f"   {label}: {absolute}")
            # TASK-847: the user-facing BawtHub link. Only this one is
            # clickable by the user — the URLs above are internal.
            public_url = artifact.get("public_url")
            if isinstance(public_url, str) and public_url:
                lines.append(f"   public (share with user): {public_url}")
        return "\n".join(lines) if rendered else ""

    async def _persist_screenshot_artifacts(
        self, content: list, session_key: str, tool_use_id: str | None,
    ) -> list[dict]:
        """Upload image blocks and return canonical upload-response envelopes."""
        artifacts: list[dict] = []
        if not self._app_api_url:
            return artifacts
        user_id = session_user(session_key)
        for block in content:
            img = self._extract_image_block(block)
            if img is None:
                if isinstance(block, dict) and block.get("type") == "image":
                    logger.warning(
                        "Screenshot image block in unrecognised shape (keys=%s)",
                        list(block.keys()),
                    )
                continue
            data_b64, mime = img
            artifact = await self._upload_data_url(
                f"data:{mime};base64,{data_b64}", user_id, tool_use_id,
            )
            if artifact:
                artifacts.append(artifact)
        return artifacts

    async def _upload_data_url(
        self, data_url: str, user_id: str, tool_use_id: str | None,
    ) -> dict | None:
        """POST image bytes and return the canonical upload-response envelope."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{self._app_api_url}/v1/uploads",
                    params={"source": "agent_attachment"},
                    headers={"X-Entity-Id": user_id},
                    json={
                        "data_url": data_url,
                        "filename": f"image-{tool_use_id or 'shot'}.png",
                    },
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "Screenshot upload failed: %s %s",
                        resp.status_code, resp.text[:200],
                    )
                    return None
                payload = resp.json() or {}
                if not isinstance(payload, dict) or not payload.get("asset_id"):
                    logger.warning("Screenshot upload returned no asset_id")
                    return None
                return payload
        except Exception:
            logger.warning("Screenshot upload error", exc_info=True)
            return None
