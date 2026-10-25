"""Read-only effective configuration inspection for a resolved bot."""

from __future__ import annotations

from typing import Any


def describe_effective_config(owner: Any, prompt: str = "") -> dict[str, Any]:
    """Describe the same prompt/configuration path a live turn consumes."""
    is_agent = owner.model_definition.get("type") in (
        "agent_backend",
        "claude-code",
    )
    bot_type = "agent" if is_agent else "chat"

    builder = owner._assemble_system_builder(prompt)
    full_prompt = builder.build()

    sections = []
    for section in builder.enabled_sections:
        metadata = section.metadata or {}
        sections.append(
            {
                "name": section.name,
                "position": section.position,
                "char_len": len(section.content),
                "source": metadata.get("source", "unknown"),
                "gate": metadata.get("gate", "unknown"),
                # TASK-1249: only when present, so existing sections stay byte-identical.
                **({"provenance": metadata["provenance"]} if metadata.get("provenance") else {}),
            }
        )

    from ..setting_definitions import SETTING_DEFINITIONS, setting_default

    setting_keys = [
        ("temperature", owner.config.TEMPERATURE),
        ("top_p", owner.config.TOP_P),
        ("max_output_tokens", setting_default("max_output_tokens", 4096)),
        ("history_tokens", setting_default("history_tokens", 12000)),
        ("summary_count", setting_default("summary_count", 5)),
        ("memory_n_results", 3),
        (
            "memory_min_relevance",
            getattr(owner.config, "MEMORY_MIN_RELEVANCE", None),
        ),
        ("agent_global_prompt_enabled", False),
    ]
    settings = []
    for key, fallback in setting_keys:
        resolved = owner.config_resolver.resolve_scalar(key, fallback=fallback)
        settings.append(
            {"key": key, "value": resolved.value, "source": resolved.source}
        )

    flags = []
    for key, definition in SETTING_DEFINITIONS.items():
        consumed = bot_type in definition.applies_to
        if key == "include_summaries":
            value, source = owner._include_summaries, "request_flag"
        else:
            resolved = owner.config_resolver.resolve_config_setting(key)
            value, source = resolved.value, resolved.source
        note = definition.help
        if not consumed:
            note = (
                f"NOT CONSUMED on this {bot_type} bot "
                f"(applies_to={list(definition.applies_to)}). {note}"
            )
        flags.append(
            {
                "key": key,
                "value": value,
                "source": source,
                "storage": definition.storage,
                "applies_to": list(definition.applies_to),
                "consumed": consumed,
                "label": definition.label,
                "note": note,
            }
        )
    flags.append(
        {
            "key": "tts_mode",
            "value": owner._tts_mode,
            "source": "request_flag",
            "storage": "request_flag",
            "applies_to": ["chat", "agent"],
            "consumed": True,
            "label": "TTS mode",
            "note": "chat: system-prompt TTS section; agent: user-message voice prefix.",
        }
    )

    downstream = []
    if is_agent:
        downstream = [
            {
                "name": "runtime_context_model_block",
                "appended_by": "agent_backends/claude_code.py",
                "order": 1,
                "note": "<runtime-context> model id block, prepended by the backend.",
            }
        ]
        model_definition = owner.model_definition
        endpoint_id = getattr(owner.bot, "endpoint_id", None)
        harness = getattr(owner.bot, "harness", None)
        if endpoint_id is not None and harness:
            try:
                current = owner.config.resolve_model(
                    endpoint_id,
                    harness=harness,
                    default=model_definition,
                )
                if isinstance(current, dict):
                    model_definition = current
            except Exception:
                pass
        provider_system_prompt = model_definition.get("provider_system_prompt")
        if isinstance(provider_system_prompt, str) and provider_system_prompt:
            downstream.append(
                {
                    "name": "provider_system_prompt",
                    "appended_by": "agent_backends/claude_code.py",
                    "order": 3,
                    "source": (
                        f"access_path:{model_definition.get('access_path', 'unknown')}"
                    ),
                    "harness": model_definition.get("harness"),
                    "char_len": len(provider_system_prompt),
                    "note": (
                        "Harness-scoped provider instructions appended after "
                        "the bot prompt."
                    ),
                }
            )
        downstream.append(
            {
                "name": "mcp_tool_context_block",
                "appended_by": "claude_code_bridge/bridge.py",
                "order": 4,
                "note": "## MCP Tool Context (bot_id + entity-id guidance).",
            }
        )

    return {
        "bot_id": owner.bot_id,
        "user_id": owner.user_id,
        "bot_type": bot_type,
        "model_alias": owner.resolved_model_alias,
        "model_type": owner.model_definition.get("type", "unknown"),
        "prompt": {
            "total_chars": len(full_prompt),
            "section_count": len(sections),
            "sections": sections,
        },
        "settings": settings,
        "flags": flags,
        "downstream_augmentation": downstream,
    }
