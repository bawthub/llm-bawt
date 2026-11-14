from __future__ import annotations

import json
from typing import Any

from .base import ToolCallRequest, ToolFormatHandler
from ...shared.output_sanitizer import strip_tool_protocol_leakage


ANY_JSON_TYPES = ("string", "number", "boolean", "array", "object")


def _json_type(param_type: str) -> tuple[str, dict | None]:
    mapping = {
        "string": "string",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "boolean": "boolean",
    }

    if param_type.startswith("list[") or param_type.startswith("array["):
        inner = param_type[param_type.find("[") + 1 : -1]
        item_type = mapping.get(inner, "string")
        return "array", {"type": item_type}

    json_type = mapping.get(param_type, "string")
    return json_type, None


def _load_json_payload(payload: str) -> dict[str, Any] | None:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    try:
        import json5  # type: ignore

        return json5.loads(payload)
    except Exception:
        return None


class NativeOpenAIFormatHandler(ToolFormatHandler):
    """Native OpenAI tools schema handling."""

    def get_system_prompt(self, tools: list) -> str:
        tool_names = []
        for tool in tools or []:
            name = getattr(tool, "name", None)
            if name:
                tool_names.append(name)
        tools_list = ", ".join(tool_names) if tool_names else "(none)"
        home_guidance = ""
        # Check if HA native tools are present (they start with "Hass" or "Get")
        ha_native_names = [n for n in tool_names if n.startswith("Hass") or n in ("GetLiveContext", "GetDateTime")]
        if ha_native_names:
            home_guidance = (
                "\nCRITICAL — Home Assistant device control:\n"
                "- You MUST call a Hass* tool for ANY smart home action (turn on/off, dim, open/close, set temperature, etc.).\n"
                "- NEVER say you performed a device action unless you ACTUALLY called a tool and received a result.\n"
                "- If the user asks to control a device, your FIRST response must be a tool call — not text.\n"
                "- After calling a tool, report the actual result: success/failure, which entities were affected, and any errors.\n"
                "- For multiple devices, call the tool once per device (e.g., two HassTurnOff calls for two lights).\n"
                "- Tools use friendly device NAMES (e.g., name='kitchen lights'), NOT entity IDs.\n"
                "- Use name, area, and/or floor parameters to target devices.\n"
                "- HassTurnOn opens covers/blinds. HassTurnOff closes them.\n"
                "- HassLightSet sets brightness (0-100), color, or color temperature.\n"
                "- HassSetPosition sets cover/blind position (0-100).\n"
                "- For device state queries, use GetLiveContext.\n"
                "- Execute tool calls immediately — do not describe what you plan to do.\n"
            )
        elif "home" in tool_names:
            home_guidance = (
                "Home tool guidance:\n"
                "- For natural names like 'sunroom lights', call home(action='query', pattern='sunroom', domain='light') first.\n"
                "- Use exact entity IDs returned by query in subsequent home(action='get'/'set') calls.\n"
                "- Covers/blinds use state='close' or state='open' (NOT 'off'/'on') with action='set'.\n"
                "- Lights and switches use state='on', 'off', or 'toggle'.\n"
                "- Locks use state='lock' or state='unlock'.\n"
                "- If a set/get call reports not found, run query and retry with the suggested exact ID.\n"
                "- If asked for current home status, call home(action='status') before answering.\n"
                "- If asked for 'raw output', return the exact tool output verbatim; do not invent or normalize JSON fields.\n"
                "- Execute tool calls immediately when the user confirms — do not describe what you're about to do without calling the tool.\n"
            )
        return (
            "## Tools\n\n"
            "You have access to tools. Use them when needed to answer the user.\n"
            f"Available tool names: {tools_list}\n\n"
            "The 'Available tool names' list above is authoritative for this turn.\n"
            "If asked to list tools, list exactly those names and do not omit any.\n\n"
            "Rules:\n"
            "1. Call a tool when you need missing or precise information.\n"
            "2. ALWAYS call a tool to perform real-world actions (device control, searches). NEVER say you did something without actually calling the tool.\n"
            "3. Do not invent tool results. Report what the tool actually returned.\n"
            "4. If no tool is needed, respond normally.\n"
            "5. If tool results contain URLs, copy each URL exactly as provided (no rewriting or guessing).\n"
            f"{home_guidance}"
        )

    def get_stop_sequences(self) -> list[str]:
        return []

    def get_tools_schema(self, tools: list) -> list[dict[str, Any]]:
        schema: list[dict[str, Any]] = []
        for tool in tools or []:
            if not hasattr(tool, "name"):
                continue
            properties: dict[str, Any] = {}
            required: list[str] = []
            for param in getattr(tool, "parameters", []) or []:
                if param.type == "any":
                    # Accept any JSON value (string, number, bool, list or object).
                    prop: dict[str, Any] = {
                        "type": list(ANY_JSON_TYPES),
                        "items": {},
                        "description": param.description,
                    }
                else:
                    json_type, items = _json_type(param.type)
                    prop = {
                        "type": json_type,
                        "description": param.description,
                    }
                    if items:
                        prop["items"] = items
                if param.default is not None:
                    prop["default"] = param.default
                properties[param.name] = prop
                if param.required:
                    required.append(param.name)

            schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )

        return schema

    def parse_response(self, response: str) -> tuple[list[ToolCallRequest], str]:
        if not response:
            return [], ""

        if isinstance(response, dict):
            tool_calls = response.get("tool_calls") or []
            content = response.get("content") or ""
            return self._parse_tool_calls(tool_calls), content

        return [], response.strip()

    def _parse_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolCallRequest]:
        calls: list[ToolCallRequest] = []
        for call in tool_calls or []:
            function = call.get("function") or {}
            name = function.get("name")
            args = function.get("arguments")
            parsed_args = {}
            if isinstance(args, str):
                payload = _load_json_payload(args)
                if isinstance(payload, dict):
                    parsed_args = payload
            elif isinstance(args, dict):
                parsed_args = args
            if name:
                calls.append(
                    ToolCallRequest(
                        name=name,
                        arguments=parsed_args,
                        tool_call_id=call.get("id"),
                        raw_text=None,
                    )
                )

        return calls

    def format_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if error:
            content = f"ERROR: {error}"
        elif isinstance(result, (dict, list)):
            content = json.dumps(result, indent=2, default=str)
        else:
            content = self._normalize_result(result)
        message = {"role": "tool", "content": content}
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        return message

    def sanitize_response(self, response: str) -> str:
        if not response:
            return ""
        return strip_tool_protocol_leakage(response)

    def _normalize_result(self, result: Any) -> str:
        result_str = str(result)
        if "<tool_result" in result_str:
            import re

            result_str = re.sub(r"<tool_result[^>]*>", "", result_str, flags=re.IGNORECASE)
            result_str = re.sub(r"</tool_result>", "", result_str, flags=re.IGNORECASE)
            result_str = re.sub(r"\\[IMPORTANT:.*?\\]", "", result_str, flags=re.DOTALL)
        return result_str.strip()
