"""OpenAI-only compatibility for tool schemas, not tool input validation."""

from typing import Any

_SCHEMA_MAPS = {"$defs", "definitions", "properties", "patternProperties", "dependentSchemas", "dependencies"}
_SCHEMA_LISTS = {"oneOf", "anyOf", "allOf", "prefixItems"}
_SCHEMA_VALUES = {
    "additionalProperties", "additionalItems", "contains", "propertyNames",
    "unevaluatedItems", "unevaluatedProperties", "not", "if", "then", "else",
    "contentSchema",
}


def _normalize_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema  # Boolean schemas and non-schema values in legacy dependencies.

    normalized = dict(schema)
    for key, value in schema.items():
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            normalized[key] = {name: _normalize_schema(sub) for name, sub in value.items()}
        elif (key in _SCHEMA_LISTS or key == "items") and isinstance(value, list):
            normalized[key] = [_normalize_schema(sub) for sub in value]
        elif key in _SCHEMA_VALUES or key == "items":
            normalized[key] = _normalize_schema(value)

    if "oneOf" in normalized:
        branches = normalized.pop("oneOf")
        if "anyOf" in normalized:
            # Sibling keywords are conjunctive. Don't replace the existing
            # anyOf or combine the two branch lists into one wider union.
            normalized["allOf"] = [*(normalized.get("allOf") or []), {"anyOf": branches}]
        else:
            normalized["anyOf"] = branches
    return normalized


def normalize_openai_tool_schemas(body: dict) -> None:
    """Approximate oneOf as anyOf for decoding, leaving the SDK schema intact.

    Only walk schema positions: property/definition names and instance data
    in defaults, examples or enums may themselves contain the string "oneOf".
    """
    for tool in body.get("tools", []):
        if tool.get("type") == "function" and "parameters" in tool:
            tool["parameters"] = _normalize_schema(tool["parameters"])
