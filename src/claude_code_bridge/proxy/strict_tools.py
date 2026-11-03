"""Per-tool OpenAI ``strict`` function schemas for the Responses path.

TASK-1966: non-strict JSON function calls on gpt-6.x occasionally degenerate
into an endless run of empty/whitespace argument deltas after a short burst of
real arguments. Strict mode (constrained decoding against the schema) is meant
to bound that. It is opt-in per catalog endpoint (``responses_strict_tools``)
and applied per tool: a schema converts only when strict mode can express it
without loosening what the CLI accepts; anything else stays non-strict.

Conversion (OpenAI structured-outputs subset):

* every object gets ``additionalProperties: false`` and lists all of its
  properties in ``required``;
* a property that was optional becomes ``anyOf: [<schema>, {"type": "null"}]``
  so the model can still "omit" it by sending ``null``;
* annotation-only keywords (``default``, ``examples``, ``$schema`` …) and
  validation keywords outside the strict subset are dropped. The CLI validates
  the returned input against its own schema, so dropping a constraint only
  loosens decoding, never execution;
* ``oneOf`` becomes ``anyOf`` (equivalent for decoding).

Not convertible (tool stays non-strict): unions with an object branch (the
null cleanup could not tell which branch was chosen), open maps (an object without
``properties`` or with ``additionalProperties`` other than ``false``),
``allOf``/``not``/conditionals/``patternProperties``/``$ref``, a non-object
root, nesting deeper than OpenAI's limit, or too many properties.

On the way back, :func:`strip_strict_nulls` removes the ``null`` the model
sent for a field that was optional in the ORIGINAL schema, so the CLI sees the
field as absent — exactly as if the tool had been non-strict.
"""

from __future__ import annotations

import copy
from typing import Any

STRICT_TOOLS_HEADER = "X-LLM-Bawt-Strict-Tools"

_MAX_DEPTH = 10
_MAX_PROPERTIES = 5000

# Keywords strict mode understands; everything else is dropped or blocks.
_KEEP = frozenset({
    "type", "description", "properties", "required", "additionalProperties",
    "items", "enum", "const", "anyOf",
    "pattern", "format", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "multipleOf", "minItems", "maxItems",
})
_BLOCKING = frozenset({
    "allOf", "not", "if", "then", "else", "patternProperties",
    "dependentRequired", "dependentSchemas", "unevaluatedProperties",
    "unevaluatedItems", "$ref", "$defs", "definitions", "prefixItems",
    "contains", "propertyNames",
})
_STRICT_FORMATS = frozenset({
    "date-time", "time", "date", "duration", "email", "hostname",
    "ipv4", "ipv6", "uuid",
})


class _NotStrict(Exception):
    pass


def parse_strict_tools(value: object) -> bool:
    """Canonical boolean for the catalog field / proxy header (default off)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _allows_null(schema: dict) -> bool:
    t = schema.get("type")
    if t == "null" or (isinstance(t, list) and "null" in t):
        return True
    if "enum" in schema and None in (schema.get("enum") or []):
        return True
    return any(
        isinstance(b, dict) and _allows_null(b)
        for b in (schema.get("anyOf") or []) + (schema.get("oneOf") or [])
    )


def _is_object_schema(schema: dict) -> bool:
    t = schema.get("type")
    return (
        t == "object" or (isinstance(t, list) and "object" in t)
        or "properties" in schema or "additionalProperties" in schema
    )


def _convert(schema: Any, depth: int, counter: list[int]) -> dict:
    if not isinstance(schema, dict):
        raise _NotStrict("non-object schema node")
    if depth > _MAX_DEPTH:
        raise _NotStrict("too deep")
    if _BLOCKING & schema.keys():
        raise _NotStrict(f"keyword {sorted(_BLOCKING & schema.keys())[0]}")
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "oneOf":
            key = "anyOf"
        if key not in _KEEP:
            continue
        if key == "format" and value not in _STRICT_FORMATS:
            continue
        out[key] = value

    branches = out.get("anyOf")
    if branches is not None:
        if not isinstance(branches, list) or not branches:
            raise _NotStrict("bad anyOf")
        if any(isinstance(b, dict) and _is_object_schema(b) for b in branches):
            # Null cleanup on the way back cannot know which branch the model
            # chose, so object unions stay non-strict (TASK-1966 review).
            raise _NotStrict("object branch in union")
        out["anyOf"] = [_convert(b, depth + 1, counter) for b in branches]

    t = out.get("type")
    is_object = t == "object" or (isinstance(t, list) and "object" in t) or "properties" in out
    if is_object:
        props = out.get("properties")
        if out.get("additionalProperties", False) is not False:
            raise _NotStrict("open map")
        if props is None:
            # ``{"type":"object"}`` with nothing declared is an open map unless
            # it explicitly forbids extra keys. An explicit empty
            # ``properties`` is a no-argument tool and stays closed.
            if schema.get("additionalProperties") is not False:
                raise _NotStrict("open map")
            props = {}
        if not isinstance(props, dict):
            raise _NotStrict("bad properties")
        counter[0] += len(props)
        if counter[0] > _MAX_PROPERTIES:
            raise _NotStrict("too many properties")
        required = set(schema.get("required") or [])
        new_props: dict[str, Any] = {}
        for name, sub in props.items():
            converted = _convert(sub, depth + 1, counter)
            if name not in required and not _allows_null(converted):
                converted = {"anyOf": [converted, {"type": "null"}]}
            new_props[name] = converted
        out["properties"] = new_props
        out["required"] = list(new_props)
        out["additionalProperties"] = False
        if "type" not in out and "anyOf" not in out:
            out["type"] = "object"

    if "items" in out:
        out["items"] = _convert(out["items"], depth + 1, counter)
    return out


def to_strict_parameters(schema: Any) -> dict | None:
    """Return a strict-mode copy of a tool's input schema, or ``None``."""
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        return None
    try:
        converted = _convert(copy.deepcopy(schema), 1, [0])
    except _NotStrict:
        return None
    if "anyOf" in converted:
        return None  # strict roots must be a plain object
    converted["type"] = "object"
    return converted


def strip_strict_nulls(value: Any, schema: Any) -> Any:
    """Drop ``null`` for fields optional in the ORIGINAL schema, recursively.

    Only the ``null`` the converter itself made possible is removed. A
    ``null`` on a required field, or on an optional field whose original
    schema already allows ``null`` (e.g. an explicit "clear"), is kept: the
    model sent it on purpose and the CLI's own validation decides.
    """
    if not isinstance(schema, dict):
        return value
    if isinstance(value, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return value
        required = set(schema.get("required") or [])
        out: dict[str, Any] = {}
        for key, item in value.items():
            sub = props.get(key)
            if (item is None and key in props and key not in required
                    and isinstance(sub, dict) and not _allows_null(sub)):
                continue
            out[key] = strip_strict_nulls(item, sub)
        return out
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [strip_strict_nulls(item, schema["items"]) for item in value]
    return value
