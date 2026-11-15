"""Strict, deliberately bounded JSON Schema subset for the operations catalog.

Unsupported keywords are rejected, not silently ignored. Objects are closed
unless ``additionalProperties: true`` is explicit; ``{}`` at the root means
no arguments. No external references or dependencies.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any


class ArgValidationError(ValueError):
    def __init__(self, violations: list[str]):
        self.violations = list(violations)
        super().__init__("args validation failed: " + "; ".join(violations))


_TYPES = {"object": dict, "array": list, "string": str, "integer": int,
          "number": (int, float), "boolean": bool, "null": type(None)}
_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items",
             "enum", "minimum", "maximum", "minLength", "maxLength", "pattern",
             "minItems", "maxItems", "description", "title", "default", "x-sensitive"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _object_json(raw: str | dict, label: str) -> dict:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        canonical_json(value)
    except (ValueError, TypeError) as exc:
        raise ArgValidationError([f"{label} is not valid JSON: {exc}"]) from exc
    if not isinstance(value, dict):
        raise ArgValidationError([f"{label} must be an object"])
    return value


def _check_schema(schema: Any, path: str, *, root: bool = False) -> None:
    def fail(message):
        raise ArgValidationError([f"{path}: {message}"])
    if not isinstance(schema, dict):
        fail("schema must be an object")
    unknown = set(schema) - _KEYWORDS
    if unknown:
        fail(f"unsupported schema keywords: {sorted(unknown)}")
    kind = schema.get("type", "object" if root or "properties" in schema else None)
    if not isinstance(kind, str) or kind not in _TYPES:
        fail("schema type must be one supported type")
    if root and kind != "object":
        fail("root schema type must be object")
    for keywords, expected in (({"properties", "required", "additionalProperties"}, "object"),
                               ({"minLength", "maxLength", "pattern"}, "string"),
                               ({"minItems", "maxItems", "items"}, "array")):
        if keywords & schema.keys() and kind != expected:
            fail(f"{sorted(keywords & schema.keys())} requires {expected} type")
    if {"minimum", "maximum"} & schema.keys() and kind not in ("integer", "number"):
        fail("numeric bounds require integer or number type")
    for key in ("description", "title"):
        if key in schema and not isinstance(schema[key], str):
            fail(f"{key} must be a string")
    if "x-sensitive" in schema and type(schema["x-sensitive"]) is not bool:
        fail("x-sensitive must be boolean")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        fail("additionalProperties must be boolean")
    if "properties" in schema and kind != "object":
        fail("properties requires object type")
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        fail("properties must be an object")
    for key, sub in props.items():
        _check_schema(sub, f"{path}.{key}")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(x, str) or x not in props for x in required):
        fail("required must list declared property names")
    if len(set(required)) != len(required):
        fail("required contains duplicate names")
    if kind == "array":
        if "items" not in schema:
            fail("array schema requires items")
        _check_schema(schema["items"], path + "[]")
    elif "items" in schema:
        fail("items requires array type")
    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            fail(f"{key} must be a nonnegative integer")
    for key in ("minimum", "maximum"):
        if key in schema and (type(schema[key]) not in (int, float) or not math.isfinite(schema[key])):
            fail(f"{key} must be a finite number")
    for low, high in (("minimum", "maximum"), ("minLength", "maxLength"), ("minItems", "maxItems")):
        if low in schema and high in schema and schema[low] > schema[high]:
            fail(f"{low} exceeds {high}")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (re.error, TypeError):
            fail("pattern is not a valid regex")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        fail("enum must be a nonempty array")
    if "default" in schema:
        problems = _validate(schema["default"], schema, path + ".default")
        if problems:
            raise ArgValidationError(problems)


def _validate(value: Any, schema: dict, path: str, *, partial: bool = False) -> list[str]:
    kind = schema.get("type", "object")
    if not isinstance(value, _TYPES[kind]) or (kind in ("integer", "number") and isinstance(value, bool)):
        return [f"{path}: expected {kind}, got {type(value).__name__}"]
    errors = []
    if "enum" in schema and not any(canonical_json(value) == canonical_json(x) for x in schema["enum"]):
        errors.append(f"{path}: value not in enum")
    if kind == "object":
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not True:
            errors += [f"{path}: unknown property {k!r}" for k in value if k not in props]
        if not partial:
            errors += [f"{path}: required property {k!r} missing" for k in schema.get("required", []) if k not in value]
        for k in value.keys() & props.keys():
            errors += _validate(value[k], props[k], f"{path}.{k}")
    if kind == "array":
        for i, item in enumerate(value):
            errors += _validate(item, schema["items"], f"{path}[{i}]")
    for low, high, amount in (("minLength", "maxLength", len(value) if kind == "string" else None),
                              ("minItems", "maxItems", len(value) if kind == "array" else None),
                              ("minimum", "maximum", value if kind in ("integer", "number") else None)):
        if amount is not None:
            if low in schema and amount < schema[low]:
                errors.append(f"{path}: below {low}")
            if high in schema and amount > schema[high]:
                errors.append(f"{path}: above {high}")
    if kind == "string" and "pattern" in schema and not re.search(schema["pattern"], value):
        errors.append(f"{path}: does not match pattern")
    return errors


def validate_catalog(schema_json: str, defaults_json: str = "{}") -> tuple[dict, dict]:
    schema = _object_json(schema_json, "operation schema")
    defaults = _object_json(defaults_json, "operation defaults")
    _check_schema(schema, "schema", root=True)
    errors = _validate(defaults, schema, "defaults", partial=True)
    if errors:
        raise ArgValidationError(errors)
    return schema, defaults


def validate_args(args: dict[str, Any], schema_json: str, defaults_json: str | None = None) -> dict[str, Any]:
    schema, defaults = validate_catalog(schema_json, defaults_json if defaults_json is not None else "{}")
    if not isinstance(args, dict):
        raise ArgValidationError(["args must be an object"])
    supplied = _object_json(args, "args")
    merged = {**defaults, **supplied}
    errors = _validate(merged, schema, "args")
    if errors:
        raise ArgValidationError(errors)
    return json.loads(canonical_json(merged))
