"""Tests for :mod:`llm_bawt.ops.validation` (TASK-639).

Verifies the minimal in-house JSON Schema validator:

* unknown properties are rejected by default, including when the schema omits
  ``additionalProperties: false``; only explicit true opens an object
* required properties are enforced
* type / enum / minLength / maxLength / pattern / minimum / maximum
* defaults fill absent declared keys but never introduce undeclared keys
"""

from __future__ import annotations

import json
import pytest

from llm_bawt.ops.validation import ArgValidationError, validate_args


def _schema(**over):
    body = {"type": "object", "additionalProperties": False, "properties": {}}
    body.update(over)
    return json.dumps(body)


def test_empty_schema_accepts_empty_args():
    assert validate_args({}, _schema()) == {}


@pytest.mark.parametrize("additional", [None, False])
def test_objects_remain_closed_by_default_at_every_depth(additional):
    nested = {"type": "object", "properties": {}}
    if additional is not None:
        nested["additionalProperties"] = additional
    with pytest.raises(ArgValidationError, match="unknown property 'unknown'"):
        validate_args({"child": {"unknown": 1}}, _schema(properties={"child": nested}))


def test_explicit_open_object_preserves_json_but_validates_declared_properties():
    nested = {"type": "object", "additionalProperties": True,
              "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    schema = _schema(properties={"child": nested})
    args = {"child": {"count": 1, "expected_runtime_sha": "a" * 40, "extra": {"items": [True, None]}}}
    assert validate_args(args, schema) == args
    with pytest.raises(ArgValidationError, match="expected integer"):
        validate_args({"child": {"count": "1"}}, schema)
    with pytest.raises(ArgValidationError, match="required property 'count'"):
        validate_args({"child": {"extra": True}}, schema)
    with pytest.raises(ArgValidationError, match="unknown property 'outside'"):
        validate_args({**args, "outside": True}, schema)


def test_unknown_property_is_rejected_even_when_schema_permits():
    # Even without additionalProperties=false in the schema, the validator
    # rejects unknown keys. TASK-639 invariant.
    permissive = json.dumps({
        "type": "object",
        # NOTE: no additionalProperties declared
        "properties": {"a": {"type": "string"}},
    })
    raised = False
    try:
        validate_args({"a": "x", "unknown": 1}, permissive)
    except ArgValidationError as exc:
        raised = "unknown property" in "; ".join(exc.violations)
    assert raised


def test_required_property_enforced():
    schema = _schema(
        properties={"target": {"type": "string"}},
        required=["target"],
    )
    raised = False
    try:
        validate_args({}, schema)
    except ArgValidationError as exc:
        raised = "required property 'target' missing" in "; ".join(exc.violations)
    assert raised


def test_type_mismatch_reported():
    schema = _schema(properties={"n": {"type": "integer"}})
    raised = False
    try:
        validate_args({"n": "not-a-number"}, schema)
    except ArgValidationError as exc:
        raised = "expected integer" in "; ".join(exc.violations)
    assert raised


def test_integer_type_rejects_boolean():
    # bool is a subclass of int in Python but JSON Schema treats them apart.
    schema = _schema(properties={"n": {"type": "integer"}})
    raised = False
    try:
        validate_args({"n": True}, schema)
    except ArgValidationError as exc:
        raised = "bool" in "; ".join(exc.violations)
    assert raised


def test_enum_enforced():
    schema = _schema(
        properties={
            "service": {"type": "string", "enum": ["a", "b", "c"]},
        }
    )
    raised = False
    try:
        validate_args({"service": "d"}, schema)
    except ArgValidationError as exc:
        raised = "not in enum" in "; ".join(exc.violations)
    assert raised
    # And a valid enum passes.
    assert validate_args({"service": "b"}, schema) == {"service": "b"}


def test_string_length_constraints():
    schema = _schema(
        properties={"s": {"type": "string", "minLength": 2, "maxLength": 4}}
    )
    raised = False
    try:
        validate_args({"s": "x"}, schema)
    except ArgValidationError as exc:
        raised = "below minLength" in "; ".join(exc.violations)
    assert raised
    raised = False
    try:
        validate_args({"s": "xxxxxx"}, schema)
    except ArgValidationError as exc:
        raised = "above maxLength" in "; ".join(exc.violations)
    assert raised
    assert validate_args({"s": "xxx"}, schema) == {"s": "xxx"}


def test_pattern_enforced():
    schema = _schema(properties={"tag": {"type": "string", "pattern": "^v\\d+$"}})
    raised = False
    try:
        validate_args({"tag": "abc"}, schema)
    except ArgValidationError as exc:
        raised = "does not match pattern" in "; ".join(exc.violations)
    assert raised
    assert validate_args({"tag": "v42"}, schema) == {"tag": "v42"}


def test_numeric_bounds():
    schema = _schema(properties={"n": {"type": "number", "minimum": 0, "maximum": 100}})
    raised = False
    try:
        validate_args({"n": -1}, schema)
    except ArgValidationError as exc:
        raised = "below minimum" in "; ".join(exc.violations)
    assert raised


def test_defaults_fill_declared_missing_keys_only():
    schema = _schema(properties={"level": {"type": "string"}, "n": {"type": "integer"}})
    defaults = json.dumps({"level": "info", "n": 3, "extra": "nope"})
    with pytest.raises(ArgValidationError, match="unknown property"):
        validate_args({"n": 5}, schema, defaults)
    assert validate_args({"n": 5}, schema, '{"level":"info","n":3}') == {"level": "info", "n": 5}


def test_multiple_violations_all_reported():
    schema = _schema(
        properties={
            "svc": {"type": "string", "enum": ["a", "b"]},
            "n": {"type": "integer"},
        },
        required=["svc", "n"],
    )
    caught = None
    try:
        validate_args({"svc": "z"}, schema)
    except ArgValidationError as exc:
        caught = exc
    assert caught is not None
    assert len(caught.violations) >= 2  # bad enum + missing required 'n'


@pytest.mark.parametrize("schema", [{"type": []}, {"type": {}}, {"properties": {"x": {"type": ["string", "null"]}}}])
def test_non_scalar_schema_type_rejected_as_validation_error(schema):
    with pytest.raises(ArgValidationError):
        validate_args({}, json.dumps(schema))


def test_non_dict_args_rejected():
    raised = False
    try:
        validate_args("nope", _schema())  # type: ignore[arg-type]
    except ArgValidationError:
        raised = True
    assert raised
