"""Validate tool-call arguments against the tool's JSON schema.

Covers the JSON Schema subset tool definitions use: type (incl. lists of types), properties,
required, additionalProperties, items, enum, const, minimum/maximum (and exclusive),
minLength/maxLength, minItems/maxItems. Errors are short sentences a model can act on.
"""

from __future__ import annotations

from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _is(value: Any, type_name: str) -> bool:
    if type_name in ("integer", "number") and isinstance(value, bool):
        return False                    # JSON true is not a number
    if type_name == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, _TYPES.get(type_name, (object,)))


def validate(schema: dict | None, value: Any, path: str = "") -> list[str]:
    """Return a list of problems; empty when `value` satisfies `schema`."""
    if not schema:
        return []
    where = f"'{path}'" if path else "the arguments"
    errors: list[str] = []

    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        if not any(_is(value, n) for n in names):
            got = type(value).__name__ if value is not None else "null"
            return [f"{where} must be {' or '.join(names)}, got {got}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where} must be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{where} must be {schema['const']!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{where} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{where} must be <= {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{where} must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{where} must be < {schema['exclusiveMaximum']}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{where} must have at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{where} must have at most {schema['maxLength']} characters")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{where} must have at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{where} must have at most {schema['maxItems']} items")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                errors += validate(items, item, f"{path}[{i}]")

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"'{_join(path, name)}' is required")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in properties:
                errors += validate(properties[name], item, _join(path, name))
            elif extra is False:
                allowed = ", ".join(properties) or "none"
                errors.append(f"'{_join(path, name)}' is not a known argument (known: {allowed})")
            elif isinstance(extra, dict):
                errors += validate(extra, item, _join(path, name))
    return errors


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name
