"""YAML validation for asset and scene definitions."""

import difflib
from typing import Any


# Known keys for asset YAML files
ASSET_KNOWN_KEYS = {
    "name", "origin", "size", "parts", "attachments", "room",
}

PART_KNOWN_KEYS = {
    "primitive", "size", "anchor", "offset", "material",
    "attach_to", "at", "from", "rotation", "bevel",
}

# Known keys for scene YAML files
SCENE_KNOWN_KEYS = {
    "name", "size", "slots", "place", "compose", "attachments",
}

SLOT_KNOWN_KEYS = {
    "anchor", "offset", "facing", "position",
}

PLACEMENT_KNOWN_KEYS = {
    "asset", "scene", "slot", "attach_to", "at",
}

# Known primitive types
KNOWN_PRIMITIVES = {"cube", "cylinder", "sphere", "cone", "plane", "room"}

# Known facing directions
KNOWN_FACINGS = {"center", "outward", "north", "south", "east", "west"}


def _suggest(value: str, known: set[str] | list[str]) -> str:
    """Suggest a close match for a misspelled value."""
    matches = difflib.get_close_matches(value, known, n=1, cutoff=0.5)
    if matches:
        return f" Did you mean '{matches[0]}'?"
    return ""


class ValidationError(Exception):
    """Raised when YAML validation fails."""
    pass


def validate_asset_yaml(data: dict[str, Any]) -> list[str]:
    """Validate an asset YAML definition.

    Returns a list of warning messages for unknown keys.
    Raises ValidationError for structural problems.
    """
    warnings = []

    if not isinstance(data, dict):
        raise ValidationError("Asset YAML must be a mapping")

    if "size" not in data:
        raise ValidationError("Asset YAML requires 'size' field")

    # Check top-level keys
    for key in data:
        if key not in ASSET_KNOWN_KEYS:
            msg = f"Unknown asset key '{key}'.{_suggest(key, ASSET_KNOWN_KEYS)}"
            warnings.append(msg)

    # Validate parts
    parts = data.get("parts", {})
    if parts and not isinstance(parts, dict):
        raise ValidationError("'parts' must be a mapping")

    for part_name, part_def in (parts or {}).items():
        if not isinstance(part_def, dict):
            raise ValidationError(f"Part '{part_name}' must be a mapping")

        if "primitive" not in part_def:
            raise ValidationError(f"Part '{part_name}' requires 'primitive' field")

        prim_type = part_def["primitive"]
        if prim_type not in KNOWN_PRIMITIVES:
            msg = f"Unknown primitive '{prim_type}' in part '{part_name}'.{_suggest(prim_type, KNOWN_PRIMITIVES)}"
            raise ValidationError(msg)

        if "size" not in part_def:
            raise ValidationError(f"Part '{part_name}' requires 'size' field")

        for key in part_def:
            if key not in PART_KNOWN_KEYS:
                msg = f"Unknown key '{key}' in part '{part_name}'.{_suggest(key, PART_KNOWN_KEYS)}"
                warnings.append(msg)

    return warnings


def validate_scene_yaml(data: dict[str, Any]) -> list[str]:
    """Validate a scene YAML definition.

    Returns a list of warning messages for unknown keys.
    Raises ValidationError for structural problems.
    """
    warnings = []

    if not isinstance(data, dict):
        raise ValidationError("Scene YAML must be a mapping")

    # Check top-level keys
    for key in data:
        if key not in SCENE_KNOWN_KEYS:
            msg = f"Unknown scene key '{key}'.{_suggest(key, SCENE_KNOWN_KEYS)}"
            warnings.append(msg)

    # Validate slots
    slots = data.get("slots", {})
    if slots and not isinstance(slots, dict):
        raise ValidationError("'slots' must be a mapping")

    for slot_name, slot_def in (slots or {}).items():
        if not isinstance(slot_def, dict):
            raise ValidationError(f"Slot '{slot_name}' must be a mapping")

        for key in slot_def:
            if key not in SLOT_KNOWN_KEYS:
                msg = f"Unknown key '{key}' in slot '{slot_name}'.{_suggest(key, SLOT_KNOWN_KEYS)}"
                warnings.append(msg)

        if "facing" in slot_def and slot_def["facing"] not in KNOWN_FACINGS:
            facing = slot_def["facing"]
            msg = f"Unknown facing '{facing}' in slot '{slot_name}'.{_suggest(facing, KNOWN_FACINGS)}"
            warnings.append(msg)

    # Validate placements
    place_data = data.get("place", data.get("compose", {}))
    if place_data and not isinstance(place_data, dict):
        raise ValidationError("'place' must be a mapping")

    for obj_name, obj_def in (place_data or {}).items():
        if not isinstance(obj_def, dict):
            raise ValidationError(f"Placement '{obj_name}' must be a mapping")

        if "asset" not in obj_def and "scene" not in obj_def:
            raise ValidationError(
                f"Placement '{obj_name}' requires 'asset' or 'scene' field"
            )

        for key in obj_def:
            if key not in PLACEMENT_KNOWN_KEYS:
                msg = f"Unknown key '{key}' in placement '{obj_name}'.{_suggest(key, PLACEMENT_KNOWN_KEYS)}"
                warnings.append(msg)

    return warnings


def pre_validate_asset_references(
    data: dict[str, Any],
) -> list[str]:
    """Pre-validate all attach_to/at references in an asset YAML.

    Returns a list of error messages for bad references (empty = all OK).
    """
    errors = []
    parts = data.get("parts", {})
    part_names = set(parts.keys())

    for part_name, part_def in parts.items():
        if "attach_to" in part_def:
            parent_name = part_def["attach_to"]
            if parent_name not in part_names:
                suggestion = _suggest(parent_name, part_names)
                errors.append(
                    f"Part '{part_name}' attaches to unknown part '{parent_name}'.{suggestion}"
                )

    return errors


def pre_validate_scene_references(
    data: dict[str, Any],
) -> list[str]:
    """Pre-validate all slot/attach_to references in a scene YAML.

    Returns a list of error messages for bad references (empty = all OK).
    """
    errors = []
    slots = set(data.get("slots", {}).keys())
    place_data = {**data.get("compose", {}), **data.get("place", {})}
    obj_names = set(place_data.keys())

    for obj_name, obj_def in place_data.items():
        if "slot" in obj_def:
            slot_name = obj_def["slot"]
            if slot_name not in slots:
                suggestion = _suggest(slot_name, slots)
                errors.append(
                    f"Placement '{obj_name}' references unknown slot '{slot_name}'.{suggestion}"
                )

        if "attach_to" in obj_def:
            target_name = obj_def["attach_to"]
            if target_name not in obj_names:
                suggestion = _suggest(target_name, obj_names)
                errors.append(
                    f"Placement '{obj_name}' attaches to unknown object '{target_name}'.{suggestion}"
                )

    return errors
