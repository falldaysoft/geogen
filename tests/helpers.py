"""What several test modules share (import as ``from helpers import ...``; tests/ is on sys.path)."""

import json
import struct
from pathlib import Path


def gltf_json(path) -> dict:
    """The JSON chunk of a ``.glb``."""
    data = Path(path).read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


_validators: dict[str, object] = {}


def schema_validate(instance, schema: dict) -> None:
    """``jsonschema.validate(instance, schema)``, with the validator kept per schema.

    ``jsonschema.validate`` checks the schema itself against its metaschema on every call, which
    for the extras schema costs over a second: a loop over a scene's nodes spent minutes on it."""
    import jsonschema

    key = json.dumps(schema, sort_keys=True)
    validator = _validators.get(key)
    if validator is None:
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
        validator = _validators[key] = cls(schema)
    error = jsonschema.exceptions.best_match(validator.iter_errors(instance))
    if error is not None:
        raise error
