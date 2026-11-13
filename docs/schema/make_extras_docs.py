"""Write what mirrors docs/schema/geogen-extras.v1.schema.json from it: GeogenSceneBuilder.KNOWN_KEYS
(runtime/godot/addons/geogen/scene_builder.gd) and the "Node extras" table of runtime/godot/README.md.
Run after adding or changing a key of extras.geogen: python docs/schema/make_extras_docs.py
(tests/test_extras_contract.py checks they agree).

A table row is the key, the scripts that read it and what it is for. The last two come from the key's
"x-godot" annotation in the schema ({"by": "`scripts/x.gd`", "for": "..."}, both optional): without "by"
the Godot runtime doesn't read the key, and without "for" the row gives the first sentence of the key's
description."""

import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "docs/schema/geogen-extras.v1.schema.json"
BUILDER = ROOT / "runtime/godot/addons/geogen/scene_builder.gd"
README = ROOT / "runtime/godot/README.md"
BEGIN, END = "<!-- extras table: docs/schema/make_extras_docs.py -->", "<!-- end extras table -->"
SENTENCE = 160      # a row's description is cut at a word once its first sentence is longer


def properties() -> dict:
    return json.loads(SCHEMA.read_text())["properties"]["geogen"]["properties"]


def known_keys(props: dict) -> str:
    """The GDScript constant, wrapped like the rest of the file."""
    body = textwrap.fill(", ".join(f'"{key}"' for key in props), width=112, initial_indent="    ",
                         subsequent_indent="    ", break_long_words=False)
    return "const KNOWN_KEYS := [\n" + body + ",\n]"


def first_sentence(description: str) -> str:
    sentence = re.split(r"(?<!\.\.)(?<!e\.g)(?<!i\.e)\.\s+(?=[A-Z'])", description, maxsplit=1)[0]
    if not sentence.endswith("..."):
        sentence = sentence.rstrip(".")
    if len(sentence) > SENTENCE:
        sentence = sentence[:SENTENCE].rsplit(" ", 1)[0].rstrip(",;:(") + " ..."
    return sentence.replace("|", "\\|")


def table(props: dict) -> str:
    rows = ["| Key | Read by | For |", "|---|---|---|"]
    for key, spec in props.items():
        note = spec.get("x-godot", {})
        rows.append(f"| `{key}` | {note.get('by', '-')} | {note.get('for') or first_sentence(spec.get('description', ''))} |")
    return "\n".join(rows)


def generated() -> dict[Path, str]:
    """{file: what it should hold}."""
    props = properties()
    builder = re.sub(r"const KNOWN_KEYS := \[.*?\]", lambda _: known_keys(props), BUILDER.read_text(), count=1, flags=re.S)
    readme = README.read_text()
    head, rest = readme.split(BEGIN, 1)
    readme = head + BEGIN + "\n" + table(props) + "\n" + END + rest.split(END, 1)[1]
    return {BUILDER: builder, README: readme}


if __name__ == "__main__":
    for path, text in generated().items():
        if path.read_text() != text:
            path.write_text(text)
            print("wrote", path.relative_to(ROOT))
