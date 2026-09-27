"""primitive: text (generators/text.py): layout, glyph regions, meshes and YAML."""

import string

import numpy as np
import pytest

from geogen.core import meshops
from geogen.generators.text import TextGenerator, layout
from geogen.layout.loader import LayoutLoader


def _bounds(shapes):
    pts = np.vstack([s.outer for s in shapes])
    return pts.min(axis=0), pts.max(axis=0)


def test_cap_height_sets_the_scale():
    lo, hi = _bounds(layout("H", height=0.3))
    assert hi[1] - lo[1] == pytest.approx(0.3, abs=0.002)
    assert lo[1] == pytest.approx(0.0, abs=1e-6)          # baseline at y = 0


def _width(shapes) -> float:
    lo, hi = _bounds(shapes)
    return float(hi[0] - lo[0])


def test_width_grows_with_the_text_and_tracking():
    w1 = _width(layout("HOTEL", height=0.2))
    w2 = _width(layout("HOTELHOTEL", height=0.2))
    assert w2 > 1.9 * w1
    spaced = _width(layout("HOTEL", height=0.2, tracking=0.1))
    assert spaced > w1 + 0.4 * 0.2 * 0.9     # four gaps of ~0.1 em


@pytest.mark.parametrize("align", ["left", "center", "right"])
def test_lines_align_within_the_block(align):
    shapes = layout("WIDE LINE\nI", height=0.2, align=align)
    # The second line (the I) sits below the first; find it by y.
    lo, hi = _bounds(shapes)
    i_shapes = [s for s in shapes if s.outer[:, 1].max() < -0.05]
    i_lo, i_hi = _bounds(i_shapes)
    centre = (i_lo[0] + i_hi[0]) / 2
    expected = {"left": lo[0] + (i_hi[0] - i_lo[0]) / 2, "center": (lo[0] + hi[0]) / 2,
                "right": hi[0] - (i_hi[0] - i_lo[0]) / 2}[align]
    assert centre == pytest.approx(expected, abs=0.03)     # side bearings: a few cm at 0.2 m


def test_counters_are_holes():
    holes = {c: sum(len(s.holes) for s in layout(c)) for c in "ABO8&HiI"}
    assert holes["A"] == 1 and holes["B"] == 2 and holes["O"] == 1 and holes["8"] == 2
    assert holes["H"] == 0 and holes["I"] == 0
    assert len(layout("i")) == 2                          # dot and stem are separate regions


@pytest.mark.parametrize("font", ["sans", "serif"])
def test_every_glyph_extrudes_watertight(font):
    text = string.ascii_letters + string.digits + "&%#@?!.,;:'\"()-+/ ÅéüçÑ€"
    for char in text.replace(" ", ""):
        mesh = TextGenerator(text=char, font=font, height=0.3, depth=0.04).generate()
        report = meshops.validate(mesh)
        assert not report.issues, (font, char, report.issues)


def test_mesh_is_centred_and_faces_z():
    mesh = TextGenerator(text="Hotel", height=0.25, depth=0.05, bevel=0.005).generate()
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    assert np.allclose((lo + hi) / 2, 0, atol=1e-9)
    assert hi[2] - lo[2] == pytest.approx(0.05)
    assert hi[0] - lo[0] > 3 * (hi[1] - lo[1])            # reads along X
    assert not meshops.validate(mesh).issues


def test_bad_font_and_align_are_rejected():
    with pytest.raises(ValueError, match="Unknown font"):
        TextGenerator(text="x", font="comic").generate()
    with pytest.raises(ValueError, match="align"):
        TextGenerator(text="x", align="justify").generate()


def test_text_part_and_text_params(tmp_path):
    path = tmp_path / "sign.yaml"
    path.write_text("""
name: sign
size: [1, 1, 1]
bounds: geometry
params:
  label: { default: Hotel, type: text }
  floor: { default: 3 }
parts:
  words:
    primitive: text
    text: "{label}"
    height: 0.2
    anchor: bottom_center
  number:
    primitive: text
    text: "Floor {floor}"
    height: 0.1
    anchor: bottom_center
    offset: [0, 0.4, 0]
""")
    loader = LayoutLoader()
    short = loader.load(path).find("words").size
    long = loader.load(path, params={"label": "Grand Hotel Ballroom"}).find("words").size
    assert long[0] > 2.5 * short[0] and short[1] == pytest.approx(long[1], abs=0.08)
    number = loader.load(path, params={"floor": 12}).find("number")
    assert number.size[0] > loader.load(path).find("number").size[0]   # "Floor 12" is wider than "Floor 3"
