"""Viewer section tools: cutaway, storey filter, interaction state preview (no GL needed)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from geogen.layout import LayoutLoader
from geogen.viewer.qt_viewer import _copy_with_interactions, describe_node
from test_building import SMALL


def test_copy_with_interactions_moves_only_the_copy():
    from geogen.layout.interactions import apply_state
    original = LayoutLoader().load("assets/nightstand.yaml")
    before = original.find("drawer_front").world_transform().copy()
    copy = _copy_with_interactions(original)
    apply_state(copy, copy.interactions[0], "open")
    assert np.allclose(original.find("drawer_front").world_transform(), before)
    assert copy.find("drawer_front").world_transform()[2, 3] > before[2, 3] + 0.2


def test_inspector_lists_tags_rooms_and_interactions():
    text = describe_node(LayoutLoader().load("assets/nightstand.yaml"))
    assert "Tags      furniture.nightstand" in text
    assert "Interact  drawer: closed -> open" in text
    building = LayoutLoader().load_string(SMALL)
    assert "storey" in describe_node(building.find("storey_1"))
    assert "room" in describe_node(building.find("storey_1").find("corridor"))


def test_section_filters_facade_per_storey():
    from PyQt6.QtWidgets import QApplication
    from geogen.viewer.qt_viewer import ViewerWindow
    app = QApplication.instance() or QApplication([])
    yaml = SMALL.replace("  roof: { parapet: 1.0 }", "  facade: { style: modern }\n  roof: { parapet: 1.0 }")
    window = ViewerWindow({"b": lambda: LayoutLoader().load_string(yaml)}, "b", watch_dir=None)
    try:
        window._storey_combo.setCurrentIndex(window._storey_combo.findData(1))
        names = {n.name for n in window._section_scene(window._current_scene).iter_nodes()}
        assert {"facade_0", "facade_1"} <= names and "facade_2" not in names
    finally:
        window.close()


def test_section_filters_storeys(qtbot=None):
    from PyQt6.QtWidgets import QApplication
    from geogen.viewer.qt_viewer import ViewerWindow
    app = QApplication.instance() or QApplication([])
    scenes = {"small": lambda: LayoutLoader().load_string(SMALL)}
    window = ViewerWindow(scenes, "small", watch_dir=None)
    try:
        combo = window._storey_combo
        assert [combo.itemData(i) for i in range(combo.count())] == [None, 0, 1, 2]
        combo.setCurrentIndex(combo.findData(0))
        shown = window._section_scene(window._current_scene)
        names = {n.name for n in shown.iter_nodes()}
        assert "storey_0" in names and "storey_1" not in names and "roof" not in names
        window._cutaway_act.setChecked(True)
        shown = window._section_scene(window._current_scene)
        assert "ceiling" not in {n.name for n in shown.iter_nodes()}
        assert window._current_scene.find("ceiling") is not None   # the loaded scene is untouched
    finally:
        window.close()
