"""Interactive Qt viewer: scene browser, node inspector, hot reload.

Controls
  Left drag            orbit            Right/middle or Shift+left drag   pan
  Wheel                zoom to cursor   Click                             select node
  Double-click / F     frame selection  A / Home                          frame all
  1 2 3 4 7 0          front / back / right / left / top / iso views
  W wireframe  G grid  S shadows  H ground  M cycle display mode  Esc clear selection
  Ctrl+R reload  Ctrl+S screenshot  Ctrl+F find scene
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
from PyQt6.QtCore import QFileSystemWatcher, Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QShortcut, QSurfaceFormat
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QSplitter,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import meshops
from ..core.node import SceneNode
from .gl_view import DISPLAY_MODES, GLView

SceneFactory = Callable[[], SceneNode]
DEFAULT_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"

VIEW_KEYS = {
    Qt.Key.Key_1: "front",
    Qt.Key.Key_2: "back",
    Qt.Key.Key_3: "right",
    Qt.Key.Key_4: "left",
    Qt.Key.Key_7: "top",
    Qt.Key.Key_0: "iso",
}


def describe_node(node: SceneNode) -> str:
    """Human-readable summary of a node subtree for the inspector panel."""
    meshes = [(n, m) for n, m in node.iter_meshes()]
    verts = sum(len(m.vertices) for _, m in meshes)
    tris = sum(len(m.faces) for _, m in meshes)
    lines = [f"{node.name}", ""]
    if meshes:
        lo = np.min([m.vertices.min(axis=0) for _, m in meshes], axis=0)
        hi = np.max([m.vertices.max(axis=0) for _, m in meshes], axis=0)
        size = hi - lo
        lines.append(f"Size      {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m")
        lines.append(f"Min       ({lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f})")
    lines.append(f"Nodes     {sum(1 for _ in node.iter_nodes())}")
    lines.append(f"Meshes    {len(meshes)}")
    lines.append(f"Vertices  {verts:,}")
    lines.append(f"Triangles {tris:,}")

    t = node.transform
    lines.append("")
    lines.append(f"Position  ({t.translation[0]:.3f}, {t.translation[1]:.3f}, {t.translation[2]:.3f})")
    rot = np.degrees(t.rotation)
    lines.append(f"Rotation  ({rot[0]:.1f}, {rot[1]:.1f}, {rot[2]:.1f}) deg")

    materials = sorted({m.material.name for _, m in meshes if m.material is not None})
    if materials:
        lines.append(f"Materials {', '.join(materials)}")
    if node.attachments:
        lines.append(f"Attach    {', '.join(sorted(node.attachments))}")
    if node.surfaces:
        lines.append(f"Surfaces  {', '.join(sorted(node.surfaces))}")
    if node.tags:
        lines.append(f"Tags      {', '.join(node.tags)}")
    for key in ("room", "storey", "collider", "walkable", "joint", "light", "gate", "stairs", "footprint",
                "clearance"):
        if key in node.meta:
            lines.append(f"{key:<9} {node.meta[key]}")
    for interaction in node.interactions:
        lines.append(f"Interact  {interaction.name}: {' -> '.join(interaction.states)} (initial {interaction.initial})")
        for motion in interaction.motions:
            parts = ", ".join(p.name for p in motion.parts)
            lines.append(f"            {motion.kind} {list(np.round(motion.axis, 3))} about "
                         f"{list(np.round(motion.pivot, 3))}: {parts}")

    if node.mesh is not None:
        report = meshops.validate(node.mesh)
        lines.append("")
        lines.append("Mesh check")
        lines.append(f"  watertight  {'yes' if report.watertight else f'no ({report.boundary_edges} open edges)'}")
        lines.append(f"  normals     {'explicit' if node.mesh.normals is not None else 'computed'}")
        lines.append(f"  uvs         {'yes' if node.mesh.uvs is not None else 'none'}")
        if report.issues:
            lines.extend(f"  ! {issue}" for issue in report.issues)
        else:
            lines.append("  no defects")

    # Interiors: layout QA (overlaps, door swings, reachability) for nodes holding rooms.
    if any(isinstance(n.meta.get("room"), dict) for n in node.iter_nodes()):
        from ..layout.qa import check_layout

        issues = check_layout(node)
        lines.append("")
        lines.append("Layout check")
        lines.extend(f"  ! {issue}" for issue in issues) if issues else lines.append("  no problems")
    return "\n".join(lines)


def _copy_with_interactions(root: SceneNode) -> SceneNode:
    """Deep copy whose interactions move the copied parts (not the originals)."""
    import copy as _copy

    shown = root.copy(deep=True)
    mapping = {id(a): b for a, b in zip(root.iter_nodes(), shown.iter_nodes())}
    for node in shown.iter_nodes():
        if not node.interactions:
            continue
        remapped = []
        for interaction in node.interactions:
            clone = _copy.copy(interaction)
            clone.targets = [mapping.get(id(t), t) for t in interaction.targets]
            clone.motions = []
            for motion in interaction.motions:
                m = _copy.copy(motion)
                m.parts = [mapping.get(id(p), p) for p in motion.parts]
                clone.motions.append(m)
            remapped.append(clone)
        node.interactions = remapped
    return shown


class ViewerWindow(QMainWindow):
    """Main viewer window."""

    def __init__(
        self,
        scenes: dict[str, SceneFactory],
        default_scene: str | None = None,
        watch_dir: Path | None = DEFAULT_ASSETS_DIR,
    ) -> None:
        super().__init__()
        self._scenes = scenes
        self._current_name: str | None = None
        self._current_scene: SceneNode | None = None
        self._tree_items: dict[int, QTreeWidgetItem] = {}

        self.setWindowTitle("Geogen Viewer")
        self.resize(1400, 900)
        self._build_ui()
        self._build_toolbar()
        self._build_shortcuts()

        self._watcher = QFileSystemWatcher(self)
        self._reload_timer = QTimer(self, singleShot=True, interval=250)
        self._reload_timer.timeout.connect(lambda: self.reload(keep_camera=True))
        if watch_dir is not None and watch_dir.is_dir():
            self._watch_dir = watch_dir
            self._refresh_watch_list()
            self._watcher.fileChanged.connect(self._on_file_changed)
            self._watcher.directoryChanged.connect(self._on_file_changed)

        names = sorted(scenes)
        self.load_scene(default_scene if default_scene in scenes else names[0])

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        self._filter = QLineEdit(placeholderText="Filter scenes…  (Ctrl+F)")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        self._filter.returnPressed.connect(self._load_first_visible)
        left_layout.addWidget(self._filter)
        self._scene_list = QListWidget()
        self._scene_list.addItems(sorted(self._scenes))
        self._scene_list.currentTextChanged.connect(lambda name: name and self.load_scene(name))
        left_layout.addWidget(self._scene_list, 2)
        left_layout.addWidget(QLabel("Nodes"))
        self._node_tree = QTreeWidget()
        self._node_tree.setHeaderLabels(["Node", "Tris"])
        self._node_tree.setColumnWidth(0, 190)
        self._node_tree.currentItemChanged.connect(self._on_tree_selection)
        self._node_tree.itemDoubleClicked.connect(lambda *_: self._view.frame_selected())
        left_layout.addWidget(self._node_tree, 3)
        left.setMinimumWidth(240)

        self._view = GLView()
        self._view.nodeClicked.connect(self._on_view_clicked)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.addWidget(QLabel("Inspector"))
        self._inspector = QPlainTextEdit(readOnly=True)
        self._inspector.setStyleSheet("font-family: Menlo, Monaco, monospace; font-size: 11px;")
        right_layout.addWidget(self._inspector)
        right.setMinimumWidth(260)

        splitter.addWidget(left)
        splitter.addWidget(self._view)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 880, 280])

        self._status = QLabel()
        self.statusBar().addWidget(self._status, 1)
        self.statusBar().addPermanentWidget(QLabel(
            "drag: orbit · right/shift: pan · wheel: zoom · click: select · F: frame · 1-4/7/0: views · W/G/S/H/M"
        ))

    def _build_toolbar(self) -> None:
        bar = QToolBar("View")
        bar.setMovable(False)
        self.addToolBar(bar)

        def action(text: str, slot, shortcut: str | None = None, checkable=False, checked=False) -> QAction:
            act = QAction(text, self, checkable=checkable)
            if checkable:
                act.setChecked(checked)
                act.toggled.connect(slot)
            else:
                act.triggered.connect(slot)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            bar.addAction(act)
            return act

        action("Reload", lambda: self.reload(keep_camera=True), "Ctrl+R")
        self._auto_reload = QCheckBox("Auto")
        self._auto_reload.setChecked(True)
        self._auto_reload.setToolTip("Reload automatically when YAML assets change")
        bar.addWidget(self._auto_reload)
        bar.addSeparator()
        action("Frame", self._view.frame_selected)
        for label, preset in (("Front", "front"), ("Side", "right"), ("Top", "top"), ("Iso", "iso")):
            action(label, lambda _=False, p=preset: self._view.set_view(p))
        bar.addSeparator()
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(DISPLAY_MODES)
        self._mode_combo.currentIndexChanged.connect(self._set_display_mode)
        bar.addWidget(self._mode_combo)
        self._wire_act = action("Wireframe", self._toggle("wireframe"), checkable=True)
        self._grid_act = action("Grid", self._toggle("show_grid"), checkable=True, checked=True)
        self._shadow_act = action("Shadows", self._toggle("shadows"), checkable=True, checked=True)
        self._ground_act = action("Ground", self._toggle("show_ground"), checkable=True, checked=True)
        self._night_act = action("Night", self._toggle("night_mode"), "N", checkable=True)
        self._night_act.setToolTip("Dim the sun; light interiors with the scene's fixtures (N)")
        bar.addSeparator()
        action("Screenshot", self.save_screenshot, "Ctrl+S")

        # Section tools for buildings and interiors.
        section = QToolBar("Section")
        section.setMovable(False)
        self.addToolBar(section)
        self._cutaway_act = QAction("Cutaway", self, checkable=True)
        self._cutaway_act.setToolTip("Hide ceilings, roofs and ceiling lights (C)")
        self._cutaway_act.setShortcut(QKeySequence("C"))
        self._cutaway_act.toggled.connect(lambda _=False: self._apply_section())
        section.addAction(self._cutaway_act)
        section.addWidget(QLabel(" Storey "))
        self._storey_combo = QComboBox()
        self._storey_combo.setToolTip("Show storeys up to this one (buildings)")
        self._storey_combo.currentIndexChanged.connect(lambda _=0: self._apply_section())
        section.addWidget(self._storey_combo)
        section.addWidget(QLabel(" Interactions "))
        self._state_combo = QComboBox()
        self._state_combo.setToolTip("Pose every interaction (doors, drawers, lifts) in a state")
        self._state_combo.currentIndexChanged.connect(lambda _=0: self._apply_section())
        section.addWidget(self._state_combo)

    def _section_scene(self, root: SceneNode) -> SceneNode:
        """The scene as filtered/posed by the Section toolbar (a copy when changed)."""
        from ..layout.interactions import apply_state
        from ..render import cutaway

        storey = self._storey_combo.currentData() if self._storey_combo.count() else None
        state = self._state_combo.currentData() if self._state_combo.count() else None
        if not self._cutaway_act.isChecked() and storey is None and state is None:
            return root
        shown = _copy_with_interactions(root)
        if storey is not None:
            for node in list(shown.iter_nodes()):
                info = node.meta.get("storey")
                if isinstance(info, dict) and info.get("index", 0) > storey and node.parent is not None:
                    node.parent.remove_child(node)
                if node.name == "roof" and node.parent is not None and "roof" in node.tags:
                    node.parent.remove_child(node)
        if state is not None:
            for node in shown.iter_nodes():
                for interaction in node.interactions:
                    if state in interaction.states:
                        apply_state(node, interaction, state)
        if self._cutaway_act.isChecked():
            shown = cutaway(shown)
        return shown

    def _apply_section(self) -> None:
        if self._current_scene is not None:
            self._view.set_scene(self._section_scene(self._current_scene), reframe=False)
            self._view.update()

    def _refresh_section_controls(self, root: SceneNode) -> None:
        storeys = sorted({n.meta["storey"]["index"] for n in root.iter_nodes()
                          if isinstance(n.meta.get("storey"), dict)})
        states = sorted({s for n in root.iter_nodes() for i in n.interactions for s in i.states})
        for combo, items, blank in ((self._storey_combo, [(f"≤ {k}", k) for k in storeys], "All"),
                                    (self._state_combo, [(s, s) for s in states], "As authored")):
            combo.blockSignals(True)
            previous = combo.currentData()
            combo.clear()
            combo.addItem(blank, None)
            for label, data in items:
                combo.addItem(label, data)
            index = combo.findData(previous)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.setEnabled(bool(items))
            combo.blockSignals(False)

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+F"), self, activated=lambda: (self._filter.setFocus(), self._filter.selectAll()))

    def _toggle(self, attr: str):
        def slot(checked: bool) -> None:
            setattr(self._view, attr, checked)
            self._view.update()
        return slot

    def _set_display_mode(self, index: int) -> None:
        self._view.display_mode = index
        self._view.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            return super().keyPressEvent(event)
        if key in VIEW_KEYS:
            self._view.set_view(VIEW_KEYS[key])
        elif key == Qt.Key.Key_F:
            self._view.frame_selected()
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Home):
            self._view.frame_all()
        elif key == Qt.Key.Key_W:
            self._wire_act.toggle()
        elif key == Qt.Key.Key_G:
            self._grid_act.toggle()
        elif key == Qt.Key.Key_S:
            self._shadow_act.toggle()
        elif key == Qt.Key.Key_H:
            self._ground_act.toggle()
        elif key == Qt.Key.Key_M:
            self._mode_combo.setCurrentIndex((self._mode_combo.currentIndex() + 1) % len(DISPLAY_MODES))
        elif key == Qt.Key.Key_Escape:
            self._select(None)
        else:
            super().keyPressEvent(event)

    # --------------------------------------------------------------- scenes

    def _apply_filter(self, text: str) -> None:
        needle = text.lower()
        for i in range(self._scene_list.count()):
            item = self._scene_list.item(i)
            item.setHidden(needle not in item.text().lower())

    def _load_first_visible(self) -> None:
        for i in range(self._scene_list.count()):
            item = self._scene_list.item(i)
            if not item.isHidden():
                self._scene_list.setCurrentItem(item)
                self._view.setFocus()
                return

    def load_scene(self, name: str, keep_camera: bool = False) -> bool:
        """Build and show a scene. On error the previous scene stays visible."""
        start = time.perf_counter()
        try:
            root = self._scenes[name]()
        except Exception as exc:
            self._show_error(name, exc)
            return False
        elapsed = (time.perf_counter() - start) * 1000

        selected_name = self._view._selected_root.name if (keep_camera and self._view._selected_root) else None
        self._current_name, self._current_scene = name, root
        self._refresh_section_controls(root)
        self._view.set_fixture_source(root)
        self._view.set_scene(self._section_scene(root), reframe=not keep_camera)
        self._rebuild_tree()
        items = self._scene_list.findItems(name, Qt.MatchFlag.MatchExactly)
        if items and self._scene_list.currentItem() is not items[0]:
            self._scene_list.blockSignals(True)
            self._scene_list.setCurrentItem(items[0])
            self._scene_list.blockSignals(False)

        tris = sum(len(m.faces) for _, m in root.iter_meshes())
        meshes = sum(1 for _ in root.iter_meshes())
        self._status.setStyleSheet("")
        self._status.setText(f"{name}  ·  {meshes} meshes  ·  {tris:,} triangles  ·  built in {elapsed:.0f} ms")
        self.setWindowTitle(f"Geogen Viewer — {name}")
        restored = root.find(selected_name) if selected_name else None
        self._select(restored or None)
        return True

    def reload(self, keep_camera: bool = True) -> None:
        if self._current_name is None:
            return
        # YAML is parsed on every factory call, so re-running it picks up edits.
        if self.load_scene(self._current_name, keep_camera=keep_camera):
            self._status.setText(self._status.text() + "  ·  reloaded")

    def _show_error(self, name: str, exc: Exception) -> None:
        self._status.setStyleSheet("color: #c62828;")
        self._status.setText(f"Failed to build '{name}': {exc}")
        self._inspector.setPlainText(f"Error building '{name}'\n\n{traceback.format_exc()}")

    def _refresh_watch_list(self) -> None:
        paths = [str(p) for p in self._watch_dir.rglob("*.yaml")]
        paths += [str(p) for p in self._watch_dir.rglob("*") if p.is_dir()] + [str(self._watch_dir)]
        existing = set(self._watcher.files()) | set(self._watcher.directories())
        new = [p for p in paths if p not in existing]
        if new:
            self._watcher.addPaths(new)

    def _on_file_changed(self, _path: str) -> None:
        # Editors often save by replacing the file, which drops it from the watcher.
        self._refresh_watch_list()
        if self._auto_reload.isChecked():
            self._reload_timer.start()

    # ------------------------------------------------------------- selection

    def _rebuild_tree(self) -> None:
        self._node_tree.blockSignals(True)
        self._node_tree.clear()
        self._tree_items.clear()

        def add(parent, node: SceneNode) -> None:
            tris = sum(len(m.faces) for _, m in node.iter_meshes())
            item = QTreeWidgetItem(parent, [node.name, f"{tris:,}"])
            item.setData(0, Qt.ItemDataRole.UserRole, node)
            self._tree_items[id(node)] = item
            for child in node.children:
                add(item, child)

        if self._current_scene is not None:
            add(self._node_tree, self._current_scene)
            self._node_tree.expandToDepth(1)
        self._node_tree.blockSignals(False)

    def _on_tree_selection(self, current: QTreeWidgetItem | None, _previous=None) -> None:
        node = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        self._select(node, from_tree=True)

    def _on_view_clicked(self, node: SceneNode | None) -> None:
        self._select(node)

    def _select(self, node: SceneNode | None, from_tree: bool = False) -> None:
        # Selecting the root highlights everything, which isn't useful.
        highlight = node if node is not self._current_scene else None
        self._view.select(highlight)
        if not from_tree:
            self._node_tree.blockSignals(True)
            item = self._tree_items.get(id(node)) if node is not None else None
            self._node_tree.setCurrentItem(item)
            if item is not None:
                self._node_tree.scrollToItem(item)
            self._node_tree.blockSignals(False)
        target = node or self._current_scene
        self._inspector.setPlainText(describe_node(target) if target is not None else "")

    # ------------------------------------------------------------ screenshot

    def save_screenshot(self, path: str | None = None) -> None:
        if not path:
            default = f"{self._current_name or 'geogen'}.png"
            path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", default, "PNG images (*.png)")
            if not path:
                return
        self._view.grabFramebuffer().save(path)
        self._status.setText(f"Saved screenshot to {path}")


def run_viewer(
    scenes: dict[str, SceneFactory],
    default_scene: str | None = None,
    screenshot: str | None = None,
    view: str | None = None,
    display_mode: int = 0,
    cutaway: bool = False,
    storey: int | None = None,
    state: str | None = None,
    night: bool = False,
) -> None:
    """Run the Qt viewer. With ``screenshot``, save one frame to that path and exit."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication.instance() or QApplication(sys.argv)
    window = ViewerWindow(scenes, default_scene)
    window.show()

    if view:
        window._view.set_view(view)
    if display_mode:
        window._mode_combo.setCurrentIndex(display_mode)
    if cutaway:
        window._cutaway_act.setChecked(True)
    if night:
        window._night_act.setChecked(True)
    if storey is not None and window._storey_combo.findData(storey) >= 0:
        window._storey_combo.setCurrentIndex(window._storey_combo.findData(storey))
    if state is not None and window._state_combo.findData(state) >= 0:
        window._state_combo.setCurrentIndex(window._state_combo.findData(state))

    if screenshot:
        def grab() -> None:
            window._view.grabFramebuffer().save(screenshot)
            print(f"Saved viewer screenshot to {screenshot}")
            app.quit()
        QTimer.singleShot(1500, grab)
        app.exec()
        return
    sys.exit(app.exec())
