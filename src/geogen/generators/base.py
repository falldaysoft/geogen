"""Base classes and protocols for geometry generators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode

if TYPE_CHECKING:
    from ..layout.attachments import AttachmentPoint
    from ..layout.surfaces import Surface


@runtime_checkable
class Generator(Protocol):
    """Protocol for geometry generators.

    Any class with a generate() method returning a Mesh satisfies this protocol.
    """

    def generate(self) -> Mesh:
        """Generate and return mesh geometry."""
        ...


def box_attachment_points(size: np.ndarray, radial: bool = False) -> dict[str, "AttachmentPoint"]:
    """Attachment points at the middle of each face of a box of ``size`` (x, y, z) about the
    origin: top, bottom, left (-X), right (+X), front (+Z), back (-Z). ``radial``: a round
    part's, at its radius (half the smaller of x and z) instead of the box's sides."""
    from ..layout.attachments import AttachmentPoint

    hx, hy, hz = (float(v) / 2 for v in size)
    if radial:
        hx = hz = min(hx, hz)
    points = {"top": ([0, hy, 0], "north"), "bottom": ([0, -hy, 0], "north"), "left": ([-hx, 0, 0], "west"),
              "right": ([hx, 0, 0], "east"), "front": ([0, 0, hz], "south"), "back": ([0, 0, -hz], "north")}
    return {name: AttachmentPoint(name=name, anchor="center", offset=np.array(offset), facing=facing)
            for name, (offset, facing) in points.items()}


def box_surfaces(size: np.ndarray) -> dict[str, "Surface"]:
    """The six faces of a box of ``size`` about the origin as surfaces.

    For each wall-like face, u is horizontal (left → right when viewed
    from outside) and v is vertical (+Y is up). Normals point outward.
    For top/bottom, u follows +X and v follows the remaining horizontal.
    """
    from ..layout.surfaces import Surface

    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

    return {
        # Front (+Z). Viewed from +Z, right is +X.
        "front": Surface(
            name="front",
            origin=np.array([-hx, -hy, hz]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 1.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            u_extent=sx, v_extent=sy,
        ),
        # Back (-Z). Viewed from -Z, right is -X.
        "back": Surface(
            name="back",
            origin=np.array([hx, -hy, -hz]),
            u_axis=np.array([-1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 1.0, 0.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            u_extent=sx, v_extent=sy,
        ),
        # Left (-X). Viewed from -X, right is -Z.
        "left": Surface(
            name="left",
            origin=np.array([-hx, -hy, hz]),
            u_axis=np.array([0.0, 0.0, -1.0]),
            v_axis=np.array([0.0, 1.0, 0.0]),
            normal=np.array([-1.0, 0.0, 0.0]),
            u_extent=sz, v_extent=sy,
        ),
        # Right (+X). Viewed from +X, right is +Z.
        "right": Surface(
            name="right",
            origin=np.array([hx, -hy, -hz]),
            u_axis=np.array([0.0, 0.0, 1.0]),
            v_axis=np.array([0.0, 1.0, 0.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            u_extent=sz, v_extent=sy,
        ),
        # Top (+Y). u along +X, v along +Z (back → front).
        "top": Surface(
            name="top",
            origin=np.array([-hx, hy, -hz]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            u_extent=sx, v_extent=sz,
        ),
        # Bottom (-Y). u along +X, v along +Z.
        "bottom": Surface(
            name="bottom",
            origin=np.array([-hx, -hy, -hz]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            u_extent=sx, v_extent=sz,
        ),
    }


class MeshGenerator(ABC):
    """Abstract base class for mesh generators.

    Provides a standard interface for generators that produce Mesh objects.
    Subclasses implement generate() to create specific geometry.
    """

    @abstractmethod
    def generate(self) -> Mesh:
        """Generate and return mesh geometry.

        Returns:
            A Mesh object containing the generated geometry.
        """
        pass

    def get_attachment_points(self, size: np.ndarray) -> dict[str, "AttachmentPoint"]:
        """Generate standard attachment points for this primitive given its actual size.

        Subclasses should override to provide shape-specific attachment points.
        Default implementation returns an empty dict.

        Args:
            size: Actual size [width, height, depth]

        Returns:
            Dictionary of attachment point name -> AttachmentPoint
        """
        return {}

    def get_surfaces(self, size: np.ndarray) -> dict[str, "Surface"]:
        """Generate named surfaces (2D regions) for this primitive.

        Surfaces expose 2D regions of a primitive (walls, floors, etc.)
        that other objects can be placed on with (u, v) coordinates.

        Subclasses should override to provide shape-specific surfaces.
        Default implementation returns an empty dict.

        Args:
            size: Actual size [width, height, depth]

        Returns:
            Dictionary of surface name -> Surface
        """
        return {}

    def to_node(self, name: str | None = None) -> SceneNode:
        """Generate geometry and wrap it in a SceneNode.

        Args:
            name: Optional name for the node. Defaults to generator class name.

        Returns:
            SceneNode containing the generated mesh.
        """
        node_name = name or self.__class__.__name__
        return SceneNode(name=node_name, mesh=self.generate())


class CompositeGenerator(ABC):
    """Abstract base class for generators that produce scene hierarchies.

    Use this for complex objects made of multiple parts (e.g., a chair
    made of legs, seat, and back).
    """

    @abstractmethod
    def generate(self) -> SceneNode:
        """Generate and return a scene hierarchy.

        Returns:
            Root SceneNode of the generated hierarchy.
        """
        pass
