"""SceneNode class for hierarchical scene composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import numpy as np
from numpy.typing import NDArray

from .mesh import Mesh
from .transform import Transform

if TYPE_CHECKING:
    from ..layout.attachments import AttachmentPoint


@dataclass
class SceneNode:
    """A node in the scene hierarchy.

    Each node has a local transform, optional mesh, and can have children.
    This enables hierarchical composition where child transforms are relative
    to their parent.

    Attachment points can be defined on nodes to specify where other objects
    can be connected. Use get_attachment() to resolve attachment points to
    world-space transforms.

    Example:
        # Create a chair from components
        chair = SceneNode("chair")
        chair.add_child(SceneNode("seat", mesh=seat_mesh))
        leg = SceneNode("leg", mesh=leg_mesh)
        for i, pos in enumerate(leg_positions):
            leg_instance = SceneNode(f"leg_{i}", mesh=leg_mesh)
            leg_instance.transform.translation = pos
            chair.add_child(leg_instance)
    """

    name: str
    transform: Transform = field(default_factory=Transform)
    mesh: Mesh | None = None
    children: list[SceneNode] = field(default_factory=list)
    parent: SceneNode | None = field(default=None, repr=False)
    attachments: dict[str, AttachmentPoint] = field(default_factory=dict)
    size: NDArray[np.float64] | None = field(default=None, repr=False)

    def add_child(self, node: SceneNode) -> SceneNode:
        """Add a child node.

        Args:
            node: The node to add as a child

        Returns:
            The added node (for chaining)
        """
        node.parent = self
        self.children.append(node)
        return node

    def remove_child(self, node: SceneNode) -> bool:
        """Remove a child node.

        Args:
            node: The node to remove

        Returns:
            True if the node was found and removed
        """
        if node in self.children:
            node.parent = None
            self.children.remove(node)
            return True
        return False

    def world_transform(self) -> NDArray[np.float64]:
        """Compute the world transformation matrix.

        Traverses up the parent chain and combines transforms.

        Returns:
            4x4 transformation matrix in world space
        """
        if self.parent is None:
            return self.transform.to_matrix()
        return self.parent.world_transform() @ self.transform.to_matrix()

    def world_mesh(self) -> Mesh | None:
        """Get the mesh transformed to world space.

        Returns:
            Transformed mesh or None if this node has no mesh
        """
        if self.mesh is None:
            return None
        return self.mesh.transform(self.world_transform())

    def iter_nodes(self, include_self: bool = True) -> Iterator[SceneNode]:
        """Iterate over this node and all descendants (depth-first).

        Args:
            include_self: Whether to include this node in the iteration

        Yields:
            SceneNode instances
        """
        if include_self:
            yield self
        for child in self.children:
            yield from child.iter_nodes(include_self=True)

    def iter_meshes(self) -> Iterator[tuple[SceneNode, Mesh]]:
        """Iterate over all nodes with meshes, yielding world-space meshes.

        Yields:
            Tuples of (node, world_space_mesh)
        """
        for node in self.iter_nodes():
            if node.mesh is not None:
                world_mesh = node.world_mesh()
                if world_mesh is not None:
                    yield node, world_mesh

    def flatten(self) -> Mesh:
        """Flatten the entire hierarchy into a single merged mesh.

        All meshes are transformed to world space and merged.

        Returns:
            Single Mesh containing all geometry
        """
        meshes = [mesh for _, mesh in self.iter_meshes()]
        return Mesh.merge(meshes)

    def find(self, name: str) -> SceneNode | None:
        """Find a descendant node by name.

        Args:
            name: The name to search for

        Returns:
            The first matching node, or None
        """
        for node in self.iter_nodes():
            if node.name == name:
                return node
        return None

    def find_all(self, name: str) -> list[SceneNode]:
        """Find all descendant nodes with the given name.

        Args:
            name: The name to search for

        Returns:
            List of matching nodes
        """
        return [node for node in self.iter_nodes() if node.name == name]

    @property
    def depth(self) -> int:
        """Get the depth of this node in the hierarchy (root = 0)."""
        if self.parent is None:
            return 0
        return self.parent.depth + 1

    @property
    def root(self) -> SceneNode:
        """Get the root node of this hierarchy."""
        if self.parent is None:
            return self
        return self.parent.root

    def get_attachment(self, name: str) -> Transform | None:
        """Get the world-space transform for a named attachment point.

        Args:
            name: Name of the attachment point

        Returns:
            Transform in world space, or None if attachment not found
        """
        attachment = self.attachments.get(name)
        if attachment is None:
            return None

        if self.size is None:
            raise ValueError(
                f"Cannot resolve attachment '{name}': node '{self.name}' has no size"
            )

        # Get local attachment transform
        local_transform = attachment.resolve(self.size)

        # Combine with node's world transform
        world_matrix = self.world_transform() @ local_transform.to_matrix()
        return Transform.from_matrix(world_matrix)

    def list_attachments(self) -> list[str]:
        """List all attachment point names on this node."""
        return list(self.attachments.keys())

    def copy(self, deep: bool = True) -> SceneNode:
        """Create a copy of this node.

        Args:
            deep: If True, recursively copy children

        Returns:
            New SceneNode with copied data
        """
        new_node = SceneNode(
            name=self.name,
            transform=self.transform.copy(),
            mesh=self.mesh,  # Meshes are typically shared, not copied
            attachments=self.attachments.copy(),
            size=self.size.copy() if self.size is not None else None,
        )
        if deep:
            for child in self.children:
                new_node.add_child(child.copy(deep=True))
        return new_node

    def __repr__(self) -> str:
        mesh_str = f", mesh={self.mesh.face_count}f" if self.mesh else ""
        children_str = f", children={len(self.children)}" if self.children else ""
        return f"SceneNode({self.name!r}{mesh_str}{children_str})"
