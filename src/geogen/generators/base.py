"""Base classes and protocols for geometry generators."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from ..core.mesh import Mesh
from ..core.node import SceneNode


@runtime_checkable
class Generator(Protocol):
    """Protocol for geometry generators.

    Any class with a generate() method returning a Mesh satisfies this protocol.
    """

    def generate(self) -> Mesh:
        """Generate and return mesh geometry."""
        ...


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
