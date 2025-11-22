"""Geometry utilities for face winding and normal computation.

This module provides helper functions to ensure consistent face winding
across all geometry generators. The convention used is:

- Counter-clockwise winding (when viewed from outside) = outward normal
- For triangle (A, B, C), normal direction is (B-A) × (C-A)
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def compute_triangle_normal(
    v0: NDArray[np.float64],
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the normal vector for a triangle.

    Uses the cross product (v1-v0) × (v2-v0) to determine normal direction.
    Counter-clockwise winding (when viewed from the normal direction) is assumed.

    Args:
        v0, v1, v2: The three vertices of the triangle

    Returns:
        Normalized normal vector (unit length)
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    length = np.linalg.norm(normal)
    if length > 1e-10:
        return normal / length
    return np.array([0.0, 1.0, 0.0])  # Degenerate triangle fallback


def make_cap_faces(
    center_idx: int,
    ring_indices: Sequence[int],
    normal_direction: NDArray[np.float64],
    vertices: NDArray[np.float64],
) -> list[list[int]]:
    """Generate triangular faces for a circular cap with correct winding.

    Creates triangles from a center vertex to consecutive pairs of ring vertices.
    Winding is determined by the desired normal direction.

    Args:
        center_idx: Index of the center vertex
        ring_indices: Indices of vertices around the ring (in CCW order when viewed from +Y)
        normal_direction: Desired outward normal direction for the cap
        vertices: The vertex array (needed to verify winding)

    Returns:
        List of face index triples with correct winding
    """
    faces = []
    n = len(ring_indices)
    center = vertices[center_idx]

    for i in range(n):
        i_next = (i + 1) % n
        idx0 = ring_indices[i]
        idx1 = ring_indices[i_next]

        # Try one winding
        v0, v1, v2 = center, vertices[idx0], vertices[idx1]
        normal = compute_triangle_normal(v0, v1, v2)

        # Check if normal aligns with desired direction
        if np.dot(normal, normal_direction) > 0:
            faces.append([center_idx, idx0, idx1])
        else:
            # Reverse winding
            faces.append([center_idx, idx1, idx0])

    return faces


def make_tube_faces(
    top_ring_indices: Sequence[int],
    bottom_ring_indices: Sequence[int],
    vertices: NDArray[np.float64],
) -> list[list[int]]:
    """Generate triangular faces for a tube/cylinder side with correct winding.

    Creates quads (as triangle pairs) connecting two rings of vertices.
    Normals will point radially outward from the axis.

    Args:
        top_ring_indices: Indices of the top ring (in CCW order when viewed from +Y)
        bottom_ring_indices: Indices of the bottom ring (same order as top)
        vertices: The vertex array (needed to compute outward direction)

    Returns:
        List of face index triples with outward-facing normals
    """
    faces = []
    n = len(top_ring_indices)

    for i in range(n):
        i_next = (i + 1) % n

        top0 = top_ring_indices[i]
        top1 = top_ring_indices[i_next]
        bot0 = bottom_ring_indices[i]
        bot1 = bottom_ring_indices[i_next]

        # Compute the outward direction for this face (radially outward)
        # Use the midpoint of the quad
        mid = (vertices[top0] + vertices[top1] + vertices[bot0] + vertices[bot1]) / 4
        # Outward is away from the Y axis
        outward = np.array([mid[0], 0.0, mid[2]])
        outward_len = np.linalg.norm(outward)
        if outward_len > 1e-10:
            outward = outward / outward_len
        else:
            outward = np.array([1.0, 0.0, 0.0])

        # First triangle: top0, top1, bot0
        v0, v1, v2 = vertices[top0], vertices[top1], vertices[bot0]
        normal = compute_triangle_normal(v0, v1, v2)

        if np.dot(normal, outward) > 0:
            faces.append([top0, top1, bot0])
        else:
            faces.append([top0, bot0, top1])

        # Second triangle: top1, bot1, bot0
        v0, v1, v2 = vertices[top1], vertices[bot1], vertices[bot0]
        normal = compute_triangle_normal(v0, v1, v2)

        if np.dot(normal, outward) > 0:
            faces.append([top1, bot1, bot0])
        else:
            faces.append([top1, bot0, bot1])

    return faces


def make_cone_side_faces(
    apex_idx: int,
    ring_indices: Sequence[int],
    vertices: NDArray[np.float64],
) -> list[list[int]]:
    """Generate triangular faces for a cone side with correct winding.

    Creates triangles from the apex to consecutive pairs of base ring vertices.
    Normals will point outward from the cone surface.

    Args:
        apex_idx: Index of the apex vertex
        ring_indices: Indices of the base ring (in CCW order when viewed from +Y)
        vertices: The vertex array

    Returns:
        List of face index triples with outward-facing normals
    """
    faces = []
    n = len(ring_indices)
    apex = vertices[apex_idx]

    for i in range(n):
        i_next = (i + 1) % n
        idx0 = ring_indices[i]
        idx1 = ring_indices[i_next]

        # Compute outward direction for this face
        # Midpoint of the triangle base, projected radially
        base_mid = (vertices[idx0] + vertices[idx1]) / 2
        outward = np.array([base_mid[0], 0.0, base_mid[2]])
        outward_len = np.linalg.norm(outward)
        if outward_len > 1e-10:
            outward = outward / outward_len
        else:
            outward = np.array([1.0, 0.0, 0.0])

        # Try winding
        v0, v1, v2 = apex, vertices[idx0], vertices[idx1]
        normal = compute_triangle_normal(v0, v1, v2)

        if np.dot(normal, outward) > 0:
            faces.append([apex_idx, idx0, idx1])
        else:
            faces.append([apex_idx, idx1, idx0])

    return faces


def verify_outward_normals(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    center: NDArray[np.float64] | None = None,
) -> tuple[bool, list[int]]:
    """Verify that all face normals point outward from a center point.

    Args:
        vertices: Vertex array
        faces: Face index array (Nx3)
        center: Center point to measure "outward" from. If None, uses centroid.

    Returns:
        Tuple of (all_valid, list_of_bad_face_indices)
    """
    if center is None:
        center = vertices.mean(axis=0)

    bad_faces = []

    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normal = compute_triangle_normal(v0, v1, v2)

        # Face center
        face_center = (v0 + v1 + v2) / 3

        # Direction from mesh center to face center
        outward = face_center - center
        outward_len = np.linalg.norm(outward)
        if outward_len > 1e-10:
            outward = outward / outward_len

            # Normal should point in same general direction as outward
            if np.dot(normal, outward) < 0:
                bad_faces.append(i)

    return len(bad_faces) == 0, bad_faces
