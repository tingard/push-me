from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

_SYMMETRY_TOL = 1e-6


@dataclass(frozen=True)
class ShapeDef:
    name: str
    outline: np.ndarray
    symmetry_order: int
    convex_parts: list[np.ndarray]


def _signed_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _centroid(polygon: np.ndarray) -> np.ndarray:
    x, y = polygon[:, 0], polygon[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2.0
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * a)
    return np.array([cx, cy])


def _ensure_ccw(polygon: np.ndarray) -> np.ndarray:
    return polygon if _signed_area(polygon) > 0 else polygon[::-1].copy()


def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s], [s, c]])
    return points @ r.T


def _verify_symmetry(name: str, outline: np.ndarray, order: int) -> None:
    rotated = _rotate(outline, 2 * np.pi / order)
    cost = np.linalg.norm(rotated[:, None, :] - outline[None, :, :], axis=-1)
    row, col = linear_sum_assignment(cost)
    max_err = cost[row, col].max()
    if max_err > _SYMMETRY_TOL:
        raise AssertionError(
            f"{name}: claimed symmetry_order={order} does not hold "
            f"(max vertex mismatch {max_err:.6g} > tol {_SYMMETRY_TOL:.1g})"
        )


def _tetromino(
    name: str,
    outline_cells: list[tuple[float, float]],
    part_cells: list[list[tuple[float, float]]],
    symmetry_order: int,
) -> ShapeDef:
    outline_raw = np.asarray(outline_cells, dtype=float)
    offset = _centroid(outline_raw)
    outline = _ensure_ccw(outline_raw - offset)
    parts = [_ensure_ccw(np.asarray(cells, dtype=float) - offset) for cells in part_cells]
    return ShapeDef(name=name, outline=outline, symmetry_order=symmetry_order, convex_parts=parts)


def _regular_polygon(name: str, n: int, symmetry_order: int) -> ShapeDef:
    angles = 2 * np.pi * np.arange(n) / n
    outline = np.column_stack([np.cos(angles), np.sin(angles)])
    return ShapeDef(name=name, outline=outline, symmetry_order=symmetry_order, convex_parts=[outline.copy()])


SHAPES: dict[str, ShapeDef] = {
    "t_tetromino": _tetromino(
        "t_tetromino",
        [(0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (1, 2), (1, 1), (0, 1)],
        [[(0, 0), (3, 0), (3, 1), (0, 1)], [(1, 1), (2, 1), (2, 2), (1, 2)]],
        symmetry_order=1,
    ),
    "l_tetromino": _tetromino(
        "l_tetromino",
        [(0, 0), (2, 0), (2, 1), (1, 1), (1, 3), (0, 3)],
        [[(0, 0), (2, 0), (2, 1), (0, 1)], [(0, 1), (1, 1), (1, 3), (0, 3)]],
        symmetry_order=1,
    ),
    "s_tetromino": _tetromino(
        "s_tetromino",
        [(0, 0), (2, 0), (2, 1), (3, 1), (3, 2), (1, 2), (1, 1), (0, 1)],
        [[(0, 0), (2, 0), (2, 1), (0, 1)], [(1, 1), (3, 1), (3, 2), (1, 2)]],
        symmetry_order=2,
    ),
    "z_tetromino": _tetromino(
        "z_tetromino",
        [(1, 0), (3, 0), (3, 1), (2, 1), (2, 2), (0, 2), (0, 1), (1, 1)],
        [[(1, 0), (3, 0), (3, 1), (1, 1)], [(0, 1), (2, 1), (2, 2), (0, 2)]],
        symmetry_order=2,
    ),
    "triangle": _regular_polygon("triangle", n=3, symmetry_order=3),
    "square": _regular_polygon("square", n=4, symmetry_order=4),
    "pentagon": _regular_polygon("pentagon", n=5, symmetry_order=5),
    "hexagon": _regular_polygon("hexagon", n=6, symmetry_order=6),
    "octagon": _regular_polygon("octagon", n=8, symmetry_order=8),
}

for _shape in SHAPES.values():
    _verify_symmetry(_shape.name, _shape.outline, _shape.symmetry_order)
del _shape


def make_shape(name: str, area: float) -> ShapeDef:
    base = SHAPES[name]
    scale = np.sqrt(area / abs(_signed_area(base.outline)))
    return ShapeDef(
        name=base.name,
        outline=base.outline * scale,
        symmetry_order=base.symmetry_order,
        convex_parts=[part * scale for part in base.convex_parts],
    )
