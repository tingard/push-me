from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.optimize import linear_sum_assignment

from push_me.shapes import SHAPES, make_shape

EXPECTED_SYMMETRY_ORDERS = {
    "t_tetromino": 1,
    "l_tetromino": 1,
    "s_tetromino": 2,
    "z_tetromino": 2,
    "triangle": 3,
    "square": 4,
    "pentagon": 5,
    "hexagon": 6,
    "octagon": 8,
}


def _polygon_signed_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    x, y = polygon[:, 0], polygon[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2.0
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * a)
    return np.array([cx, cy])


def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s], [s, c]])
    return points @ r.T


def _vertex_sets_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    row, col = linear_sum_assignment(cost)
    return bool(cost[row, col].max() <= tol)


def _is_convex(polygon: np.ndarray) -> bool:
    n = len(polygon)
    signs = []
    for i in range(n):
        a, b, c = polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-9:
            signs.append(cross > 0)
    return len(set(signs)) <= 1


def test_registry_has_exactly_the_required_shapes():
    assert set(SHAPES) == set(EXPECTED_SYMMETRY_ORDERS)


@pytest.mark.parametrize("name, order", sorted(EXPECTED_SYMMETRY_ORDERS.items()))
def test_symmetry_order_matches_spec_table(name, order):
    assert SHAPES[name].symmetry_order == order


@given(st.sampled_from(sorted(SHAPES)))
def test_rotation_by_symmetry_order_maps_outline_onto_itself(name):
    shape = SHAPES[name]
    rotated = _rotate(shape.outline, 2 * np.pi / shape.symmetry_order)
    assert _vertex_sets_match(rotated, shape.outline)


def test_outline_centroid_is_at_origin():
    for shape in SHAPES.values():
        cx, cy = _polygon_centroid(shape.outline)
        assert cx == pytest.approx(0.0, abs=1e-9)
        assert cy == pytest.approx(0.0, abs=1e-9)


def test_outline_is_ccw():
    for shape in SHAPES.values():
        assert _polygon_signed_area(shape.outline) > 0, (
            f"{shape.name} outline is not CCW"
        )


def test_convex_parts_are_individually_convex():
    for shape in SHAPES.values():
        for i, part in enumerate(shape.convex_parts):
            assert _is_convex(part), f"{shape.name}: convex_parts[{i}] is not convex"


def test_convex_parts_area_sums_to_outline_area():
    for shape in SHAPES.values():
        outline_area = abs(_polygon_signed_area(shape.outline))
        parts_area = sum(abs(_polygon_signed_area(part)) for part in shape.convex_parts)
        assert parts_area == pytest.approx(outline_area, rel=1e-6)


@given(st.sampled_from(sorted(SHAPES)), st.floats(min_value=1.0, max_value=1e6))
def test_make_shape_normalises_outline_area(name, area):
    shape = make_shape(name, area)
    computed = abs(_polygon_signed_area(shape.outline))
    assert computed == pytest.approx(area, rel=1e-6)


@given(st.sampled_from(sorted(SHAPES)), st.floats(min_value=1.0, max_value=1e6))
def test_make_shape_preserves_convex_part_area_sum(name, area):
    shape = make_shape(name, area)
    outline_area = abs(_polygon_signed_area(shape.outline))
    parts_area = sum(abs(_polygon_signed_area(part)) for part in shape.convex_parts)
    assert parts_area == pytest.approx(outline_area, rel=1e-6)


@given(st.sampled_from(sorted(SHAPES)), st.floats(min_value=1.0, max_value=1e6))
def test_make_shape_preserves_symmetry_order_and_name(name, area):
    shape = make_shape(name, area)
    assert shape.name == name
    assert shape.symmetry_order == SHAPES[name].symmetry_order


def test_make_shape_does_not_mutate_registry():
    before = SHAPES["square"].outline.copy()
    make_shape("square", 999.0)
    assert np.array_equal(SHAPES["square"].outline, before)
