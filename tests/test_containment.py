from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from push_me.geometry import (
    containment_error,
    contains,
    min_area_rect,
    rect_corners,
    rects_overlap,
    transform_to_rect_frame,
)
from push_me.shapes import SHAPES

_SHAPE_NAMES = sorted(SHAPES)
_angles = st.floats(min_value=0, max_value=2 * np.pi)
_coords = st.floats(min_value=-50, max_value=50)
_extents = st.floats(min_value=1, max_value=50)


@dataclass
class _Rect:
    center: np.ndarray
    angle: float
    half_extents: np.ndarray


def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s], [s, c]])
    return points @ r.T


# ---- min_area_rect ----


def test_min_area_rect_known_axis_aligned_rectangle():
    outline = np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=float)
    half_extents, _phi = min_area_rect(outline)
    assert sorted(half_extents) == pytest.approx([1.0, 2.0])
    assert 4 * half_extents[0] * half_extents[1] == pytest.approx(8.0)


@given(st.sampled_from(_SHAPE_NAMES), _angles)
def test_min_area_rect_is_self_consistent(name, theta):
    outline = _rotate(SHAPES[name].outline, theta)
    half_extents, phi = min_area_rect(outline)
    local = _rotate(outline, -phi)
    extents = (local.max(axis=0) - local.min(axis=0)) / 2.0
    assert extents == pytest.approx(half_extents, abs=1e-6)


@given(st.sampled_from(_SHAPE_NAMES), _angles)
def test_min_area_rect_area_is_rotation_invariant(name, theta):
    outline = SHAPES[name].outline
    he0, _ = min_area_rect(outline)
    he1, _ = min_area_rect(_rotate(outline, theta))
    assert 4 * he1[0] * he1[1] == pytest.approx(4 * he0[0] * he0[1], rel=1e-5)


@given(st.sampled_from(_SHAPE_NAMES))
def test_min_area_rect_area_does_not_exceed_axis_aligned_bounding_box(name):
    outline = SHAPES[name].outline
    half_extents, _ = min_area_rect(outline)
    min_area = 4 * half_extents[0] * half_extents[1]
    aabb = outline.max(axis=0) - outline.min(axis=0)
    assert min_area <= aabb[0] * aabb[1] + 1e-9


# ---- transform_to_rect_frame ----


def test_transform_to_rect_frame_rect_center_maps_to_origin():
    rect = _Rect(
        center=np.array([3.0, -2.0]), angle=0.5, half_extents=np.array([1.0, 1.0])
    )
    local = transform_to_rect_frame(rect.center[None, :], rect)
    assert local == pytest.approx(np.zeros((1, 2)), abs=1e-9)


def test_transform_to_rect_frame_axis_aligned_corners():
    rect = _Rect(center=np.zeros(2), angle=0.0, half_extents=np.array([2.0, 3.0]))
    corners = np.array([[2, 3], [-2, 3], [-2, -3], [2, -3]], dtype=float)
    local = transform_to_rect_frame(corners, rect)
    assert local == pytest.approx(corners, abs=1e-9)


# ---- contains / containment_error ----


def test_contains_and_error_agree_on_known_examples():
    rect = _Rect(center=np.zeros(2), angle=0.0, half_extents=np.array([2.0, 1.0]))
    inside = np.array([[0.0, 0.0], [1.9, 0.9]])
    outside = np.array([[0.0, 0.0], [2.1, 0.0]])
    boundary = np.array([[2.0, 1.0], [-2.0, -1.0]])

    assert contains(rect, inside) is True
    assert contains(rect, outside) is False
    assert contains(rect, boundary) is True
    assert containment_error(rect, inside) <= 0
    assert containment_error(rect, outside) > 0
    assert containment_error(rect, boundary) == pytest.approx(0.0, abs=1e-9)


@given(st.sampled_from(_SHAPE_NAMES), _coords, _coords, _angles, _extents, _extents)
def test_contains_agrees_with_containment_error_sign(name, cx, cy, angle, hw, hh):
    rect = _Rect(
        center=np.array([cx, cy]), angle=angle, half_extents=np.array([hw, hh])
    )
    outline = SHAPES[name].outline + rect.center
    err = containment_error(rect, outline)
    assert contains(rect, outline) == (err <= 0)


@given(
    _coords, _coords, _angles, _extents, _extents, st.floats(min_value=0, max_value=10)
)
def test_containment_error_is_monotonic_in_margin(cx, cy, angle, hw, hh, extra_margin):
    point = np.array([[cx + 0.1, cy + 0.1]])
    small = _Rect(
        center=np.array([cx, cy]), angle=angle, half_extents=np.array([hw, hh])
    )
    big = _Rect(
        center=np.array([cx, cy]),
        angle=angle,
        half_extents=np.array([hw + extra_margin, hh + extra_margin]),
    )
    assert containment_error(big, point) <= containment_error(small, point) + 1e-9


@given(st.sampled_from(_SHAPE_NAMES), _coords, _coords, _angles)
def test_contains_is_vertex_order_invariant(name, cx, cy, angle):
    rng = np.random.default_rng(0)
    outline = SHAPES[name].outline + np.array([cx, cy])
    rect = _Rect(
        center=np.array([cx, cy]), angle=angle, half_extents=np.array([100.0, 100.0])
    )
    permuted = outline[rng.permutation(len(outline))]
    assert contains(rect, outline) == contains(rect, permuted)
    assert containment_error(rect, outline) == pytest.approx(
        containment_error(rect, permuted)
    )


@given(
    st.sampled_from(_SHAPE_NAMES),
    _coords,
    _coords,
    _angles,
    _extents,
    _extents,
    _coords,
    _coords,
    _angles,
)
def test_contains_is_rigid_transform_invariant(
    name, cx, cy, angle, hw, hh, dx, dy, dtheta
):
    outline = SHAPES[name].outline + np.array([cx, cy])
    rect = _Rect(
        center=np.array([cx, cy]), angle=angle, half_extents=np.array([hw, hh])
    )

    shift = np.array([dx, dy])
    moved_rect = _Rect(
        center=_rotate(rect.center[None, :], dtheta)[0] + shift,
        angle=rect.angle + dtheta,
        half_extents=rect.half_extents,
    )
    moved_outline = _rotate(outline, dtheta) + shift

    # contains() is a step function of this continuous quantity, which is the actual invariant; a discrete equality check can flip from FP rounding right at an exact boundary.
    assert containment_error(rect, outline) == pytest.approx(
        containment_error(moved_rect, moved_outline), abs=1e-6
    )


# ---- rect_corners / rects_overlap ----


def test_rect_corners_axis_aligned():
    rect = _Rect(
        center=np.array([1.0, 2.0]), angle=0.0, half_extents=np.array([3.0, 4.0])
    )
    corners = rect_corners(rect)
    expected = np.array([[4, 6], [-2, 6], [-2, -2], [4, -2]], dtype=float)
    assert sorted(corners.tolist()) == sorted(expected.tolist())


def test_rects_overlap_known_cases():
    a = _Rect(center=np.zeros(2), angle=0.0, half_extents=np.array([1.0, 1.0]))
    touching = _Rect(
        center=np.array([2.0, 0.0]), angle=0.0, half_extents=np.array([1.0, 1.0])
    )
    overlapping = _Rect(
        center=np.array([1.5, 0.0]), angle=0.0, half_extents=np.array([1.0, 1.0])
    )
    far = _Rect(
        center=np.array([10.0, 10.0]), angle=0.0, half_extents=np.array([1.0, 1.0])
    )

    assert rects_overlap(a, touching) is True
    assert rects_overlap(a, overlapping) is True
    assert rects_overlap(a, far) is False


@given(
    _coords,
    _coords,
    _angles,
    _extents,
    _extents,
    _coords,
    _coords,
    _angles,
    _extents,
    _extents,
)
def test_rects_overlap_is_symmetric(cx1, cy1, a1, hw1, hh1, cx2, cy2, a2, hw2, hh2):
    r1 = _Rect(center=np.array([cx1, cy1]), angle=a1, half_extents=np.array([hw1, hh1]))
    r2 = _Rect(center=np.array([cx2, cy2]), angle=a2, half_extents=np.array([hw2, hh2]))
    assert rects_overlap(r1, r2) == rects_overlap(r2, r1)


@given(_coords, _coords, _angles, _extents, _extents)
def test_rects_overlap_self(cx, cy, angle, hw, hh):
    rect = _Rect(
        center=np.array([cx, cy]), angle=angle, half_extents=np.array([hw, hh])
    )
    assert rects_overlap(rect, rect) is True
