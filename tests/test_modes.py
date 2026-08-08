from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from push_me.geometry import containment_error, contains
from push_me.goals import make_goal_rect
from push_me.shapes import SHAPES, make_shape

_MARGIN = 24.0
_SHAPE_AREA = 4000.0


def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s], [s, c]])
    return points @ r.T


@given(st.sampled_from(sorted(SHAPES)))
def test_all_symmetric_orientations_are_contained_with_equal_error(name):
    shape = make_shape(name, _SHAPE_AREA)
    base_pose = (256.0, 256.0, 0.3)
    rect = make_goal_rect(shape, margin=_MARGIN, pose=base_pose)
    gx, gy, gtheta = base_pose

    errors = []
    for k in range(shape.symmetry_order):
        theta = gtheta + k * (2 * np.pi / shape.symmetry_order)
        outline_world = _rotate(shape.outline, theta) + np.array([gx, gy])
        assert contains(rect, outline_world), f"{name} orientation {k} not contained"
        errors.append(containment_error(rect, outline_world))

    assert errors == pytest.approx([errors[0]] * len(errors), abs=1e-6)


def test_hexagon_at_margin_24_all_six_orientations_contained_with_equal_error():
    shape = make_shape("hexagon", _SHAPE_AREA)
    assert shape.symmetry_order == 6
    base_pose = (256.0, 256.0, 0.0)
    rect = make_goal_rect(shape, margin=_MARGIN, pose=base_pose)
    gx, gy, gtheta = base_pose

    errors = []
    for k in range(6):
        theta = gtheta + k * (2 * np.pi / 6)
        outline_world = _rotate(shape.outline, theta) + np.array([gx, gy])
        assert contains(rect, outline_world)
        errors.append(containment_error(rect, outline_world))

    assert errors == pytest.approx([errors[0]] * 6, abs=1e-6)
