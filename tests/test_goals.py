from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from push_me.geometry import min_area_rect, rects_overlap
from push_me.goals import make_goal_rect, sample_goal_rects
from push_me.shapes import SHAPES, make_shape

_SHAPE_AREA = 4000.0
_ARENA_SIZE = 512.0


def test_make_goal_rect_size_and_orientation():
    shape = make_shape("square", _SHAPE_AREA)
    half_extents, phi = min_area_rect(shape.outline)
    rect = make_goal_rect(shape, margin=8.0, pose=(100.0, 200.0, 0.7))

    assert rect.center == pytest.approx(np.array([100.0, 200.0]))
    assert rect.angle == pytest.approx(0.7 + phi)
    assert rect.half_extents == pytest.approx(half_extents + 8.0)
    assert rect.accepts == {"square"}


@given(st.floats(min_value=0.0, max_value=20.0))
def test_make_goal_rect_inflates_by_exactly_margin(margin):
    shape = make_shape("hexagon", _SHAPE_AREA)
    base_half_extents, _phi = min_area_rect(shape.outline)
    rect = make_goal_rect(shape, margin=margin, pose=(0.0, 0.0, 0.0))
    assert rect.half_extents == pytest.approx(base_half_extents + margin)


def test_sample_goal_rects_returns_one_per_shape():
    rng = np.random.default_rng(0)
    shapes = [make_shape(name, _SHAPE_AREA) for name in ("square", "square", "square")]
    rects = sample_goal_rects(rng, shapes, margin=8.0, arena_size=_ARENA_SIZE)
    assert len(rects) == 3


@settings(max_examples=25)
@given(st.integers(min_value=1, max_value=4), st.integers(min_value=0, max_value=1000))
def test_sampled_goal_rects_never_overlap_each_other(n_objects, seed):
    rng = np.random.default_rng(seed)
    shapes = [make_shape("square", _SHAPE_AREA) for _ in range(n_objects)]
    rects = sample_goal_rects(rng, shapes, margin=8.0, arena_size=_ARENA_SIZE)
    for a, b in itertools.combinations(rects, 2):
        assert not rects_overlap(a, b)


@settings(max_examples=25)
@given(st.integers(min_value=1, max_value=4), st.integers(min_value=0, max_value=1000))
def test_sampled_goal_rects_stay_within_arena(n_objects, seed):
    rng = np.random.default_rng(seed)
    shapes = [make_shape("square", _SHAPE_AREA) for _ in range(n_objects)]
    rects = sample_goal_rects(rng, shapes, margin=8.0, arena_size=_ARENA_SIZE)
    for rect in rects:
        from push_me.geometry import rect_corners

        corners = rect_corners(rect)
        assert np.all(corners >= -1e-6)
        assert np.all(corners <= _ARENA_SIZE + 1e-6)


def test_sample_goal_rects_uses_only_the_given_generator(monkeypatch):
    import numpy as _np

    def _boom(*args, **kwargs):
        raise AssertionError("global np.random must not be used")

    monkeypatch.setattr(_np.random, "uniform", _boom)
    monkeypatch.setattr(_np.random, "seed", _boom)

    rng = np.random.default_rng(0)
    shapes = [make_shape(name, _SHAPE_AREA) for name in SHAPES]
    sample_goal_rects(rng, shapes, margin=8.0, arena_size=_ARENA_SIZE)


def test_sample_goal_rects_raises_when_infeasible():
    rng = np.random.default_rng(0)
    huge = make_shape("hexagon", _SHAPE_AREA * 1000)
    with pytest.raises(RuntimeError):
        sample_goal_rects(rng, [huge, huge, huge], margin=8.0, arena_size=_ARENA_SIZE, max_attempts=20)
