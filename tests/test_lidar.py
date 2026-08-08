from __future__ import annotations

import numpy as np
import pymunk
import pytest
from hypothesis import given
from hypothesis import strategies as st

from push_me.lidar import N_HIT_CLASSES, HitClass, cast_rays


def _static_segment(
    space: pymunk.Space, a, b, collision_type: HitClass, object_index: int | None = None
) -> None:
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, a, b, radius=0.0)
    shape.collision_type = collision_type
    if object_index is not None:
        shape.user_data = object_index
    space.add(body, shape)


def test_cast_rays_output_shapes():
    space = pymunk.Space()
    features, hit_object_index = cast_rays(
        space, origin=(0.0, 0.0), n_rays=16, lidar_range=100.0
    )
    assert features.shape == (16, 1 + N_HIT_CLASSES)
    assert hit_object_index.shape == (16,)


def test_ray_hits_wall_at_known_distance():
    space = pymunk.Space()
    _static_segment(space, (10, -100), (10, 100), HitClass.WALL)
    # n_rays=4 -> angles [0, pi/2, pi, 3pi/2]; ray 0 points along +x straight at the wall.
    features, hit_object_index = cast_rays(
        space, origin=(0.0, 0.0), n_rays=4, lidar_range=100.0
    )
    assert features[0, 0] == pytest.approx(0.1)
    assert features[0, 1 + HitClass.WALL] == 1.0
    assert features[0, 1 + HitClass.NONE] == 0.0
    assert features[0, 1 + HitClass.OBJECT] == 0.0
    assert hit_object_index[0] == -1


def test_ray_with_nothing_in_its_path_reports_max_range_and_none():
    space = pymunk.Space()
    _static_segment(space, (10, -100), (10, 100), HitClass.WALL)
    # ray 2 (angle pi) points along -x, away from the wall entirely.
    features, hit_object_index = cast_rays(
        space, origin=(0.0, 0.0), n_rays=4, lidar_range=100.0
    )
    assert features[2, 0] == pytest.approx(1.0)
    assert features[2, 1 + HitClass.NONE] == 1.0
    assert features[2, 1 + HitClass.WALL] == 0.0
    assert features[2, 1 + HitClass.OBJECT] == 0.0
    assert hit_object_index[2] == -1


def test_nearer_object_occludes_farther_wall_on_the_same_ray():
    space = pymunk.Space()
    _static_segment(space, (50, -100), (50, 100), HitClass.WALL)
    _static_segment(space, (20, -100), (20, 100), HitClass.OBJECT, object_index=7)
    features, hit_object_index = cast_rays(
        space, origin=(0.0, 0.0), n_rays=4, lidar_range=100.0
    )
    assert features[0, 0] == pytest.approx(0.2)
    assert features[0, 1 + HitClass.OBJECT] == 1.0
    assert features[0, 1 + HitClass.WALL] == 0.0
    assert hit_object_index[0] == 7


def test_ray_index_corresponds_to_uniform_world_frame_angle():
    space = pymunk.Space()
    # place an object exactly along the angle of ray index 3 out of 8 (3 * 2*pi/8 = 3*pi/4)
    angle = 3 * (2 * np.pi / 8)
    px, py = 30 * np.cos(angle), 30 * np.sin(angle)
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Circle(body, radius=1.0, offset=(px, py))
    shape.collision_type = HitClass.OBJECT
    shape.user_data = 2
    space.add(body, shape)

    features, hit_object_index = cast_rays(
        space, origin=(0.0, 0.0), n_rays=8, lidar_range=100.0
    )
    assert features[3, 1 + HitClass.OBJECT] == 1.0
    assert hit_object_index[3] == 2
    for i in (0, 1, 2, 4, 5, 6, 7):
        assert features[i, 1 + HitClass.NONE] == 1.0
        assert hit_object_index[i] == -1


def test_multiple_objects_report_distinct_indices():
    space = pymunk.Space()
    _static_segment(space, (10, -100), (10, 100), HitClass.OBJECT, object_index=0)
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, (-10, -100), (-10, 100), radius=0.0)
    shape.collision_type = HitClass.OBJECT
    shape.user_data = 1
    space.add(body, shape)

    _, hit_object_index = cast_rays(
        space, origin=(0.0, 0.0), n_rays=4, lidar_range=100.0
    )
    assert hit_object_index[0] == 0
    assert hit_object_index[2] == 1


@given(
    st.integers(min_value=1, max_value=32), st.floats(min_value=1.0, max_value=500.0)
)
def test_empty_space_reports_max_range_and_none_for_every_ray(n_rays, lidar_range):
    space = pymunk.Space()
    features, hit_object_index = cast_rays(
        space, origin=(0.0, 0.0), n_rays=n_rays, lidar_range=lidar_range
    )
    assert np.all(features[:, 0] == 1.0)
    assert np.all(features[:, 1 + HitClass.NONE] == 1.0)
    assert np.all(hit_object_index == -1)


def test_shape_filter_excludes_masked_shapes():
    space = pymunk.Space()
    excluded_category = 0b10
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, (5, -100), (5, 100), radius=0.0)
    shape.collision_type = HitClass.OBJECT
    shape.user_data = 3
    shape.filter = pymunk.ShapeFilter(categories=excluded_category)
    space.add(body, shape)
    _static_segment(space, (50, -100), (50, 100), HitClass.WALL)

    query_filter = pymunk.ShapeFilter(
        mask=pymunk.ShapeFilter.ALL_MASKS() ^ excluded_category
    )
    features, hit_object_index = cast_rays(
        space, origin=(0.0, 0.0), n_rays=4, lidar_range=100.0, shape_filter=query_filter
    )

    assert features[0, 0] == pytest.approx(0.5)
    assert features[0, 1 + HitClass.WALL] == 1.0
    assert hit_object_index[0] == -1
