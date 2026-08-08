from __future__ import annotations

import dataclasses

from push_me.config import PushTPOConfig

EXPECTED_DEFAULTS = {
    "n_objects": 1,
    "shapes": ["t_tetromino"],
    "shape_sampling": "fixed",
    "assignment_mode": "free",
    "goal_margin": 8.0,
    "obs_mode": "lidar",
    "n_rays": 128,
    "lidar_range": 300.0,
    "occluder_walls": 0,
    "arena_size": 512.0,
    "shape_area": 4000.0,
    "pusher_radius": 15.0,
    "sim_substeps": 10,
    "dt": 0.01,
    "damping": 0.05,
    "action_mode": "absolute",
    "max_delta": 30.0,
    "kp": 100.0,
    "kd": 10.0,
    "max_push_force": 500.0,
    "max_steps": 1000,
    "success_hold_steps": 10,
    "sparse_only": False,
    "success_bonus": 1.0,
    "dense_weight": 0.01,
    "traps": False,
    "n_traps": 0,
    "seed": None,
}


def test_field_set_matches_spec_exactly():
    field_names = {f.name for f in dataclasses.fields(PushTPOConfig)}
    assert field_names == set(EXPECTED_DEFAULTS)


def test_default_values_match_spec_table():
    config = PushTPOConfig()
    for name, expected in EXPECTED_DEFAULTS.items():
        assert getattr(config, name) == expected, f"{name}: expected {expected!r}"


def test_shapes_default_is_independent_per_instance():
    a, b = PushTPOConfig(), PushTPOConfig()
    assert a.shapes == b.shapes
    assert a.shapes is not b.shapes
    a.shapes.append("square")
    assert b.shapes == ["t_tetromino"]


def test_keyword_overrides_apply():
    config = PushTPOConfig(
        n_objects=3, shapes=["square", "square", "square"], goal_margin=24.0
    )
    assert config.n_objects == 3
    assert config.shapes == ["square", "square", "square"]
    assert config.goal_margin == 24.0
    assert config.obs_mode == "lidar"
