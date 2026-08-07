from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from hypothesis import given, strategies as st

from push_me.config import PushTPOConfig
from push_me.env import PushTPOEnv
from push_me.geometry import min_area_rect
from push_me.shapes import SHAPES

from conftest import pymunk_settings

_EXPECTED_INFO_KEYS = {
    "is_success",
    "containment_errors",
    "assignment",
    "object_poses",
    "steps_since_observed",
    "achieved_mode",
    "n_objects_trapped",
}


def _place_object_at_rect(env: PushTPOEnv, obj_idx: int, rect_idx: int, mode: int = 0) -> None:
    rect = env._goal_rects[rect_idx]
    shape = env._object_shapes[obj_idx]
    body = env._object_bodies[obj_idx]
    _half_extents, phi = min_area_rect(shape.outline)
    step = 2 * np.pi / shape.symmetry_order
    body.position = tuple(rect.center)
    body.angle = (rect.angle - phi) + mode * step
    body.velocity = (0, 0)
    body.angular_velocity = 0.0


# ---- info dict / observation contract ----


def test_reset_info_has_exactly_the_spec_keys():
    env = PushTPOEnv(PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full", seed=0))
    _obs, info = env.reset(seed=0)
    assert set(info) == _EXPECTED_INFO_KEYS


def test_step_info_has_exactly_the_spec_keys():
    env = PushTPOEnv(PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="lidar", seed=0))
    env.reset(seed=0)
    _obs, _r, _term, _trunc, info = env.step(np.array([0.0, 0.0]))
    assert set(info) == _EXPECTED_INFO_KEYS


@pymunk_settings
@given(
    st.sampled_from(["full", "lidar"]),
    st.integers(min_value=1, max_value=3),
)
def test_observation_matches_declared_space(obs_mode, n_objects):
    cfg = PushTPOConfig(n_objects=n_objects, shapes=["square"], obs_mode=obs_mode, seed=0, n_rays=16)
    env = PushTPOEnv(cfg)
    obs, _info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    obs, _r, _term, _trunc, _info = env.step(np.array([0.3, -0.4], dtype=np.float32))
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_gymnasium_check_env_full():
    env = PushTPOEnv(PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full", seed=0))
    check_env(env, skip_render_check=True)


def test_gymnasium_check_env_lidar():
    env = PushTPOEnv(PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="lidar", seed=0))
    check_env(env, skip_render_check=True)


def test_object_poses_is_ground_truth_position_and_angle():
    env = PushTPOEnv(PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="lidar", seed=0))
    _obs, info = env.reset(seed=0)
    for i, body in enumerate(env._object_bodies):
        assert info["object_poses"][i] == pytest.approx([body.position.x, body.position.y, body.angle])


# ---- reward ----


def test_reward_matches_hand_computed_value_for_known_placement():
    cfg = PushTPOConfig(
        n_objects=1, shapes=["square"], shape_area=4000.0, goal_margin=8.0, obs_mode="full", seed=5
    )
    env = PushTPOEnv(cfg)
    env.reset(seed=5)
    _place_object_at_rect(env, 0, 0)

    _obs, reward, _term, _trunc, info = env.step(np.array([0.0, 0.0]))

    error = info["containment_errors"][0]
    expected = cfg.success_bonus * float(info["is_success"]) - cfg.dense_weight * error
    assert reward == pytest.approx(expected, abs=1e-9)
    assert info["is_success"] is True
    assert error < 0


def test_sparse_only_drops_the_dense_term():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", seed=6, sparse_only=True)
    env = PushTPOEnv(cfg)
    env.reset(seed=6)
    # do not place object at its goal -> not a success, dense term (if present) would be nonzero
    _obs, reward, _term, _trunc, info = env.step(np.array([0.0, 0.0]))
    assert info["is_success"] is False
    assert reward == pytest.approx(0.0)


# ---- termination ----


def test_success_hold_steps_terminates_at_exact_count():
    cfg = PushTPOConfig(
        n_objects=1, shapes=["square"], obs_mode="full", seed=7, success_hold_steps=5, max_steps=300
    )
    env = PushTPOEnv(cfg)
    env.reset(seed=7)
    _place_object_at_rect(env, 0, 0)

    for step in range(1, 5):
        _obs, _r, terminated, _trunc, info = env.step(np.array([0.0, 0.0]))
        assert info["is_success"] is True
        assert terminated is False, f"terminated early at step {step}"

    _obs, _r, terminated, _trunc, info = env.step(np.array([0.0, 0.0]))
    assert terminated is True


def test_success_streak_resets_if_object_leaves_goal():
    cfg = PushTPOConfig(
        n_objects=1, shapes=["square"], obs_mode="full", seed=8, success_hold_steps=3, max_steps=300
    )
    env = PushTPOEnv(cfg)
    env.reset(seed=8)
    _place_object_at_rect(env, 0, 0)
    env.step(np.array([0.0, 0.0]))
    env.step(np.array([0.0, 0.0]))
    # yank the object far away from its goal, breaking the streak
    env._object_bodies[0].position = (1.0, 1.0)
    env._object_bodies[0].velocity = (0, 0)
    _obs, _r, terminated, _trunc, info = env.step(np.array([0.0, 0.0]))
    assert info["is_success"] is False
    assert terminated is False
    assert env._success_streak == 0


def test_max_steps_truncates_without_success():
    cfg = PushTPOConfig(n_objects=1, shapes=["t_tetromino"], obs_mode="full", seed=9, max_steps=15)
    env = PushTPOEnv(cfg)
    env.reset(seed=9)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _obs, _r, terminated, truncated, _info = env.step(np.array([0.0, 0.0]))
        steps += 1
    assert steps == 15
    assert truncated is True
    assert terminated is False


# ---- assignment modes ----


def test_fixed_assignment_mode_pairs_object_i_with_rect_i():
    cfg = PushTPOConfig(
        n_objects=3, shapes=["square"], assignment_mode="fixed", obs_mode="full", seed=10
    )
    env = PushTPOEnv(cfg)
    _obs, info = env.reset(seed=10)
    assert info["assignment"].tolist() == [0, 1, 2]


# ---- achieved_mode ----


@pymunk_settings
@given(st.sampled_from(sorted(SHAPES)))
def test_achieved_mode_matches_symmetric_orientation_placed(name):
    cfg = PushTPOConfig(n_objects=1, shapes=[name], shape_area=4000.0, obs_mode="full", seed=11)
    env = PushTPOEnv(cfg)
    env.reset(seed=11)
    n = env._object_shapes[0].symmetry_order
    for k in range(n):
        _place_object_at_rect(env, 0, 0, mode=k)
        _obs, _r, _term, _trunc, info = env.step(np.array([0.0, 0.0]))
        assert info["achieved_mode"][0] == k, f"{name} mode {k}: got {info['achieved_mode'][0]}"


# ---- traps ----


def test_n_objects_trapped_counts_object_in_pocket():
    cfg = PushTPOConfig(
        n_objects=1, shapes=["square"], obs_mode="full", seed=12, traps=True, n_traps=1
    )
    env = PushTPOEnv(cfg)
    env.reset(seed=12)
    trap = env._traps[0]
    body = env._object_bodies[0]
    body.position = tuple(trap.center)
    body.angle = trap.angle
    body.velocity = (0, 0)

    _obs, _r, _term, _trunc, info = env.step(np.array([0.0, 0.0]))
    assert info["n_objects_trapped"] == 1


def test_no_traps_by_default_means_zero_trapped():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", seed=13)
    env = PushTPOEnv(cfg)
    env.reset(seed=13)
    _obs, _r, _term, _trunc, info = env.step(np.array([0.0, 0.0]))
    assert info["n_objects_trapped"] == 0


# ---- action modes ----


def test_absolute_action_extremes_target_arena_corners():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", action_mode="absolute", seed=14)
    env = PushTPOEnv(cfg)
    env.reset(seed=14)
    env.step(np.array([-1.0, -1.0]))
    assert env._target == pytest.approx([0.0, 0.0])
    env.step(np.array([1.0, 1.0]))
    assert env._target == pytest.approx([cfg.arena_size, cfg.arena_size])


def test_delta_action_scales_by_max_delta():
    cfg = PushTPOConfig(
        n_objects=1, shapes=["square"], obs_mode="full", action_mode="delta", max_delta=30.0, seed=15
    )
    env = PushTPOEnv(cfg)
    env.reset(seed=15)
    assert env._pusher_body is not None
    pos_before = np.array(env._pusher_body.position)
    env.step(np.array([1.0, 0.0]))
    assert env._target == pytest.approx(pos_before + np.array([30.0, 0.0]))


# ---- determinism smoke (thorough hypothesis-based version lands in Phase 8) ----


def test_same_seed_same_actions_gives_identical_trajectory():
    cfg = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="lidar", seed=None, n_rays=16)
    actions = [np.array([0.2, -0.3]), np.array([-0.5, 0.1]), np.array([0.0, 0.9])]

    def rollout():
        env = PushTPOEnv(cfg)
        obs, info = env.reset(seed=42)
        trace = [obs]
        for a in actions:
            obs, reward, terminated, truncated, info = env.step(a)
            trace.append(obs)
            trace.append(np.array([reward]))
        return trace

    trace_a = rollout()
    trace_b = rollout()
    for a, b in zip(trace_a, trace_b):
        assert np.array_equal(a, b)


# ---- compute_observation(mode): dual-obs recording for demo collection (SPEC.md section 13) ----


def test_compute_observation_full_matches_native_full_mode():
    cfg = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full", seed=20)
    env = PushTPOEnv(cfg)
    obs, _info = env.reset(seed=20)
    assert np.array_equal(env.compute_observation("full"), obs)


def test_compute_observation_lidar_matches_native_lidar_mode():
    cfg = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="lidar", n_rays=16, seed=21)
    env = PushTPOEnv(cfg)
    obs, _info = env.reset(seed=21)
    assert np.array_equal(env.compute_observation("lidar"), obs)


def test_compute_observation_lidar_available_while_running_in_full_mode():
    cfg = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full", n_rays=16, seed=22)
    env = PushTPOEnv(cfg)
    env.reset(seed=22)
    lidar_obs = env.compute_observation("lidar")
    expected_dim = 4 + cfg.n_rays * 4 + cfg.n_objects * (6 + 9)
    assert lidar_obs.shape == (expected_dim,)


def test_compute_observation_full_available_while_running_in_lidar_mode():
    cfg = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="lidar", n_rays=16, seed=23)
    env = PushTPOEnv(cfg)
    env.reset(seed=23)
    full_obs = env.compute_observation("full")
    expected_dim = 4 + cfg.n_objects * (7 + 9) + cfg.n_objects * 6
    assert full_obs.shape == (expected_dim,)


def test_compute_observation_probe_does_not_mutate_steps_since_observed():
    # Checked immediately after the probe calls, on the internal counter directly --
    # info["steps_since_observed"] after a real step() wouldn't catch a regression
    # here, since obs_mode="full" unconditionally zeroes it every step regardless.
    cfg = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full", n_rays=16, seed=24)
    env = PushTPOEnv(cfg)
    env.reset(seed=24)
    before = env._steps_since_observed.copy()

    env.compute_observation("lidar")
    env.compute_observation("lidar")

    assert np.array_equal(env._steps_since_observed, before)


def test_compute_observation_unknown_mode_raises():
    cfg = PushTPOConfig(n_objects=1, obs_mode="full", seed=25)
    env = PushTPOEnv(cfg)
    env.reset(seed=25)
    with pytest.raises(ValueError):
        env.compute_observation("bogus")


# ---- usable dynamics: guards the density=1.0 -> 0.002 fix from Phase 12 ----


def test_pusher_can_cross_most_of_the_arena_within_a_third_of_max_steps():
    cfg = PushTPOConfig(n_objects=1, obs_mode="full", seed=0, max_steps=300)
    env = PushTPOEnv(cfg)
    env.reset(seed=0)
    assert env._pusher_body is not None
    start = np.array(env._pusher_body.position)
    corner = np.array([cfg.arena_size, cfg.arena_size]) if start[0] < cfg.arena_size / 2 else np.zeros(2)
    action = corner / cfg.arena_size * 2 - 1
    for _ in range(100):
        env.step(action)
    distance = np.linalg.norm(np.array(env._pusher_body.position) - start)
    assert distance > cfg.arena_size * 0.5, f"pusher only moved {distance:.1f} units in 100 steps"


def test_pusher_can_push_an_object_a_meaningful_distance():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", seed=0, max_steps=300)
    env = PushTPOEnv(cfg)
    env.reset(seed=0)
    body = env._object_bodies[0]
    start = np.array(body.position)
    assert env._pusher_body is not None
    env._pusher_body.position = tuple(start - np.array([0.0, 60.0]))
    env._pusher_body.velocity = (0, 0)
    body.velocity = (0, 0)
    target = np.clip(start + np.array([0.0, 300.0]), 0, cfg.arena_size)
    action = target / cfg.arena_size * 2 - 1
    for _ in range(60):
        env.step(action)
    distance = np.linalg.norm(np.array(body.position) - start)
    assert distance > 30.0, f"object only moved {distance:.1f} units after 60 steps of sustained pushing"
