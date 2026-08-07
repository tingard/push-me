from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import push_me  # noqa: F401  (side effect: registers the presets with gymnasium)

_EXPECTED = {
    "PushTPO-Full-Single-v0": dict(obs_mode="full", n_objects=1, shapes=["t_tetromino"], goal_margin=8.0),
    "PushTPO-Lidar-Single-v0": dict(obs_mode="lidar", n_objects=1, shapes=["t_tetromino"], goal_margin=8.0),
    "PushTPO-Full-Multimodal-v0": dict(obs_mode="full", n_objects=1, shapes=["hexagon"], goal_margin=24.0),
    "PushTPO-Lidar-Multimodal-v0": dict(obs_mode="lidar", n_objects=1, shapes=["hexagon"], goal_margin=24.0),
    "PushTPO-Lidar-Multi3-v0": dict(
        obs_mode="lidar", n_objects=3, shapes=["square"], goal_margin=24.0, assignment_mode="free"
    ),
    "PushTPO-Lidar-Trap-v0": dict(
        obs_mode="lidar",
        n_objects=3,
        shapes=["square"],
        goal_margin=24.0,
        assignment_mode="free",
        traps=True,
    ),
}


@pytest.mark.parametrize("preset_id", sorted(_EXPECTED))
def test_preset_is_registered_and_makeable(preset_id):
    env = gym.make(preset_id)
    assert env is not None
    env.close()


@pytest.mark.parametrize("preset_id, expected", sorted(_EXPECTED.items()))
def test_preset_config_matches_spec_table(preset_id, expected):
    env = gym.make(preset_id).unwrapped
    for key, value in expected.items():
        assert getattr(env.config, key) == value, f"{preset_id}.{key}"


def test_trap_preset_actually_enables_traps():
    env = gym.make("PushTPO-Lidar-Trap-v0").unwrapped
    assert env.config.traps is True
    assert env.config.n_traps > 0


@pytest.mark.parametrize("preset_id", sorted(_EXPECTED))
def test_preset_env_runs_a_few_steps(preset_id):
    env = gym.make(preset_id)
    obs, info = env.reset(seed=0)
    assert obs in env.observation_space
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()


@pytest.mark.parametrize("preset_id", sorted(_EXPECTED))
def test_two_makes_of_the_same_preset_have_independent_configs(preset_id):
    env_a = gym.make(preset_id).unwrapped
    env_b = gym.make(preset_id).unwrapped
    assert env_a.config is not env_b.config
    env_a.config.n_objects = 999
    assert env_b.config.n_objects != 999


def test_factorial_four_presets_cover_the_two_by_two_of_obs_mode_and_margin():
    combos = {
        (_EXPECTED[p]["obs_mode"], _EXPECTED[p]["goal_margin"])
        for p in (
            "PushTPO-Full-Single-v0",
            "PushTPO-Lidar-Single-v0",
            "PushTPO-Full-Multimodal-v0",
            "PushTPO-Lidar-Multimodal-v0",
        )
    }
    assert combos == {("full", 8.0), ("lidar", 8.0), ("full", 24.0), ("lidar", 24.0)}
