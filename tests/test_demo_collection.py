from __future__ import annotations

import numpy as np
import pygame

from push_me.config import PushTPOConfig
from push_me.demo_collection import RECORDED_FIELDS, EpisodeOutcome, collect_one_episode
from push_me.env import PushTPOEnv


def _fixed_action_source(action=(0.0, 0.0)):
    arr = np.array(action, dtype=np.float32)
    return lambda: arr


def test_collect_one_episode_kept_records_all_fields_with_matching_length(
    sdl_dummy_driver,
):
    cfg = PushTPOConfig(
        n_objects=1, shapes=["square"], obs_mode="full", n_rays=16, max_steps=5
    )
    env = PushTPOEnv(cfg, render_mode="human")

    outcome, steps = collect_one_episode(env, seed=0, get_action=_fixed_action_source())

    assert outcome is EpisodeOutcome.KEPT
    assert steps is not None
    assert set(steps) == set(RECORDED_FIELDS)
    lengths = {len(arr) for arr in steps.values()}
    assert lengths == {5}
    env.close()


def test_collect_one_episode_records_both_observation_modes_at_correct_dims(
    sdl_dummy_driver,
):
    cfg = PushTPOConfig(
        n_objects=2, shapes=["square"], obs_mode="full", n_rays=16, max_steps=3
    )
    env = PushTPOEnv(cfg, render_mode="human")

    _outcome, steps = collect_one_episode(
        env, seed=1, get_action=_fixed_action_source()
    )
    assert steps is not None

    expected_full_dim = 4 + cfg.n_objects * (7 + 9) + cfg.n_objects * 6
    expected_lidar_dim = 4 + cfg.n_rays * 4 + cfg.n_objects * (6 + 9)
    assert steps["obs_full"].shape == (3, expected_full_dim)
    assert steps["obs_lidar"].shape == (3, expected_lidar_dim)
    assert steps["action"].shape == (3, 2)
    env.close()


def test_collect_one_episode_discarded_on_n_returns_none(sdl_dummy_driver):
    cfg = PushTPOConfig(n_objects=1, obs_mode="full", max_steps=100, seed=0)
    env = PushTPOEnv(cfg, render_mode="human")

    step_count = 0

    def get_action():
        nonlocal step_count
        step_count += 1
        if step_count == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
        return np.array([0.0, 0.0], dtype=np.float32)

    outcome, steps = collect_one_episode(env, seed=2, get_action=get_action)
    assert outcome is EpisodeOutcome.DISCARDED
    assert steps is None
    env.close()


def test_collect_one_episode_quit_mid_episode_returns_quit_outcome(sdl_dummy_driver):
    cfg = PushTPOConfig(n_objects=1, obs_mode="full", max_steps=100, seed=0)
    env = PushTPOEnv(cfg, render_mode="human")

    step_count = 0

    def get_action():
        nonlocal step_count
        step_count += 1
        if step_count == 3:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        return np.array([0.0, 0.0], dtype=np.float32)

    outcome, steps = collect_one_episode(env, seed=3, get_action=get_action)
    assert outcome is EpisodeOutcome.QUIT
    assert steps is None


def test_collect_one_episode_achieved_mode_and_assignment_are_recorded_per_step(
    sdl_dummy_driver,
):
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", max_steps=4)
    env = PushTPOEnv(cfg, render_mode="human")

    _outcome, steps = collect_one_episode(
        env, seed=4, get_action=_fixed_action_source()
    )
    assert steps is not None
    assert steps["achieved_mode"].shape == (4, 1)
    assert steps["assignment"].shape == (4, 1)
    env.close()
