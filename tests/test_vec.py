from __future__ import annotations

import numpy as np
import pytest

from push_me.config import PushTPOConfig
from push_me.vec import make_vec_env


def test_make_vec_env_has_requested_num_envs():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full")
    vec_env = make_vec_env(cfg, n_envs=3)
    try:
        assert vec_env.num_envs == 3
    finally:
        vec_env.close()


def test_vec_env_reset_and_step_shapes():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="lidar", n_rays=16)
    vec_env = make_vec_env(cfg, n_envs=2)
    try:
        obs, infos = vec_env.reset(seed=0)
        assert obs.shape == (2,) + vec_env.single_observation_space.shape

        actions = np.stack([vec_env.single_action_space.sample() for _ in range(2)])
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        assert obs.shape == (2,) + vec_env.single_observation_space.shape
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
    finally:
        vec_env.close()


def test_vec_env_sub_envs_are_independent_not_lockstep_identical():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full")
    vec_env = make_vec_env(cfg, n_envs=4)
    try:
        obs, _infos = vec_env.reset(seed=0)
        assert not np.array_equal(obs[0], obs[1]), "sub-envs should not share identical seeded state"
    finally:
        vec_env.close()


def test_vec_env_config_sharing_does_not_corrupt_across_sub_envs():
    cfg = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full")
    vec_env = make_vec_env(cfg, n_envs=2)
    try:
        vec_env.reset(seed=0)
        assert cfg.n_objects == 2, "make_vec_env must not mutate the caller's config"
    finally:
        vec_env.close()
