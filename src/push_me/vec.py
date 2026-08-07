from __future__ import annotations

import functools

from gymnasium.vector import AsyncVectorEnv

from push_me.config import PushTPOConfig
from push_me.env import PushTPOEnv


def make_vec_env(config: PushTPOConfig, n_envs: int) -> AsyncVectorEnv:
    env_fns = [functools.partial(PushTPOEnv, config) for _ in range(n_envs)]
    return AsyncVectorEnv(env_fns)
