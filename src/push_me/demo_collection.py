from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto

import numpy as np
import pygame

from push_me.env import PushTPOEnv

_INFO_FIELDS = (
    "is_success",
    "containment_errors",
    "assignment",
    "object_poses",
    "steps_since_observed",
    "achieved_mode",
    "n_objects_trapped",
)
RECORDED_FIELDS = (
    "obs_full",
    "obs_lidar",
    "action",
    "reward",
    "terminated",
    "truncated",
) + _INFO_FIELDS


class EpisodeOutcome(Enum):
    KEPT = auto()
    DISCARDED = auto()
    QUIT = auto()


def collect_one_episode(
    env: PushTPOEnv, seed: int, get_action: Callable[[], np.ndarray]
) -> tuple[EpisodeOutcome, dict[str, np.ndarray] | None]:
    env.reset(seed=seed)
    if env._renderer is not None:
        env._renderer.consume_unhandled_keys()
    buffers: dict[str, list] = {key: [] for key in RECORDED_FIELDS}

    while True:
        action = get_action()
        obs_full = env.compute_observation("full")
        obs_lidar = env.compute_observation("lidar")
        # step() already auto-renders in human mode, draining pygame's event queue; a
        # second render() call here would recreate a fresh window right after Esc closes it.
        _obs, reward, terminated, truncated, info = env.step(action)
        if env._renderer is None:
            return EpisodeOutcome.QUIT, None
        if pygame.K_n in env._renderer.consume_unhandled_keys():
            return EpisodeOutcome.DISCARDED, None

        buffers["obs_full"].append(obs_full)
        buffers["obs_lidar"].append(obs_lidar)
        buffers["action"].append(np.asarray(action, dtype=np.float32))
        buffers["reward"].append(np.float32(reward))
        buffers["terminated"].append(terminated)
        buffers["truncated"].append(truncated)
        for key in _INFO_FIELDS:
            buffers[key].append(info[key])

        if terminated or truncated:
            break

    return EpisodeOutcome.KEPT, {
        key: np.stack(values) for key, values in buffers.items()
    }
