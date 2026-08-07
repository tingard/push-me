from __future__ import annotations

import argparse

import gymnasium as gym
import pygame

import push_me  # noqa: F401  (side effect: registers the presets with gymnasium)
from push_me.teleop import mouse_to_action

_PRESET_IDS = [
    "PushTPO-Full-Single-v0",
    "PushTPO-Lidar-Single-v0",
    "PushTPO-Full-Multimodal-v0",
    "PushTPO-Lidar-Multimodal-v0",
    "PushTPO-Lidar-Multi3-v0",
    "PushTPO-Lidar-Trap-v0",
]


def _make_env(preset_id: str, seed: int | None):
    env = gym.make(preset_id, render_mode="human").unwrapped
    env.reset(seed=seed)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Manually play a PushT-PO preset with mouse teleop.")
    parser.add_argument("--preset", default="PushTPO-Lidar-Multi3-v0", choices=_PRESET_IDS)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    preset_index = _PRESET_IDS.index(args.preset)
    env = _make_env(_PRESET_IDS[preset_index], args.seed)

    print(
        f"playing {_PRESET_IDS[preset_index]}\n"
        "mouse drives the pusher -- Space=pause R=reset L=lidar B=belief "
        "G=ground-truth Tab=next preset Esc=quit"
    )

    while True:
        renderer = env._renderer
        if renderer is None:
            break

        if renderer.consume_preset_cycle_request():
            env.close()
            preset_index = (preset_index + 1) % len(_PRESET_IDS)
            env = _make_env(_PRESET_IDS[preset_index], args.seed)
            print(f"switched to {_PRESET_IDS[preset_index]}")
            continue

        action = mouse_to_action(pygame.mouse.get_pos(), env.config.arena_size)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            env.reset(seed=args.seed)

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
