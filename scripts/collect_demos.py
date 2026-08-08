from __future__ import annotations

import argparse
from typing import cast

import gymnasium as gym
import pygame

import push_me  # noqa: F401  (side effect: registers the presets with gymnasium)
from push_me.demo_collection import EpisodeOutcome, collect_one_episode
from push_me.demo_storage import ReplayBufferWriter, summarize_achieved_modes
from push_me.env import PushTPOEnv
from push_me.teleop import mouse_to_action


def _await_next_rollout(env: PushTPOEnv) -> bool:
    """Hold the current frame until Space starts the next rollout -- otherwise the
    first actions of a fresh episode chase wherever the mouse was left from the
    previous one. Returns False if the operator quit (Esc / window close) instead."""
    env._ensure_renderer()
    assert env._renderer is not None
    env._renderer.paused = True
    while True:
        env.render()
        if env._renderer is None:
            return False
        if not env._renderer.paused:
            return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect teleoperated demonstrations for behaviour cloning."
    )
    parser.add_argument("--preset", default="PushTPO-Lidar-Multi3-v0")
    parser.add_argument("--output", required=True, help="path to the zarr store")
    parser.add_argument(
        "--n-episodes", type=int, default=10, help="number of KEPT episodes to collect"
    )
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    env = cast(PushTPOEnv, gym.make(args.preset, render_mode="human").unwrapped)
    # _await_next_rollout below assumes physics state already exists (it only skips
    # drawing while paused, not while uninitialised) -- collect_one_episode's first
    # call re-resets with the same seed, so this isn't wasted: same seed means the
    # previewed layout is bit-identical to what actually gets collected (§14).
    env.reset(seed=args.seed_start)
    writer = ReplayBufferWriter(args.output, env.config)

    print(
        f"collecting {args.n_episodes} episodes of {args.preset} -> {args.output}\n"
        "mouse drives the pusher -- N discards the current episode, Esc quits\n"
        "each rollout starts paused -- reposition the mouse, then press Space"
    )

    def get_action():
        return mouse_to_action(pygame.mouse.get_pos(), env.config.arena_size)

    kept = 0
    seed = args.seed_start
    while kept < args.n_episodes:
        if not _await_next_rollout(env):
            print("quit requested, stopping early")
            break

        outcome, steps = collect_one_episode(env, seed=seed, get_action=get_action)

        if outcome is EpisodeOutcome.QUIT:
            print("quit requested, stopping early")
            break
        if outcome is EpisodeOutcome.DISCARDED:
            print(f"episode discarded (seed={seed})")
            seed += 1
            continue

        assert steps is not None
        writer.append_episode(seed=seed, steps=steps)
        kept += 1
        print(
            f"kept episode {kept}/{args.n_episodes} (seed={seed}, {len(steps['action'])} steps)"
        )
        seed += 1

    env.close()

    if writer.n_episodes > 0:
        counts = summarize_achieved_modes(args.output)
        total = sum(counts.values())
        print(
            "\nachieved-mode distribution across kept episodes "
            "(operator preference confound -- SPEC.md section 13):"
        )
        for mode, count in sorted(counts.items()):
            print(f"  mode {mode}: {count}/{total} ({count / total:.0%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
