from __future__ import annotations

import argparse
import os
import time

import numpy as np

from push_me.config import PushTPOConfig
from push_me.env import PushTPOEnv
from push_me.vec import make_vec_env

_BENCHMARK_CONFIG = PushTPOConfig(n_objects=1, shapes=["t_tetromino"], obs_mode="lidar", n_rays=64)


def run_single_process_benchmark(n_steps: int, warmup_steps: int) -> float:
    env = PushTPOEnv(_BENCHMARK_CONFIG)
    env.reset(seed=0)
    action_space = env.action_space

    for _ in range(warmup_steps):
        env.step(action_space.sample())

    start = time.perf_counter()
    for _ in range(n_steps):
        env.step(action_space.sample())
    elapsed = time.perf_counter() - start

    return n_steps / elapsed


def run_vec_env_benchmark(n_envs: int, n_steps: int, warmup_steps: int) -> float:
    vec_env = make_vec_env(_BENCHMARK_CONFIG, n_envs=n_envs)
    try:
        vec_env.reset(seed=0)
        action_space = vec_env.single_action_space

        for _ in range(warmup_steps):
            actions = np.stack([action_space.sample() for _ in range(n_envs)])
            vec_env.step(actions)

        start = time.perf_counter()
        for _ in range(n_steps):
            actions = np.stack([action_space.sample() for _ in range(n_envs)])
            vec_env.step(actions)
        elapsed = time.perf_counter() - start
    finally:
        vec_env.close()

    total_steps = n_steps * n_envs
    return (total_steps / elapsed) / n_envs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PushTPOEnv step throughput. Pass/fail is gated on the raw "
            "single-process rate (SPEC.md's acceptance test #5: 'benchmark_speed.py "
            "reports >= 2000 steps/sec/core headless'), since that isolates the "
            "environment's own computational cost from gymnasium.vector.AsyncVectorEnv's "
            "inter-process communication overhead, which doesn't scale linearly with "
            "worker count for an environment this cheap per step. The AsyncVectorEnv "
            "number is reported for transparency, not gated on."
        )
    )
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--min-steps-per-sec-per-core", type=float, default=2000.0)
    parser.add_argument("--n-envs", type=int, default=os.cpu_count() or 1, help="for the reported AsyncVectorEnv number only")
    parser.add_argument("--skip-vec-env", action="store_true", help="skip the AsyncVectorEnv comparison run")
    args = parser.parse_args()

    single_rate = run_single_process_benchmark(args.n_steps, args.warmup_steps)
    print(
        f"single-process: {single_rate:.1f} steps/sec/core "
        f"(target={args.min_steps_per_sec_per_core:.0f}) [gates pass/fail]"
    )

    if not args.skip_vec_env:
        vec_rate = run_vec_env_benchmark(args.n_envs, args.n_steps, args.warmup_steps)
        print(
            f"AsyncVectorEnv (n_envs={args.n_envs}): {vec_rate:.1f} steps/sec/core "
            "[reported for transparency, not gated]"
        )

    if single_rate < args.min_steps_per_sec_per_core:
        print("FAIL: single-process rate below target")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
