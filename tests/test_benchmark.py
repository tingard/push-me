from __future__ import annotations

import pathlib
import subprocess
import sys
import time

from push_me.config import PushTPOConfig
from push_me.env import PushTPOEnv

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_env_step_throughput_regression_guard():
    # Not SPEC.md's 2000/sec/core acceptance target (verified by manually running
    # scripts/benchmark_speed.py) -- a much lower floor, just to catch a severe
    # regression without being flaky on slower/shared CI hardware.
    cfg = PushTPOConfig(n_objects=1, shapes=["t_tetromino"], obs_mode="lidar", n_rays=64)
    env = PushTPOEnv(cfg)
    env.reset(seed=0)
    for _ in range(10):
        env.step(env.action_space.sample())

    n = 200
    start = time.perf_counter()
    for _ in range(n):
        env.step(env.action_space.sample())
    elapsed = time.perf_counter() - start

    rate = n / elapsed
    assert rate > 500, f"env.step() throughput regressed badly: {rate:.1f} steps/sec (expected > 500)"


def test_benchmark_speed_script_runs_end_to_end():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_speed.py",
            "--skip-vec-env",
            "--n-steps",
            "50",
            "--warmup-steps",
            "5",
            "--min-steps-per-sec-per-core",
            "100",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
