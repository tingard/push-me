from __future__ import annotations

import pathlib

import numpy as np
from conftest import pymunk_settings
from hypothesis import given
from hypothesis import strategies as st

from push_me.config import PushTPOConfig
from push_me.env import PushTPOEnv

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "push_me"

_actions = st.lists(
    st.tuples(
        st.floats(min_value=-1, max_value=1, allow_nan=False),
        st.floats(min_value=-1, max_value=1, allow_nan=False),
    ).map(lambda t: np.array(t, dtype=np.float32)),
    min_size=0,
    max_size=8,
)


def _copy_info(info: dict) -> dict:
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in info.items()}


def _rollout(
    config: PushTPOConfig, seed: int, actions: list[np.ndarray]
) -> list[tuple]:
    env = PushTPOEnv(config)
    obs, info = env.reset(seed=seed)
    trace: list[tuple[np.ndarray, float | None, bool | None, bool | None, dict]] = [
        (obs.copy(), None, None, None, _copy_info(info))
    ]
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        trace.append((obs.copy(), reward, terminated, truncated, _copy_info(info)))
    return trace


def _assert_traces_equal(trace_a: list[tuple], trace_b: list[tuple]) -> None:
    assert len(trace_a) == len(trace_b)
    for (obs_a, r_a, term_a, trunc_a, info_a), (
        obs_b,
        r_b,
        term_b,
        trunc_b,
        info_b,
    ) in zip(trace_a, trace_b):
        assert np.array_equal(obs_a, obs_b)
        assert r_a == r_b
        assert term_a == term_b
        assert trunc_a == trunc_b
        assert set(info_a) == set(info_b)
        for key in info_a:
            va, vb = info_a[key], info_b[key]
            if isinstance(va, np.ndarray):
                assert np.array_equal(va, vb), f"info[{key!r}] mismatch"
            else:
                assert va == vb, f"info[{key!r}] mismatch"


# ---- source audit: this is what makes the bit-identical claim meaningful ----


def test_source_has_no_stray_global_randomness():
    offenders = []
    for path in _SRC_DIR.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if (
                "np.random." in line
                and "np.random.default_rng" not in line
                and "np.random.Generator" not in line
            ):
                offenders.append(
                    f"{path.relative_to(_SRC_DIR.parent.parent)}:{lineno}: {stripped}"
                )
            if stripped.startswith(("import random", "from random import")):
                offenders.append(
                    f"{path.relative_to(_SRC_DIR.parent.parent)}:{lineno}: {stripped}"
                )
    assert not offenders, "found stray global-randomness usage:\n" + "\n".join(
        offenders
    )


# ---- seeding actually does something (a precondition for the bit-identical claim to be meaningful) ----


def test_different_seeds_produce_different_initial_observations():
    config = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full")
    obs_a, _info_a = PushTPOEnv(config).reset(seed=1)
    obs_b, _info_b = PushTPOEnv(config).reset(seed=2)
    assert not np.array_equal(obs_a, obs_b)


def test_reset_without_seed_continues_the_existing_stream():
    config = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full", seed=123)
    env = PushTPOEnv(config)
    obs1, _info1 = env.reset()
    obs2, _info2 = env.reset()
    assert not np.array_equal(obs1, obs2)


def test_reset_with_explicit_seed_resets_the_stream():
    config = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full")
    env = PushTPOEnv(config)
    obs1, _info1 = env.reset(seed=7)
    env.reset()
    obs2, _info2 = env.reset(seed=7)
    assert np.array_equal(obs1, obs2)


# ---- the acceptance test: bit-identical replay for any seed / action sequence ----


@pymunk_settings
@given(st.integers(min_value=0, max_value=10_000), _actions)
def test_bit_identical_replay_full_obs(seed, actions):
    config = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full")
    _assert_traces_equal(
        _rollout(config, seed, actions), _rollout(config, seed, actions)
    )


@pymunk_settings
@given(st.integers(min_value=0, max_value=10_000), _actions)
def test_bit_identical_replay_lidar_obs(seed, actions):
    config = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="lidar", n_rays=16)
    _assert_traces_equal(
        _rollout(config, seed, actions), _rollout(config, seed, actions)
    )


@pymunk_settings
@given(st.integers(min_value=0, max_value=10_000), _actions)
def test_bit_identical_replay_with_traps_and_multiple_objects(seed, actions):
    config = PushTPOConfig(
        n_objects=3,
        shapes=["square"],
        obs_mode="lidar",
        n_rays=16,
        traps=True,
        n_traps=2,
    )
    _assert_traces_equal(
        _rollout(config, seed, actions), _rollout(config, seed, actions)
    )


@pymunk_settings
@given(st.integers(min_value=0, max_value=10_000), _actions)
def test_bit_identical_replay_delta_action_mode(seed, actions):
    config = PushTPOConfig(
        n_objects=1, shapes=["hexagon"], obs_mode="full", action_mode="delta"
    )
    _assert_traces_equal(
        _rollout(config, seed, actions), _rollout(config, seed, actions)
    )
