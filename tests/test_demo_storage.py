from __future__ import annotations

import numpy as np
import pytest
import zarr

from push_me.config import PushTPOConfig
from push_me.demo_storage import ReplayBufferWriter, read_config, read_episode, summarize_achieved_modes


def _episode(n_steps: int, obj_mode_value: int) -> dict[str, np.ndarray]:
    return {
        "obs_full": np.random.default_rng(0).normal(size=(n_steps, 5)).astype(np.float32),
        "action": np.random.default_rng(0).normal(size=(n_steps, 2)).astype(np.float32),
        "achieved_mode": np.full((n_steps, 1), obj_mode_value, dtype=np.int64),
    }


def test_write_creates_diffusion_policy_style_zarr_v2_layout(tmp_path):
    path = tmp_path / "demos.zarr"
    writer = ReplayBufferWriter(path, PushTPOConfig(n_objects=1))
    writer.append_episode(seed=0, steps=_episode(10, obj_mode_value=1))

    assert (path / ".zgroup").exists()
    assert (path / "data" / ".zgroup").exists()
    assert (path / "meta" / ".zgroup").exists()
    assert (path / "data" / "obs_full" / ".zarray").exists()
    assert (path / "meta" / "episode_ends" / ".zarray").exists()


def test_append_episode_grows_data_arrays_and_episode_ends(tmp_path):
    path = tmp_path / "demos.zarr"
    writer = ReplayBufferWriter(path, PushTPOConfig(n_objects=1))
    writer.append_episode(seed=1, steps=_episode(10, 0))
    writer.append_episode(seed=2, steps=_episode(15, 1))

    root = zarr.open_group(str(path), mode="r")
    assert root["data"]["obs_full"].shape == (25, 5)
    assert root["data"]["action"].shape == (25, 2)
    assert root["meta"]["episode_ends"][:].tolist() == [10, 25]
    assert root["meta"]["episode_seeds"][:].tolist() == [1, 2]
    assert writer.n_episodes == 2
    assert writer.n_steps == 25


def test_append_episode_rejects_mismatched_field_lengths(tmp_path):
    writer = ReplayBufferWriter(tmp_path / "demos.zarr", PushTPOConfig(n_objects=1))
    with pytest.raises(ValueError):
        writer.append_episode(
            seed=0,
            steps={"obs_full": np.zeros((10, 5), dtype=np.float32), "action": np.zeros((9, 2), dtype=np.float32)},
        )


def test_append_empty_episode_is_a_no_op(tmp_path):
    writer = ReplayBufferWriter(tmp_path / "demos.zarr", PushTPOConfig(n_objects=1))
    writer.append_episode(seed=0, steps={"obs_full": np.zeros((0, 5), dtype=np.float32)})
    assert writer.n_episodes == 0


def test_config_round_trips_through_attrs(tmp_path):
    path = tmp_path / "demos.zarr"
    original = PushTPOConfig(n_objects=3, shapes=["square"], goal_margin=24.0)
    writer = ReplayBufferWriter(path, original)
    writer.append_episode(seed=0, steps=_episode(5, 0))

    loaded = read_config(path)
    assert loaded == original


def test_read_episode_slices_out_the_correct_range(tmp_path):
    path = tmp_path / "demos.zarr"
    writer = ReplayBufferWriter(path, PushTPOConfig(n_objects=1))
    writer.append_episode(seed=0, steps=_episode(10, 0))
    writer.append_episode(seed=1, steps=_episode(6, 1))

    ep0 = read_episode(path, 0)
    ep1 = read_episode(path, 1)
    assert ep0["obs_full"].shape == (10, 5)
    assert ep1["obs_full"].shape == (6, 5)
    assert np.all(ep0["achieved_mode"] == 0)
    assert np.all(ep1["achieved_mode"] == 1)


def test_summarize_achieved_modes_counts_final_step_per_episode(tmp_path):
    path = tmp_path / "demos.zarr"
    writer = ReplayBufferWriter(path, PushTPOConfig(n_objects=1))
    writer.append_episode(seed=0, steps=_episode(5, obj_mode_value=0))
    writer.append_episode(seed=1, steps=_episode(5, obj_mode_value=0))
    writer.append_episode(seed=2, steps=_episode(5, obj_mode_value=2))

    counts = summarize_achieved_modes(path)
    assert counts == {0: 2, 2: 1}


def test_multiple_episodes_survive_reopening_the_store(tmp_path):
    path = tmp_path / "demos.zarr"
    writer = ReplayBufferWriter(path, PushTPOConfig(n_objects=1))
    writer.append_episode(seed=0, steps=_episode(4, 0))
    del writer

    writer2 = ReplayBufferWriter(path, PushTPOConfig(n_objects=1))
    writer2.append_episode(seed=1, steps=_episode(6, 1))

    root = zarr.open_group(str(path), mode="r")
    assert root["meta"]["episode_ends"][:].tolist() == [4, 10]
