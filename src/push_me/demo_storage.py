from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import zarr

from push_me.config import PushTPOConfig


def _last_or_zero(array: zarr.Array) -> int:
    return int(np.asarray(array[-1])) if array.shape[0] else 0


class ReplayBufferWriter:
    def __init__(self, path: str | Path, config: PushTPOConfig):
        self._root = zarr.open_group(str(path), mode="a", zarr_format=2)
        self._data = self._root.require_group("data")
        self._meta = self._root.require_group("meta")
        if "config" not in self._root.attrs:
            self._root.attrs["config"] = json.dumps(dataclasses.asdict(config))
        self._ensure_meta_array("episode_ends")
        self._ensure_meta_array("episode_seeds")

    def _ensure_meta_array(self, name: str) -> None:
        if name not in self._meta:
            self._meta.create_array(name, shape=(0,), chunks=(1024,), dtype="int64")

    @property
    def n_episodes(self) -> int:
        return self._meta.get_array("episode_ends").shape[0]

    @property
    def n_steps(self) -> int:
        return _last_or_zero(self._meta.get_array("episode_ends"))

    def append_episode(self, seed: int, steps: dict[str, np.ndarray]) -> None:
        lengths = {len(arr) for arr in steps.values()}
        if len(lengths) > 1:
            raise ValueError(f"all recorded fields must have the same episode length, got {lengths}")
        n_new = lengths.pop() if lengths else 0
        if n_new == 0:
            return

        for key, arr in steps.items():
            arr = np.asarray(arr)
            if key not in self._data:
                self._data.create_array(
                    key, shape=(0,) + arr.shape[1:], chunks=(1024,) + arr.shape[1:], dtype=arr.dtype
                )
            array = self._data.get_array(key)
            old_len = array.shape[0]
            array.resize((old_len + n_new,) + array.shape[1:])
            array[old_len : old_len + n_new] = arr

        ends = self._meta.get_array("episode_ends")
        prev_end = _last_or_zero(ends)
        ends.resize((ends.shape[0] + 1,))
        ends[-1] = prev_end + n_new

        seeds = self._meta.get_array("episode_seeds")
        seeds.resize((seeds.shape[0] + 1,))
        seeds[-1] = seed


def read_config(path: str | Path) -> PushTPOConfig:
    root = zarr.open_group(str(path), mode="r")
    return PushTPOConfig(**json.loads(cast(str, root.attrs["config"])))


def read_episode(path: str | Path, index: int) -> dict[str, np.ndarray]:
    root = zarr.open_group(str(path), mode="r")
    data = root.require_group("data")
    ends = np.asarray(root.require_group("meta").get_array("episode_ends")[:])
    start = int(ends[index - 1]) if index > 0 else 0
    end = int(ends[index])
    return {key: np.asarray(data.get_array(key)[start:end]) for key in data.array_keys()}


def summarize_achieved_modes(path: str | Path) -> dict[int, int]:
    root = zarr.open_group(str(path), mode="r")
    ends = np.asarray(root.require_group("meta").get_array("episode_ends")[:])
    achieved_mode = np.asarray(root.require_group("data").get_array("achieved_mode")[:])

    counts: Counter[int] = Counter()
    for end in ends:
        for mode in np.atleast_1d(achieved_mode[int(end) - 1]):
            counts[int(mode)] += 1
    return dict(counts)
