from __future__ import annotations

import numpy as np

from push_me.render import ARENA_PIXELS


def mouse_to_action(mouse_px: tuple[int, int], arena_size: float) -> np.ndarray:
    mx = float(np.clip(mouse_px[0], 0, ARENA_PIXELS))
    my = float(np.clip(mouse_px[1], 0, ARENA_PIXELS))
    wx = mx / ARENA_PIXELS * arena_size
    wy = (ARENA_PIXELS - my) / ARENA_PIXELS * arena_size
    action = np.array([wx / arena_size * 2.0 - 1.0, wy / arena_size * 2.0 - 1.0], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)
