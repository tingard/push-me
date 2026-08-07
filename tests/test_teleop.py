from __future__ import annotations

import pytest

from push_me.render import ARENA_PIXELS
from push_me.teleop import mouse_to_action


def test_mouse_to_action_maps_corners():
    arena_size = 512.0
    # pixel origin (top-left) is world (0, arena_size) after the y-flip
    assert mouse_to_action((0, 0), arena_size) == pytest.approx([-1.0, 1.0])
    assert mouse_to_action((ARENA_PIXELS, ARENA_PIXELS), arena_size) == pytest.approx([1.0, -1.0])
    assert mouse_to_action((ARENA_PIXELS // 2, ARENA_PIXELS // 2), arena_size) == pytest.approx(
        [0.0, 0.0], abs=1e-2
    )


def test_mouse_to_action_scales_with_arena_size():
    action = mouse_to_action((ARENA_PIXELS, 0), arena_size=256.0)
    assert action == pytest.approx([1.0, 1.0])


def test_mouse_to_action_clips_positions_outside_the_arena_panel_region():
    action = mouse_to_action((ARENA_PIXELS + 100, -50), arena_size=512.0)
    assert action[0] == pytest.approx(1.0)
    assert action[1] == pytest.approx(1.0)
    assert action.min() >= -1.0 and action.max() <= 1.0
