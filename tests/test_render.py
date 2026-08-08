from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from push_me.config import PushTPOConfig
from push_me.env import PushTPOEnv
from push_me.geometry import min_area_rect
from push_me.render import BeliefMarker, Renderer


def _renderer(env: PushTPOEnv) -> Renderer:
    assert env._renderer is not None
    return env._renderer


def _enable_ground_truth_and_belief(env: PushTPOEnv) -> None:
    # the canonical snapshot scenes are meant to show ground truth + belief overlay
    # regardless of the teleop-facing default (both off, to avoid biasing demo collection)
    env._ensure_renderer()
    renderer = _renderer(env)
    renderer.show_ground_truth = True
    renderer.show_belief = True


_REPO_ROOT_PYTHON_CHECK = """
import sys
from push_me.config import PushTPOConfig
from push_me.env import PushTPOEnv
env = PushTPOEnv(PushTPOConfig(n_objects=1, obs_mode="full"))
env.reset(seed=0)
env.step(env.action_space.sample())
env.close()
assert "pygame" not in sys.modules, "pygame was imported despite render_mode=None"
print("OK")
"""


def test_render_mode_none_never_imports_pygame():
    result = subprocess.run(
        [sys.executable, "-c", _REPO_ROOT_PYTHON_CHECK],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_invalid_render_mode_raises():
    with pytest.raises(ValueError):
        PushTPOEnv(PushTPOConfig(n_objects=1, obs_mode="full"), render_mode="bogus")


def test_rgb_array_frame_shape_and_dtype():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (512, 752, 3)
    assert frame.dtype == np.uint8
    env.close()


def test_human_mode_runs_headless_via_sdl_dummy_driver(sdl_dummy_driver):
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="human")
    env.reset(seed=0)
    for _ in range(3):
        env.step(env.action_space.sample())
    env.close()


def test_belief_overlay_changes_the_rendered_frame():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.reset(seed=0)
    frame_without = env.render()

    def belief_fn():
        body = env._object_bodies[0]
        pose = np.array([body.position.x + 40, body.position.y + 40, body.angle])
        return [
            BeliefMarker(
                pose=pose, shape_name=env._object_shapes[0].name, confidence=1.0
            )
        ]

    env.set_belief_overlay(belief_fn)
    _renderer(env).show_belief = True
    frame_with = env.render()
    assert frame_without is not None and frame_with is not None
    assert not np.array_equal(frame_without, frame_with)
    env.close()


def test_toggling_ground_truth_on_changes_the_frame():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.reset(seed=0)
    frame_off = env.render()
    _renderer(env).show_ground_truth = True
    frame_on = env.render()
    assert frame_on is not None and frame_off is not None
    assert not np.array_equal(frame_on, frame_off)
    env.close()


def test_traps_and_occluders_render_without_crashing():
    cfg = PushTPOConfig(
        n_objects=2,
        shapes=["square"],
        obs_mode="lidar",
        n_rays=16,
        seed=0,
        traps=True,
        n_traps=1,
        occluder_walls=2,
    )
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    frame = env.render()
    assert frame is not None
    assert frame.shape == (512, 752, 3)
    env.close()


def test_set_belief_overlay_before_reset_does_not_crash():
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.set_belief_overlay(lambda: None)
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (512, 752, 3)
    env.close()


def test_close_is_idempotent():
    cfg = PushTPOConfig(n_objects=1, obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.reset(seed=0)
    env.render()
    env.close()
    env.close()


# ---- canonical snapshot scenes (SPEC.md acceptance test #6 is scene 5) ----


def test_snapshot_baseline_reset(assert_matches_snapshot, sdl_dummy_driver):
    cfg = PushTPOConfig(n_objects=1, shapes=["t_tetromino"], obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.reset(seed=0)
    _enable_ground_truth_and_belief(env)
    frame = env.render()
    assert_matches_snapshot("01_baseline_reset", frame)
    env.close()


def test_snapshot_lidar_rays_with_occlusion(assert_matches_snapshot, sdl_dummy_driver):
    cfg = PushTPOConfig(
        n_objects=1, shapes=["square"], obs_mode="lidar", n_rays=32, seed=1
    )
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.reset(seed=1)
    assert env._pusher_body is not None
    _enable_ground_truth_and_belief(env)

    # place the object well within lidar_range so a ray actually hits it (orange) --
    # a random reset placement isn't guaranteed to be in range, and the point of
    # this scene is to show the object-occlusion hit class, not just walls/misses.
    pusher_pos = np.array(env._pusher_body.position)
    env._object_bodies[0].position = tuple(pusher_pos + np.array([80.0, 0.0]))
    env._object_bodies[0].velocity = (0, 0)

    env.step(np.array([0.0, 0.0], dtype=np.float32))
    frame = env.render()
    assert_matches_snapshot("02_lidar_rays", frame)
    env.close()


def test_snapshot_satisfied_goal_turns_green(assert_matches_snapshot, sdl_dummy_driver):
    cfg = PushTPOConfig(n_objects=1, shapes=["square"], obs_mode="full", seed=2)
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.reset(seed=2)
    _enable_ground_truth_and_belief(env)

    rect = env._goal_rects[0]
    shape = env._object_shapes[0]
    _half_extents, phi = min_area_rect(shape.outline)
    body = env._object_bodies[0]
    body.position = tuple(rect.center)
    body.angle = rect.angle - phi
    body.velocity = (0, 0)
    env.step(np.array([0.0, 0.0], dtype=np.float32))

    frame = env.render()
    assert_matches_snapshot("03_satisfied_goal", frame)
    env.close()


def test_snapshot_multi_object_scene(assert_matches_snapshot, sdl_dummy_driver):
    cfg = PushTPOConfig(
        n_objects=3, shapes=["square"], obs_mode="lidar", n_rays=32, seed=3
    )
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.reset(seed=3)
    _enable_ground_truth_and_belief(env)
    for i in range(15):
        env.step(np.array([np.sin(i * 0.3), np.cos(i * 0.3)], dtype=np.float32))
    frame = env.render()
    assert_matches_snapshot("04_multi_object", frame)
    env.close()


def test_snapshot_belief_overlay_ground_truth_plus_noise(
    assert_matches_snapshot, sdl_dummy_driver
):
    cfg = PushTPOConfig(n_objects=2, shapes=["square"], obs_mode="full", seed=4)
    env = PushTPOEnv(cfg, render_mode="rgb_array")
    env.reset(seed=4)

    def belief_fn():
        rng = np.random.default_rng(123)
        markers = []
        for i, body in enumerate(env._object_bodies):
            dx, dy = rng.normal(0, 5, size=2)
            dtheta = rng.normal(0, 0.1)
            pose = np.array(
                [body.position.x + dx, body.position.y + dy, body.angle + dtheta]
            )
            markers.append(
                BeliefMarker(
                    pose=pose,
                    shape_name=env._object_shapes[i].name,
                    confidence=0.7,
                    label=f"b{i}",
                )
            )
        return markers

    env.set_belief_overlay(belief_fn)
    _enable_ground_truth_and_belief(env)
    frame = env.render()
    assert_matches_snapshot("05_belief_overlay", frame)
    env.close()


# ---- keyboard controls (SPEC.md section 12's key table) ----


def _press(key: int) -> None:
    import pygame

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))


def test_space_toggles_pause(sdl_dummy_driver):
    import pygame

    cfg = PushTPOConfig(n_objects=1, obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="human")
    env.reset(seed=0)
    assert _renderer(env).paused is False
    _press(pygame.K_SPACE)
    env.render()
    assert _renderer(env).paused is True
    _press(pygame.K_SPACE)
    env.render()
    assert _renderer(env).paused is False
    env.close()


def test_l_toggles_lidar_rays(sdl_dummy_driver):
    import pygame

    cfg = PushTPOConfig(n_objects=1, obs_mode="lidar", n_rays=16, seed=0)
    env = PushTPOEnv(cfg, render_mode="human")
    env.reset(seed=0)
    assert _renderer(env).show_lidar is True
    _press(pygame.K_l)
    env.render()
    assert _renderer(env).show_lidar is False
    env.close()


def test_b_toggles_belief_overlay(sdl_dummy_driver):
    import pygame

    cfg = PushTPOConfig(n_objects=1, obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="human")
    env.reset(seed=0)
    assert _renderer(env).show_belief is False
    _press(pygame.K_b)
    env.render()
    assert _renderer(env).show_belief is True
    env.close()


def test_g_toggles_ground_truth(sdl_dummy_driver):
    import pygame

    cfg = PushTPOConfig(n_objects=1, obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="human")
    env.reset(seed=0)
    assert _renderer(env).show_ground_truth is False
    _press(pygame.K_g)
    env.render()
    assert _renderer(env).show_ground_truth is True
    env.close()


def test_r_resets_the_episode(sdl_dummy_driver):
    import pygame

    cfg = PushTPOConfig(n_objects=1, obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="human")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    assert env._step_count == 2
    _press(pygame.K_r)
    env.render()
    assert env._step_count == 0
    assert _renderer(env).paused is True
    env.close()


def test_tab_sets_preset_cycle_request_for_play_py_to_consume(sdl_dummy_driver):
    import pygame

    cfg = PushTPOConfig(n_objects=1, obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="human")
    env.reset(seed=0)
    assert _renderer(env).consume_preset_cycle_request() is False
    _press(pygame.K_TAB)
    env.render()
    assert _renderer(env).consume_preset_cycle_request() is True
    assert _renderer(env).consume_preset_cycle_request() is False
    env.close()


def test_unrecognised_keys_are_queued_for_external_consumers(sdl_dummy_driver):
    import pygame

    cfg = PushTPOConfig(n_objects=1, obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="human")
    env.reset(seed=0)
    assert _renderer(env).consume_unhandled_keys() == []
    _press(pygame.K_n)
    env.render()
    assert _renderer(env).consume_unhandled_keys() == [pygame.K_n]
    assert _renderer(env).consume_unhandled_keys() == []
    env.close()


def test_recognised_keys_are_not_queued_as_unhandled(sdl_dummy_driver):
    import pygame

    cfg = PushTPOConfig(n_objects=1, obs_mode="lidar", n_rays=16, seed=0)
    env = PushTPOEnv(cfg, render_mode="human")
    env.reset(seed=0)
    _press(pygame.K_l)
    env.render()
    assert _renderer(env).show_lidar is False
    assert _renderer(env).consume_unhandled_keys() == []
    env.close()


def test_escape_closes_the_renderer(sdl_dummy_driver):
    import pygame

    cfg = PushTPOConfig(n_objects=1, obs_mode="full", seed=0)
    env = PushTPOEnv(cfg, render_mode="human")
    env.reset(seed=0)
    _press(pygame.K_ESCAPE)
    env.render()
    assert env._renderer is None
