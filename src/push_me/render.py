from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pygame

from push_me.geometry import rect_corners
from push_me.lidar import HitClass
from push_me.shapes import make_shape

ARENA_PIXELS = 512
PANEL_WIDTH = 240
WINDOW_SIZE = (ARENA_PIXELS + PANEL_WIDTH, ARENA_PIXELS)
FPS = 30

_ARENA_BG = (30, 30, 35)
_PANEL_BG = (18, 18, 22)
_PANEL_TEXT = (230, 230, 230)
_WALL_COLOR = (200, 200, 210)
_TRAP_COLOR = (150, 40, 40)
_PUSHER_COLOR = (240, 240, 240)
_TARGET_LINE_COLOR = (140, 140, 140)
_BELIEF_COLOR = (255, 210, 60)
_OBJECT_COLOR = (190, 190, 190)
_LIDAR_COLORS = {
    HitClass.NONE: (110, 110, 110),
    HitClass.WALL: (80, 140, 240),
    HitClass.OBJECT: (240, 150, 60),
}
_RECT_PALETTE = [
    (231, 76, 60),
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (149, 165, 166),
]


@dataclass
class BeliefMarker:
    pose: np.ndarray
    shape_name: str
    confidence: float = 1.0
    label: str = ""


def _rect_color(i: int) -> tuple[int, int, int]:
    return _RECT_PALETTE[i % len(_RECT_PALETTE)]


class Renderer:
    def __init__(self, env, render_mode: str):
        self.env = env
        self.render_mode = render_mode

        pygame.font.init()
        if render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode(WINDOW_SIZE)
            pygame.display.set_caption("PushT-PO")
        else:
            self.screen = pygame.Surface(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 13)

        self.show_lidar = True
        self.show_belief = False
        self.show_ground_truth = False
        self.paused = False
        self.preset_cycle_requested = False
        self.close_requested = False
        self._unhandled_keys: list[int] = []
        self._belief_overlay_fn: Callable[[], list[BeliefMarker] | None] | None = None

    def set_belief_overlay(self, fn: Callable[[], list[BeliefMarker] | None] | None) -> None:
        self._belief_overlay_fn = fn

    def consume_preset_cycle_request(self) -> bool:
        requested, self.preset_cycle_requested = self.preset_cycle_requested, False
        return requested

    def consume_unhandled_keys(self) -> list[int]:
        keys, self._unhandled_keys = self._unhandled_keys, []
        return keys

    def close(self) -> None:
        if self.render_mode == "human":
            pygame.display.quit()
        pygame.font.quit()

    # ---- coordinate transform: world [0, arena_size]^2 -> pixels [0, ARENA_PIXELS]^2, y flipped ----

    def _scale(self) -> float:
        return ARENA_PIXELS / self.env.config.arena_size

    def _to_px(self, world_xy: np.ndarray) -> np.ndarray:
        world_xy = np.asarray(world_xy, dtype=np.float64)
        s = self._scale()
        px = world_xy[..., 0] * s
        py = ARENA_PIXELS - world_xy[..., 1] * s
        return np.stack([px, py], axis=-1)

    # ---- top-level render ----

    def render(self, last_reward: float | None) -> np.ndarray | None:
        if self.render_mode == "human":
            self._handle_events()
            if self.close_requested:
                return None
            if self.paused:
                self.clock.tick(FPS)
                return None

        self._draw_background()
        self._draw_walls_and_occluders()
        self._draw_traps()
        self._draw_goal_rects()
        if self.show_ground_truth:
            self._draw_objects()
        if self.show_lidar:
            self._draw_lidar_hits()
        self._draw_pusher()
        if self.show_belief:
            self._draw_belief_overlay()
        self._draw_panel(last_reward)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
            return None
        return self._surface_to_array()

    def _surface_to_array(self) -> np.ndarray:
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    # ---- events / keyboard ----

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close_requested = True
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)

    def _handle_keydown(self, key: int) -> None:
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.env.reset()
            # pause on reset: the mouse is still wherever it was for the previous
            # rollout, and stepping immediately would drive the pusher there
            self.paused = True
        elif key == pygame.K_l:
            self.show_lidar = not self.show_lidar
        elif key == pygame.K_b:
            self.show_belief = not self.show_belief
        elif key == pygame.K_g:
            self.show_ground_truth = not self.show_ground_truth
        elif key == pygame.K_TAB:
            self.preset_cycle_requested = True
        elif key == pygame.K_ESCAPE:
            self.close_requested = True
        else:
            self._unhandled_keys.append(key)

    # ---- draw order (SPEC.md section 12) ----

    def _draw_background(self) -> None:
        self.screen.fill(_PANEL_BG)
        self.screen.fill(_ARENA_BG, pygame.Rect(0, 0, ARENA_PIXELS, ARENA_PIXELS))

    def _draw_walls_and_occluders(self) -> None:
        for seg in self.env._wall_segments + self.env._occluder_segments:
            a = self._to_px(np.array(seg.a))
            b = self._to_px(np.array(seg.b))
            pygame.draw.line(self.screen, _WALL_COLOR, tuple(a), tuple(b), 3)

    def _draw_traps(self) -> None:
        env = self.env
        for seg in env._trap_wall_segments:
            a = self._to_px(np.array(seg.a))
            b = self._to_px(np.array(seg.b))
            pygame.draw.line(self.screen, _TRAP_COLOR, tuple(a), tuple(b), 3)
        for trap in env._traps:
            corners_px = self._to_px(rect_corners(trap))
            overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            pygame.draw.polygon(overlay, (*_TRAP_COLOR, 100), [tuple(p) for p in corners_px])
            self.screen.blit(overlay, (0, 0))

    def _draw_goal_rects(self) -> None:
        env = self.env
        info = env._last_info
        rect_colors = [_rect_color(i) for i in range(len(env._goal_rects))]

        # the outlines below are fair game unconditionally -- goal rects are given directly
        # in the observation (SPEC.md §7) -- but which rect currently holds a correctly
        # placed object is not observed in either obs_mode (containment/assignment are only
        # ever computed into `info`, never `obs`), so gate the reveal behind show_ground_truth
        # the same as the object outlines, or it leaks an unfair success cue during teleop
        satisfied_rects = set()
        if self.show_ground_truth and info is not None:
            assignment = info["assignment"]
            errors = info["containment_errors"]
            for obj_i, rect_i in enumerate(assignment):
                if errors[obj_i] <= 0:
                    satisfied_rects.add(int(rect_i))

        for i, rect in enumerate(env._goal_rects):
            corners_px = [tuple(p) for p in self._to_px(rect_corners(rect))]
            fill = (60, 200, 110, 110) if i in satisfied_rects else (*rect_colors[i], 55)
            overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            pygame.draw.polygon(overlay, fill, corners_px)
            self.screen.blit(overlay, (0, 0))
            pygame.draw.polygon(self.screen, rect_colors[i], corners_px, width=2)

    def _draw_objects(self) -> None:
        # not coloured to match an assigned goal rect -- which specific box an object
        # ends up in doesn't matter for success (assignment is resolved after the fact,
        # order-agnostically), so implying an object->box binding via colour is noise
        env = self.env
        for i in range(env.config.n_objects):
            outline_px = [tuple(p) for p in self._to_px(env._object_outline_world(i))]
            pygame.draw.polygon(self.screen, _OBJECT_COLOR, outline_px)

    def _draw_lidar_hits(self) -> None:
        env = self.env
        if env.config.obs_mode != "lidar" or env._lidar_features is None:
            return
        n_rays = env.config.n_rays
        lidar_range = env.config.lidar_range
        origin_world = np.array(env._pusher_body.position)
        angles = 2 * np.pi * np.arange(n_rays) / n_rays
        for i, angle in enumerate(angles):
            dist_norm = env._lidar_features[i, 0]
            hit_class = HitClass(int(np.argmax(env._lidar_features[i, 1:])))
            hit_world = origin_world + dist_norm * lidar_range * np.array([np.cos(angle), np.sin(angle)])
            hit_px = tuple(self._to_px(hit_world))
            pygame.draw.circle(self.screen, _LIDAR_COLORS[hit_class], hit_px, 2)

    def _draw_pusher(self) -> None:
        env = self.env
        pos_px = tuple(self._to_px(np.array(env._pusher_body.position)))
        radius_px = env.config.pusher_radius * self._scale()
        pygame.draw.circle(self.screen, _PUSHER_COLOR, pos_px, radius_px, width=2)
        target_px = tuple(self._to_px(env._target))
        pygame.draw.line(self.screen, _TARGET_LINE_COLOR, pos_px, target_px, 1)

    def _draw_belief_overlay(self) -> None:
        if self._belief_overlay_fn is None:
            return
        markers = self._belief_overlay_fn()
        if not markers:
            return
        for marker in markers:
            shape = make_shape(marker.shape_name, self.env.config.shape_area)
            x, y, theta = marker.pose
            c, s = np.cos(theta), np.sin(theta)
            r_mat = np.array([[c, -s], [s, c]])
            outline_world = shape.outline @ r_mat.T + np.array([x, y])
            corners_px = self._to_px(outline_world)
            alpha = int(np.clip(marker.confidence, 0.0, 1.0) * 255)
            self._draw_dashed_polygon(corners_px, (*_BELIEF_COLOR, alpha))
            if marker.label:
                label_px = tuple(self._to_px(np.array([x, y])))
                text = self.font.render(marker.label, True, _BELIEF_COLOR)
                self.screen.blit(text, label_px)

    def _draw_dashed_polygon(self, points_px: np.ndarray, color: tuple[int, int, int, int]) -> None:
        overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
        n = len(points_px)
        for i in range(n):
            self._draw_dashed_line(overlay, points_px[i], points_px[(i + 1) % n], color)
        self.screen.blit(overlay, (0, 0))

    def _draw_dashed_line(
        self, surface: pygame.Surface, a: np.ndarray, b: np.ndarray, color, dash_len: float = 6, gap_len: float = 4
    ) -> None:
        length = float(np.linalg.norm(b - a))
        if length == 0:
            return
        direction = (b - a) / length
        pos = 0.0
        drawing = True
        while pos < length:
            seg_end = min(pos + (dash_len if drawing else gap_len), length)
            if drawing:
                pygame.draw.line(surface, color, tuple(a + direction * pos), tuple(a + direction * seg_end), 2)
            pos = seg_end
            drawing = not drawing

    def _draw_panel(self, last_reward: float | None) -> None:
        env = self.env
        pygame.draw.rect(self.screen, _PANEL_BG, pygame.Rect(ARENA_PIXELS, 0, PANEL_WIDTH, ARENA_PIXELS))
        x0 = ARENA_PIXELS + 10
        y = 10

        def line(text: str, color=_PANEL_TEXT) -> None:
            nonlocal y
            self.screen.blit(self.font.render(text, True, color), (x0, y))
            y += 16

        line(f"step: {env._step_count}")
        line(f"reward: {last_reward:+.3f}" if last_reward is not None else "reward: -")

        info = env._last_info
        if info is None:
            return

        y += 6
        line(f"success: {info['is_success']}")
        y += 6
        line("containment error:")
        for i, e in enumerate(info["containment_errors"]):
            line(f"  obj{i}: {e:+.1f}")
            bar_w = int(np.clip(abs(e), 0, PANEL_WIDTH - 40))
            bar_color = (200, 70, 70) if e > 0 else (70, 200, 110)
            pygame.draw.rect(self.screen, bar_color, pygame.Rect(x0 + 10, y, bar_w, 6))
            y += 10
        y += 6
        line("assignment (obj -> rect):")
        for i, r in enumerate(info["assignment"]):
            line(f"  {i} -> {r}")
        y += 6
        line("steps since observed:")
        for i, s in enumerate(info["steps_since_observed"]):
            line(f"  obj{i}: {s}")
        y += 6
        line("achieved mode:")
        for i, m in enumerate(info["achieved_mode"]):
            line(f"  obj{i}: {m}")
        if env.config.traps:
            y += 6
            line(f"trapped: {info['n_objects_trapped']}")
