from __future__ import annotations

import copy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pymunk
from gymnasium import spaces

from push_me.config import PushTPOConfig
from push_me.geometry import RectLike, containment_error, min_area_rect, rect_corners, rects_overlap, contains
from push_me.goals import GoalRect, resolve_assignment, sample_goal_rects
from push_me.lidar import N_HIT_CLASSES, HitClass, cast_rays
from push_me.shapes import SHAPES, ShapeDef, make_shape

_SHAPE_NAMES = sorted(SHAPES)
_N_SHAPES = len(_SHAPE_NAMES)
_SHAPE_INDEX = {name: i for i, name in enumerate(_SHAPE_NAMES)}

_FRICTION = 0.6
_ELASTICITY = 0.0
# calibrated against kp/kd/max_push_force/damping so max-force pusher crosses the arena in ~40 steps, not ~21000 (density=1.0 was unusably heavy)
_OBJECT_DENSITY = 0.002

_CATEGORY_WALL = 0b001
_CATEGORY_OBJECT = 0b010
_CATEGORY_PUSHER = 0b100


@dataclass
class _Pose:
    center: np.ndarray
    angle: float
    half_extents: np.ndarray


def _polygon_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _one_hot(index: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[index] = 1.0
    return v


def _normalise_position(pos: np.ndarray, arena_size: float) -> np.ndarray:
    return np.asarray(pos, dtype=np.float32) / arena_size * 2.0 - 1.0


def _make_object_body(shape: ShapeDef, density: float) -> tuple[pymunk.Body, list[pymunk.Poly]]:
    total_mass = 0.0
    total_moment = 0.0
    for part in shape.convex_parts:
        mass = density * _polygon_area(part)
        total_mass += mass
        total_moment += pymunk.moment_for_poly(mass, part.tolist())
    body = pymunk.Body(total_mass, total_moment)
    polys = [pymunk.Poly(body, part.tolist()) for part in shape.convex_parts]
    return body, polys


class PushTPOEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: PushTPOConfig, render_mode: str | None = None):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unknown render_mode: {render_mode!r}")
        self.config = copy.deepcopy(config)
        self.render_mode = render_mode
        self._renderer = None
        self._last_reward: float | None = None
        self._last_info: dict | None = None
        self._rng = np.random.default_rng(config.seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = self._build_observation_space()

        self._space: pymunk.Space | None = None
        self._pusher_body: pymunk.Body | None = None
        self._object_bodies: list[pymunk.Body] = []
        self._object_shapes: list[ShapeDef] = []
        self._object_min_rects: list[tuple[np.ndarray, float]] = []
        self._goal_rects: list[GoalRect] = []
        self._traps: list[_Pose] = []
        self._wall_segments: list[pymunk.Segment] = []
        self._occluder_segments: list[pymunk.Segment] = []
        self._trap_wall_segments: list[pymunk.Segment] = []
        self._target = np.zeros(2, dtype=np.float64)
        self._lidar_features: np.ndarray | None = None
        self._steps_since_observed = np.zeros(config.n_objects, dtype=np.int64)
        self._step_count = 0
        self._success_streak = 0

    # ---- gymnasium API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.config

        self._space = pymunk.Space()
        self._space.damping = cfg.damping
        self._space.gravity = (0.0, 0.0)

        self._build_walls()
        shape_names = self._resolve_shape_names()
        self._object_shapes = [make_shape(name, cfg.shape_area) for name in shape_names]
        self._object_min_rects = [min_area_rect(shape.outline) for shape in self._object_shapes]
        self._goal_rects = sample_goal_rects(self._rng, self._object_shapes, cfg.goal_margin, cfg.arena_size)

        pusher_pose = self._build_pusher()
        taken: list[RectLike] = [*self._goal_rects, pusher_pose]
        self._object_bodies = self._build_objects(taken=taken)
        self._traps = self._build_traps() if cfg.traps else []
        self._build_occluder_walls()

        assert self._pusher_body is not None
        self._target = np.array(self._pusher_body.position, dtype=np.float64)
        self._step_count = 0
        self._success_streak = 0
        self._last_reward = None
        self._steps_since_observed = np.zeros(cfg.n_objects, dtype=np.int64)
        self._refresh_perception()

        obs = self._compute_observation()
        self._last_info = self._compute_info()

        if self.render_mode == "human":
            self.render()

        return obs, self._last_info

    def step(self, action):
        assert self._space is not None
        assert self._pusher_body is not None
        cfg = self.config
        action = np.asarray(action, dtype=np.float64)
        if cfg.action_mode == "absolute":
            self._target = (action + 1.0) / 2.0 * cfg.arena_size
        elif cfg.action_mode == "delta":
            self._target = np.array(self._pusher_body.position) + action * cfg.max_delta
        else:
            raise ValueError(f"unknown action_mode: {cfg.action_mode!r}")

        for _ in range(cfg.sim_substeps):
            self._apply_pd_force()
            self._space.step(cfg.dt)

        self._step_count += 1
        self._refresh_perception()
        obs = self._compute_observation()
        info = self._compute_info()
        self._last_info = info

        is_success = bool(info["is_success"])
        self._success_streak = self._success_streak + 1 if is_success else 0
        terminated = self._success_streak >= cfg.success_hold_steps
        truncated = self._step_count >= cfg.max_steps

        dense_term = 0.0 if cfg.sparse_only else cfg.dense_weight * float(np.sum(info["containment_errors"]))
        reward = cfg.success_bonus * float(is_success) - dense_term
        self._last_reward = reward

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        self._ensure_renderer()
        assert self._renderer is not None
        result = self._renderer.render(self._last_reward)
        if self._renderer.close_requested:
            self.close()
        return result

    def set_belief_overlay(self, fn) -> None:
        self._ensure_renderer()
        assert self._renderer is not None
        self._renderer.set_belief_overlay(fn)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _ensure_renderer(self) -> None:
        if self._renderer is None:
            if self.render_mode is None:
                raise ValueError("cannot render/set a belief overlay when render_mode is None")
            from push_me.render import Renderer

            self._renderer = Renderer(self, self.render_mode)

    # ---- construction ----

    def _build_observation_space(self) -> spaces.Box:
        cfg = self.config
        if cfg.obs_mode == "full":
            dim = 4 + cfg.n_objects * (7 + _N_SHAPES) + cfg.n_objects * 6
        elif cfg.obs_mode == "lidar":
            dim = 4 + cfg.n_rays * (1 + N_HIT_CLASSES) + cfg.n_objects * (6 + _N_SHAPES)
        else:
            raise ValueError(f"unknown obs_mode: {cfg.obs_mode!r}")
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def _resolve_shape_names(self) -> list[str]:
        cfg = self.config
        if cfg.shape_sampling == "fixed":
            names = cfg.shapes
            if len(names) == cfg.n_objects:
                return list(names)
            return [names[i % len(names)] for i in range(cfg.n_objects)]
        elif cfg.shape_sampling == "random":
            indices = self._rng.integers(0, len(cfg.shapes), size=cfg.n_objects)
            return [cfg.shapes[i] for i in indices]
        raise ValueError(f"unknown shape_sampling: {cfg.shape_sampling!r}")

    def _build_walls(self) -> None:
        assert self._space is not None
        cfg = self.config
        s = cfg.arena_size
        corners = [(0, 0), (s, 0), (s, s), (0, s)]
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        segs = []
        for i in range(4):
            seg = pymunk.Segment(body, corners[i], corners[(i + 1) % 4], radius=0.0)
            seg.friction = _FRICTION
            seg.elasticity = _ELASTICITY
            seg.collision_type = HitClass.WALL
            seg.filter = pymunk.ShapeFilter(categories=_CATEGORY_WALL)
            segs.append(seg)
        self._space.add(body, *segs)
        self._wall_segments = segs

    def _build_occluder_walls(self) -> None:
        assert self._space is not None
        cfg = self.config
        self._occluder_segments = []
        for _ in range(cfg.occluder_walls):
            cx = self._rng.uniform(0, cfg.arena_size)
            cy = self._rng.uniform(0, cfg.arena_size)
            angle = self._rng.uniform(0, 2 * np.pi)
            half_len = 0.075 * cfg.arena_size
            d = np.array([np.cos(angle), np.sin(angle)]) * half_len
            a = tuple(np.array([cx, cy]) - d)
            b = tuple(np.array([cx, cy]) + d)
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            seg = pymunk.Segment(body, a, b, radius=0.0)
            seg.friction = _FRICTION
            seg.elasticity = _ELASTICITY
            seg.collision_type = HitClass.WALL
            seg.filter = pymunk.ShapeFilter(categories=_CATEGORY_WALL)
            self._space.add(body, seg)
            self._occluder_segments.append(seg)

    def _sample_free_pose(
        self, half_extents: np.ndarray, taken: list[RectLike], max_attempts: int = 1000
    ) -> _Pose:
        cfg = self.config
        for _ in range(max_attempts):
            candidate = _Pose(
                center=np.array([self._rng.uniform(0, cfg.arena_size), self._rng.uniform(0, cfg.arena_size)]),
                angle=self._rng.uniform(0, 2 * np.pi),
                half_extents=half_extents,
            )
            corners = rect_corners(candidate)
            if not (np.all(corners >= 0) and np.all(corners <= cfg.arena_size)):
                continue
            if any(rects_overlap(candidate, other) for other in taken):
                continue
            return candidate
        raise RuntimeError(
            f"could not find a free placement after {max_attempts} attempts "
            f"(arena_size={cfg.arena_size}); try fewer objects/traps or a smaller margin"
        )

    def _build_pusher(self) -> _Pose:
        assert self._space is not None
        cfg = self.config
        r = cfg.pusher_radius
        pose = self._sample_free_pose(np.array([r, r]), taken=list(self._goal_rects))
        mass = _OBJECT_DENSITY * np.pi * r * r
        moment = pymunk.moment_for_circle(mass, 0.0, r)
        body = pymunk.Body(mass, moment)
        body.position = tuple(pose.center)
        shape = pymunk.Circle(body, r)
        shape.friction = _FRICTION
        shape.elasticity = _ELASTICITY
        shape.filter = pymunk.ShapeFilter(categories=_CATEGORY_PUSHER)
        self._space.add(body, shape)
        self._pusher_body = body
        return pose

    def _build_objects(self, taken: list[RectLike]) -> list[pymunk.Body]:
        assert self._space is not None
        bodies = []
        for idx, shape in enumerate(self._object_shapes):
            half_extents, _phi = self._object_min_rects[idx]
            pose = self._sample_free_pose(half_extents, taken)
            body, polys = _make_object_body(shape, _OBJECT_DENSITY)
            body.position = tuple(pose.center)
            body.angle = pose.angle
            for poly in polys:
                poly.friction = _FRICTION
                poly.elasticity = _ELASTICITY
                poly.collision_type = HitClass.OBJECT
                poly.user_data = idx
                poly.filter = pymunk.ShapeFilter(categories=_CATEGORY_OBJECT)
            self._space.add(body, *polys)
            taken.append(pose)
            bodies.append(body)
        return bodies

    def _build_traps(self) -> list[_Pose]:
        assert self._space is not None
        cfg = self.config
        r = cfg.pusher_radius
        w_out, w_in, depth = 3.0 * r, 0.6 * r, 2.0 * r
        w_pocket, pocket_depth = 1.5 * r, 2.0 * r
        half_h = (depth + pocket_depth) / 2.0
        throat_y = half_h - depth

        local_segments = [
            ((-w_out, half_h), (-w_in, throat_y)),
            ((w_out, half_h), (w_in, throat_y)),
            ((-w_in, throat_y), (-w_pocket, -half_h)),
            ((w_in, throat_y), (w_pocket, -half_h)),
            ((-w_pocket, -half_h), (w_pocket, -half_h)),
        ]
        pocket_local_center = np.array([0.0, (throat_y - half_h) / 2.0])
        pocket_local_half_extents = np.array([w_pocket, (throat_y + half_h) / 2.0])
        bbox_half_extents = np.array([max(w_out, w_pocket), half_h])

        taken: list[RectLike] = list(self._goal_rects)
        traps = []
        self._trap_wall_segments = []
        for _ in range(cfg.n_traps):
            pose = self._sample_free_pose(bbox_half_extents, taken)
            c, s = np.cos(pose.angle), np.sin(pose.angle)
            r_mat = np.array([[c, -s], [s, c]])

            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            segs = []
            for a, b in local_segments:
                wa = tuple(np.array(a) @ r_mat.T + pose.center)
                wb = tuple(np.array(b) @ r_mat.T + pose.center)
                seg = pymunk.Segment(body, wa, wb, radius=0.0)
                seg.friction = _FRICTION
                seg.elasticity = _ELASTICITY
                seg.collision_type = HitClass.WALL
                seg.filter = pymunk.ShapeFilter(categories=_CATEGORY_WALL)
                segs.append(seg)
            self._space.add(body, *segs)
            self._trap_wall_segments.extend(segs)

            pocket_center = pocket_local_center @ r_mat.T + pose.center
            traps.append(_Pose(center=pocket_center, angle=pose.angle, half_extents=pocket_local_half_extents))
            taken.append(pose)
        return traps

    # ---- physics stepping ----

    def _apply_pd_force(self) -> None:
        cfg = self.config
        body = self._pusher_body
        assert body is not None
        pos = np.array(body.position)
        vel = np.array(body.velocity)
        force = cfg.kp * (self._target - pos) - cfg.kd * vel
        norm = float(np.linalg.norm(force))
        if norm > cfg.max_push_force:
            force = force / norm * cfg.max_push_force
        body.force = tuple(force)

    # ---- perception ----

    def _cast_lidar(self) -> tuple[np.ndarray, np.ndarray]:
        assert self._pusher_body is not None
        assert self._space is not None
        cfg = self.config
        origin = tuple(self._pusher_body.position)
        query_filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ _CATEGORY_PUSHER)
        return cast_rays(self._space, origin, cfg.n_rays, cfg.lidar_range, query_filter)

    def _refresh_perception(self) -> None:
        cfg = self.config
        if cfg.obs_mode == "lidar":
            self._lidar_features, hit_object_index = self._cast_lidar()
            self._steps_since_observed += 1
            seen = hit_object_index[hit_object_index >= 0]
            self._steps_since_observed[seen] = 0
        else:
            self._lidar_features = None
            self._steps_since_observed[:] = 0

    def _object_outline_world(self, i: int) -> np.ndarray:
        shape = self._object_shapes[i]
        body = self._object_bodies[i]
        c, s = np.cos(body.angle), np.sin(body.angle)
        r_mat = np.array([[c, -s], [s, c]])
        return shape.outline @ r_mat.T + np.array(body.position)

    # ---- observation ----

    def _compute_observation(self) -> np.ndarray:
        if self.config.obs_mode == "full":
            return self._full_observation()
        return self._lidar_observation()

    def compute_observation(self, obs_mode: str) -> np.ndarray:
        # lets demo collection record both obs modes per step regardless of self.config.obs_mode
        if obs_mode == "full":
            return self._full_observation()
        if obs_mode == "lidar":
            if self.config.obs_mode == "lidar":
                return self._lidar_observation()
            features, _hit_object_index = self._cast_lidar()
            return self._lidar_observation(features)
        raise ValueError(f"unknown obs_mode: {obs_mode!r}")

    def _pusher_state(self) -> np.ndarray:
        assert self._pusher_body is not None
        cfg = self.config
        pos = _normalise_position(np.array(self._pusher_body.position), cfg.arena_size)
        vel = np.array(self._pusher_body.velocity, dtype=np.float32)
        return np.concatenate([pos, vel]).astype(np.float32)

    def _object_full_features(self, i: int) -> np.ndarray:
        cfg = self.config
        body = self._object_bodies[i]
        pos = _normalise_position(np.array(body.position), cfg.arena_size)
        theta = body.angle
        vel = np.array(body.velocity, dtype=np.float32)
        omega = np.array([body.angular_velocity], dtype=np.float32)
        onehot = _one_hot(_SHAPE_INDEX[self._object_shapes[i].name], _N_SHAPES)
        return np.concatenate([pos, [np.cos(theta), np.sin(theta)], vel, omega, onehot]).astype(np.float32)

    def _goal_rect_full_features(self, rect: GoalRect) -> np.ndarray:
        cfg = self.config
        center = _normalise_position(rect.center, cfg.arena_size)
        return np.concatenate(
            [center, [np.cos(rect.angle), np.sin(rect.angle)], rect.half_extents]
        ).astype(np.float32)

    def _full_observation(self) -> np.ndarray:
        cfg = self.config
        parts = [self._pusher_state()]
        parts += [self._object_full_features(i) for i in range(cfg.n_objects)]
        parts += [self._goal_rect_full_features(rect) for rect in self._goal_rects]
        return np.concatenate(parts).astype(np.float32)

    def _goal_rect_lidar_features(self, rect: GoalRect) -> np.ndarray:
        cfg = self.config
        center = _normalise_position(rect.center, cfg.arena_size)
        accepts_onehot = np.zeros(_N_SHAPES, dtype=np.float32)
        for name in rect.accepts:
            accepts_onehot[_SHAPE_INDEX[name]] = 1.0
        return np.concatenate(
            [center, [np.cos(rect.angle), np.sin(rect.angle)], rect.half_extents, accepts_onehot]
        ).astype(np.float32)

    def _lidar_observation(self, lidar_features: np.ndarray | None = None) -> np.ndarray:
        features = self._lidar_features if lidar_features is None else lidar_features
        assert features is not None
        parts = [self._pusher_state(), features.flatten().astype(np.float32)]
        parts += [self._goal_rect_lidar_features(rect) for rect in self._goal_rects]
        return np.concatenate(parts).astype(np.float32)

    # ---- reward / info ----

    def _achieved_mode(self, obj_idx: int, rect_idx: int) -> int:
        shape = self._object_shapes[obj_idx]
        rect = self._goal_rects[rect_idx]
        _half_extents, phi = self._object_min_rects[obj_idx]
        theta_obj = self._object_bodies[obj_idx].angle
        theta_rel = theta_obj - (rect.angle - phi)
        n = shape.symmetry_order
        step = 2 * np.pi / n
        return int(round(np.mod(theta_rel, 2 * np.pi) / step)) % n

    def _count_trapped_objects(self) -> int:
        if not self._traps:
            return 0
        count = 0
        for body in self._object_bodies:
            pos = np.array(body.position)[None, :]
            if any(contains(trap_rect, pos) for trap_rect in self._traps):
                count += 1
        return count

    def _compute_info(self) -> dict:
        cfg = self.config
        outlines_world = [self._object_outline_world(i) for i in range(cfg.n_objects)]
        cost = np.array(
            [[containment_error(rect, outline) for rect in self._goal_rects] for outline in outlines_world]
        )
        assignment, errors = resolve_assignment(cost, mode=cfg.assignment_mode)
        is_success = bool(np.all(errors <= 0))
        achieved_mode = np.array(
            [self._achieved_mode(i, int(assignment[i])) for i in range(cfg.n_objects)]
        )
        object_poses = np.array(
            [[*self._object_bodies[i].position, self._object_bodies[i].angle] for i in range(cfg.n_objects)]
        )
        return {
            "is_success": is_success,
            "containment_errors": errors,
            "assignment": assignment,
            "object_poses": object_poses,
            "steps_since_observed": self._steps_since_observed.copy(),
            "achieved_mode": achieved_mode,
            "n_objects_trapped": self._count_trapped_objects(),
        }
