from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from push_me.geometry import min_area_rect, rect_corners, rects_overlap
from push_me.shapes import ShapeDef


@dataclass
class GoalRect:
    center: np.ndarray
    angle: float
    half_extents: np.ndarray
    accepts: set[str]


def make_goal_rect(shape: ShapeDef, margin: float, pose: tuple[float, float, float]) -> GoalRect:
    half_extents, phi = min_area_rect(shape.outline)
    gx, gy, gtheta = pose
    return GoalRect(
        center=np.array([gx, gy]),
        angle=gtheta + phi,
        half_extents=half_extents + margin,
        accepts={shape.name},
    )


def _within_arena(rect: GoalRect, arena_size: float) -> bool:
    corners = rect_corners(rect)
    return bool(np.all(corners >= 0) and np.all(corners <= arena_size))


def sample_goal_rects(
    rng: np.random.Generator,
    shapes: list[ShapeDef],
    margin: float,
    arena_size: float,
    max_attempts: int = 1000,
) -> list[GoalRect]:
    placed: list[GoalRect] = []
    for shape in shapes:
        for _attempt in range(max_attempts):
            pose = (
                float(rng.uniform(0, arena_size)),
                float(rng.uniform(0, arena_size)),
                float(rng.uniform(0, 2 * np.pi)),
            )
            candidate = make_goal_rect(shape, margin, pose)
            if not _within_arena(candidate, arena_size):
                continue
            if any(rects_overlap(candidate, other) for other in placed):
                continue
            placed.append(candidate)
            break
        else:
            raise RuntimeError(
                f"could not place goal rect for {shape.name!r} after {max_attempts} attempts "
                f"(arena_size={arena_size}, margin={margin}); try a smaller margin, "
                "fewer objects, or a larger arena"
            )
    return placed


def resolve_assignment(cost: np.ndarray, mode: str = "free") -> tuple[np.ndarray, np.ndarray]:
    k = cost.shape[0]
    if mode == "fixed":
        assignment = np.arange(k)
    elif mode == "free":
        row, col = linear_sum_assignment(cost)
        assignment = col[np.argsort(row)]
    else:
        raise ValueError(f"unknown assignment_mode: {mode!r}")
    errors = cost[np.arange(k), assignment]
    return assignment, errors
