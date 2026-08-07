from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.spatial import ConvexHull


class RectLike(Protocol):
    center: np.ndarray
    angle: float
    half_extents: np.ndarray


def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s], [s, c]])
    return points @ r.T


def _convex_hull(outline: np.ndarray) -> np.ndarray:
    if len(outline) <= 3:
        return outline
    hull = ConvexHull(outline)
    return outline[hull.vertices]


def min_area_rect(outline: np.ndarray) -> tuple[np.ndarray, float]:
    hull = _convex_hull(outline)
    n = len(hull)
    if n == 0:
        raise ValueError("outline must have at least one point")
    best_area = np.inf
    best_half_extents: np.ndarray | None = None
    best_angle = 0.0
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        angle = float(np.arctan2(edge[1], edge[0]))
        local = _rotate(hull, -angle)
        extents = local.max(axis=0) - local.min(axis=0)
        area = extents[0] * extents[1]
        if area < best_area:
            best_area = area
            best_half_extents = extents / 2.0
            best_angle = angle
    assert best_half_extents is not None
    return best_half_extents, best_angle


def transform_to_rect_frame(outline_world: np.ndarray, rect: RectLike) -> np.ndarray:
    c, s = np.cos(-rect.angle), np.sin(-rect.angle)
    r = np.array([[c, -s], [s, c]])
    return (outline_world - rect.center) @ r.T


def contains(rect: RectLike, outline_world: np.ndarray) -> bool:
    local = transform_to_rect_frame(outline_world, rect)
    return bool(np.all(np.abs(local) <= rect.half_extents))


def containment_error(rect: RectLike, outline_world: np.ndarray) -> float:
    local = transform_to_rect_frame(outline_world, rect)
    excess = np.abs(local) - rect.half_extents
    return float(np.max(excess))


def rect_corners(rect: RectLike) -> np.ndarray:
    hw, hh = rect.half_extents
    local_corners = np.array([[hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh]])
    c, s = np.cos(rect.angle), np.sin(rect.angle)
    r = np.array([[c, -s], [s, c]])
    return local_corners @ r.T + rect.center


def rects_overlap(rect_a: RectLike, rect_b: RectLike) -> bool:
    corners_a = rect_corners(rect_a)
    corners_b = rect_corners(rect_b)
    axes = [
        np.array([np.cos(rect_a.angle), np.sin(rect_a.angle)]),
        np.array([-np.sin(rect_a.angle), np.cos(rect_a.angle)]),
        np.array([np.cos(rect_b.angle), np.sin(rect_b.angle)]),
        np.array([-np.sin(rect_b.angle), np.cos(rect_b.angle)]),
    ]
    for axis in axes:
        proj_a = corners_a @ axis
        proj_b = corners_b @ axis
        if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
            return False
    return True
