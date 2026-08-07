from __future__ import annotations

from enum import IntEnum

import numpy as np
import pymunk


class HitClass(IntEnum):
    NONE = 0
    WALL = 1
    OBJECT = 2


N_HIT_CLASSES = len(HitClass)


def cast_rays(
    space: pymunk.Space,
    origin: tuple[float, float],
    n_rays: int,
    lidar_range: float,
    shape_filter: pymunk.ShapeFilter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if shape_filter is None:
        shape_filter = pymunk.ShapeFilter()

    angles = 2 * np.pi * np.arange(n_rays) / n_rays
    features = np.zeros((n_rays, 1 + N_HIT_CLASSES), dtype=float)
    hit_object_index = np.full(n_rays, -1, dtype=int)
    ox, oy = origin

    for i, angle in enumerate(angles):
        end = (ox + lidar_range * np.cos(angle), oy + lidar_range * np.sin(angle))
        hit = space.segment_query_first((ox, oy), end, 0.0, shape_filter)
        if hit is None:
            features[i, 0] = 1.0
            features[i, 1 + HitClass.NONE] = 1.0
        else:
            hit_class = HitClass(hit.shape.collision_type)
            features[i, 0] = hit.alpha
            features[i, 1 + hit_class] = 1.0
            if hit_class == HitClass.OBJECT:
                hit_object_index[i] = hit.shape.user_data

    return features, hit_object_index
