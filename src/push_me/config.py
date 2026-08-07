from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PushTPOConfig:
    n_objects: int = 1
    shapes: list[str] = field(default_factory=lambda: ["t_tetromino"])
    shape_sampling: str = "fixed"
    assignment_mode: str = "free"
    goal_margin: float = 8.0

    obs_mode: str = "lidar"
    n_rays: int = 64
    lidar_range: float = 150.0
    occluder_walls: int = 0

    arena_size: float = 512.0
    shape_area: float = 4000.0
    pusher_radius: float = 15.0
    sim_substeps: int = 10
    dt: float = 0.01
    damping: float = 0.05

    action_mode: str = "absolute"
    max_delta: float = 30.0
    kp: float = 100.0
    kd: float = 10.0
    max_push_force: float = 500.0

    max_steps: int = 300
    success_hold_steps: int = 10
    sparse_only: bool = False
    success_bonus: float = 1.0
    dense_weight: float = 0.01

    traps: bool = False
    n_traps: int = 0

    seed: int | None = None
