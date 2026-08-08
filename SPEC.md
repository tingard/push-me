# PushT-PO — Environment Design Spec

A partially-observable, multi-object, multi-modal-goal variant of PushT.
Target audience: a coding agent implementing this from scratch.

---

## 1. Purpose

This environment exists to test two claims. Every design decision below serves one of them.

**Claim A — Goal sets beat goal images.**
A goal specified as a *region* admits many valid end states. A policy conditioned on a
single goal image must arbitrarily commit to one of them, and pays for it. We want an
environment where the number and size of valid end states are both tunable, so the
baseline's disadvantage can be plotted as a curve rather than asserted at a point.

**Claim B — A recurrent belief state matches frame-stacking, and keeps working past its horizon.**
Under partial observation, information must be retained across time. Frame stacking has a
fixed window; a recurrent belief does not. We want the required memory horizon to be a
first-class config parameter so performance can be plotted against it.

Two things follow from this that are easy to get wrong:

- The dense reward **must be a minimum over valid goal modes**, never a distance to one
  canonical pose. A reward that secretly prefers one mode destroys Claim A.
- Partial observability **must be parametric**, not binary. A single "occluded" flag gives
  one data point; a horizon knob gives a curve.

---

## 2. Core design: how multi-modality is generated

A goal is a **rotated rectangle** sized to the object's minimum-area bounding rectangle,
inflated by a margin. An object satisfies its goal when its entire outline lies inside
that rectangle.

This yields two orthogonal dials.

### Dial 1 — Mode count, via shape symmetry

If a shape has rotational symmetry of order *n*, then *n* distinct orientations fit the
same bounding rectangle. These are genuinely distinct end states that a goal image cannot
express simultaneously.

| Shape | Rotational symmetry order | Valid orientations per goal rect |
|---|---|---|
| T-tetromino | 1 | 1 |
| L-tetromino | 1 | 1 |
| S/Z-tetromino | 2 | 2 |
| Equilateral triangle | 3 | 3 |
| Square | 4 | 4 |
| Regular pentagon | 5 | 5 |
| Regular hexagon | 6 | 6 |
| Regular octagon | 8 | 8 |

Note the T is included deliberately as the *n = 1* control: it is the case where goal-image
conditioning loses nothing, and your method should show no advantage there. A benchmark
where your method wins everywhere is less convincing than one where it wins exactly where
theory says it should.

### Dial 2 — Mode size, via margin

`goal_margin` inflates the bounding rectangle. Small margin ⇒ each mode is a near-point.
Large margin ⇒ each mode is a manifold of valid translations and rotations.

Sweep this. The expected result is that goal-image baselines are fine at margin ≈ 0 and
degrade monotonically as margin grows, while a goal-set method stays flat.

### Dial 3 — Assignment, via multiple objects

With `n_objects = k` and `assignment_mode = "free"`, any object may occupy any
size-compatible goal rectangle. With *k* identical shapes this multiplies the mode count by
*k!*.

Total valid end states ≈ `k! × (symmetry_order)^k` for identical shapes under free
assignment. With 3 squares that is 6 × 64 = 384 distinct valid configurations from one
config line.

`assignment_mode = "fixed"` pairs object *i* with rectangle *i* (colour-matched in the
renderer) as an ablation that removes the permutation component while keeping the rotational one.

---

## 3. Physics and arena

- **Engine:** `pymunk` (Chipmunk2D), matching the original PushT so published baseline
  numbers stay comparable.
- **Arena:** square, side `arena_size` (default 512 units), enclosed by four static
  segment walls with `friction = 0.6`, `elasticity = 0.0`.
- **Timestep:** `dt = 1/100 s`, with `sim_substeps = 10` physics steps per environment
  step ⇒ control rate 10 Hz. Fixed, never variable.
- **Damping:** `space.damping = 0.05` (heavy — objects stop when not pushed; this is
  quasi-static pushing, not billiards).
- **Gravity:** zero. Top-down plane.

### Pusher (agent)

A circle body, radius `pusher_radius` (default 15), `friction = 0.6`. Kinematic-ish:
driven by a PD controller toward a commanded target, so it can be blocked by objects
rather than teleporting through them.

```
force = kp * (target_pos - body.position) - kd * body.velocity
```

Defaults `kp = 100.0`, `kd = 10.0`, force magnitude clipped to `max_push_force = 500`.

### Objects

Each object is a rigid body built from a **convex decomposition** of its outline (pymunk
requires convex shapes; concave outlines such as T, L, S, Z must be split into convex
pieces attached to the same body). Mass scaled so all shapes have equal density.

**Critical normalisation:** every shape is scaled to a common canonical **area**
(`shape_area`, default 4000 units²) before use. Without this, a hexagon and a T-tetromino
have wildly different push difficulty and results are uninterpretable across shapes.

---

## 4. Shape registry

Each shape is defined by an outline polygon in canonical orientation, centred on its
centroid.

```python
@dataclass(frozen=True)
class ShapeDef:
    name: str
    outline: np.ndarray  # (V, 2) float, CCW, centroid at origin
    symmetry_order: int  # rotational symmetry; drives mode count
    convex_parts: list[np.ndarray]  # convex decomposition for pymunk
```

Required entries: `t_tetromino`, `l_tetromino`, `s_tetromino`, `z_tetromino`,
`triangle`, `square`, `pentagon`, `hexagon`, `octagon`.

Registry API:

```python
SHAPES: dict[str, ShapeDef]
def make_shape(name: str, area: float) -> ShapeDef   # returns area-normalised copy
```

Implementation notes:

- Tetrominoes are built from unit cells then scaled to `area`.
- Regular polygons generated from `n` vertices on a circle, then scaled to `area`.
- `symmetry_order` must be **verified numerically at import time**, not hardcoded by hand:
  rotate the outline by `2π/n` and assert the vertex set matches within tolerance. This
  catches an entire class of silent benchmark bugs.
- Convex decomposition: use ear-clipping plus merging, or hand-specify the parts for the
  eight tetromino-like shapes (they are simple enough — a T is two rectangles).

---

## 5. Goal specification

### Construction

For object with outline `O` and pose `(x, y, θ)`:

1. Compute `minAreaRect(O)` in canonical orientation via rotating calipers ⇒ half-extents
   `(hw, hh)` and the rectangle's own orientation offset `φ`.
2. Inflate: `(hw + margin, hh + margin)`.
3. Sample a goal pose `(gx, gy, gθ)` uniformly in the arena, rejecting placements that
   overlap other goal rectangles or walls.

```python
@dataclass
class GoalRect:
    center: np.ndarray  # (2,)
    angle: float  # radians
    half_extents: np.ndarray  # (2,) already inflated
    accepts: set[str]  # shape names this rect is sized for
```

### Containment test

Transform every vertex of the object's current outline into the rectangle's local frame
and check all lie within the half-extents:

```python
def contains(rect: GoalRect, outline_world: np.ndarray) -> bool:
    c, s = np.cos(-rect.angle), np.sin(-rect.angle)
    R = np.array([[c, -s], [s, c]])
    local = (outline_world - rect.center) @ R.T
    return bool(np.all(np.abs(local) <= rect.half_extents))
```

Exact, cheap, and works for concave outlines because the rectangle is convex.

### Containment margin (for dense reward)

Signed penetration depth — how far outside the rectangle the worst vertex sits:

```python
def containment_error(rect, outline_world) -> float:
    local = transform_to_rect_frame(outline_world, rect)
    excess = np.abs(local) - rect.half_extents
    return float(np.max(excess))  # <= 0 means contained
```

---

## 6. Reward

```
r_t = success_bonus * is_success   -   dense_weight * sum_i min_assignment E_i
```

where `E_i` is `containment_error` for object *i* under the **best available assignment**.

**Assignment resolution matters.** With `assignment_mode = "free"` and *k* objects, compute
the *k × k* cost matrix of containment errors and solve with the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`) to get the minimum-cost matching. Do **not**
greedily match, and do **not** fix the pairing — a fixed pairing silently collapses the
factorial mode structure and invalidates Claim A.

Rotational modes need no explicit enumeration: the containment test is already invariant
to them, since all *n* symmetric orientations produce identical containment error.

Defaults: `success_bonus = 1.0`, `dense_weight = 0.01`, and a `sparse_only` flag for the
harder setting.

Episode terminates on success held for `success_hold_steps = 10` consecutive steps
(prevents credit for transient fly-throughs), or on `max_steps` (default 1000).

---

## 7. Observation

Two modes, selected by `obs_mode`.

### `obs_mode = "full"`

```
[pusher_x, pusher_y, pusher_vx, pusher_vy,
 for each object: [x, y, cos θ, sin θ, vx, vy, ω, shape_onehot...],
 for each goal rect: [cx, cy, cos α, sin α, hw, hh]]
```

All positions normalised to `[-1, 1]` by `arena_size`.

### `obs_mode = "lidar"`

The agent carries a ray sensor. This is the partially-observed setting.

- `n_rays` rays (default 128), uniformly spaced in `[0, 2π)`, **fixed in world frame**
  (not pusher-relative — avoids conflating rotation-equivariance with memory).
- Range `lidar_range` (default 300 units). **This is the primary memory-horizon knob.**
- Each ray cast with `space.segment_query_first`; returns first hit only, so objects
  occlude one another naturally.

Per ray: `[normalised_distance, hit_class_onehot]` where `hit_class ∈ {none, wall, object}`.
`normalised_distance = 1.0` when nothing is hit within range.

**Goal rectangles are NOT sensed by lidar.** They are abstract task specification, not
physical objects, and are appended to the observation in full:

```
[pusher_x, pusher_y, pusher_vx, pusher_vy,
 ray_features (n_rays × 4),
 for each goal rect: [cx, cy, cos α, sin α, hw, hh, accepts_onehot...]]
```

This is the right split for a belief-based method: the *goal* is given, the *world state*
must be inferred.

### Memory horizon

Required horizon is governed by the ratio `arena_size / lidar_range` together with
`n_objects`. Small range plus several objects means the agent cannot see all objects at
once and must retain their poses while attending elsewhere.

Provide a derived, logged diagnostic — `mean_steps_since_last_observed` per object,
measured over a random-policy rollout — so the horizon can be reported empirically rather
than inferred from config. This is the x-axis of the Claim B plot.

Optionally set `occluder_walls = k` to add *k* static interior wall segments, which raises
the horizon further at fixed lidar range.

---

## 8. Action space

`Box(low=-1, high=1, shape=(2,))`.

- `action_mode = "absolute"` (default, matches original PushT): action maps linearly to a
  target position in the arena.
- `action_mode = "delta"`: action is a displacement from the current pusher position,
  scaled by `max_delta` (default 30 units).

Action chunking is a policy-side concern, not an environment concern — do not bake it in.

---

## 9. Irreversibility (optional)

`traps = True` adds `n_traps` static concave wedge geometries. An object pushed into a
wedge cannot be extracted by a circular pusher. This provides a direct test of whether a
least-action planner walks into an absorbing state, since the trap sits on a low-cost path.

Log `n_objects_trapped` at episode end as a separate metric. It should not be folded into
the reward — the point is to observe whether the method avoids traps unprompted.

---

## 10. Configuration

```python
@dataclass
class PushTPOConfig:
    # task structure
    n_objects: int = 1
    shapes: list[str] = field(default_factory=lambda: ["t_tetromino"])
    shape_sampling: str = "fixed"  # "fixed" | "random"
    assignment_mode: str = "free"  # "free" | "fixed"
    goal_margin: float = 8.0  # Dial 2

    # observability
    obs_mode: str = "lidar"  # "lidar" | "full"
    n_rays: int = 128
    lidar_range: float = 300.0  # Dial: memory horizon
    occluder_walls: int = 0

    # arena and physics
    arena_size: float = 512.0
    shape_area: float = 4000.0
    pusher_radius: float = 15.0
    sim_substeps: int = 10
    dt: float = 0.01
    damping: float = 0.05

    # control
    action_mode: str = "absolute"  # "absolute" | "delta"
    max_delta: float = 30.0
    kp: float = 100.0
    kd: float = 10.0
    max_push_force: float = 500.0

    # episode
    max_steps: int = 1000
    success_hold_steps: int = 10
    sparse_only: bool = False
    success_bonus: float = 1.0
    dense_weight: float = 0.01

    # hazards
    traps: bool = False
    n_traps: int = 0

    seed: int | None = None
```

Register standard variants as named presets so experiments are reproducible by name:

| Preset | Config |
|---|---|
| `PushTPO-Full-Single-v0` | full obs, 1 T, margin 8 — sanity baseline |
| `PushTPO-Lidar-Single-v0` | lidar, 1 T, margin 8 — tests Claim B only |
| `PushTPO-Full-Multimodal-v0` | full obs, 1 hexagon, margin 24 — tests Claim A only |
| `PushTPO-Lidar-Multimodal-v0` | lidar, 1 hexagon, margin 24 — both |
| `PushTPO-Lidar-Multi3-v0` | lidar, 3 squares, free assignment, margin 24 — the dishwasher |
| `PushTPO-Lidar-Trap-v0` | as Multi3 plus traps |

The 2 × 2 of the first four is the factorial design. Run it, or claims cannot be attributed.

---

## 11. Gymnasium API

Standard `gymnasium.Env`:

```python
env = PushTPOEnv(config)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

`info` must always include, for logging and probing:

```python
{
    "is_success": bool,
    "containment_errors": np.ndarray,  # (k,) per object, best assignment
    "assignment": np.ndarray,  # (k,) object -> rect index
    "object_poses": np.ndarray,  # (k, 3) GROUND TRUTH — probing only
    "steps_since_observed": np.ndarray,  # (k,) for horizon diagnostics
    "achieved_mode": np.ndarray,  # (k,) rotational mode index 0..n-1
    "n_objects_trapped": int,
}
```

`object_poses` is ground truth and **must never enter the observation** in lidar mode. It
exists so the belief-probe experiment (linear decode of true pose from the learned belief
vector, as a function of `steps_since_observed`) can be run. Guard it behind an assertion
in the training loop.

`achieved_mode` is computed as `round(θ_object_relative_to_rect / (2π / symmetry_order))`
and is what mode-diversity entropy is calculated from.

---

## 12. Pygame visualiser

`render_mode ∈ {None, "human", "rgb_array"}`. `None` must involve zero pygame imports so
headless vectorised training is fast.

Layout: arena at 512×512, plus a 240px right-hand panel for diagnostics.

Draw order:

1. Arena background, walls, traps (dark hatched).
2. **Goal rectangles** — outline only, 2px, colour-coded per rect; semi-transparent fill
   that turns green when currently satisfied.
3. **Objects** — filled polygons, single neutral colour, independent of which goal rect
   they're currently matched to. Which specific box an object lands in never mattered for
   success (the assignment in §6 is resolved after the fact, order-agnostically); colouring
   objects to imply an object→box binding was misleading, not informative, and unreadable
   for an operator who can't distinguish the goal-rect palette.
4. **Lidar hits** — a small dot at each ray's hit point, colour by hit class
   (grey = miss, blue = wall, orange = object). Drawn as points rather than lines from the
   pusher so the higher-resolution ray fan (128 rays by default) doesn't turn into visual
   clutter. Toggle with `L`.
5. **Pusher** — circle, plus a faint line to its commanded target.
6. **Belief overlay hook** (see below). Toggle with `B`.
7. Panel: step count, reward, per-object containment error bars, assignment table,
   `steps_since_observed` per object, achieved mode indices.

### Belief overlay hook

Leave an explicit, documented injection point:

```python
env.set_belief_overlay(fn)  # fn: () -> list[BeliefMarker] | None


@dataclass
class BeliefMarker:
    pose: np.ndarray  # (3,) x, y, theta — predicted object pose
    shape_name: str
    confidence: float = 1.0  # drives alpha
    label: str = ""
```

Rendered as dashed outlines at the believed poses, alpha by confidence. Being able to watch
a believed pose drift away from ground truth while an object is unobserved is the single
most useful debugging affordance in this whole environment. Build it early.

### Keyboard

| Key | Action |
|---|---|
| `Space` | pause / resume |
| `R` | reset episode |
| `L` | toggle lidar hits |
| `B` | toggle belief overlay (off by default) |
| `G` | toggle ground-truth object outlines (off by default, to avoid biasing teleop demonstrations with information the lidar policy can't see; on to sanity-check against ground truth) |
| `Tab` | cycle presets |
| `Esc` | quit |

---

## 13. Demonstration collection

`scripts/collect_demos.py` — mouse position drives the pusher target directly, matching
the original PushT teleop so operators behave comparably.

Per episode record: config, seed, and per-step `(obs, action, info)`. Store as a single
`zarr` or HDF5 store with episode boundary indices, following the Diffusion Policy dataset
layout so existing dataloaders work unmodified.

Two operator-facing controls that matter:

- `N` discards the current episode (operators make mistakes; unflagged failures poison BC).
- Record **which mode the operator chose**, from `info["achieved_mode"]` and
  `info["assignment"]`. Operator mode preference is a confound: if humans always pick mode
  0, a goal-image baseline trained on those demos looks artificially good. Report the
  demonstration mode distribution alongside results.

Collect demos in `obs_mode = "full"` while also recording the lidar observation, so the
same demonstrations serve both observability conditions. This requires computing both
observations each step — cheap, and it removes an entire confound from the Claim B
comparison.

---

## 14. Determinism and vectorisation

- All randomness through a single `np.random.Generator` seeded from `reset(seed=...)`.
  No global `random` or `np.random` calls anywhere.
- Given identical seed and action sequence, trajectories must be bit-identical. Add this
  as a test.
- Provide `make_vec_env(config, n_envs)` over `gymnasium.vector.AsyncVectorEnv`. Target
  ≥ 2000 env-steps/sec/core with `render_mode=None`, `n_rays=64`, `n_objects=1`. If
  ray-casting dominates, batch the queries or reduce `n_rays` — do not silently drop
  substeps, which changes the physics.

---

## 15. File layout

```
pusht_po/
├── __init__.py           # gymnasium registration of presets
├── config.py             # PushTPOConfig
├── shapes.py             # ShapeDef, SHAPES registry, symmetry verification
├── geometry.py           # minAreaRect, containment, transforms, convex decomposition
├── goals.py              # GoalRect, sampling, Hungarian assignment
├── env.py                # PushTPOEnv
├── lidar.py              # ray casting
├── render.py             # pygame visualiser, BeliefMarker
├── vec.py                # make_vec_env
└── tests/
    ├── test_shapes.py         # symmetry orders verified numerically
    ├── test_containment.py    # known-answer cases, rotated rects
    ├── test_assignment.py     # Hungarian matches brute force for k <= 5
    ├── test_determinism.py    # seed replay is bit-identical
    └── test_modes.py          # all n symmetric orientations report contained
scripts/
├── collect_demos.py
├── play.py               # keyboard/mouse manual play
└── benchmark_speed.py
```

---

## 16. Acceptance tests

The environment is done when:

1. `test_modes.py` passes: for a hexagon at margin 24, all 6 symmetric orientations are
   reported contained with equal containment error.
2. `test_assignment.py` passes: Hungarian assignment matches brute-force permutation search
   for *k* ≤ 5.
3. A human can reliably solve `PushTPO-Lidar-Multi3-v0` via `play.py` — if a human cannot
   solve it with the same observations, the task is broken, not hard.
4. Seed replay is bit-identical.
5. `benchmark_speed.py` reports ≥ 2000 steps/sec/core headless.
6. The belief overlay renders dashed outlines from an injected dummy function that simply
   returns ground-truth poses plus Gaussian noise.

---

## 17. Deliberately out of scope

Not the environment's job; keep it out so the environment stays a fair referee:

- Any policy, belief model, or training loop.
- Action chunking, frame stacking, observation history — all wrappers, all policy-side.
- Reward shaping beyond the min-over-assignment containment term.
- Curriculum or auto-difficulty. Sweeps are run externally over explicit configs.
