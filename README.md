# PushT-PO

A partially-observable, multi-object, multi-modal-goal variant of [PushT](https://github.com/huggingface/gym-pusht), built as a `gymnasium.Env` on `pymunk` physics.

## What this adds over PushT

The original PushT task is: push one T-shaped block into one fixed target pose, fully observed. PushT-PO keeps the same physics and pushing mechanic but extends it along three axes that vanilla PushT doesn't have:

- **Goals are regions, not poses.** Success is "does the object's outline lie inside this rectangle," not "does it match this exact pose." The rectangle's size (`goal_margin`) and the object's rotational symmetry both mean many distinct end poses can satisfy the same goal — vanilla PushT has exactly one correct pose.
- **Multiple objects with free assignment.** `n_objects` can be > 1, and with `assignment_mode="free"` any object may land in any compatible goal rectangle — the environment resolves the best object-to-goal matching itself (Hungarian algorithm on containment error), rather than pinning object *i* to goal *i*.
- **Partial observability is a tunable dial, not on/off.** `obs_mode="lidar"` replaces full state with a ray-cast sensor of configurable range and count, so objects can go unseen for a controllable, measurable number of steps (`steps_since_observed` is logged directly). Optional interior `occluder_walls` push this further. Vanilla PushT is always fully observed.
- **Reward respects the multi-modality.** The dense reward is a minimum over valid goal assignments/orientations, never a distance to one canonical pose — so it doesn't secretly reward one arbitrary "correct" solution over the others.
- **Optional irreversibility.** `traps=True` adds concave wedges that can permanently trap a pushed object, for testing whether a policy walks into absorbing states it can't recover from.

## How multi-modality is generated

A goal is a rotated rectangle sized to an object's minimum-area bounding rectangle, inflated by a margin. An object satisfies its goal when its entire outline lies inside that rectangle. Three orthogonal dials fall out of this:

1. **Shape symmetry** — an object with rotational symmetry of order *n* has *n* distinct orientations that fit the same goal rectangle (a T-tetromino has 1, a hexagon has 6).
2. **Goal margin** — inflating the rectangle turns each mode from a near-point into a manifold of valid translations and rotations.
3. **Free assignment** — with *k* identical objects and `assignment_mode="free"`, any object may occupy any compatible rectangle, multiplying the mode count by *k!*.

## Install

Requires Python 3.13+. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Quick start

```python
import gymnasium as gym
import push_me  # registers the presets

env = gym.make("PushTPO-Lidar-Multi3-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

### Presets

| Preset | Config |
|---|---|
| `PushTPO-Full-Single-v0` | full obs, 1 T, margin 8 — closest to vanilla PushT |
| `PushTPO-Lidar-Single-v0` | lidar obs, 1 T, margin 8 — adds partial observability only |
| `PushTPO-Full-Multimodal-v0` | full obs, 1 hexagon, margin 24 — adds goal multi-modality only |
| `PushTPO-Lidar-Multimodal-v0` | lidar obs, 1 hexagon, margin 24 — both together |
| `PushTPO-Lidar-Multi3-v0` | lidar obs, 3 squares, free assignment, margin 24 — the dishwasher |
| `PushTPO-Lidar-Trap-v0` | as Multi3, plus traps — irreversibility |

### Manual play

```bash
uv run scripts/play.py --preset PushTPO-Lidar-Multi3-v0
```

Mouse teleoperates the pusher. `Space` pause/resume, `R` reset, `L` toggle lidar hits, `B` toggle belief overlay, `G` toggle ground-truth outlines, `Tab` cycle presets, `Esc` quit.

### Collect demonstrations

```bash
uv run scripts/collect_demos.py --preset PushTPO-Lidar-Multi3-v0 --output demos.zarr --n-episodes 50
```

### Benchmark throughput

```bash
uv run scripts/benchmark_speed.py
```

## Development

```bash
uv run pytest
```

All randomness is routed through a single seeded `np.random.Generator`; identical seed and action sequence must produce bit-identical trajectories (`tests/test_determinism.py`).

## License

MIT — see [LICENSE](LICENSE).
