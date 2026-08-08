# PushT-PO

A partially-observable, multi-object, multi-modal-goal variant of [PushT](https://diffusion-policy.cs.columbia.edu/), built as a `gymnasium.Env` on `pymunk` physics.

## Why this exists

PushT-PO exists to test two specific claims, not to be a general-purpose benchmark:

**Claim A — goal sets beat goal images.** A goal specified as a *region* admits many valid end states. A policy conditioned on a single goal image must arbitrarily commit to one of them, and pays for it. Most pushing benchmarks bake in a single canonical target pose, which makes this claim untestable — there's nothing to be ambiguous about. Here, the number and size of valid end states are both tunable config parameters, so the goal-image baseline's disadvantage can be plotted as a curve instead of asserted at a single point.

**Claim B — a recurrent belief state matches frame-stacking, and keeps working past its horizon.** Under partial observation, information about unseen objects must be retained across time. Frame stacking has a fixed window; a recurrent belief doesn't. Rather than a single "occluded: yes/no" flag, the required memory horizon (lidar range vs. arena size, object count, optional occluder walls) is a first-class, continuously tunable knob, so performance can be plotted against it.

Two things follow that are easy to get wrong and are treated as invariants throughout the codebase:

- The dense reward is a **minimum over valid goal modes**, never a distance to one canonical pose — a reward that secretly prefers one mode would quietly destroy Claim A.
- Partial observability is **parametric** (a range knob, occluder count), not a binary flag — a single data point can't be swept into a curve.

See [`SPEC.md`](SPEC.md) for the full design rationale and implementation spec.

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

| Preset | Config | Tests |
|---|---|---|
| `PushTPO-Full-Single-v0` | full obs, 1 T, margin 8 | sanity baseline |
| `PushTPO-Lidar-Single-v0` | lidar, 1 T, margin 8 | Claim B only |
| `PushTPO-Full-Multimodal-v0` | full obs, 1 hexagon, margin 24 | Claim A only |
| `PushTPO-Lidar-Multimodal-v0` | lidar, 1 hexagon, margin 24 | both |
| `PushTPO-Lidar-Multi3-v0` | lidar, 3 squares, free assignment, margin 24 | the dishwasher |
| `PushTPO-Lidar-Trap-v0` | as Multi3, plus traps | irreversibility |

The first four form a 2×2 factorial design (obs mode × margin) — run all four together, or claims about which one wins can't be attributed to a cause.

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
